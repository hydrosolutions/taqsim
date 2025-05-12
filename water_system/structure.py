"""
This module defines the various types of nodes that can exist in a water system simulation.

The module includes a base Node class and several specialized node types such as:
- SupplyNode: Represents a water supply source in the system.
- SinkNode: Represents a point where water exits the system with minimum flow requirements.
- DemandNode: Represents a point of water demand in the system.
- StorageNode: Represents a reservoir or storage facility in the system.
- HydroWorks: Represents a node that redistributes water using fixed distribution parameters.

Each node type has its own behavior for handling water inflows and outflows.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Union

class TimeSeriesImport:
    """
    Base class for importing and managing time series data from CSV files.
    Provides common functionality for nodes that need to import time-based data.
    """
    
    def _initialize_time_series(
        self,
        id: str,
        csv_file: Optional[str] = None,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
        num_time_steps: int = 0,
        column_name: str = 'Q'
    ) -> List[float]:
        """
        Initialize time series data from CSV file.
        
        Args:
            id (str): Node identifier for error messages
            csv_file (str): Path to CSV file
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            column_name (str): Name of the column in CSV to import
            
        Returns:
            list: Time series data from CSV, or None if import failed
        """
        # If all CSV parameters are provided, try to import data
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            try:
                ts_data = self.import_time_series(csv_file, start_year, start_month, 
                                                num_time_steps, column_name)
                
                # Check if data is valid
                if not (ts_data.empty or 
                    ts_data['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(ts_data[column_name]) < num_time_steps):
                    return ts_data[column_name].tolist()
                
                # Print warning for invalid data
                print(f"Warning: Insufficient data in CSV file for node '{id}'")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not ts_data.empty:
                    print(f"Available data range: {ts_data['Date'].min():%Y-%m} to {ts_data['Date'].max():%Y-%m}")   
                
            except Exception as e:
                raise ValueError(f"Warning: Time series data import failed for node '{id}': {str(e)}")
            
        return [0.0] * num_time_steps  # Default to zero if import fails or is invalid


    def import_time_series(
        self,
        csv_file: str,
        start_year: int,
        start_month: int,
        num_time_steps: int,
        column_name: str
    ) -> pd.DataFrame:
        """
        Import time series data from a CSV file for a specified time period.
        
        Args:
            csv_file (str): Path to the CSV file 
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            column_name (str): Name of the column to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            time_series = pd.read_csv(csv_file, parse_dates=['Date'])
            
            if 'Date' not in time_series.columns or column_name not in time_series.columns:
                raise ValueError(f"CSV file must contain 'Date' and '{column_name}' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return time_series[(time_series['Date'] >= start_date) & (time_series['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import time series data: {str(e)}")

class Node:
    """
    Base class for all types of nodes in the water system.

    Attributes:
        id (str): A unique identifier for the node.
        inflow_edges (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflow_edges (dict): A dictionary of outflow edges, keyed by the target node's id.
        easting (float): The easting coordinate of the node.
        northing (float): The northing coordinate of the node.
    """

    def __init__(self, id: str, easting: Optional[float] = None, northing: Optional[float] = None) -> None:
        """
        Initialize a Node object.

        Args:
            id (str): A unique identifier for the node.
            easting (float, optional): The easting coordinate of the node.
            northing (float, optional): The northing coordinate of the node.

        Raises:
            ValueError: If id is empty or coordinates are invalid.
        """
        if not id or not isinstance(id, str):
            raise ValueError(f"Invalid node ID: {id}")
        
        if easting is None or northing is None:
            raise ValueError(f"Missing coordinate value for node {id}: easting={easting}, northing={northing}")
        if not isinstance(easting, (int, float)) or not isinstance(northing, (int, float)):
            raise ValueError(f"Invalid coordinate type for node {id}: easting={easting}, northing={northing}")


        self.id = id
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.outflow_edges = {}  # Dictionary of outflow edges
        self.easting = easting # easting coordinate of the node. Defaults to None.
        self.northing = northing # northing coordinate of the node. Defaults to None.

    def add_inflow_edge(self, edge):
        """
        Add an inflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflow_edges[edge.source.id] = edge

    def add_outflow_edge(self, edge):
        """
        Add an outflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an outflow.
        """
        self.outflow_edges[edge.target.id] = edge

    def update(self, time_step: int, dt: float):
        """
        Update the node's state for the given time step.

        This method should be overridden by subclasses to implement
        specific behavior for each node type.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): Number of seconds in time step.
        """
        pass

class SupplyNode(Node, TimeSeriesImport):
    """
    Represents a water supply source in the system.

    Attributes:
        supply_rates (list): A list of supply rates for each time step.
        default_supply_rate (float): The default supply rate if not specified for a time step.
        supply_history (list): A record of actual supply amounts for each time step.
    """

    def __init__(
        self,
        id: str,
        constant_supply_rate: Optional[float] = None,
        easting: Optional[float] = None,
        northing: Optional[float] = None,
        csv_file: Optional[str] = None,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
        num_time_steps: int = 0
    ) -> None:
        """
        Initialize a SupplyNode object.

        Args:
            id (str): A unique identifier for the node.
            supply_rates (list, optional): A list of supply rates for each time step. Defaults to None.
            default_supply_rate (float, optional): The default supply rate. Defaults to 0.
            easting (float, optional): The easting coordinate of the node. Defaults to None.
            northing (float, optional): The northing coordinate of the node. Defaults to None.
            csv_file (str, optional): Path to CSV file containing supply data. Defaults to None.
            start_year (int, optional): Starting year for CSV data import. Defaults to None.
            start_month (int, optional): Starting month (1-12) for CSV data import. Defaults to None.
            num_time_steps (int, optional): Number of time steps to import from CSV. Defaults to None.
        """
        super().__init__(id, easting, northing)
        self.supply_history = []

        # Try to import time series data first
        imported_data = None
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            imported_data = self._initialize_time_series(
                id, csv_file, start_year, start_month, num_time_steps, 'Q'
            )
        
        # Decision tree for setting supply_rates
        if imported_data is not None:
            self.supply_rates = imported_data
        elif constant_supply_rate is not None:
            self.supply_rates = [constant_supply_rate]*num_time_steps    
        else:
            self.supply_rates = [0]*num_time_steps

    def update(self, time_step: int, dt: float) -> None:
        """
        Update the SupplyNode's state for the given time step.

        This method calculates the current supply rate and distributes it among outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            current_supply_rate = self.supply_rates[time_step]
            self.supply_history.append(current_supply_rate)

            total_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
            if total_capacity > 0:
                edge_flow = current_supply_rate / total_capacity
                for edge in self.outflow_edges.values():
                    edge.update(time_step, edge.capacity * edge_flow)
            else:
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
        except Exception as e:
            raise ValueError(f"Failed to update supply node {self.id}: {str(e)}")

class SinkNode(Node, TimeSeriesImport):
    """
    Represents a point where water exits the system with minimum flow requirements.
    Minimum flows can be specified either as a constant or loaded from a CSV file.
    
    Attributes:
        id (str): Unique identifier for the node
        min_flows (list): List of minimum required flow rates for each timestep
        flow_history (list): Record of actual flows for each timestep
        flow_deficits (list): Record of flow requirement deficits for each timestep
        weight (float): Weight factor for minimum flow violations in optimization
    """

    def __init__(
        self,
        id: str,
        constant_min_flow: Optional[float] = None,
        easting: Optional[float] = None,
        northing: Optional[float] = None,
        weight: float = 1.0,
        csv_file: Optional[str] = None,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
        num_time_steps: int = 0
    ) -> None:
        """
        Initialize a SinkNode object.

        Args:
            id (str): Unique identifier for the node
            min_flow (float, optional): Constant minimum required flow rate in m³/s
            easting (float, optional): Easting coordinate
            northing (float, optional): Northing coordinate
            weight (float, optional): Weight factor for minimum flow violations
            csv_file (str, optional): Path to CSV file containing minimum flow data
            start_year (int, optional): Starting year for CSV data import
            start_month (int, optional): Starting month (1-12) for CSV data import
            num_time_steps (int, optional): Number of time steps to import from CSV
        """
        super().__init__(id, easting, northing)
        
        if weight <= 0:
            raise ValueError("Weight must be positive")
        self.weight = weight
        
        # Initialize tracking lists
        self.flow_history = []
        self.flow_deficits = []

        # Try to import time series data first
        imported_data = None
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            imported_data = self._initialize_time_series(
                id, csv_file, start_year, start_month, num_time_steps, 'Q'
            )
        
        # Decision tree for setting supply_rates
        if imported_data is not None:
            self.min_flows = imported_data
        elif constant_min_flow is not None:
            self.min_flows = [constant_min_flow]*num_time_steps    
        else:
            self.min_flows = [0]*num_time_steps
            
    def update(self, time_step: int, dt: float) -> None:
        """
        Update the SinkNode's state for the given time step.
        Calculates actual flow and deficit relative to minimum requirement.

        Args:
            time_step (int): The current time step of the simulation
            dt (float): The duration of the time step in seconds
        """
        try:
            # Calculate total inflow for this timestep
            total_inflow = sum(edge.get_edge_flow_after_losses(time_step) for edge in self.inflow_edges.values())
            
            # Record the actual flow and deficit if any
            self.flow_history.append(total_inflow)
            self.flow_deficits.append(max(0, self.min_flows[time_step] - total_inflow))
            
        except Exception as e:
            raise ValueError(f"Failed to update sink node {self.id}: {str(e)}")
    
class DemandNode(Node, TimeSeriesImport):
    """
    Represents a point of water demand in the system.

    Attributes:
        demand_rates (list): A list of demand rates for each time step.
        satisfied_demand (list): A record of satisfied demand for each time step.
        excess_flow (list): A record of excess flow for each time step.
    """

    def __init__(
        self,
        id: str,
        easting: Optional[float] = None,
        northing: Optional[float] = None,
        constant_demand_rate: Optional[float] = None,
        non_consumptive_rate: float = 0.0,
        csv_file: Optional[str] = None,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
        num_time_steps: int = 0,
        field_efficiency: float = 1.0,
        conveyance_efficiency: float = 1.0,
        weight: int = 1
    ) -> None:
        """
        Initialize a DemandNode object.

        Args:
            id (str): A unique identifier for the node.
            demand_rates (list or float): Either a list of demand rates for each time step,
                                          or a constant demand rate.
        """
        super().__init__(id, easting, northing)

        # Validate field efficiency
        if not isinstance(field_efficiency, (int, float)):
            raise ValueError("Field efficiency must be a number")
        if field_efficiency <= 0 or field_efficiency > 1:
            raise ValueError("Field efficiency must be between 0 and 1")
        self.field_efficiency = field_efficiency

        # Validate conveyance efficiency
        if not isinstance(conveyance_efficiency, (int, float)):
            raise ValueError("Conveyance efficiency must be a number")
        if conveyance_efficiency <= 0 or conveyance_efficiency > 1:
            raise ValueError("Conveyance efficiency must be between 0 and 1")
        self.conveyance_efficiency = conveyance_efficiency

        # Validate weight
        if not isinstance(weight, (int, float)):
            raise ValueError("Weight must be a number")
        if weight <= 0:
            raise ValueError("Weight must be positive")
        self.weight = weight

        # Validate non consumptive flow
        if non_consumptive_rate is not None:
            if not isinstance(non_consumptive_rate, (int, float)):
                raise ValueError("Non-consumptive rate must be a number")
            if non_consumptive_rate < 0:
                raise ValueError("Non-consumptive rate cannot be negative")
            self.non_consumptive_rate = non_consumptive_rate
        
        # Initialize tracking lists
        self.satisfied_consumptive_demand = []
        self.satisfied_non_consumptive_demand = []
        self.satisfied_demand_total = []

        # Try to import time series data first
        imported_data = None
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            imported_data = self._initialize_time_series(
                id, csv_file, start_year, start_month, num_time_steps, id
            )
        
        # Decision tree for setting demand_rates
        if imported_data is not None:
            # Validate imported data
            if any(rate < 0 for rate in imported_data):
                raise ValueError("Demand rates cannot be negative")
            if any(rate < self.non_consumptive_rate for rate in imported_data):
                raise ValueError("Demand rates cannot be less than non-consumptive rate")
            
            # Apply efficiency factors to imported data
            self.demand_rates = [rate/(self.field_efficiency*self.conveyance_efficiency) 
                               for rate in imported_data]
            
        elif constant_demand_rate is not None:
            # Validate constant_demand_rate
            if constant_demand_rate < 0:
                raise ValueError("Demand rate cannot be negative")
            if constant_demand_rate < self.non_consumptive_rate:
                raise ValueError("Demand rate cannot be less than non-consumptive rate")
            
            # Apply efficiency factors to constant rate
            demand_rate = constant_demand_rate/(self.field_efficiency*self.conveyance_efficiency)
            self.demand_rates = [demand_rate] * num_time_steps
        else:
            # Default to zero demand if no other information provided
            self.demand_rates = [0] * num_time_steps

    def update(self, time_step: int, dt: float) -> None:
        """
        Update the DemandNode's state for the given time step.

        This method calculates the satisfied demand and excess flow, and distributes
        excess water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            total_inflow = sum(edge.get_edge_flow_after_losses(time_step) for edge in self.inflow_edges.values())
            current_demand = self.demand_rates[time_step] # Total demand for this timestep (consumptive + non-consumptive)
            non_consumptive_rate = self.non_consumptive_rate # Non-consumptive demand
            
            # Satisfy consumptive demand first
            consumptive_demand = current_demand - non_consumptive_rate
            satisfied_consumptive = min(total_inflow, consumptive_demand)
            satisfied_consumptive = max(0, satisfied_consumptive)  # Ensure non-negative
            self.satisfied_consumptive_demand.append(satisfied_consumptive)
            
            # Then handle non-consumptive demand from remaining flow
            remaining_flow = max(0, total_inflow - satisfied_consumptive)
            satisfied_non_consumptive = min(remaining_flow, non_consumptive_rate)
            self.satisfied_non_consumptive_demand.append(satisfied_non_consumptive)
            
            # Calculate excess after satisfying both demands
            total_satisfied = satisfied_consumptive + satisfied_non_consumptive
            self.satisfied_demand_total.append(total_satisfied)

            # Forward flow to outflow edges (excess + satisfied non-consumptive)
            total_forward_flow = remaining_flow
            total_outflow_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
            
            if total_outflow_capacity > 0:
                for edge in self.outflow_edges.values():
                    edge_flow = (edge.capacity / total_outflow_capacity) * total_forward_flow
                    edge.update(time_step, edge_flow)
            else:
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
        except Exception as e:
            raise ValueError(f"Failed to update demand node {self.id}: {str(e)}")

class StorageNode(Node, TimeSeriesImport):  

    def __init__(
        self,
        id: str,
        easting: float,
        northing: float,
        hv_file: str,
        evaporation_file: Optional[str] = None,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
        num_time_steps: int = 0,
        initial_storage: float = 0.0,
        dead_storage: float = 0.0,
        buffer_coef: float = 0.0
    ) -> None:
        """
        Initialize a StorageNode object.

        Args:
            id (str): Unique identifier for the node
            hv_file (str): Path to CSV file containing height-volume-area relationships
            initial_storage (float, optional): Initial storage volume. Defaults to 0.
            easting (float, optional): Easting coordinate
            northing (float, optional): Northing coordinate
            evaporation_file (str, optional): Path to CSV file containing monthly evaporation rates [mm/month]
            start_year (int, optional): Starting year for evaporation data
            start_month (int, optional): Starting month (1-12) for evaporation data
            num_time_steps (int, optional): Number of time steps to import from evaporation data
            dead_storage (float): Dead storage volume (V0) [m³]
        """
        # Call parent class (Node) initialization
        super().__init__(id, easting, northing)
        
        # Initialize StorageNode specific attributes
        self._level_to_volume = None
        self._volume_to_level = None
        self.evaporation_losses = []
        # Load height-volume-area data
        self._load_hv_data(hv_file)
        
        if buffer_coef < 0 and buffer_coef > 1:
            raise ValueError(f"buffer_coef ({buffer_coef}) must be between 0 and 1")

        self.buffer_coef = buffer_coef  # Buffer coefficient for low storage

 
        self.dead_storage = dead_storage  # Dead storage volume [m³]
        self.dead_storage_level = self.get_level_from_volume(dead_storage)
        
        # Initialize evaporation rates using TimeSeriesImport
        imported_data = None
        if evaporation_file is not None and all(param is not None for param in [start_year, start_month, num_time_steps]):
            imported_data = self._initialize_time_series(
                id, evaporation_file, start_year, start_month, num_time_steps, 'Evaporation'
            )
        
        self.evaporation_rates = imported_data if imported_data is not None else [0] * num_time_steps


        # Validate initial storage against capacity
        if initial_storage > self.capacity:
            raise ValueError(f"Initial storage ({initial_storage} m³) exceeds maximum capacity ({self.capacity} m³)")
        
        # Initialize storage attributes
        self.storage = [initial_storage]
        self.spillway_register = []
        self.water_level = [self.get_level_from_volume(initial_storage)]

    def set_release_params(self, params: Dict[str, Union[float, List[float]]]) -> None:
        """
        Set and validate release function parameters.
        Now supports both monthly and annual parameters.
        
        Args:
            params (dict): Monthly release parameters with each parameter being either:
                         - a single float (used for all months)
                         - a list of 12 floats (one per month)
        """
        # Validate parameters
        required_params = ['Vr', 'V1', 'V2']
        if not all(key in params for key in required_params):
            missing = [key for key in required_params if key not in params]
            raise ValueError(f"Missing release parameters: {missing}")
            
        # Convert single values to monthly lists
        monthly_params = {}
        for param, value in params.items():
            if isinstance(value, (int, float)):
                monthly_params[param] = [float(value)] * 12
            elif isinstance(value, (list, np.ndarray)) and len(value) == 12:
                monthly_params[param] = [float(v) for v in value]
            else:
                raise ValueError(f"Parameter {param} must be a number or list of 12 numbers")

        # Validate monthly parameters
        for month in range(12):
            Vr = monthly_params['Vr'][month]
            V1 = monthly_params['V1'][month]
            V2 = monthly_params['V2'][month]
            
            # Check volume relationships
            if Vr < 0:
                raise ValueError(f"Month {month+1}: Vr ({Vr}) cannot be negative")
                
            if V1 <= self.dead_storage:
                raise ValueError(f"Month {month+1}: V1 ({V1}) must be greater than dead storage ({self.dead_storage})")

            if V1 >= V2:
                raise ValueError(f"Month {month+1}: V1 ({V1}) must be less than V2 ({V2})")

            if V2 <= V1:
                raise ValueError(f"Month {month+1}: V2 ({V2}) must be greater than V1 ({V1})")
                
            if V2 > self.capacity:
                raise ValueError(f"Month {month+1}: V2 ({V2}) cannot exceed reservoir capacity ({self.capacity})")


        # Store parameters
        self.release_params = monthly_params

    def calculate_release(self, volume: float, time_step: int, dt: float) -> float:
        """
        Calculate the reservoir release based on current water level.
        
        Args:
            water_level (float): Current water level [m]
            time_step (int): Current time step
            
        Returns:
            float: Calculated release rate [m³/s]
        """
        current_month = time_step % 12
        Vr = self.release_params['Vr'][current_month]  # Target release volume
        V1 = self.release_params['V1'][current_month]  # Top of buffer zone
        V2 = self.release_params['V2'][current_month]  # Top of conservation zone
        buffer_coef = self.buffer_coef
        
        # Case 1: Below dead storage
        if volume <= self.dead_storage:
            return 0
            
        # Case 2: In buffer zone
        elif volume < V1:
            # Calculate release based on buffer coefficient
            buffer_release = min(buffer_coef * (volume - self.dead_storage), Vr)
            # Convert from volume to rate
            return buffer_release
            
        # Case 3: In conservation zone
        elif volume < V2:
            # Calculate the buffer zone contribution at V1
            buffer_contrib = buffer_coef * (V1 - self.dead_storage)
            # Add excess over V1
            conservation_release = min(Vr, volume - V1 + buffer_contrib)
            # Convert from volume to rate
            return conservation_release
            
        # Case 4: Above conservation zone
        else:
            # Release at least target volume, plus any excess above V2, limited by capacity
            flood_release = min(max(Vr, volume - V2), sum(edge.capacity for edge in self.outflow_edges.values()) * dt)
            # Convert from volume to rate
            return flood_release
        
        return release
    
    def _load_hv_data(self, csv_path: str) -> None:
        """Load and validate height-volume-area relationship data."""
        try:
            # Read and validate CSV
            df = pd.read_csv(csv_path, sep=';')
            
            # Check required columns
            required_cols = ['h', 'v']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Sort by height and remove duplicates
            df = df.sort_values('h').drop_duplicates(subset=['h'])
            
            # Convert elevations to water levels (depth above ground)
            min_waterlevel = df['h'].min()
            max_waterlevel = df['h'].max()

            # Set capacity from maximum volume in survey data
            self.capacity = float(df['v'].max())
            
            # Store survey data in dictionary
            self.hv_data = {
                'waterlevels': df['h'].values,         # Original elevations m asl. 
                'volumes': df['v'].values,
                'min_waterlevel': min_waterlevel,              # Ground level m asl. 
                'max_waterlevel': max_waterlevel,              # Maximum water level m asl.
                'max_depth': max_waterlevel - min_waterlevel   # Maximum water depth
            }
            
            # Initialize interpolation functions
            self._initialize_interpolators()
            
        except Exception as e:
            raise ValueError(f"Error loading hv data from CSV file: {str(e)}")

    def _initialize_interpolators(self):
        """Initialize interpolation functions for height-volume-area relationships."""
        try:
            if self.hv_data is None:
                raise ValueError("Height-volume data not loaded")

            # Create height to volume interpolator
            self._level_to_volume = interp1d(
                self.hv_data['waterlevels'],
                self.hv_data['volumes'],
                kind='linear',
                bounds_error=False,  # Allow extrapolation
            )
            
            # Create volume to height interpolator
            self._volume_to_level = interp1d(
                self.hv_data['volumes'],
                self.hv_data['waterlevels'],
                kind='linear',
                bounds_error=False,
            )
            
        except Exception as e:
            raise Exception(f"Error creating interpolation functions: {str(e)}")

    def get_volume_from_level(self, waterlevel: float) -> float:
        """
        Get storage volume for a given water level.
        
        Args:
            water_level (float): Water level in m asl.
            
        Returns:
            float: Corresponding storage volume [m³]
        """
        if not self.hv_data:
            print(f'{self.id} volume can not be determined from water level: Height-Volume relation is missing!')
            return 0.0

        if self._level_to_volume is None:
            raise ValueError("No level-volume relationship available")
        return float(self._level_to_volume(waterlevel))
        
    def get_level_from_volume(self, volume: float) -> float:
        """
        Get water level for a given storage volume.
        
        Args:
            volume (float): Storage volume [m³]
            
        Returns:
            float: Corresponding water level above ground [m]
        """
        if not self.hv_data:
            print(f'{self.id} water level can not be determined from volume: Height-Volume relation is missing!')
            return 0.0

        if self._volume_to_level is None:
            raise ValueError("No volume-level relationship available")
        return float(self._volume_to_level(volume))

    def get_evaporation_loss(self, time_step: int) -> float:
        """
        Get the evaporation loss for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the evaporation loss.

        Returns:
            float: The evaporation loss in m³ for the specified time step, or 0 if not available.
        """
        if time_step < len(self.evaporation_losses):
            return self.evaporation_losses[time_step]
        return 0.0

    def update(self, time_step: int, dt: float) -> None:
        """
        Update the StorageNode's state for the given time step.

        This method calculates the new storage level based on inflows, outflows,
        and evaporation losses, and distributes available water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The length of the time step in seconds.
        """
        try:
            inflow = np.sum([edge.get_edge_flow_after_losses(time_step) for edge in self.inflow_edges.values()])
            previous_storage = self.storage[-1]
            
            # Convert flow rates (m³/s) to volumes (m³) for the time step
            inflow_volume = inflow * dt
            
            # Calculate evaporation loss
            previous_water_level = self.get_level_from_volume(previous_storage)
            new_water_level = max((previous_water_level - (self.evaporation_rates[time_step] / 1000)),self.hv_data['min_waterlevel'] )  # Convert mm to m

            evap_loss = previous_storage-self.get_volume_from_level(new_water_level)
            self.evaporation_losses.append(evap_loss)
            
             # Calculate available water after evaporation
            available_water = previous_storage + inflow_volume -evap_loss
            available_water = max(0, available_water)  # Ensure non-negative storage
            
            # Calculate current water level and desired release
            requested_outflow_volume = self.calculate_release(available_water, time_step, dt)

            # Limit actual outflow to available water
            actual_outflow_volume = min(available_water, requested_outflow_volume)
            
             # Calculate total outflow capacity
            #total_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
            # Distribute actual outflow among edges proportionally
            if actual_outflow_volume > 0:
                for edge in self.outflow_edges.values():
                    edge_flow_rate = actual_outflow_volume / dt
                    edge.update(time_step, edge_flow_rate)
            else:
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
            
            # Calculate new storage
            new_storage = available_water - actual_outflow_volume

            # Check if new storage exceeds maximum and handle spillway
            if new_storage > self.capacity:
                excess_volume = new_storage - self.capacity
                new_storage = self.capacity
            else:
                excess_volume = 0
            # Log the spill event in the storage node's spillway register
            self.spillway_register.append(excess_volume)
            
            self.storage.append(new_storage)
            if self.hv_data:
                self.water_level.append(self.get_level_from_volume(new_storage))
        except Exception as e:
            # Log the error and attempt to maintain last known state
            print(f"Error updating storage node {self.id}: {str(e)}")

    def get_storage(self, time_step):
        """
        Get the storage level for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the storage level.

        Returns:
            float: The storage level in cubic meters for the specified time step, or the last known storage level if out of range.
        """
        if time_step < len(self.storage):
            return self.storage[time_step]
        return self.storage[-1]

class HydroWorks(Node):
    """
    Represents a point where water can be redistributed using fixed distribution parameters.
    Each outflow edge has a distribution parameter for each timestep.
    All distribution parameters for a timestep must sum to 1.

    Attributes:
        id (str): Unique identifier for the node
        distribution_params (dict): Maps edge IDs to their distribution parameters for each timestep
    """

    def __init__(self, id: str, easting: float, northing: float) -> None:
        """
        Initialize a HydroWorks node with default distribution parameters.

        Args:
            id (str): Unique identifier for the node
            easting (float, optional): Easting coordinate
            northing (float, optional): Northing coordinate
        """
        super().__init__(id, easting, northing)
        self.distribution_params = {}
        self.spill_register = []

    def set_distribution_parameters(self, parameters: Dict[str, Union[float, List[float]]]) -> None:
        """
        Set distribution parameters for multiple edges.

        Args:
            parameters (dict): Dictionary mapping edge IDs to either:
                             - a list of 12 monthly values
                             - a single float (will be used for all months)

        Raises:
            ValueError: If parameters are invalid or don't sum to 1 for any month
            KeyError: If any edge ID is not found in outflow edges
        """
        # Verify all edges exist
        for node_id in parameters:
            if node_id not in self.outflow_edges:
                raise KeyError(f"Edge to node {node_id} not found in outflow edges")

        # Create temporary dictionary to store new parameters
        new_params = {}
        
        # Process and validate parameters
        for node_id, params in parameters.items():
            if isinstance(params, (int, float)):
                # If single value provided, use it for all months
                new_params[node_id] = np.full(12, float(params))
            elif isinstance(params, (list, np.ndarray)) and len(params) == 12:
                # If monthly values provided, convert to array
                new_params[node_id] = np.array(params, dtype=float)
            else:
                raise ValueError(
                    f"Parameters for edge to {node_id} must be either a single value "
                    f"or a list of 12 monthly values"
                )
            
        # Verify all parameters are valid
        for node_id, params in new_params.items():
            if not np.all((0 <= params) & (params <= 1)):
                raise ValueError(
                    f"Distribution parameters for edge to {node_id} must be between 0 and 1"
                )
        
        # Verify parameters sum to 1 for each month
        for month in range(12):
            total = sum(params[month] for params in new_params.values())
            total += sum(
                self.distribution_params[node_id][month] 
                for node_id in self.outflow_edges 
                if node_id not in new_params
            )
            
            if not np.isclose(total, 1.0, atol=1e-10):  # Allow for small floating point errors
                raise ValueError(f"Distribution parameters for month {month + 1} sum to {total}, must sum to 1")
    
        # Update parameters
        self.distribution_params.update(new_params)

    def update(self, time_step: int, dt: float) -> None:

        try:
            # Calculate total inflow
            total_inflow = np.sum([edge.get_edge_flow_after_losses(time_step) for edge in self.inflow_edges.values()])
            
            # Verify distribution parameters are properly set
            if not self.distribution_params:
                raise ValueError("Distribution parameters not set")
            
            # Get current month's parameters (assuming monthly time steps)
            current_month = time_step % 12
            
            # Track total spill for this time step
            total_spill = 0
            
            # First pass: Calculate target flows and identify overflows
            target_flows = {}
            overflow_volume = 0
            
            for edge_id, edge in self.outflow_edges.items():
                # Calculate target flow based on distribution parameter
                target_flow = float(total_inflow * self.distribution_params[edge_id][current_month])
                
                # Check if target exceeds capacity
                if target_flow > edge.capacity:
                    overflow = target_flow - edge.capacity
                    overflow_volume += overflow
                    target_flows[edge_id] = edge.capacity
                else:
                    target_flows[edge_id] = target_flow
            
            # Second pass: Redistribute overflow if any exists
            if overflow_volume > 0:
                # Calculate remaining capacity for each edge
                remaining_capacity = {}
                total_remaining_capacity = 0
                
                for edge_id, edge in self.outflow_edges.items():
                    remaining = edge.capacity - target_flows[edge_id]
                    if remaining > 0:
                        remaining_capacity[edge_id] = remaining
                        total_remaining_capacity += remaining
                
                # Redistribute overflow proportionally to available capacity
                if total_remaining_capacity > 0:
                    redistributed_overflow = 0
                    for edge_id, remaining in remaining_capacity.items():
                        # Calculate proportion of overflow to add to this edge
                        proportion = remaining / total_remaining_capacity
                        additional_flow = overflow_volume * proportion
                        
                        # Ensure we don't exceed capacity
                        new_flow = min(target_flows[edge_id] + additional_flow, self.outflow_edges[edge_id].capacity)
                        redistributed_overflow += (new_flow - target_flows[edge_id])
                        target_flows[edge_id] = new_flow
                    
                    # Calculate any remaining spill after redistribution
                    total_spill = (overflow_volume - redistributed_overflow) * dt
                else:
                    # If no remaining capacity, all overflow is spilled
                    total_spill = overflow_volume * dt
            
            # Apply the final flows to edges
            for edge_id, flow in target_flows.items():
                self.outflow_edges[edge_id].update(time_step, flow)
            
            if abs(total_spill) < 1e-5:  # Consider values less than 5e-10 m³ as zero
                total_spill = 0
            # Record total spill for this time step
            self.spill_register.append(total_spill)
                
        except Exception as e:
            raise ValueError(f"Failed to update hydroworks node {self.id}: {str(e)}")
        
class RunoffNode(Node, TimeSeriesImport):
    """
    Represents a runoff generation area in the water system using a simple runoff coefficient approach.
    
    Attributes:
        id (str): A unique identifier for the node.
        area (float): The catchment area in square kilometers.
        runoff_coefficient (float): Proportion of rainfall that becomes runoff (0-1).
        rainfall_data (list): A list of rainfall depths for each time step in mm.
        runoff_history (list): A record of generated runoff for each time step in m³/s.
    """

    def __init__(
        self,
        id: str,
        area: float,
        runoff_coefficient: float,
        easting: float,
        northing: float,
        rainfall_csv: str,
        start_year: int,
        start_month: int,
        num_time_steps: int = 0
    ) -> None:
        """
        Initialize a RunoffNode object.

        Args:
            id (str): A unique identifier for the node.
            area (float): The catchment area in square kilometers.
            runoff_coefficient (float): Proportion of rainfall that becomes runoff (0-1).
            easting (float, optional): The easting coordinate of the node.
            northing (float, optional): The northing coordinate of the node.
            rainfall_csv (str, optional): Path to CSV file containing rainfall data.
            start_year (int, optional): Starting year for CSV data import.
            start_month (int, optional): Starting month (1-12) for CSV data import.
            num_time_steps (int, optional): Number of time steps to import from CSV.
        """
        super().__init__(id, easting, northing)
        
        # Validate inputs
        if area <= 0:
            raise ValueError(f"Area must be positive, got {area}")
        if not (0 <= runoff_coefficient <= 1):
            raise ValueError(f"Runoff coefficient must be between 0 and 1, got {runoff_coefficient}")
            
        self.area = area  # km²
        self.runoff_coefficient = runoff_coefficient
        self.runoff_history = []
        
        # Import rainfall data from CSV if provided
        self.rainfall_data = self._initialize_time_series(
            id, rainfall_csv, start_year, start_month, num_time_steps, 'Precipitation'
        ) if rainfall_csv else []
        
        # Initialize with zeros if no data provided
        if not self.rainfall_data and num_time_steps:
            self.rainfall_data = [0] * num_time_steps
            print(f"Warning: No rainfall data provided for RunoffNode '{id}'. Initializing with zeros.")
    
    def calculate_runoff(self, rainfall: float, dt: float) -> float:
        """
        Calculate runoff using a simple runoff coefficient approach.
        
        Args:
            rainfall (float): Rainfall depth in mm for the current time step
            dt (float): Time step duration in seconds
            
        Returns:
            float: Runoff rate in m³/s
        """
        if rainfall <= 0:
            return 0
            
        # Calculate runoff volume
        # Convert rainfall from mm to m and multiply by area (km²) to get volume in m³
        # Then apply runoff coefficient to determine how much becomes runoff
        volume = (rainfall / 1000) * self.area * 1e6 * self.runoff_coefficient
        
        # Convert volume to flow rate (m³/s) based on the time step duration
        runoff_rate = volume / dt
        
        return runoff_rate
    
    def get_rainfall(self, time_step: int) -> float:
        """
        Get the rainfall for a specific time step.
        
        Args:
            time_step (int): The time step for which to retrieve the rainfall
            
        Returns:
            float: Rainfall in mm or 0 if not available
        """
        if time_step < len(self.rainfall_data):
            return self.rainfall_data[time_step]
        return 0
    
    def update(self, time_step: int, dt: float) -> None:
        """
        Update the RunoffNode's state for the given time step.
        
        Args:
            time_step (int): The current time step of the simulation
            dt (float): The duration of the time step in seconds
        """
        try:
            # Get rainfall for current time step
            rainfall = self.get_rainfall(time_step)
            
            # Calculate runoff
            runoff = self.calculate_runoff(rainfall, dt)
            
            # Store runoff in history
            self.runoff_history.append(runoff)
            
            # Update outflow edges
            total_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
            
            if total_capacity > 0:
                # Distribute runoff proportionally based on edge capacities
                for edge in self.outflow_edges.values():
                    edge_flow = (edge.capacity / total_capacity) * runoff
                    edge.update(time_step, min(edge_flow, edge.capacity))
            else:
                # If no capacity, set flow to zero
                for edge in self.outflow_edges.values():
                    edge.update(time_step, 0)
                    
        except Exception as e:
            print(f"Error updating RunoffNode {self.id}: {str(e)}")
            self.runoff_history.append(0)
            for edge in self.outflow_edges.values():
                edge.update(time_step, 0)
    
    def get_runoff(self, time_step: int) -> float:  
        """
        Get the runoff rate for a specific time step.
        
        Args:
            time_step (int): The time step for which to retrieve the runoff
            
        Returns:
            float: Runoff rate in m³/s or 0 if not available
        """
        if time_step < len(self.runoff_history):
            return self.runoff_history[time_step]
        return 0
    
    def get_total_runoff_volume(self, dt: float) -> float:
        """
        Calculate the total runoff volume across all time steps.
        
        Args:
            dt (float): The duration of each time step in seconds
            
        Returns:
            float: Total runoff volume in m³
        """
        return sum(runoff * dt for runoff in self.runoff_history)