"""
This module defines the various types of nodes that can exist in a water system simulation.

The module provides specialized node classes for water system modeling:
- SupplyNode: Water sources (rivers, intakes) that provide inflow to the system.
- SinkNode: Terminal points where water exits the system with minimum flow requirements.
- DemandNode: Consumption points (agriculture, domestic, industrial) with specific water demands.
- StorageNode: Reservoirs and storage facilities with volume-based operation rules.
- HydroWorks: Distribution nodes that split water among multiple targets based on predefined rules.
- RunoffNode: Surface runoff sources that generate flow based on rainfall and catchment characteristics.

Each node type implements specialized behavior for handling inflows and outflows, while
maintaining compatibility with the overall water system architecture.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Union, Any
from .validation_functions import validate_node_id, validate_coordinates, validate_positive_integer,validate_positive_float, validate_probability, validate_nonnegativity_int_or_float, validate_dataframe_period


def initialize_time_series(
    id: str,
    csv_file: Optional[str] = None,
    start_year: Optional[int] = None,
    start_month: Optional[int] = None,
    num_time_steps: int = 0,
    column_name: str = 'Q'
) -> List[float]:
    """
    Initialize time series data from a CSV file for a specified time period.
    
    Args:
        id (str): Node identifier for error messages and logging
        csv_file (str): Path to CSV file containing time series data
        start_year (int): Starting year for the data extraction
        start_month (int): Starting month (1-12) for the data extraction
        num_time_steps (int): Number of time steps to import
        column_name (str): Name of the column in CSV to import (default: 'Q' for flow)
        
    Returns:
        list: Time series data extracted from the CSV, or zeros if import failed
        
    Note:
        If CSV import fails or parameters are missing, returns an array of zeros
        with length equal to num_time_steps to ensure simulation can continue.
    """
    # If all CSV parameters are provided, try to import data
    if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
        try:
            # Read the CSV file into a pandas DataFrame
            time_series = pd.read_csv(csv_file, parse_dates=['Date'])
            
            # Validate the DataFrame structure
            validate_dataframe_period(time_series, start_year, start_month, num_time_steps, column_name)
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            filtered_data = time_series[(time_series['Date'] >= start_date) & (time_series['Date'] < end_date)]
            
            return filtered_data[column_name].tolist()
            
        except FileNotFoundError:
            raise ValueError(f"Data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import time series data for node '{id}': {str(e)}")
            
    return [0.0] * num_time_steps  # Default to zero if import fails or is invalid

class SupplyNode:
    """
    Represents a water supply source in the system.
    
    A SupplyNode serves as an entry point for water into the system, such as a river,
    a water intake, or an external inflow. It can have a constant supply rate or
    a time-varying rate loaded from a CSV file.

    Attributes:
        id (str): A unique identifier for the node.
        easting (float): The easting coordinate of the node (UTM coordinate system).
        northing (float): The northing coordinate of the node (UTM coordinate system).
        supply_rates (list): A list of water supply rates [m³/s] for each time step.
        supply_history (list): A record of actual supply amounts [m³/s] for each time step.
        outflow_edge: The single outflow edge from this node.
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
            constant_supply_rate (float, optional): A constant water supply rate [m³/s].
            easting (float): The easting coordinate of the node in UTM system.
            northing (float): The northing coordinate of the node in UTM system.
            csv_file (str, optional): Path to CSV file containing time-varying supply data.
            start_year (int, optional): Starting year for CSV data import.
            start_month (int, optional): Starting month (1-12) for CSV data import.
            num_time_steps (int): Number of time steps to simulate.
        
        Note:
            Supply data priority: CSV file (if valid) > constant_supply_rate > zeros.
            This allows flexible specification of supply rates based on available data.
        """
        # Validate Node ID
        validate_node_id(id, "SupplyNode")
        # Validate coordinates
        validate_coordinates(easting, northing, id)

        self.id = id
        self.easting = easting  # easting coordinate of the node in UTM system
        self.northing = northing  # northing coordinate of the node in UTM system
        self.outflow_edge = None  # Single outflow edge
        self.supply_history = []

        # Try to import time series data first
        imported_data = None
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            imported_data = initialize_time_series(
                id, csv_file, start_year, start_month, num_time_steps, 'Q'
            )
        
        # Decision tree for setting supply_rates
        if imported_data is not None:
            self.supply_rates = imported_data
        elif constant_supply_rate is not None:
            self.supply_rates = [constant_supply_rate]*num_time_steps    
        else:
            self.supply_rates = [0]*num_time_steps

    def add_outflow_edge(self, edge):
        """
        Set the outflow edge from this node.
        
        Args:
            edge (Edge): The edge to be set as the outflow.
            
        Raises:
            ValueError: If an outflow edge is already set.
        """
        if self.outflow_edge is not None:
            raise ValueError(f"SupplyNode {self.id} already has an outflow edge. Only one outflow edge is allowed.")
        self.outflow_edge = edge

    def update(self, time_step: int, dt: float) -> None:
        """
        Update the SupplyNode's state for the given time step.
        
        This method calculates the current supply rate and sends it to the outflow edge.

        Args:
            time_step (int): The current time step index of the simulation.
            dt (float): The duration of the time step in seconds.
            
        Raises:
            ValueError: If there's an error in updating the supply node.
        """
        try:
            # Get the supply rate for the current time step
            current_supply_rate = self.supply_rates[time_step]
            self.supply_history.append(current_supply_rate)

            # Update the outflow edge
            if self.outflow_edge is not None:
                # The flow is limited by the edge capacity
                flow = min(current_supply_rate, self.outflow_edge.capacity)
                self.outflow_edge.update(time_step, flow)
            else:
                # No outflow edge
                pass
                
        except Exception as e:
            raise ValueError(f"Failed to update supply node {self.id}: {str(e)}")

class SinkNode:
    """
    Represents a point where water exits the system with minimum flow requirements.
    
    A SinkNode acts as a terminal point in the water system where flow must meet or exceed
    minimum requirements. It can be used to model environmental flow demands, downstream
    water rights, or other system outflow constraints. Minimum flows can be specified as 
    a constant or loaded from a CSV file.
    
    Attributes:
        id (str): Unique identifier for the node
        easting (float): The easting coordinate of the node (UTM coordinate system).
        northing (float): The northing coordinate of the node (UTM coordinate system).
        min_flows (list): List of minimum required flow rates [m³/s] for each timestep
        flow_history (list): Record of actual flows [m³/s] for each timestep
        flow_deficits (list): Record of flow requirement deficits [m³/s] for each timestep
        weight (float): Weight factor for minimum flow violations in optimization
        inflow_edges (dict): A dictionary of inflow edges, keyed by the source node's id.
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
            constant_min_flow (float, optional): Constant minimum required flow rate [m³/s]
            easting (float): Easting coordinate in UTM system
            northing (float): Northing coordinate in UTM system
            weight (float): Weight factor for minimum flow violations in optimization objective function
            csv_file (str, optional): Path to CSV file containing minimum flow requirements
            start_year (int, optional): Starting year for CSV data import
            start_month (int, optional): Starting month (1-12) for CSV data import
            num_time_steps (int): Number of time steps to simulate
            
        Raises:
            ValueError: If weight is non-positive
            
        Note:
            Minimum flow data priority: CSV file (if valid) > constant_min_flow > zeros.
            This allows flexible specification of minimum flows based on available data.
        """
        # Validate Node ID
        validate_node_id(id, "SinkNode")
        # Validate coordinates
        validate_coordinates(easting, northing, id)
        # Validate weight
        validate_positive_integer(weight, "SinkNode weight")

        self.id = id
        self.easting = easting
        self.northing = northing
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.weight = weight
        
        # Initialize tracking lists
        self.flow_history = []
        self.flow_deficits = []

        # Try to import time series data first
        imported_data = None
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            imported_data = initialize_time_series(
                id, csv_file, start_year, start_month, num_time_steps, 'Q'
            )
        
        # Decision tree for setting supply_rates
        if imported_data is not None:
            self.min_flows = imported_data
        elif constant_min_flow is not None:
            self.min_flows = [constant_min_flow]*num_time_steps    
        else:
            self.min_flows = [0]*num_time_steps
    
    def add_inflow_edge(self, edge):
        """
        Add an inflow edge to the node.
        
        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflow_edges[edge.source.id] = edge
            
    def update(self, time_step: int, dt: float) -> None:
        """
        Update the SinkNode's state for the given time step.
        
        Calculates actual flow and deficit relative to minimum flow requirement.
        Flow deficits are tracked for optimization and reporting purposes.

        Args:
            time_step (int): The current time step index of the simulation
            dt (float): The duration of the time step in seconds
            
        Raises:
            ValueError: If there's an error in updating the sink node
        """
        try:
            # Calculate total inflow for this timestep from all incoming edges
            total_inflow = sum(edge.get_edge_flow_after_losses(time_step) for edge in self.inflow_edges.values())
            
            # Record the actual flow and deficit (if any)
            self.flow_history.append(total_inflow)
            self.flow_deficits.append(max(0, self.min_flows[time_step] - total_inflow))
            
        except Exception as e:
            raise ValueError(f"Failed to update sink node {self.id}: {str(e)}")

class DemandNode:
    """
    Represents a point of water demand in the system.
    
    A DemandNode models water consumption for uses like irrigation, municipal supply,
    or industrial processes. It can include both consumptive use (water that is removed
    from the system) and non-consumptive use (water that returns to the system).
    Efficiency factors can be applied to model system losses.

    Attributes:
        id (str): A unique identifier for the node.
        easting (float): The easting coordinate of the node (UTM coordinate system).
        northing (float): The northing coordinate of the node (UTM coordinate system).
        demand_rates (list): List of total demand rates [m³/s] for each time step
        field_efficiency (float): Efficiency of water use at the field/end-use level (0-1)
        conveyance_efficiency (float): Efficiency of the water delivery system (0-1)
        non_consumptive_rate (float): Flow rate [m³/s] that returns to the system
        weight (float): Weight factor for demand shortfalls in optimization
        inflow_edges (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflow_edge: The single outflow edge from this node.
        satisfied_consumptive_demand (list): Record of met consumptive demand [m³/s]
        satisfied_non_consumptive_demand (list): Record of met non-consumptive demand [m³/s]
        satisfied_demand_total (list): Record of total satisfied demand [m³/s]
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
            id (str): A unique identifier for the node
            easting (float): Easting coordinate in UTM system
            northing (float): Northing coordinate in UTM system
            constant_demand_rate (float, optional): Constant water demand rate [m³/s]
            non_consumptive_rate (float): Flow rate that returns to system [m³/s]
            csv_file (str, optional): Path to CSV file with time-varying demand data
            start_year (int, optional): Starting year for CSV data import
            start_month (int, optional): Starting month (1-12) for CSV data import
            num_time_steps (int): Number of time steps to simulate
            field_efficiency (float): Efficiency of water use at field/end-use (0-1)
            conveyance_efficiency (float): Efficiency of water delivery system (0-1)
            weight (int): Weight factor for demand shortfalls in optimization
            
        Raises:
            ValueError: If efficiency values are invalid, weight is non-positive,
                      or non-consumptive rate is negative
                      
        Note:
            The demand_rates are adjusted by efficiency factors to represent gross
            water requirements. When efficiencies < 1, more water is required to 
            satisfy the same net demand.
        """
        # Validate Node ID
        validate_node_id(id, "DemandNode")
        # Validate coordinates
        validate_coordinates(easting, northing, id)
        # Validate weight
        validate_positive_integer(weight, "DemandNode weight")
        # Validate field efficiency
        validate_probability(field_efficiency, "field_efficiency")
        # Validate conveyance efficiency
        validate_probability(conveyance_efficiency, "conveyance_efficiency")
        
        self.id = id
        self.easting = easting
        self.northing = northing
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.outflow_edge = None  # Single outflow edge
        self.weight = weight  # Weight factor for demand shortfalls in optimization
        self.field_efficiency = field_efficiency  # Field efficiency (0-1)
        self.conveyance_efficiency = conveyance_efficiency # Conveyance efficiency (0-1)

        # Validate non consumptive flow
        if non_consumptive_rate is not None:
            validate_nonnegativity_int_or_float(non_consumptive_rate, "non_consumptive_rate")
            self.non_consumptive_rate = non_consumptive_rate
        
        # Initialize tracking lists
        self.satisfied_consumptive_demand = []
        self.satisfied_non_consumptive_demand = []
        self.satisfied_demand_total = []

        # Try to import time series data first
        imported_data = None
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            imported_data = initialize_time_series(
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
            validate_nonnegativity_int_or_float(constant_demand_rate, "constant_demand_rate")
            if constant_demand_rate < self.non_consumptive_rate:
                raise ValueError("Demand rate cannot be less than non-consumptive rate")
            
            # Apply efficiency factors to constant rate
            demand_rate = constant_demand_rate/(self.field_efficiency*self.conveyance_efficiency)
            self.demand_rates = [demand_rate] * num_time_steps
        else:
            # Default to zero demand if no other information provided
            self.demand_rates = [0] * num_time_steps

    def add_inflow_edge(self, edge):
        """
        Add an inflow edge to the node.
        
        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflow_edges[edge.source.id] = edge

    def add_outflow_edge(self, edge):
        """
        Set the outflow edge from this node.
        
        Args:
            edge (Edge): The edge to be set as the outflow.
            
        Raises:
            ValueError: If an outflow edge is already set.
        """
        if self.outflow_edge is not None:
            raise ValueError(f"DemandNode {self.id} already has an outflow edge. Only one outflow edge is allowed.")
        self.outflow_edge = edge

    def update(self, time_step: int, dt: float) -> None:
        """
        Update the DemandNode's state for the given time step.
        
        This method:
        1. Calculates total inflow from all incoming edges
        2. Satisfies consumptive demand first (water that leaves the system)
        3. Satisfies non-consumptive demand (water that returns to the system)
        4. Routes any remaining water to outflow edge

        Args:
            time_step (int): The current time step index of the simulation
            dt (float): The duration of the time step in seconds
            
        Raises:
            ValueError: If there's an error in updating the demand node
        """
        try:
            # Calculate total inflow from all incoming edges
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

            # Forward flow to outflow edge (excess + satisfied non-consumptive)
            total_forward_flow = remaining_flow
            
            # Update the outflow edge with the forwarded flow
            if self.outflow_edge is not None:
                self.outflow_edge.update(time_step, min(total_forward_flow, self.outflow_edge.capacity))
                
        except Exception as e:
            raise ValueError(f"Failed to update demand node {self.id}: {str(e)}")

class StorageNode:  
    """
    Represents a reservoir or storage facility in the water system.
    
    A StorageNode models water storage and controlled release, such as a reservoir or a lake.
    It tracks water volumes, surface elevations, and manages releases based on operating rules.
    It also accounts for evaporation losses and spillway flows when capacity is exceeded.

    Attributes:
        id (str): Unique identifier for the node
        easting (float): The easting coordinate of the node (UTM coordinate system).
        northing (float): The northing coordinate of the node (UTM coordinate system).
        capacity (float): Maximum storage volume [m³]
        dead_storage (float): Minimum operational storage volume [m³]
        storage (list): Record of storage volumes [m³] for each time step
        water_level (list): Record of water surface elevations [m.a.s.l.] for each time step
        evaporation_rates (list): Monthly evaporation rates [mm/month]
        evaporation_losses (list): Record of evaporation volume losses [m³]
        spillway_register (list): Record of excess water volumes [m³] spilled
        release_params (dict): Parameters controlling reservoir release policy
        buffer_coef (float): Coefficient for controlling releases at low storage levels
        inflow_edges (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflow_edge: The single outflow edge from this node.
    """

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
            easting (float): Easting coordinate in UTM system
            northing (float): Northing coordinate in UTM system
            hv_file (str): Path to CSV file containing height-volume relationship
            evaporation_file (str, optional): Path to CSV file with monthly evaporation rates [mm/month]
            start_year (int, optional): Starting year for evaporation data
            start_month (int, optional): Starting month (1-12) for evaporation data
            num_time_steps (int): Number of time steps to simulate
            initial_storage (float): Initial storage volume [m³]
            dead_storage (float): Minimum operational storage volume [m³]
            buffer_coef (float): Coefficient for controlling releases at low storage (0-1)
            
        Raises:
            ValueError: If buffer_coef is outside the valid range [0,1],
                      or hv_file cannot be loaded properly
                      
        Note:
            The StorageNode imports height-volume-area relationships from a CSV file.
            The relationship is used to calculate water levels from volumes and vice versa.
            Evaporation losses are calculated based on water surface area and evaporation rates.
        """
        # Validate Node ID
        validate_node_id(id, "StorageNode")
        # Validate coordinates
        validate_coordinates(easting, northing, id)
        # Validate buffer coefficient
        validate_probability(buffer_coef, "buffer_coef")
        
        self.id = id
        self.easting = easting
        self.northing = northing
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.outflow_edge = None  # Single outflow edge
        
        # Initialize StorageNode specific attributes
        self._level_to_volume = None
        self._volume_to_level = None
        self.evaporation_losses = []
        
        # Load height-volume-area data
        self._load_hv_data(hv_file)
        
        self.buffer_coef = buffer_coef  # Buffer coefficient for low storage
        self.dead_storage = dead_storage  # Dead storage volume [m³]
        self.dead_storage_level = self._volume_to_level(dead_storage)
        
        # Initialize evaporation rates using TimeSeriesImport
        imported_data = None
        if evaporation_file is not None and all(param is not None for param in [start_year, start_month, num_time_steps]):
            imported_data = initialize_time_series(
                id, evaporation_file, start_year, start_month, num_time_steps, 'Evaporation'
            )
        
        self.evaporation_rates = imported_data if imported_data is not None else [0] * num_time_steps

        # Validate initial storage against capacity
        if initial_storage > self.capacity:
            raise ValueError(f"Initial storage ({initial_storage} m³) exceeds maximum capacity ({self.capacity} m³)")
        
        # Initialize storage attributes
        self.storage = [initial_storage]
        self.spillway_register = []
        self.water_level = [self._volume_to_level(initial_storage)]

    def add_inflow_edge(self, edge):
        """
        Add an inflow edge to the node.
        
        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflow_edges[edge.source.id] = edge

    def add_outflow_edge(self, edge):
        """
        Set the outflow edge from this node.
        
        Args:
            edge (Edge): The edge to be set as the outflow.
            
        Raises:
            ValueError: If an outflow edge is already set.
        """
        if self.outflow_edge is not None:
            raise ValueError(f"StorageNode {self.id} already has an outflow edge. Only one outflow edge is allowed.")
        self.outflow_edge = edge

    def set_release_params(self, params: Dict[str, Union[float, List[float]]]) -> None:
        """
        Set and validate reservoir release policy parameters.
        
        The release policy uses a rule-curve approach with three key parameters:
        - Vr: Target monthly release volume [m³]
        - V1: Top of buffer zone [m³]
        - V2: Top of conservation zone [m³]
        
        These parameters define different operational zones in the reservoir:
        - Below dead_storage: No release
        - Between dead_storage and V1: Buffer zone (reduced releases)
        - Between V1 and V2: Conservation zone (normal operations)
        - Above V2: Flood control zone (increased releases)
        
        Args:
            params (dict): Monthly release parameters with each parameter being either:
                         - a single float (used for all months)
                         - a list of 12 floats (one per month)
                         
        Raises:
            ValueError: If parameters are missing or invalid (e.g., V1 >= V2)
            
        Note:
            This implementation supports both monthly-varying and constant parameters.
            Volume relationships must be maintained: dead_storage < V1 < V2 <= capacity
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
        Calculate the reservoir release volume based on current storage and operating rules.
        
        This method implements the reservoir release policy using rule curves.
        The release is determined by which zone the current storage falls into:
        
        1. Below dead storage: No release
        2. In buffer zone (V0 to V1): Reduced release based on buffer_coef
        3. In conservation zone (V1 to V2): Normal target release
        4. Above conservation zone (>V2): Increased release to prevent flooding
        
        Args:
            volume (float): Current storage volume [m³]
            time_step (int): Current time step index
            dt (float): Time step duration in seconds
            
        Returns:
            float: Calculated release volume [m³] for the current time step
            
        Note:
            This method returns volumes (not flow rates) to be released during
            the current time step.
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
            flood_release = min(max(Vr, volume - V2), self.outflow_edge.capacity * dt if self.outflow_edge else 0)
            # Convert from volume to rate
            return flood_release      
    
    def _load_hv_data(self, csv_path: str) -> None:
        """
        Load and validate height-volume-area relationship data from CSV file.
        
        This method loads the relationship between water level (elevation),
        storage volume, and potentially surface area from a CSV file. It creates
        interpolation functions for converting between height and volume.
        
        Args:
            csv_path (str): Path to the CSV file with height-volume-area data
            
        Raises:
            ValueError: If the CSV file cannot be read or has invalid format
            
        Note:
            The CSV must contain at least 'h' (height) and 'v' (volume) columns.
            An optional 'a' (area) column can be included for surface area data.
        """
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
            self._initialize_hv_interpolators()
            
        except Exception as e:
            raise ValueError(f"Error loading hv data from CSV file: {str(e)}")

    def _initialize_hv_interpolators(self):
        """
        Initialize interpolation functions for height-volume-area relationships.
        
        Creates two interpolation functions:
        1. level_to_volume: Converts elevation [m.a.s.l.] to volume [m³]
        2. volume_to_level: Converts volume [m³] to elevation [m.a.s.l.]
        
        Raises:
            Exception: If interpolation functions cannot be created
            
        Note:
            These interpolation functions enable converting between water level
            and storage volume in both directions during the simulation.
        """
        try:
            if self.hv_data is None:
                raise ValueError("Height-volume data not loaded")

            # Create height to volume interpolator
            self._level_to_volume = interp1d(
                self.hv_data['waterlevels'],
                self.hv_data['volumes'],
                kind='linear',
                bounds_error=True,  # Allow extrapolation
            )
            
            # Create volume to height interpolator
            self._volume_to_level = interp1d(
                self.hv_data['volumes'],
                self.hv_data['waterlevels'],
                kind='linear',
                bounds_error=True,
            )
            
        except Exception as e:
            raise Exception(f"Error creating interpolation functions: {str(e)}")

    def update(self, time_step: int, dt: float) -> None:
        """
        Update the StorageNode's state for the given time step.
        
        This method implements the reservoir water balance:
        1. Calculates total inflow from all incoming edges
        2. Accounts for evaporation losses based on water surface
        3. Determines release volume based on operating rules
        4. Handles excess water (spillway flow) if capacity is exceeded
        5. Updates storage volume and water level
        
        Args:
            time_step (int): The current time step index of the simulation
            dt (float): The duration of the time step in seconds
            
        Note:
            The update sequence prioritizes evaporation losses before releases.
            Spillway flows occur when the storage exceeds capacity after all
            other gains and losses are accounted for.
        """
        try:
            inflow = np.sum([edge.get_edge_flow_after_losses(time_step) for edge in self.inflow_edges.values()])
            previous_storage = self.storage[-1]
            
            # Convert flow rates (m³/s) to volumes (m³) for the time step
            inflow_volume = inflow * dt
            
            # Calculate evaporation loss
            previous_water_level = self._volume_to_level(previous_storage)
            new_water_level = max((previous_water_level - (self.evaporation_rates[time_step] / 1000)),self.hv_data['min_waterlevel'] )  # Convert mm to m

            evap_loss = previous_storage-self._level_to_volume(new_water_level)
            self.evaporation_losses.append(evap_loss)
            
             # Calculate available water after evaporation
            available_water = previous_storage + inflow_volume -evap_loss
            available_water = max(0, available_water)  # Ensure non-negative storage
            
            # Calculate current water level and desired release
            requested_outflow_volume = self.calculate_release(available_water, time_step, dt)

            # Limit actual outflow to available water
            actual_outflow_volume = min(available_water, requested_outflow_volume)
            
            # Update the outflow edge with the calculated outflow
            if self.outflow_edge is not None and actual_outflow_volume > 0:
                edge_flow_rate = actual_outflow_volume / dt
                self.outflow_edge.update(time_step, edge_flow_rate)
            elif self.outflow_edge is not None:
                self.outflow_edge.update(time_step, 0)
            
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
                self.water_level.append(self._volume_to_level(new_storage))
        except Exception as e:
            # Log the error and attempt to maintain last known state
            print(f"Error updating storage node {self.id}: {str(e)}")

class HydroWorks:
    """
    Represents a water distribution point that splits flow according to predefined ratios.
    
    A HydroWorks node models hydraulic structures or operational decisions that distribute
    water to multiple destinations according to specified ratios. Examples include canal
    bifurcations, distribution structures, or operational water allocation policies. 
    
    Each outflow edge has a distribution parameter (ratio) that can vary monthly.
    The distribution parameters for all outflows in a given month must sum to 1.
    Capacity constraints are respected, with overflow redistributed or spilled.

    Attributes:
        id (str): Unique identifier for the node
        easting (float): The easting coordinate of the node (UTM coordinate system).
        northing (float): The northing coordinate of the node (UTM coordinate system).
        distribution_params (dict): Maps target node IDs to their distribution parameters
        spill_register (list): Record of spill volumes that couldn't be accommodated
        inflow_edges (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflow_edges (dict): A dictionary of outflow edges, keyed by the target node's id.
    """

    def __init__(self, id: str, easting: float, northing: float) -> None:
        """
        Initialize a HydroWorks node with empty distribution parameters.

        Args:
            id (str): Unique identifier for the node
            easting (float): Easting coordinate in UTM system
            northing (float): Northing coordinate in UTM system
            
        Note:
            After initialization, the set_distribution_parameters method must be called
            to define how water should be distributed among outflow edges.
        """
        # Validate Node ID
        validate_node_id(id, "HydroWorks")
        # Validate coordinates
        validate_coordinates(easting, northing, id)

        self.id = id
        self.easting = easting
        self.northing = northing
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.outflow_edges = {}  # Dictionary of outflow edges - HydroWorks can have multiple outflows
        self.distribution_params = {}
        self.spill_register = []

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

    def set_distribution_parameters(self, parameters: Dict[str, Union[float, List[float]]]) -> None:
        """
        Set distribution parameters for multiple outflow edges.
        
        The distribution parameters define how incoming water is allocated among
        the outflow edges. For each month, the parameters across all edges must sum to 1.
        
        Args:
            parameters (dict): Dictionary mapping edge IDs to either:
                             - a list of 12 monthly values (distribution ratios)
                             - a single float (will be used for all months)

        Raises:
            ValueError: If parameters are invalid or don't sum to 1 for any month
            KeyError: If any edge ID is not found in outflow edges
            
        Note:
            The distribution parameters must be between 0 and 1, and the sum of
            parameters for all edges in a given month must equal 1.
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
        """
        Update the HydroWorks node state for the given time step.
        
        This method:
        1. Calculates total inflow from all incoming edges
        2. Distributes water to outflow edges based on distribution parameters
        3. Respects capacity constraints of outflow edges
        4. Redistributes overflow to edges with remaining capacity if possible
        5. Records spills when overflow cannot be fully redistributed
        
        Args:
            time_step (int): The current time step index of the simulation
            dt (float): The duration of the time step in seconds
            
        Raises:
            ValueError: If distribution parameters are not set or other errors occur
            
        Note:
            The distribution occurs in two passes:
            1. Initial distribution according to parameters
            2. Redistribution of overflow to remaining capacity where possible
        """
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
        
class RunoffNode:
    """
    Represents a runoff generation area in the water system.
    
    A RunoffNode models a catchment area that generates surface runoff based on
    precipitation and catchment characteristics. It uses a simple coefficient-based
    approach to convert rainfall to runoff, accounting for catchment area and
    runoff efficiency.

    Attributes:
        id (str): Unique identifier for the node
        easting (float): The easting coordinate of the node (UTM coordinate system).
        northing (float): The northing coordinate of the node (UTM coordinate system).
        area (float): Catchment area [km²]
        runoff_coefficient (float): Fraction of rainfall that becomes runoff (0-1)
        rainfall_data (list): Precipitation depths [mm] for each time step
        runoff_history (list): Record of generated runoff [m³/s] for each time step
        outflow_edge: The single outflow edge from this node.
    """

    def __init__(self, id: str, area: float, runoff_coefficient: float,
                 easting: float, northing: float, rainfall_csv: str, start_year: int, start_month: int, num_time_steps: int = 0) -> None:
        """
        Initialize a RunoffNode object.

        Args:
            id (str): Unique identifier for the node
            area (float): Catchment area [km²]
            runoff_coefficient (float): Fraction of rainfall that becomes runoff (0-1)
            easting (float): Easting coordinate in UTM system
            northing (float): Northing coordinate in UTM system
            rainfall_csv (str): Path to CSV file containing rainfall data [mm]
            start_year (int): Starting year for data extraction
            start_month (int): Starting month (1-12) for data extraction
            num_time_steps (int): Number of time steps to simulate
            
        Raises:
            ValueError: If area is non-positive or runoff_coefficient is outside [0,1]
            
        Note:
            The runoff calculation uses a simple coefficient method where:
            Runoff = Rainfall * Area * Runoff_coefficient
            This is a simplified approach that does not account for infiltration,
            evapotranspiration, or other complex hydrological processes.
        """
        # Validate Node ID
        validate_node_id(id, "RunoffNode")
        # Validate coordinates
        validate_coordinates(easting, northing, id)
        # Validate area 
        validate_positive_float(area, "area")
        # Validate runoff coefficient
        validate_probability(runoff_coefficient, "runoff_coefficient")

        self.id = id
        self.easting = easting
        self.northing = northing
        self.area = area # km²
        self.outflow_edge = None  # Single outflow edge
        self.runoff_coefficient = runoff_coefficient # Fraction of rainfall that becomes runoff
        self.runoff_history = []
        
        # Import rainfall data from CSV if provided
        self.rainfall_data = initialize_time_series(
            id, rainfall_csv, start_year, start_month, num_time_steps, 'Precipitation'
        ) if rainfall_csv else []
        
        # Initialize with zeros if no data provided
        if not self.rainfall_data and num_time_steps:
            self.rainfall_data = [0] * num_time_steps
            print(f"Warning: No rainfall data provided for RunoffNode '{id}'. Initializing with zeros.")

    def add_outflow_edge(self, edge):
        """
        Set the outflow edge from this node.
        
        Args:
            edge (Edge): The edge to be set as the outflow.
            
        Raises:
            ValueError: If an outflow edge is already set.
        """
        if self.outflow_edge is not None:
            raise ValueError(f"RunoffNode {self.id} already has an outflow edge. Only one outflow edge is allowed.")
        self.outflow_edge = edge

    def calculate_runoff(self, rainfall: float, dt: float) -> float:
        """
        Calculate runoff using a simple runoff coefficient approach.
        
        This method applies the rational formula to convert rainfall depth to runoff.
        
        Args:
            rainfall (float): Rainfall depth [mm] for the current time step
            dt (float): Time step duration [seconds]
            
        Returns:
            float: Runoff rate [m³/s]
            
        Note:
            The conversion follows these steps:
            1. Convert rainfall from mm to m
            2. Multiply by area in km² (converted to m²)
            3. Apply runoff coefficient
            4. Divide by time step duration to get flow rate
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
    
    def update(self, time_step: int, dt: float) -> None:
        """
        Update the RunoffNode's state for the given time step.
        
        This method:
        1. Gets rainfall for the current time step
        2. Calculates runoff using the coefficient method
        3. Distributes runoff to outflow edges proportionally
        
        Args:
            time_step (int): The current time step index of the simulation
            dt (float): The duration of the time step in seconds
            
        Note:
            Runoff is distributed to outflow edges proportionally based on their
            capacities, subject to capacity constraints.
        """
        try:
            # Get rainfall for current time step
            rainfall = self.rainfall_data[time_step]
            
            # Calculate runoff
            runoff = self.calculate_runoff(rainfall, dt)
            
            # Store runoff in history
            self.runoff_history.append(runoff)

            # Update the outflow edge
            if self.outflow_edge is not None:
                # The flow is limited by the edge capacity
                flow = min(runoff, self.outflow_edge.capacity)
                self.outflow_edge.update(time_step, flow)
            
            else:
                pass
                    
        except Exception as e:
            print(f"Error updating RunoffNode {self.id}: {str(e)}")
            self.runoff_history.append(0)