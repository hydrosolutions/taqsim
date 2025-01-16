"""
This module defines the various types of nodes that can exist in a water system simulation.

The module includes a base Node class and several specialized node types such as
SupplyNode, SinkNode, DemandNode, StorageNode, and HydroWorks.
Each node type has its own behavior for handling water inflows and outflows.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class Node:
    """
    Base class for all types of nodes in the water system.

    Attributes:
        id (str): A unique identifier for the node.
        inflows (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflows (dict): A dictionary of outflow edges, keyed by the target node's id.
        easting (float): The easting coordinate of the node.
        northing (float): The northing coordinate of the node.
    """

    def __init__(self, id, easting=None, northing=None):
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

    def add_inflow(self, edge):
        """
        Add an inflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflow_edges[edge.source.id] = edge

    def add_outflow(self, edge):
        """
        Add an outflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an outflow.
        """
        self.outflow_edges[edge.target.id] = edge

    def update(self, time_step, dt):
        """
        Update the node's state for the given time step.

        This method should be overridden by subclasses to implement
        specific behavior for each node type.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): Number of seconds in time step.
        """
        pass

class SupplyNode(Node):
    """
    Represents a water supply source in the system.

    Attributes:
        supply_rates (list): A list of supply rates for each time step.
        default_supply_rate (float): The default supply rate if not specified for a time step.
        supply_history (list): A record of actual supply amounts for each time step.
    """

    def __init__(self, id, supply_rates=None, default_supply_rate=0, easting=None, northing=None,
                 csv_file=None, start_year=None, start_month=None, num_time_steps=None):
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
        self.default_supply_rate = default_supply_rate
        self.supply_history = []

        self.supply_rates = self._initialize_supply_rates(
            id, csv_file, start_year, start_month, num_time_steps, supply_rates
        )

    def _initialize_supply_rates(self, id, csv_file, start_year, start_month, 
                                 num_time_steps, supply_rates):
        """
        Initialize supply rates from either CSV or direct input.
        
        Args:
            id (str): Node identifier for error messages
            csv_file (str): Path to CSV file
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            supply_rates (list): Direct supply rates input
            
        Returns:
            list: Initialized supply rates
        """
        # If all CSV parameters are provided, try to import data
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            try:
                supply = self.import_supply_data(csv_file, start_year, start_month, num_time_steps)
                
                # Check if data is valid
                if not (supply.empty or 
                    supply['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(supply['Q']) < num_time_steps):
                    return supply['Q'].tolist()
                
                # Print warning for invalid data
                print(f"Warning: Using default supply rate ({self.default_supply_rate}) for node '{id}' due to insufficient data")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not supply.empty:
                    print(f"Available data range: {supply['Date'].min():%Y-%m} to {supply['Date'].max():%Y-%m}")
                return [self.default_supply_rate] * num_time_steps
                
            except Exception as e:
                print(f"Warning: Using default supply rate ({self.default_supply_rate}) for node '{id}' due to error: {str(e)}")
                return [self.default_supply_rate] * num_time_steps
        
        # If no CSV import, use provided rates or initialize empty list
        return supply_rates if supply_rates is not None else []
            
    def import_supply_data(self, csv_file, start_year, start_month, num_time_steps):
        """
        Import supply data from a CSV file for a specified time period.
        
        Args:
            csv_file (str): Path to the CSV file containing supply data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            supply = pd.read_csv(csv_file, parse_dates=['Date'])
            
            if 'Date' not in supply.columns or 'Q' not in supply.columns:
                raise ValueError("CSV file must contain 'Date' and 'Q' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return supply[(supply['Date'] >= start_date) & (supply['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Supply data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import supply data: {str(e)}")
        
    def get_supply_rate(self, time_step):
        """
        Get the supply rate for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the supply rate.

        Returns:
            float: The supply rate for the specified time step, or the default rate if not specified.
        """
        if time_step < len(self.supply_rates):
            return self.supply_rates[time_step]
        return self.default_supply_rate

    def update(self, time_step, dt):
        """
        Update the SupplyNode's state for the given time step.

        This method calculates the current supply rate and distributes it among outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            current_supply_rate = self.get_supply_rate(time_step)
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

class SinkNode(Node):
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

    def __init__(self, id, min_flow=None, easting=None, northing=None, weight=1.0,
                 min_flow_csv_file=None, start_year=None, start_month=None, num_time_steps=None):
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

        # Initialize minimum flows based on input method
        if all(param is not None for param in [min_flow_csv_file, start_year, start_month, num_time_steps]):
            self.min_flows = self._initialize_min_flows(
                id, min_flow_csv_file, start_year, start_month, num_time_steps, min_flow
            )
        elif isinstance(min_flow, (int, float)):
            if min_flow < 0:
                raise ValueError("Minimum flow cannot be negative")
            self.min_flows = [min_flow]
        else:
            self.min_flows = [0]  # Default to no minimum flow requirement
            
    def _initialize_min_flows(self, id, csv_file, start_year, start_month, 
                            num_time_steps, default_flow):
        """
        Initialize minimum flows from CSV file.
        
        Args:
            id (str): Node identifier for error messages
            csv_file (str): Path to CSV file
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            default_flow (float): Default flow to use if CSV import fails
            
        Returns:
            list: Initialized minimum flows
        """
        try:
            flow_data = self.import_flow_data(csv_file, start_year, start_month, num_time_steps)
            
            # Check if data is valid
            if not (flow_data.empty or 
                flow_data['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                len(flow_data['Q']) < num_time_steps):
                return flow_data['Q'].tolist()
            
            # Print warning for invalid data
            print(f"Warning: Using default minimum flow ({default_flow if default_flow is not None else 0}) "
                  f"for node '{id}' due to insufficient data")
            print(f"Requested period: {start_year}-{start_month:02d} to "
                  f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
            if not flow_data.empty:
                print(f"Available data range: {flow_data['Date'].min():%Y-%m} to {flow_data['Date'].max():%Y-%m}")
            
            return [default_flow if default_flow is not None else 0] * num_time_steps
                
        except Exception as e:
            print(f"Warning: Using default minimum flow ({default_flow if default_flow is not None else 0}) "
                  f"for node '{id}' due to error: {str(e)}")
            return [default_flow if default_flow is not None else 0] * num_time_steps

    def import_flow_data(self, csv_file, start_year, start_month, num_time_steps):
        """
        Import minimum flow data from a CSV file for a specified time period.
        
        Args:
            csv_file (str): Path to the CSV file containing flow data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            flow_data = pd.read_csv(csv_file, parse_dates=['Date'])
            
            if 'Date' not in flow_data.columns or 'Q' not in flow_data.columns:
                raise ValueError("CSV file must contain 'Date' and 'Q' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return flow_data[(flow_data['Date'] >= start_date) & (flow_data['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Flow data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import flow data: {str(e)}")

    def get_min_flow(self, time_step):
        """
        Get the minimum flow requirement for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the minimum flow

        Returns:
            float: The minimum flow requirement for the specified time step,
                  or the last known requirement if out of range
        """
        if time_step < len(self.min_flows):
            return self.min_flows[time_step]
        return self.min_flows[-1]

    def update(self, time_step, dt):
        """
        Update the SinkNode's state for the given time step.
        Calculates actual flow and deficit relative to minimum requirement.

        Args:
            time_step (int): The current time step of the simulation
            dt (float): The duration of the time step in seconds
        """
        try:
            # Calculate total inflow for this timestep
            total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
            
            # Record the actual flow and deficit if any
            self.flow_history.append(total_inflow)
            self.flow_deficits.append(max(0, self.get_min_flow(time_step) - total_inflow))
            
        except Exception as e:
            raise ValueError(f"Failed to update sink node {self.id}: {str(e)}")
            
    def get_flow_deficit(self, time_step):
        """
        Get the flow deficit for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the deficit

        Returns:
            float: The flow deficit for the specified time step, or 0 if not available
        """
        if time_step < len(self.flow_deficits):
            return self.flow_deficits[time_step]
        return 0
        
    def get_total_deficit_volume(self, dt):
        """
        Calculate the total deficit volume across all time steps.

        Args:
            dt (float): The duration of each time step in seconds

        Returns:
            float: Total deficit volume in m³
        """
        return sum(deficit * dt for deficit in self.flow_deficits)
    
class DemandNode(Node):
    """
    Represents a point of water demand in the system.

    Attributes:
        demand_rates (list): A list of demand rates for each time step.
        satisfied_demand (list): A record of satisfied demand for each time step.
        excess_flow (list): A record of excess flow for each time step.
    """

    def __init__(self, id, demand_rates=None, easting=None, northing=None,
                 csv_file=None, start_year=None, start_month=None, num_time_steps=None, 
                 field_efficiency=1, conveyance_efficiency=1, weight=1.0, non_consumptive_rate=0):
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
        
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            self.demand_rates = self._initialize_demand_rates(
                id, csv_file, start_year, start_month, num_time_steps, demand_rates
            )
        elif isinstance(demand_rates, (int, float)):
            if demand_rates < 0:
                raise ValueError("Demand rate cannot be negative")
            if demand_rates < self.non_consumptive_rate:
                raise ValueError("Demand rate cannot be less than non-consumptive rate")
            self.demand_rates = [demand_rates/(self.field_efficiency*self.conveyance_efficiency)]
        elif isinstance(demand_rates, list):
            if not all(isinstance(rate, (int, float)) for rate in demand_rates):
                raise ValueError("All demand rates must be numeric values")
            if any(rate < 0 for rate in demand_rates):
                raise ValueError("Demand rates cannot be negative")
            if any(rate < self.non_consumptive_rate for rate in demand_rates):
                raise ValueError("Demand rates cannot be less than non-consumptive rate")
            self.demand_rates = [rate/(self.field_efficiency*self.conveyance_efficiency) for rate in demand_rates]
        else:
            raise ValueError("demand_rates must be a number or list of numbers or defined by CSV")
        
        self.satisfied_consumptive_demand = []
        self.satisfied_non_consumptive_demand = []
        self.satisfied_demand_total = []
        self.excess_flow = []
    
    def _initialize_demand_rates(self, id, csv_file, start_year, start_month, 
                               num_time_steps, demand_rates):
        """
        Initialize demand rates from either CSV or direct input.
        
        Args:
            id (str): Node identifier for error messages
            csv_file (str): Path to CSV file
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            demand_rates (list or float): Direct demand rates input
            
        Returns:
            list: Initialized demand rates
        """
        # If all CSV parameters are provided, try to import data
        if all(param is not None for param in [csv_file, start_year, start_month, num_time_steps]):
            try:
                demand = self.import_demand_data(csv_file, start_year, start_month, num_time_steps)
                
                # Check if data is valid
                if not (demand.empty or 
                    demand['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(demand['Q']) < num_time_steps):
                    return [rate/(self.field_efficiency*self.conveyance_efficiency) for rate in demand['Q'].tolist()]
                
                # Print warning for invalid data
                print(f"Warning: Insufficient data in csv file for node '{id}'")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not demand.empty:
                    print(f"Available data range: {demand['Date'].min():%Y-%m} to {demand['Date'].max():%Y-%m}")
            except Exception as e:
                print(f"Warning: Demand from csv could not be used for node '{id}' due to error: {str(e)}")
    
    def import_demand_data(self, csv_file, start_year, start_month, num_time_steps):
        """
        Import demand data from a CSV file for a specified time period.
        
        Args:
            csv_file (str): Path to the CSV file containing demand data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            demand = pd.read_csv(csv_file, parse_dates=['Date'])
            
            if 'Date' not in demand.columns or 'Q' not in demand.columns:
                raise ValueError("CSV file must contain 'Date' and 'Q' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return demand[(demand['Date'] >= start_date) & (demand['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Demand data file not found: {csv_file}")
        except Exception as e:
            raise ValueError(f"Failed to import demand data: {str(e)}")

    def get_demand_rate(self, time_step):
        """
        Get the demand rate for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the demand rate.

        Returns:
            float: The demand rate for the specified time step, or the last known rate if out of range.
        """
        if time_step < len(self.demand_rates):
            return self.demand_rates[time_step]
        return self.demand_rates[-1]

    def update(self, time_step, dt):
        """
        Update the DemandNode's state for the given time step.

        This method calculates the satisfied demand and excess flow, and distributes
        excess water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        try:
            total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
            current_demand = self.get_demand_rate(time_step)
            non_consumptive_rate = self.non_consumptive_rate
            
            # Satisfy consumptive demand first
            consumptive_demand = current_demand - non_consumptive_rate
            satisfied = min(total_inflow, consumptive_demand)
            satisfied = max(0, satisfied)  # Ensure non-negative
            self.satisfied_consumptive_demand.append(satisfied)
            
            # Then handle non-consumptive demand from remaining flow
            remaining_flow = max(0, total_inflow - satisfied)
            non_consumptive_satisfied = min(remaining_flow, non_consumptive_rate)
            self.satisfied_non_consumptive_demand.append(non_consumptive_satisfied)
            
            # Calculate excess after satisfying both demands
            excess = max(0, remaining_flow - non_consumptive_satisfied)
            
            total_satisfied = satisfied + non_consumptive_satisfied
            self.satisfied_demand_total.append(total_satisfied)
            self.excess_flow.append(excess)

            # Forward flow to outflow edges (excess + satisfied non-consumptive)
            total_forward_flow = excess + non_consumptive_satisfied
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

    def get_satisfied_demand(self, time_step):
        """
        Get the satisfied demand for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the satisfied demand.

        Returns:
            float: The satisfied demand for the specified time step, or 0 if not available.
        """
        if time_step < len(self.satisfied_demand):
            return self.satisfied_demand[time_step]
        return 0

class StorageNode(Node):  
    """
    Represents a water storage facility in the system.
    Now enhanced with height-volume-area relationships from survey data.

    Attributes:
        id (str): Unique identifier for the node
        capacity (float): Maximum storage capacity [m³]
        storage (list): Record of storage levels for each time step [m³]
        hva_data (dict): Dictionary containing the height-volume-area relationships
        evaporation (list): List of monthly evaporation rates [mm/month]
        evaporation_losses (list): Record of volume lost to evaporation each timestep [m³]
    """

    def __init__(self, id, hva_file, initial_storage=0, easting=None, northing=None, 
                 evaporation_file=None, start_year=None, start_month=None, num_time_steps=None, 
                 release_params=None):
        """
        Initialize a StorageNode object.

        Args:
            id (str): Unique identifier for the node
            hva_file (str): Path to CSV file containing height-volume-area relationships
            initial_storage (float, optional): Initial storage volume. Defaults to 0.
            easting (float, optional): Easting coordinate
            northing (float, optional): Northing coordinate
            evaporation_file (str, optional): Path to CSV file containing monthly evaporation rates [mm/month]
            start_year (int, optional): Starting year for evaporation data
            start_month (int, optional): Starting month (1-12) for evaporation data
            num_time_steps (int, optional): Number of time steps to import from evaporation data
            release_params (dict): Dictionary of monthly release parameters {
                'h1': list[12] or float,  # Low reservoir levels [m] for each month
                'h2': list[12] or float,  # High reservoir levels [m] for each month
                'w': list[12] or float,   # Constant releases [m³/s] for each month
                'm1': list[12] or float,  # Slopes for low level [rad] for each month
                'm2': list[12] or float   # Slopes for high level [rad] for each month
            }
        """
        # Call parent class (Node) initialization
        super().__init__(id, easting, northing)
        
        # Initialize StorageNode specific attributes
        self.hva_data = None
        self._level_to_volume = None
        self._volume_to_level = None
        self._level_to_area = None
        self.evaporation_losses = []

        # Load height-volume-area data
        self._load_hva_data(hva_file)
        # Initialize evaporation rates
        self.evaporation_rates = self._initialize_evaporation_rates(
            id, evaporation_file, start_year, start_month, num_time_steps
        )


        # Set release parameters with validation
        self.set_release_params(release_params)

        # Validate initial storage against capacity
        if initial_storage > self.capacity:
            raise ValueError(f"Initial storage ({initial_storage} m³) exceeds maximum capacity ({self.capacity} m³)")
        
        # Initialize storage attributes
        self.storage = [initial_storage]
        self.spillway_register = []
        self.water_level = [self.get_level_from_volume(initial_storage)]

    def set_release_params(self, params):
        """
        Set and validate release function parameters.
        
        Args:
            params (dict): Monthly release parameters
        """
        # Validate parameters
        required_params = ['h1', 'h2', 'w', 'm1', 'm2']
        if not all(key in params for key in required_params):
            missing = [key for key in required_params if key not in params]
            raise ValueError(f"Missing release parameters: {missing}")
            
        # Validate parameter values
        h1 = params['h1']
        h2 = params['h2']
        w =  params['w']
        m1 = params['m1']
        m2 = params['m2']
        
        # Check level bounds against HVA data
        if hasattr(self, 'hva_data'):
            min_level = self.hva_data['min_waterlevel']
            max_level = self.hva_data['max_waterlevel']
            
            if h1 < min_level or h1 > max_level:
                raise ValueError(f"h1 ({h1}) outside valid range [{min_level}, {max_level}]")
            if h2 < min_level or h2 > max_level:
                raise ValueError(f"h2 ({h2}) outside valid range [{min_level}, {max_level}]")
        
        # Check level relationships
        if h1 >= h2:
            raise ValueError(f"h1 ({h1}) must be less than h2 ({h2})")
        
        # Check slope ranges (0 to π/2 radians)
        if not (0 <= m1 < 1.571):
            raise ValueError(f"m1 ({m1}) must be between 0 and π/2")
        if not (0 <= m2 < 1.571):
            raise ValueError(f"m2 ({m2}) must be between 0 and π/2")
        
        # Check base release rate
        if w < 0:
            raise ValueError(f"w ({w}) cannot be negative")
        
        
        # Store parameters
        self.release_params = params

    def calculate_release(self, waterlevel):
        """
        Calculate the reservoir release based on current water level.
        
        Args:
            water_level (float): Current water level [m]
            time_step (int): Current time step
            
        Returns:
            float: Calculated release rate [m³/s]
        """
        h1 = self.release_params['h1']
        h2 = self.release_params['h2']
        w = self.release_params['w']
        m1 = self.release_params['m1']
        m2 = self.release_params['m2']
        
        release = sum(edge.capacity for edge in self.outflow_edges.values())
        if waterlevel < self.hva_data['max_waterlevel'] and (w + (waterlevel-h2)* np.tan(m2)) < sum(edge.capacity for edge in self.outflow_edges.values()):
            release = w + (waterlevel-h2)* np.tan(m2)
        if waterlevel <= h2 and (w<=sum(edge.capacity for edge in self.outflow_edges.values())):
            release = w
        if (waterlevel-h1)* np.tan(m1) < w and (waterlevel-h1)* np.tan(m1) < sum(edge.capacity for edge in self.outflow_edges.values()):
            release = (waterlevel-h1)* np.tan(m1)
        if waterlevel < h1:
            release = 0
        
        return release
    
    def _initialize_evaporation_rates(self, id, evaporation_file, start_year, start_month, num_time_steps):
        """
        Initialize evaporation rates from CSV file.
        
        Args:
            id (str): Node identifier for error messages
            evaporation_file (str): Path to evaporation data CSV
            start_year (int): Start year for data
            start_month (int): Start month for data
            num_time_steps (int): Number of time steps
            
        Returns:
            list: Initialized evaporation rates or None if not configured
        """
        # If no evaporation file provided, return None
        if evaporation_file is None:
            return None

        # If all parameters are provided, try to import data
        if all(param is not None for param in [evaporation_file, start_year, start_month, num_time_steps]):
            try:
                evap_data = self.import_evaporation_data(evaporation_file, start_year, start_month, num_time_steps)
                
                # Check if data is valid
                if not (evap_data.empty or 
                    evap_data['Date'].iloc[0] != pd.Timestamp(year=start_year, month=start_month, day=1) or 
                    len(evap_data['Evaporation']) < num_time_steps):
                    return evap_data['Evaporation'].tolist()
                
                # Print warning for invalid data
                print(f"Warning: No evaporation rates will be applied for node '{id}' due to insufficient data")
                print(f"Requested period: {start_year}-{start_month:02d} to "
                    f"{pd.Timestamp(year=start_year, month=start_month, day=1) + pd.DateOffset(months=num_time_steps-1):%Y-%m}")
                if not evap_data.empty:
                    print(f"Available data range: {evap_data['Date'].min():%Y-%m} to {evap_data['Date'].max():%Y-%m}")
                return None
                
            except Exception as e:
                print(f"Warning: Failed to load evaporation data for node '{id}': {str(e)}")
                return None
        
        return None

    def import_evaporation_data(self, evaporation_file, start_year, start_month, num_time_steps):
        """
        Import evaporation data from a CSV file for a specified time period.
        
        Args:
            evaporation_file (str): Path to the CSV file containing evaporation data
            start_year (int): Starting year for the data
            start_month (int): Starting month (1-12) for the data
            num_time_steps (int): Number of time steps to import
            
        Returns:
            DataFrame: Filtered DataFrame containing the requested data period
        """
        try:
            # Read the CSV file into a pandas DataFrame
            evap_df = pd.read_csv(evaporation_file, parse_dates=['Date'])
            
            if 'Date' not in evap_df.columns or 'Evaporation' not in evap_df.columns:
                raise ValueError("CSV file must contain 'Date' and 'Evaporation' columns")
        
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            
            return evap_df[(evap_df['Date'] >= start_date) & (evap_df['Date'] < end_date)]
            
        except FileNotFoundError:
            raise ValueError(f"Evaporation data file not found: {evaporation_file}")
        except Exception as e:
            raise ValueError(f"Failed to import evaporation data: {str(e)}")

    def _load_hva_data(self, csv_path):
        """Load and validate height-volume-area relationship data."""
        try:
            # Read and validate CSV
            df = pd.read_csv(csv_path, sep=';')
            
            # Check required columns
            required_cols = ['Height_m', 'Volume_m3', 'Area_m2']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Sort by height and remove duplicates
            df = df.sort_values('Height_m').drop_duplicates(subset=['Height_m'])
            
            # Convert elevations to water levels (depth above ground)
            min_waterlevel = df['Height_m'].min()
            max_waterlevel = df['Height_m'].max()

            # Set capacity from maximum volume in survey data
            self.capacity = float(df['Volume_m3'].max())
            
            # Store survey data in dictionary
            self.hva_data = {
                'waterlevels': df['Height_m'].values,         # Original elevations m asl. 
                'volumes': df['Volume_m3'].values,
                'areas': df['Area_m2'].values,
                'min_waterlevel': min_waterlevel,              # Ground level m asl. 
                'max_waterlevel': max_waterlevel,              # Maximum water level m asl.
                'max_depth': max_waterlevel - min_waterlevel   # Maximum water depth
            }
            
            # Initialize interpolation functions
            self._initialize_interpolators()
            
        except Exception as e:
            raise ValueError(f"Error loading HVA data from CSV file: {str(e)}")

    def _initialize_interpolators(self):
        """Initialize interpolation functions for height-volume-area relationships."""
        try:
            # Create height to volume interpolator
            self._level_to_volume = interp1d(
                self.hva_data['waterlevels'],
                self.hva_data['volumes'],
                kind='linear',
                bounds_error=False,  # Allow extrapolation
                fill_value=(self.hva_data['volumes'][0], self.hva_data['volumes'][-1])
            )
            
            # Create volume to height interpolator
            self._volume_to_level = interp1d(
                self.hva_data['volumes'],
                self.hva_data['waterlevels'],
                kind='linear',
                bounds_error=False,
                fill_value=(self.hva_data['waterlevels'][0], self.hva_data['waterlevels'][-1])
            )
            
            # Create height to area interpolator
            self._level_to_area = interp1d(
                self.hva_data['waterlevels'],
                self.hva_data['areas'],
                kind='linear',
                bounds_error=False,
                fill_value=(self.hva_data['areas'][0], self.hva_data['areas'][-1])
            )
            
        except Exception as e:
            raise Exception(f"Error creating interpolation functions: {str(e)}")

    def get_volume_from_level(self, waterlevel):
        """
        Get storage volume for a given water level.
        
        Args:
            water_level (float): Water level in m asl.
            
        Returns:
            float: Corresponding storage volume [m³]
        """
        if not self.hva_data:
            print(f'{self.id} volume can not be determined from water level: Height-Volume relation is missing!')
            return

        if self._level_to_volume is None:
            raise ValueError("No level-volume relationship available")
        return float(self._level_to_volume(waterlevel))
        
    def get_level_from_volume(self, volume):
        """
        Get water level for a given storage volume.
        
        Args:
            volume (float): Storage volume [m³]
            
        Returns:
            float: Corresponding water level above ground [m]
        """
        if not self.hva_data:
            print(f'{self.id} water level can not be determined from volume: Height-Volume relation is missing!')
            return

        if self._volume_to_level is None:
            raise ValueError("No volume-level relationship available")
        return float(self._volume_to_level(volume))

    def get_evaporation_loss(self, time_step):
        """
        Get the evaporation loss for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the evaporation loss.

        Returns:
            float: The evaporation loss in m³ for the specified time step, or 0 if not available.
        """
        if time_step < len(self.evaporation_losses):
            return self.evaporation_losses[time_step]
        return 0

    def update(self, time_step, dt):
        """
        Update the StorageNode's state for the given time step.

        This method calculates the new storage level based on inflows, outflows,
        and evaporation losses, and distributes available water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The length of the time step in seconds.
        """
        try:
            inflow = np.sum([edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values()])
            previous_storage = self.storage[-1]
            
            # Convert flow rates (m³/s) to volumes (m³) for the time step
            inflow_volume = inflow * dt
            
            # Calculate evaporation loss
            previous_water_level = self.get_level_from_volume(previous_storage)
            new_water_level = max((previous_water_level - (self.evaporation_rates[time_step] / 1000)),self.hva_data['min_waterlevel'] )  # Convert mm to m

            evap_loss = previous_storage-self.get_volume_from_level(new_water_level)
            self.evaporation_losses.append(evap_loss)
            
             # Calculate available water after evaporation
            available_water = previous_storage + inflow_volume -evap_loss
            available_water = max(0, available_water)  # Ensure non-negative storage
            
            # Calculate current water level and desired release
            current_level = self.get_level_from_volume(available_water)
            target_release_rate = self.calculate_release(current_level)
            
            # Convert release rate to volume
            requested_outflow_volume = target_release_rate * dt

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
            if self.hva_data:
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

    def __init__(self, id, easting=None, northing=None):
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
        self.num_timesteps = None  # Will be set when parameters are initialized

    def add_outflow(self, edge):
        """
        Add an outflow edge to the node and initialize its distribution parameter.
        Initially distributes flow equally among all edges for all timesteps.

        Args:
            edge (Edge): The edge to be added as an outflow
        """
        super().add_outflow(edge)
        
        # When adding a new edge, redistribute parameters equally among all edges
        # This will be overridden when set_distribution_parameters is called
        n_edges = len(self.outflow_edges)
        equal_distribution = 1.0 / n_edges
        
        # Initialize with equal distribution for 12 months
        self.distribution_params = {
            edge_id: np.full(12, equal_distribution)
            for edge_id in self.outflow_edges
        }

    def set_distribution_parameters(self, parameters):
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
                new_params[node_id] = float(params)
            else:
                raise ValueError(
                    f"Parameters for edge to {node_id} must be a single value"
                )
            
        # Verify all parameters are valid
        for node_id, params in new_params.items():
            if not np.all((0 <= params) & (params <= 1)):
                raise ValueError(
                    f"Distribution parameters for edge to {node_id} must be between 0 and 1"
                )
        
        # Verify parameters sum to 1
        total = np.sum([params for params in new_params.values()], axis=0)
        total += np.sum(
            [self.distribution_params[node_id] for node_id in self.outflow_edges if node_id not in new_params],
            axis=0
        )
        
        if not np.allclose(total, 1.0, atol=1e-10):  # Allow for small floating point errors
            raise ValueError(f"Distribution parameters must sum to 1. Got {total}")
    
        # Update parameters
        self.distribution_params.update(new_params)

    def update(self, time_step, dt):
        """
        Update the HydroWorks node's state for the given time step using the
        distribution parameters for that specific timestep.

        Args:
            time_step (int): The current time step of the simulation
            dt (float): The duration of the time step in seconds

        Raises:
            ValueError: If distribution parameters are not properly set
        """
        try:
            # Calculate total inflow
            total_inflow = np.sum([edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values()])
            
            # Verify distribution parameters are properly set
            if not self.distribution_params:
                raise ValueError("Distribution parameters not set")
            
            # Track total spill for this time step
            total_spill = 0
            
            # Distribute flow according to parameters, recording spills when capacity is exceeded
            for edge_id, edge in self.outflow_edges.items():
                # Calculate target flow based on distribution parameter
                target_flow = total_inflow * self.distribution_params[edge_id]
                
                # If target exceeds capacity, record the excess as spill
                if target_flow > edge.capacity:
                    actual_flow = edge.capacity
                    spill = (target_flow - edge.capacity) * dt
                    total_spill += spill
                else:
                    actual_flow = target_flow
                
                # Update edge with the actual flow
                edge.update(time_step, actual_flow)
            
            # Record total spill for this time step
            self.spill_register.append(total_spill)
                
        except Exception as e:
            raise ValueError(f"Failed to update hydroworks node {self.id}: {str(e)}")
