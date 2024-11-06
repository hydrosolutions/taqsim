"""
This module defines the various types of nodes that can exist in a water system simulation.

The module includes a base Node class and several specialized node types such as
SupplyNode, SinkNode, DemandNode, StorageNode, and HydroWorks.
Each node type has its own behavior for handling water inflows and outflows.
"""
import pandas as pd

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
        """
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
        super().__init__(id)
        self.default_supply_rate = default_supply_rate
        self.supply_history = []
        self.easting = easting
        self.northing = northing

        self.supply_rates = self._initialize_supply_rates(
            id, csv_file, start_year, start_month, num_time_steps, supply_rates
        )

    def _initialize_supply_rates(self, id, csv_file, start_year, start_month, num_time_steps, supply_rates):
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
            
            # Filter the DataFrame to find the start point
            start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
            end_date = start_date + pd.DateOffset(months=num_time_steps)
            supply = supply[(supply['Date'] >= start_date) & (supply['Date'] < end_date)]
            
            return supply
        except Exception as e:
            print(f"Error reading CSV file '{csv_file}': {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
        
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
        current_supply_rate = self.get_supply_rate(time_step)
        self.supply_history.append(current_supply_rate)

        total_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
        if total_capacity > 0:
            for edge in self.outflow_edges.values():
                edge_flow = (edge.capacity / total_capacity) * current_supply_rate
                edge.update(time_step, edge_flow)
        else:
            for edge in self.outflow_edges.values():
                edge.update(time_step, 0)

class SinkNode(Node):
    """
    Represents a point where water exits the system.
    """

    def update(self, time_step, dt):
        """
        Update the SinkNode's state for the given time step.

        This method calculates the total inflow to the sink node.
        Sink nodes remove all incoming water from the system.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.

        """
        total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
        # Sink nodes remove all incoming water from the system

class DemandNode(Node):
    """
    Represents a point of water demand in the system.

    Attributes:
        demand_rates (list): A list of demand rates for each time step.
        satisfied_demand (list): A record of satisfied demand for each time step.
        excess_flow (list): A record of excess flow for each time step.
    """

    def __init__(self, id, demand_rates, easting=None, northing=None):
        """
        Initialize a DemandNode object.

        Args:
            id (str): A unique identifier for the node.
            demand_rates (list or float): Either a list of demand rates for each time step,
                                          or a constant demand rate.
        """
        super().__init__(id)
        self.demand_rates = demand_rates if isinstance(demand_rates, list) else [demand_rates]
        self.satisfied_demand = []
        self.excess_flow = []
        self.easting = easting # easting coordinate of the node. Defaults to None.
        self.northing = northing # northing coordinate of the node. Defaults to None.

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
        total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
        current_demand = self.get_demand_rate(time_step)
        satisfied = min(total_inflow, current_demand)
        excess = max(0, total_inflow - current_demand)
        
        self.satisfied_demand.append(satisfied)
        self.excess_flow.append(excess)

        total_outflow_capacity = sum(edge.capacity for edge in self.outflow_edges.values())
        if total_outflow_capacity > 0:
            for edge in self.outflow_edges.values():
                edge_flow = (edge.capacity / total_outflow_capacity) * excess
                edge.update(time_step, edge_flow)
        else:
            for edge in self.outflow_edges.values():
                edge.update(time_step, 0)

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

    Attributes:
        capacity (float): The maximum storage capacity of the node in cubic meters.
        storage (list): A record of storage levels for each time step in cubic meters.
    """

    def __init__(self, id, capacity, initial_storage=0, easting=None, northing=None):
        """
        Initialize a StorageNode object.

        Args:
            id (str): A unique identifier for the node.
            capacity (float): The maximum storage capacity of the node in cubic meters.
            initial_storage (float, optional): The initial storage level in cubic meters. Defaults to 0.
        """
        super().__init__(id)
        self.capacity = capacity
        self.storage = [initial_storage]
        self.spillway_register = [0] # List to store spill events
        self.easting = easting # easting coordinate of the node. Defaults to None.
        self.northing = northing # northing coordinate of the node. Defaults to None.

    def update(self, time_step, dt):
        """
        Update the StorageNode's state for the given time step.

        This method calculates the new storage level based on inflows and outflows,
        and distributes available water to outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The length of the time step in seconds.
        """
        inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
        previous_storage = self.storage[-1]
        
        # Convert flow rates (m³/s) to volumes (m³) for the time step
        inflow_volume = inflow * dt
        available_water = previous_storage + inflow_volume
        
        # Calculate total requested outflow
        requested_outflow = sum(edge.capacity for edge in self.outflow_edges.values())
        
        # Convert requested outflow to volume
        requested_outflow_volume = requested_outflow * dt
        
        # Limit actual outflow to available water
        actual_outflow_volume = min(available_water, requested_outflow_volume)
        
        # Distribute actual outflow among edges proportionally
        if requested_outflow_volume > 0:
            for edge in self.outflow_edges.values():
                edge_flow_volume = (edge.capacity / requested_outflow) * actual_outflow_volume
                edge_flow_rate = edge_flow_volume / dt
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
    Represents a point where water can be redistributed, combining the functionality
    of diversion and confluence points.
    """

    def update(self, time_step, dt):
        """
        Update the HydroWorks node's state for the given time step.

        This method calculates the total inflow and distributes it among outflow edges
        based on their capacities.

        Args:
            time_step (int): The current time step of the simulation.
            dt (float): The duration of the time step in seconds.
        """
        total_inflow = sum(edge.get_edge_outflow(time_step) for edge in self.inflow_edges.values())
        total_outflow_capacity = sum(edge.capacity for edge in self.outflow_edges.values())

        if total_outflow_capacity > 0:
            for edge in self.outflow_edges.values():
                # Distribute water proportionally based on edge capacity
                edge_flow = (edge.capacity / total_outflow_capacity) * total_inflow
                edge.update(time_step, edge_flow)
        else:
            # If there's no outflow capacity, set all outflows to 0
            for edge in self.outflow_edges.values():
                edge.update(time_step, 0)