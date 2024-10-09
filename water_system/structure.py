"""
This module defines the various types of nodes that can exist in a water system simulation.

The module includes a base Node class and several specialized node types such as
SupplyNode, SinkNode, DemandNode, StorageNode, DiversionNode, and ConfluenceNode.
Each node type has its own behavior for handling water inflows and outflows.
"""

class Node:
    """
    Base class for all types of nodes in the water system.

    Attributes:
        id (str): A unique identifier for the node.
        inflows (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflows (dict): A dictionary of outflow edges, keyed by the target node's id.
    """

    def __init__(self, id):
        """
        Initialize a Node object.

        Args:
            id (str): A unique identifier for the node.
        """
        self.id = id
        self.inflows = {}  # Dictionary of inflow edges
        self.outflows = {}  # Dictionary of outflow edges

    def add_inflow(self, edge):
        """
        Add an inflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an inflow.
        """
        self.inflows[edge.source.id] = edge

    def add_outflow(self, edge):
        """
        Add an outflow edge to the node.

        Args:
            edge (Edge): The edge to be added as an outflow.
        """
        self.outflows[edge.target.id] = edge

    def update(self, time_step):
        """
        Update the node's state for the given time step.

        This method should be overridden by subclasses to implement
        specific behavior for each node type.

        Args:
            time_step (int): The current time step of the simulation.
        """
        pass

class SupplyNode(Node):
    def __init__(self, id, supply_rates=None, default_supply_rate=0):
        super().__init__(id)
        self.supply_rates = supply_rates if supply_rates is not None else []
        self.default_supply_rate = default_supply_rate
        self.supply_history = []

    def get_supply_rate(self, time_step):
        if time_step < len(self.supply_rates):
            return self.supply_rates[time_step]
        return self.default_supply_rate

    def update(self, time_step):
        current_supply_rate = self.get_supply_rate(time_step)
        self.supply_history.append(current_supply_rate)

        # Distribute supply among outflow edges
        total_capacity = sum(edge.capacity for edge in self.outflows.values())
        if total_capacity > 0:
            for edge in self.outflows.values():
                edge_flow = (edge.capacity / total_capacity) * current_supply_rate
                edge.update(time_step, edge_flow)
        else:
            for edge in self.outflows.values():
                edge.update(time_step, 0)

class SinkNode(Node):
    """
    Represents a point where water exits the system.
    """

    def update(self, time_step):
        """
        Update the SinkNode's state for the given time step.

        This method calculates the total inflow to the sink node.
        Sink nodes remove all incoming water from the system.

        Args:
            time_step (int): The current time step of the simulation.
        """
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        # Sink nodes remove all incoming water from the system

class DemandNode(Node):
    def __init__(self, id, demand_rate):
        super().__init__(id)
        self.demand_rate = demand_rate
        self.satisfied_demand = []
        self.excess_flow = []

    def update(self, time_step):
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        satisfied = min(total_inflow, self.demand_rate)
        excess = max(0, total_inflow - self.demand_rate)
        
        self.satisfied_demand.append(satisfied)
        self.excess_flow.append(excess)

        # Distribute excess water to outflow edges
        total_outflow_capacity = sum(edge.capacity for edge in self.outflows.values())
        if total_outflow_capacity > 0:
            for edge in self.outflows.values():
                edge_flow = (edge.capacity / total_outflow_capacity) * excess
                edge.update(time_step, edge_flow)
        else:
            for edge in self.outflows.values():
                edge.update(time_step, 0)

    def get_satisfied_demand(self, time_step):
        if time_step < len(self.satisfied_demand):
            return self.satisfied_demand[time_step]
        return 0
        
    def get_satisfied_demand(self, time_step):
        if time_step < len(self.satisfied_demand):
            return self.satisfied_demand[time_step]
        return 0

class StorageNode(Node):
    def __init__(self, id, capacity, initial_storage=0):
        super().__init__(id)
        self.capacity = capacity
        self.storage = [initial_storage]

    def update(self, time_step):
        inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        previous_storage = self.storage[-1]
        available_water = previous_storage + inflow
        
        # Calculate total requested outflow
        requested_outflow = sum(edge.capacity for edge in self.outflows.values())
        
        # Limit actual outflow to available water
        actual_outflow = min(available_water, requested_outflow)
        
        # Distribute actual outflow among edges proportionally
        if requested_outflow > 0:
            for edge in self.outflows.values():
                edge_flow = (edge.capacity / requested_outflow) * actual_outflow
                edge.update(time_step, edge_flow)
        else:
            for edge in self.outflows.values():
                edge.update(time_step, 0)
        
        # Calculate new storage
        new_storage = available_water - actual_outflow
        self.storage.append(min(new_storage, self.capacity))

    def get_storage(self, time_step):
        if time_step < len(self.storage):
            return self.storage[time_step]
        return self.storage[-1]

class DiversionNode(Node):
    """
    Represents a point where water can be diverted from one path to another.
    """

    def update(self, time_step):
        """
        Update the DiversionNode's state for the given time step.

        This method calculates the total inflow and distributes it among outflow edges
        based on their capacities.

        Args:
            time_step (int): The current time step of the simulation.
        """
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        total_outflow_capacity = sum(edge.capacity for edge in self.outflows.values())

        if total_outflow_capacity > 0:
            for edge in self.outflows.values():
                # Distribute water proportionally based on edge capacity
                edge_flow = (edge.capacity / total_outflow_capacity) * total_inflow
                edge.update(time_step, edge_flow)
        else:
            # If there's no outflow capacity, set all outflows to 0
            for edge in self.outflows.values():
                edge.update(time_step, 0)

class ConfluenceNode(Node):
    """
    Represents a point where multiple water flows combine.
    """

    def update(self, time_step):
        """
        Update the ConfluenceNode's state for the given time step.

        This method calculates the total inflow from all incoming edges and
        distributes it among the outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
        """
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        total_outflow_capacity = sum(edge.capacity for edge in self.outflows.values())

        if total_outflow_capacity > 0:
            for edge in self.outflows.values():
                # Distribute water proportionally based on edge capacity
                edge_flow = (edge.capacity / total_outflow_capacity) * total_inflow
                edge.update(time_step, edge_flow)
        else:
            # If there's no outflow capacity, set all outflows to 0
            for edge in self.outflows.values():
                edge.update(time_step, 0)

class HydroWorks(Node):
    """
    Represents a point where water can be redistributed, combining the functionality
    of diversion and confluence points.
    """

    def update(self, time_step):
        """
        Update the HydroWorks node's state for the given time step.

        This method calculates the total inflow and distributes it among outflow edges
        based on their capacities.

        Args:
            time_step (int): The current time step of the simulation.
        """
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        total_outflow_capacity = sum(edge.capacity for edge in self.outflows.values())

        if total_outflow_capacity > 0:
            for edge in self.outflows.values():
                # Distribute water proportionally based on edge capacity
                edge_flow = (edge.capacity / total_outflow_capacity) * total_inflow
                edge.update(time_step, edge_flow)
        else:
            # If there's no outflow capacity, set all outflows to 0
            for edge in self.outflows.values():
                edge.update(time_step, 0)