"""
This module defines the various types of nodes that can exist in a water system simulation.

The module includes a base Node class and several specialized node types such as
SupplyNode, SinkNode, DemandNode, StorageNode, DiversionNode, and ConfluenceNode.
Each node type has its own behavior for handling water inflows and outflows.
"""

from .edge import Edge

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
    """
    Represents a water supply source in the system.

    Attributes:
        supply_rate (float): The rate at which this node supplies water to the system.
    """

    def __init__(self, id, supply_rate):
        """
        Initialize a SupplyNode object.

        Args:
            id (str): A unique identifier for the node.
            supply_rate (float): The rate at which this node supplies water.
        """
        super().__init__(id)
        self.supply_rate = supply_rate

    def update(self, time_step):
        """
        Update the SupplyNode's state for the given time step.

        This method updates all outflow edges with the supply rate.

        Args:
            time_step (int): The current time step of the simulation.
        """
        for edge in self.outflows.values():
            edge.update(time_step)

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
    """
    Represents a point of water demand in the system.

    Attributes:
        demand_rate (float): The rate at which this node demands water from the system.
    """

    def __init__(self, id, demand_rate):
        """
        Initialize a DemandNode object.

        Args:
            id (str): A unique identifier for the node.
            demand_rate (float): The rate at which this node demands water.
        """
        super().__init__(id)
        self.demand_rate = demand_rate

    def update(self, time_step):
        """
        Update the DemandNode's state for the given time step.

        This method calculates the total inflow and determines how much of the
        demand is satisfied. Any excess water continues downstream.

        Args:
            time_step (int): The current time step of the simulation.
        """
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        satisfied_demand = min(total_inflow, self.demand_rate)
        # Remaining water continues downstream

class StorageNode(Node):
    """
    Represents a water storage facility in the system, such as a reservoir or lake.

    This class ensures proper water balance and prevents negative storage. It manages
    inflows, outflows, and storage capacity constraints.

    Attributes:
        capacity (float): The maximum amount of water that can be stored.
        storage (list of float): The amount of water stored at each time step.
    """

    def __init__(self, id, capacity):
        """
        Initialize a StorageNode object.

        Args:
            id (str): A unique identifier for the node.
            capacity (float): The maximum storage capacity of the node.
        """
        super().__init__(id)
        self.capacity = capacity
        self.storage = [0]  # Initialize with zero storage

    def update(self, time_step):
        """
        Update the StorageNode's state for the given time step.

        This method calculates the new storage level based on inflows, outflows,
        and storage capacity. It ensures water balance is maintained and prevents
        negative storage.

        Args:
            time_step (int): The current time step of the simulation.
        """
        # Calculate total inflow and available water
        inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        previous_storage = self.storage[-1]
        available_water = previous_storage + inflow

        # Calculate potential outflows based on edge capacities and available water
        potential_outflows = {}
        for edge in self.outflows.values():
            potential_outflows[edge] = min(edge.capacity, available_water)

        # Adjust outflows if total potential outflow exceeds available water
        total_potential_outflow = sum(potential_outflows.values())
        if total_potential_outflow > available_water:
            scale_factor = available_water / total_potential_outflow
            for edge, flow in potential_outflows.items():
                potential_outflows[edge] = flow * scale_factor

        # Update outflow edges with calculated flows
        for edge, flow in potential_outflows.items():
            edge.flow.append(flow)

        # Calculate actual outflow and new storage
        actual_outflow = sum(potential_outflows.values())
        new_storage = min(available_water - actual_outflow, self.capacity)
        self.storage.append(max(new_storage, 0))  # Ensure non-negative storage

class DiversionNode(Node):
    """
    Represents a point where water can be diverted from one path to another.
    """

    def update(self, time_step):
        """
        Update the DiversionNode's state for the given time step.

        This method calculates the total inflow and should implement logic
        to determine how water is diverted among outflow edges.

        Args:
            time_step (int): The current time step of the simulation.
        """
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflows.values())
        # Implement diversion logic here
        for edge in self.outflows.values():
            edge.update(time_step)

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
        # Distribute total inflow among outflow edges
        for edge in self.outflows.values():
            edge.update(time_step)