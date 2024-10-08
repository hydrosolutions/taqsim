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
        inflow_edges (dict): A dictionary of inflow edges, keyed by the source node's id.
        outflow_edges (dict): A dictionary of outflow edges, keyed by the target node's id.
    """

    def __init__(self, id):
        """
        Initialize a Node object.

        Args:
            id (str): A unique identifier for the node.
        """
        self.id = id
        self.inflow_edges = {}  # Dictionary of inflow edges
        self.outflow_edges = {}  # Dictionary of outflow edges

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
    Represents a water supply source in the system with time-varying supply rates.

    Attributes:
        id (str): A unique identifier for the node.
        supply_rates (list of float): The rates at which this node supplies water to the system at each time step.
        default_supply_rate (float): The default supply rate to use if no specific rate is provided for a time step.
        supply_history (list of float): The actual supply rates used at each time step of the simulation.
    """

    def __init__(self, id, supply_rates=None, default_supply_rate=0):
        """
        Initialize a SupplyNode object.

        Args:
            id (str): A unique identifier for the node.
            supply_rates (list of float, optional): A list of supply rates for each time step.
            default_supply_rate (float, optional): The default supply rate to use if no specific rate is provided.
        """
        super().__init__(id)
        self.supply_rates = supply_rates if supply_rates is not None else []
        self.default_supply_rate = default_supply_rate
        self.supply_history = []

    def get_supply_rate(self, time_step):
        """
        Get the supply rate for a specific time step.

        Args:
            time_step (int): The current time step of the simulation.

        Returns:
            float: The supply rate for the specified time step.
        """
        if time_step < len(self.supply_rates):
            return self.supply_rates[time_step]
        return self.default_supply_rate

    def update(self, time_step):
        """
        Update the SupplyNode's state for the given time step.

        This method updates all outflow edges with the current supply rate and
        records the actual supply used in the supply_history.

        Args:
            time_step (int): The current time step of the simulation.
        """
        current_supply_rate = self.get_supply_rate(time_step)
        remaining_supply = current_supply_rate
        total_supplied = 0

        for edge in self.outflow_edges.values():
            flow = min(remaining_supply, edge.capacity)
            edge.update(time_step, flow)
            remaining_supply -= flow
            total_supplied += flow

        self.supply_history.append(total_supplied)

        if remaining_supply > 0:
            print(f"Warning: Excess supply of {remaining_supply} at node {self.id} for time step {time_step}")

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
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflow_edges.values())
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
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflow_edges.values())
        satisfied_demand = min(total_inflow, self.demand_rate)
        # Remaining water continues downstream

class StorageNode(Node):
    """
    Represents a water storage facility in the system.

    Attributes:
        capacity (float): The maximum amount of water that can be stored.
        storage (list of float): The amount of water stored at each time step.
        inflow_edges (list of float): The total inflow at each time step.
        outflow_edges (list of float): The total outflow at each time step.
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
        self.storage = [0]
        self.inflow_edges = [0]
        self.outflow_edges = [0]

    def update(self, time_step):
        """
        Update the StorageNode's state for the given time step.

        This method calculates the new storage level based on inflow_edges, outflows,
        and storage capacity.

        Args:
            time_step (int): The current time step of the simulation.
        """
        inflow = sum(edge.get_flow(time_step) for edge in self.inflow_edges.values())
        self.inflow_edges.append(inflow)
        
        previous_storage = self.storage[-1]
        available_water = previous_storage + inflow
        
        for edge in self.outflow_edges.values():
            edge.update(time_step)
        
        outflow = sum(edge.get_flow(time_step) for edge in self.outflow_edges.values())
        #self.outflows.append(outflow)
        self.outflow_edges.append(outflow)
        
        new_storage = min(available_water - outflow, self.capacity)
        self.storage.append(max(new_storage, 0))

    def get_formatted_history(self):
        """
        Get a formatted string representation of the storage history.

        Returns:
            str: A formatted table showing inflow, outflow, and volume at each time step.
        """
        header = f"{'Time Step':^10}{'Inflow':^15}{'Outflow':^15}{'Volume':^15}"
        separator = "-" * 55
        rows = [header, separator]
        
        for t in range(len(self.storage)):
            row = f"{t:^10}{self.inflow_edges[t]:^15.2f}{self.outflow_edges[t]:^15.2f}{self.storage[t]:^15.2f}"
            rows.append(row)
        
        return "\n".join(rows)

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
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflow_edges.values())
        # Implement diversion logic here
        for edge in self.outflow_edges.values():
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
        total_inflow = sum(edge.get_flow(time_step) for edge in self.inflow_edges.values())
        # Distribute total inflow among outflow edges
        for edge in self.outflow_edges.values():
            edge.update(time_step)