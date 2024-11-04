import math
"""
This module defines the Edge class, which represents a connection between two nodes in a water system.

The Edge class is responsible for managing the flow of water between nodes and enforcing capacity constraints.
"""

from .structure import SupplyNode

class Edge:
    """
    Represents a connection between two nodes in a water system.

    Attributes:
        source (Node): The source node of the edge.
        target (Node): The target node of the edge.
        capacity (float): The maximum flow capacity of the edge.
        flow (list): A list of flow values for each time step of the simulation.
        length (float): The length of the canals in the irrigation/demand system [km]
    """

    def __init__(self, source, target, capacity, length=None):
        """
        Initialize an Edge object.

        Args:
            source (Node): The source node of the edge.
            target (Node): The target node of the edge.
            capacity (float): The maximum flow capacity of the edge.
        """
        self.source = source
        self.target = target
        self.capacity = capacity
        self.flow = []
        self.source.add_outflow(self)
        self.target.add_inflow(self)
        if length is not None:
            self.length=length
        else:
            self.length = self.get_edge_length() 

    def update(self, time_step, flow=None):
        """
        Update the flow of the edge for the given time step.

        If a flow value is provided, it is used (capped at the edge's capacity).
        If the source is a SupplyNode, the flow is set to the minimum of the supply rate and the edge's capacity.
        Otherwise, the flow is set to 0.

        Args:
            time_step (int): The current time step of the simulation.
            flow (float, optional): The flow value to set for this time step. Defaults to None.
        """
        if flow is not None:
            self.flow.append(min(flow, self.capacity))
        elif isinstance(self.source, SupplyNode):
            supply_rate = self.source.get_supply_rate(time_step)
            self.flow.append(min(supply_rate, self.capacity))
        else:
            # This case should not occur with the current node update methods
            self.flow.append(0)

    def get_flow(self, time_step):
        """
        Get the flow value for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the flow value.

        Returns:
            float: The flow value for the specified time step, or 0 if the time step is out of range.
        """
        if time_step < len(self.flow):
            return self.flow[time_step]
        return 0
    
    def get_edge_length(self):
        """
        Calculate the length of the edge using the easting and northing coordinates
        of the source and target nodes.

        Returns:
            float: The Euclidean distance between the source and target nodes. [m]
        """
        delta_easting = self.source.easting - self.target.easting
        delta_northing = self.source.northing - self.target.northing
        
        return math.sqrt(delta_easting**2 + delta_northing**2)