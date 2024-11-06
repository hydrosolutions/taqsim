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
        outflow (list): A list of outflow values for each time step of the simulation.
        length (float): The length of the canals in the irrigation/demand system [km]
        loss_factor (float): The loss factor per unit distance [fraction/km].
        inflow (list): A list of inflow values before losses.
        losses (list): A list of total losses for each time step.
    """

    def __init__(self, source, target, capacity, length=None, loss_factor=0.0):
        """
        Initialize an Edge object.

        Args:
            source (Node): The source node of the edge.
            target (Node): The target node of the edge.
            capacity (float): The maximum flow capacity of the edge.
            loss_factor (float, optional): The loss factor per unit distance [fraction/km]. Defaults to 0 (0% per km)
        """
        self.source = source
        self.target = target
        self.capacity = capacity
        self.loss_factor = loss_factor
        self.outflow = []  # Changed from flow to outflow
        self.inflow = []
        self.losses = []

        self.source.add_outflow(self)
        self.target.add_inflow(self)

        if length is not None:
            self.length = length
        else:
            self.length = self.get_edge_length()

    def update(self, time_step, flow=None):
        """
        Update the flow of the edge for the given time step, accounting for losses.

        If a flow value is provided, it is used (capped at the edge's capacity).
        If the source is a SupplyNode, the flow is set to the minimum of the supply rate and the edge's capacity.
        Otherwise, the flow is set to 0.

        Args:
            time_step (int): The current time step of the simulation.
            flow (float, optional): The flow value to set for this time step. Defaults to None.
        """
        if flow is not None:
            input_flow = min(flow, self.capacity)
        elif isinstance(self.source, SupplyNode):
            supply_rate = self.source.get_supply_rate(time_step)
            input_flow = min(supply_rate, self.capacity)
        else:
            input_flow = 0
            
        # Record the inflow before losses
        self.inflow.append(input_flow)
        
        # Calculate remaining flow after losses
        remaining_flow, losses = self.calculate_edge_losses(input_flow)
        
        # Record the flow after losses and the total losses
        self.outflow.append(remaining_flow)  # Changed from flow to outflow
        self.losses.append(losses)

    def get_edge_inflow(self, time_step):
        """
        Get the inflow value for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the inflow value.

        Returns:
            float: The inflow value for the specified time step, or 0 if the time step is out of range.
        """
        if time_step < len(self.inflow):
            return self.inflow[time_step]
        return 0

    def get_edge_outflow(self, time_step):
        """
        Get the outflow value for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the outflow value.

        Returns:
            float: The outflow value for the specified time step, or 0 if the time step is out of range.
        """
        if time_step < len(self.outflow):
            return self.outflow[time_step]
        return 0
    
    def get_edge_length(self):
        """
        Calculate the length of the edge using the easting and northing coordinates
        of the source and target nodes.

        Returns:
            float: The Euclidean distance between the source and target nodes. [km]
        """
        delta_easting = self.source.easting - self.target.easting
        delta_northing = self.source.northing - self.target.northing
        
        return math.sqrt(delta_easting**2 + delta_northing**2)/1000
    
    def calculate_edge_losses(self, flow):
        """
        Calculate water losses along the edge based on distance and loss factor.
        
        Args:
            flow (float): The initial flow rate entering the edge [m³/s].
            
        Returns:
            tuple: (remaining_flow, losses) where:
                - remaining_flow (float): Flow rate after losses [m³/s]
                - losses (float): Total flow lost along the edge [m³/s]
        """
        # Calculate loss fraction based on length and loss factor
        total_loss_fraction = 1 - (1 - self.loss_factor)**self.length
        
        # Ensure loss fraction doesn't exceed 1 (100%)
        total_loss_fraction = min(total_loss_fraction, 1.0)
        
        # Calculate losses and remaining flow
        losses = flow * total_loss_fraction
        remaining_flow = flow - losses
        
        return remaining_flow, losses