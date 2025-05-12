import math
"""
This module defines the Edge class, which represents a connection between two nodes in a water system.

The Edge class is responsible for managing the flow of water between nodes and enforcing capacity constraints.
"""

from .structure import Node, SupplyNode
from typing import Optional, Tuple

class Edge:
    """
    Represents a connection between two nodes in a water system.

    Attributes:
        source (Node): The source node of the edge.
        target (Node): The target node of the edge.
        capacity (float): The maximum flow capacity of the edge.
        flow_after_losses (list): A list of outflow values for each time step of the simulation.
        length (float): The length of the canals in the irrigation/demand system [km]
        loss_factor (float): The loss factor per unit distance [fraction/km].
        inflow (list): A list of inflow values before losses.
        losses (list): A list of total losses for each time step.
    """

    def __init__(
        self,
        source: Node,
        target: Node,
        capacity: float,
        length: Optional[float] = None,
        loss_factor: float = 0
    ) -> None:
        """
        Initialize an Edge object.

        Args:
            source (Node): The source node of the edge.
            target (Node): The target node of the edge.
            capacity (float): The maximum flow capacity of the edge.
            length (float, optional): The length of the edge in km. If None, calculated from coordinates.
            loss_factor (float, optional): The loss factor per unit distance [fraction/km]. Defaults to 0 (0% per km)
        
        Raises:
            ValueError: If invalid parameters are provided (negative capacity, loss_factor, or length).
            AttributeError: If nodes cannot be connected or lack required attributes.
        """
        # Validate inputs
        if capacity <= 0:
            raise ValueError("Edge capacity can not be negative")
        if loss_factor < 0 or loss_factor>1:
            raise ValueError("Edge loss factor must be between 0 and 1")
         
        self.source = source
        self.target = target
        self.capacity = capacity
        self.loss_factor = loss_factor
        self.flow_after_losses = []
        self.flow_before_losses = []
        self.losses = []

        # Try to connect nodes
        try:
            self.source.add_outflow_edge(self)
            self.target.add_inflow_edge(self)
        except AttributeError as e:
            raise AttributeError(f"Failed to connect nodes: {str(e)}")

        # Set or calculate length
        if length is not None:
            if length < 0:
                raise ValueError("Edge length cannot be negative")
            self.length = length
        else:
            try:
                self.length = self.get_edge_length()
            except AttributeError:
                print(f"Warning: Could not calculate edge length from coordinates. Using length = 0.")
                self.length = 0

    def update(self, time_step: int, flow: Optional[float] = None) -> None:
        """
        Update the flow of the edge for the given time step, accounting for losses.

        Args:
            time_step (int): The current time step of the simulation
            flow (float, optional): The flow value to set for this time step
        """
        try:
            if flow is not None:
                # Use provided flow (already calculated by source node)
                if flow < 0:
                    print(f"Warning: Negative flow value ({flow}) provided, setting to 0")
                    flow = 0
                input_flow = min(flow, self.capacity)
            elif isinstance(self.source, SupplyNode):
                # Special case for SupplyNodes which don't pre-calculate their outflows
                try:
                    supply_rate = self.source.get_supply_rate(time_step)
                    input_flow = min(supply_rate, self.capacity)
                except Exception as e:
                    print(f"Warning: Failed to get supply rate: {str(e)}. Setting input flow to 0.")
                    input_flow = 0
            else:
                input_flow = 0
            
            # Record the inflow before losses
            self.flow_before_losses.append(input_flow)
            
            # Calculate and record losses
            remaining_flow, losses = self.calculate_edge_losses(input_flow)
            self.flow_after_losses.append(remaining_flow)
            self.losses.append(losses)

        except Exception as e:
            print(f"Error updating edge flow: {str(e)}. Flow values may be incorrect.")
            self.flow_before_losses.append(0)
            self.flow_after_losses.append(0)
            self.losses.append(0)

    def get_edge_flow_before_losses(self, time_step: int) -> float:
        """
        Get the inflow value for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the inflow value.

        Returns:
            float: The inflow value for the specified time step, or 0 if the time step is out of range.
        """
        try:
            return self.flow_before_losses[time_step] if time_step < len(self.flow_before_losses) else 0
        except Exception as e:
            print(f"Error retrieving edge inflow for time step {time_step}: {str(e)}")
            return 0

    def get_edge_flow_after_losses(self, time_step: int) -> float:
        """
        Get the outflow value for a specific time step.

        Args:
            time_step (int): The time step for which to retrieve the outflow value.

        Returns:
            float: The outflow value for the specified time step, or 0 if the time step is out of range.
        """
        try:
            return self.flow_after_losses[time_step] if time_step < len(self.flow_after_losses) else 0
        except Exception as e:
            print(f"Error retrieving edge outflow for time step {time_step}: {str(e)}")
            return 0
    
    def get_edge_length(self) -> float:
        """
        Calculate the length of the edge using the easting and northing coordinates
        of the source and target nodes.

        Returns:
            float: The Euclidean distance between the source and target nodes. [km]
        
        Raises:
            AttributeError: If nodes lack coordinate attributes.
            ValueError: If coordinate values are missing.
        """
        try:
            # Check for missing coordinates
            if any(attr is None for attr in [self.source.easting, self.source.northing, 
                                           self.target.easting, self.target.northing]):
                raise ValueError("Missing coordinate values")
            
            delta_easting = self.source.easting - self.target.easting
            delta_northing = self.source.northing - self.target.northing
            
            length = math.sqrt(delta_easting**2 + delta_northing**2)/1000
            
            if length == 0:
                print("Warning: Calculated edge length is 0 - source and target at same location")
                
            return length
            
        except AttributeError as e:
            raise AttributeError(f"Missing coordinate attributes: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to calculate edge length: {str(e)}")
    
    def calculate_edge_losses(self, flow: float) -> Tuple[float, float]:
        """
        Calculate water losses along the edge based on distance and loss factor.
        
        Args:
            flow (float): The initial flow rate entering the edge [m³/s].
            
        Returns:
            tuple: (remaining_flow, losses) where:
                - remaining_flow (float): Flow rate after losses [m³/s]
                - losses (float): Total flow lost along the edge [m³/s]
        """
        try:
            # Calculate loss fraction based on length and loss factor
            total_loss_fraction = 1 - (1 - self.loss_factor)**self.length
            
            # Ensure loss fraction doesn't exceed 1 (100%)
            total_loss_fraction = min(total_loss_fraction, 1.0)
            
            # Calculate losses and remaining flow
            losses = flow * total_loss_fraction
            remaining_flow = flow - losses
        
            return remaining_flow, losses
        except Exception as e:
            raise ValueError(f"Failed to calculate edge losses: {str(e)}")