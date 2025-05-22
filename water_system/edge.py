"""
This module defines the Edge class, which represents a connection between two nodes in a water system.

The Edge class is responsible for managing the flow of water between nodes and enforcing capacity constraints.
It supports specialized node types: SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, and RunoffNode.
"""

import math
from typing import Optional, Tuple, Union, Any
from .structure import SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode
from .validation_functions import validate_nonnegativity_int_or_float, validate_probability

# Forward references for type hints - these are imported in the Edge class
# but need to be referenced in type annotations
SupplyNodeType = Any
StorageNodeType = Any
DemandNodeType = Any
SinkNodeType = Any
HydroWorksType = Any
RunoffNodeType = Any
NodeType = Union[SupplyNodeType, StorageNodeType, DemandNodeType, SinkNodeType, HydroWorksType, RunoffNodeType]

class Edge:
    """
    Represents a connection between two nodes in a water system.

    Attributes:
        source: The source node of the edge.
        target: The target node of the edge.
        capacity (float): The maximum flow capacity of the edge in m³/s.
        flow_after_losses (list): A list of outflow values for each time step of the simulation.
        flow_before_losses (list): A list of inflow values before losses.
        length (float): The length of the canals in the irrigation/demand system [km]
        loss_factor (float): The loss factor per unit distance [fraction/km].
        losses (list): A list of total losses for each time step.
        ecological_flow (float): The minimum flow that must be maintained in the edge [m³/s].
        unmet_ecological_flow (list): A list of unmet ecological flow values for each time step.
    """
    edges_with_min_flow = {}  # Format: {(source_id, target_id): min_flow}

    def __init__(
        self,
        source: NodeType,
        target: NodeType,
        capacity: float,
        length: Optional[float] = None,
        loss_factor: float = 0, 
        ecological_flow: float = 0,
    ) -> None:
        """
        Initialize an Edge object.

        Args:
            source: The source node of the edge.
            target: The target node of the edge.
            capacity (float): The maximum flow capacity of the edge in m³/s.
            length (float, optional): The length of the edge in km. If None, calculated from coordinates.
            loss_factor (float, optional): The loss factor per unit distance [fraction/km]. Defaults to 0 (0% per km)
            ecological_flow (float, optional): The minimum flow that must be maintained in the edge [m³/s]. Defaults to 0.

        Raises:
            ValueError: If invalid parameters are provided (negative capacity, loss_factor, or length).
            AttributeError: If nodes cannot be connected or lack required attributes.
        """
        # Validate capacity
        validate_nonnegativity_int_or_float(capacity, "Edge capacity")
        # Validate loss factor
        validate_probability(loss_factor, "Edge loss factor")
        # Validate min flow
        validate_nonnegativity_int_or_float(ecological_flow, "Edge min flow")
        if ecological_flow > capacity:
            raise ValueError(f"Ecological flow from {source} to {target} ({ecological_flow}) cannot exceed capacity {capacity}")

        self.source = source
        self.target = target
        self.capacity = capacity
        self.loss_factor = loss_factor
        self.flow_after_losses = []
        self.flow_before_losses = []
        self.losses = []
        self.ecological_flow = ecological_flow
        self.unmet_ecological_flow = []

        if self.ecological_flow > 0:
            # Store the minimum flow for this edge using tuple as key
            Edge.edges_with_min_flow[(self.source.id, self.target.id)] = self.ecological_flow

        # Set or calculate length
        if length is not None:
            validate_nonnegativity_int_or_float(length, "Edge length")
            self.length = length
        else:
            try:
                self.length = self.get_edge_length()
            except AttributeError:
                print(f"Warning: Could not calculate edge length from coordinates. Using length = 0.")
                self.length = 0

        # Try to connect nodes
        try:
            # Define valid node type connections
            if isinstance(source, (SupplyNode, RunoffNode)):
                # Supply and RunoffNodes should only have add_outflow_edge
                source.add_outflow_edge(self)
            elif isinstance(source, (StorageNode, DemandNode, HydroWorks)):
                # These nodes have both inflow and outflow
                source.add_outflow_edge(self)
            else:
                raise AttributeError(f"Invalid source node type: {type(source).__name__}")
            
            if isinstance(target, SinkNode):
                # SinkNodes should only have add_inflow_edge
                target.add_inflow_edge(self)
            elif isinstance(target, (StorageNode, DemandNode, HydroWorks)):
                # These nodes have both inflow and outflow
                target.add_inflow_edge(self)
            else:
                raise AttributeError(f"Invalid target node type: {type(target).__name__}")
                
        except AttributeError as e:
            raise AttributeError(f"Failed to connect nodes: {str(e)}")

    def update(self, flow: Optional[float] = None) -> None:
        """
        Update the flow of the edge for the given time step, accounting for losses.

        Args:
            time_step (int): The current time step of the simulation
            flow (float, optional): The flow value to set for this time step
        """
        try:
            if flow is None or flow < 0:
                input_flow = 0
            else:
                input_flow = min(flow, self.capacity)
            
            # Record the inflow before losses
            self.flow_before_losses.append(input_flow)
            # Check for ecological flow requirement
            if self.ecological_flow > 0:
                if input_flow < self.ecological_flow:
                    unmet_flow = self.ecological_flow - input_flow
                    self.unmet_ecological_flow.append(unmet_flow)
            
            # Calculate and record losses
            remaining_flow, losses = self.calculate_edge_losses(input_flow)
            self.flow_after_losses.append(remaining_flow)
            self.losses.append(losses)

        except Exception as e:
            print(f"Error updating edge flow: {str(e)}. Flow values may be incorrect.")
            self.flow_before_losses.append(0)
            self.flow_after_losses.append(0)
            self.losses.append(0)
    
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