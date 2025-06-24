import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from .validation_functions import (validate_node_id, validate_coordinates, 
                                   validate_positive_float, validate_nonnegativity_int_or_float)

class AquiferNode:
    """
    Represents a groundwater aquifer using simplified conceptual model.
    
    Uses porosity and max_thickness instead of storage coefficient,
    compatible with existing node-edge framework.
    
    Attributes:
        id (str): Unique identifier for the aquifer node
        easting (float): Easting coordinate (UTM) [m]
        northing (float): Northing coordinate (UTM) [m]
        area (float): Aquifer area [m²]
        max_thickness (float): Maximum saturated thickness [m]
        porosity (float): Effective porosity [-]
        initial_head (float): Initial hydraulic head [m]
        current_head (float): Current hydraulic head [m]
        current_storage (float): Current storage volume [m³]
        storage_history (list): Record of storage volumes [m³]
        head_history (list): Record of hydraulic heads [m]
        inflow_edges (dict): Dictionary of inflow edges
        outflow_edges (dict): Dictionary of outflow edges
    """
    
    # Class variable to track all instances
    all_ids = []
    
    def __init__(
        self,
        id: str,
        easting: float,
        northing: float,
        area: float,
        bottom_elevation: float,
        max_thickness: float,
        porosity: float,
        initial_head: float
    ) -> None:
        """
        Initialize an AquiferNode.
        
        Args:
            id (str): Unique identifier
            easting (float): Easting coordinate [m]
            northing (float): Northing coordinate [m] 
            area (float): Aquifer area [m²]
            bottom_elevation (float): Bottom elevation of aquifer [m asl.]
            max_thickness (float): Maximum saturated thickness [m]
            porosity (float): Effective porosity [-]
            initial_head (float): Initial hydraulic head above datum [m asl.]
        """
        # Validate inputs using existing validation functions
        validate_node_id(id, "AquiferNode")
        validate_coordinates(easting, northing, id)
        validate_positive_float(bottom_elevation, "bottom_elevation")
        validate_positive_float(area, "area")
        # Convert area from km² to m²
        area_m2 = area * 1e6
        validate_positive_float(max_thickness, "max_thickness")
        validate_positive_float(porosity, "porosity")
        validate_nonnegativity_int_or_float(initial_head, "initial_head")
        
        if porosity >= 1.0:
            raise ValueError(f"Porosity ({porosity}) must be less than 1.0")
        if initial_head > (bottom_elevation+max_thickness):
            raise ValueError(f"Initial head ({initial_head}) cannot exceed max thickness ({bottom_elevation+max_thickness})")
            
        self.id = id
        AquiferNode.all_ids.append(id)
        self.easting = easting
        self.northing = northing
        self.area = area_m2  # Store area in m² internally
        self.bottom_elevation = bottom_elevation  # Bottom elevation [m asl.]
        self.max_thickness = max_thickness
        self.porosity = porosity
        
        # Initialize state variables
        self.current_head = initial_head
        self.current_storage = self._head_to_storage(initial_head)
        
        # History tracking
        self.storage_history = [self.current_storage]
        self.head_history = [initial_head]
        self.net_inflow_history = []
        
        # Edge connections
        self.inflow_edges = {}  # From surface water, recharge, or other aquifers
        self.outflow_edges = {}  # To streams, other aquifers, or wells
    
    def _head_to_storage(self, head: float) -> float:
        """
        Convert hydraulic head to storage volume.
        
        Storage = head * area * porosity
        
        Args:
            head (float): Hydraulic head [m]
            
        Returns:
            float: Storage volume [m³]
        """
        effective_head = max(0, min(head-self.bottom_elevation, self.max_thickness))
        return effective_head * self.area * self.porosity
    
    def _storage_to_head(self, storage: float) -> float:
        """
        Convert storage volume to hydraulic head.
        
        Head = storage / (area * porosity)
        
        Args:
            storage (float): Storage volume [m³]
            
        Returns:
            float: Hydraulic head [m]
        """
        max_storage = self.max_thickness * self.area * self.porosity
        storage = max(0, min(storage, max_storage))
        return storage / (self.area * self.porosity) + self.bottom_elevation
    
    def add_inflow_edge(self, edge):
        """Add an inflow edge (recharge or inter-aquifer flow)."""
        self.inflow_edges[edge.source.id] = edge
    
    def add_outflow_edge(self, edge):
        """Add an outflow edge (discharge to streams, wells, or other aquifers)."""
        self.outflow_edges[edge.target.id] = edge
    
    def update(self, time_step: int, dt: float) -> None:
        """
        Update aquifer state using water balance equation.
        
        dS/dt = Total_Inflow - Total_Outflow
        
        Args:
            time_step (int): Current time step index
            dt (float): Time step duration [s]
        """
        try:
            # Calculate total inflows (from all incoming edges)
            total_inflow = 0
            for edge in self.inflow_edges.values():
                if hasattr(edge, 'flow_after_losses') and time_step < len(edge.flow_after_losses):
                    total_inflow += edge.flow_after_losses[time_step]
                elif hasattr(edge, 'flow_history') and time_step < len(edge.flow_history):
                    total_inflow += edge.flow_history[time_step]
            
            # Calculate total outflows (this will be updated by outgoing edges)
            total_outflow = 0
            for edge in self.outflow_edges.values():
                if hasattr(edge, 'calculate_flow'):
                    # For GroundwaterEdge, calculate flow based on current conditions
                    outflow = edge.calculate_flow(time_step)
                    total_outflow += outflow
                    # Update the edge with calculated flow
                    edge.update(outflow)
            
            # Water balance: dS/dt = In - Out
            net_flow = total_inflow - total_outflow
            storage_change = net_flow * dt
            new_storage = max(0, self.current_storage + storage_change)
            
            # Update state variables
            self.current_storage = new_storage
            self.current_head = self._storage_to_head(new_storage)
            
            # Record history
            self.storage_history.append(new_storage)
            self.head_history.append(self.current_head)
            self.net_inflow_history.append(net_flow)
            
        except Exception as e:
            print(f"Error updating aquifer node {self.id}: {str(e)}")
            # Maintain last known state
            self.storage_history.append(self.current_storage)
            self.head_history.append(self.current_head)
            self.net_inflow_history.append(0)

class GroundwaterEdge:
    """
    Groundwater edge for different types of groundwater flow.
    
    Supports multiple flow types:
    1. Horizontal flow (Darcy): Q = K * A * (h1 - h2) / L
    2. Recharge flow: Q = fraction * source_discharge OR fraction * rainfall_volume
    3. Pumping flow: Q = specified rate
    4. Sink flow: Q = a * S^b
    """
    
    def __init__(
        self,
        source,
        target, 
        edge_type: str = "horizontal",
        conductivity: Optional[float] = None,
        area: Optional[float] = None,
        length: Optional[float] = None,
        recharge_fraction: Optional[float] = None,
        fixed_head: Optional[float] = None
    ) -> None:
        """
        Initialize groundwater edge with different flow types.
        
        Args:
            source: Source node (AquiferNode, SupplyNode, etc.)
            target: Target node (AquiferNode, WellNode, or SinkNode)
            edge_type (str): Type of flow - "horizontal", "recharge", or "pumping"
            conductivity (float, optional): Hydraulic conductivity [m/s] (for horizontal flow)
            area (float, optional): Cross-sectional area [m²] (for horizontal flow)
            length (float, optional): Distance between nodes [m] (for horizontal flow)
            recharge_fraction (float, optional): Fraction of source discharge (for recharge flow)
        """
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.flow_after_losses = []
        self.flow_before_losses = []
        self.losses = []
        
        # Validate inputs based on edge type
        if edge_type == "horizontal":
            if conductivity is None or area is None:
                raise ValueError("Horizontal flow requires conductivity and area parameters")
            validate_positive_float(conductivity, "conductivity")
            validate_positive_float(area, "area")
            if length is not None:
                validate_positive_float(length, "length")
                
            self.conductivity = conductivity  # K [m/s]
            self.area = area  # A [m²]
            self.length = length if length is not None else self._calculate_length()
            
        elif edge_type == "recharge":
            if recharge_fraction is None:
                raise ValueError("Recharge flow requires recharge_fraction parameter")
            if not (0 <= recharge_fraction <= 1):
                raise ValueError(f"Recharge fraction ({recharge_fraction}) must be between 0 and 1")
            self.recharge_fraction = recharge_fraction
            
        elif edge_type == "pumping":
            # Pumping flow is controlled by the target (well) node
            pass
        elif edge_type == "sink":
            if conductivity is None or area is None:
                raise ValueError("Horizontal flow requires conductivity and area parameters")
            validate_positive_float(conductivity, "conductivity")
            validate_positive_float(area, "area")
            if length is not None:
                validate_positive_float(length, "length")
            if fixed_head is None:
                raise ValueError("Sink flow requires fixed_head parameter")
            validate_positive_float(fixed_head, "fixed_head")
                
            self.conductivity = conductivity  # K [m/s]
            self.area = area  # A [m²]
            self.length = length if length is not None else self._calculate_length()
            self.fixed_head = fixed_head  # Fixed head at the sink node
            
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}. Must be 'horizontal', 'recharge', or 'pumping'")
        
        # Flow tracking
        self.flow_history = []
        
        # Connect to nodes
        if hasattr(source, 'add_outflow_edge'):
            source.add_outflow_edge(self)
        if hasattr(target, 'add_inflow_edge'):
            target.add_inflow_edge(self)
    
    def _calculate_length(self) -> float:
        """
        Calculate distance between source and target nodes.
        
        Returns:
            float: Distance [m]
        """
        try:
            if (hasattr(self.source, 'easting') and hasattr(self.source, 'northing') and
                hasattr(self.target, 'easting') and hasattr(self.target, 'northing')):
                
                dx = self.source.easting - self.target.easting
                dy = self.source.northing - self.target.northing
                return np.sqrt(dx**2 + dy**2)
            else:
                print(f"Warning: Could not calculate length for edge {self.source.id}->{self.target.id}. Using default 1000m.")
                return 1000.0  # Default length
        except Exception as e:
            print(f"Error calculating edge length: {e}. Using default 1000m.")
            return 1000.0
    
    def calculate_flow(self, time_step: int) -> float:
        """
        Calculate flow based on edge type.
        
        Args:
            time_step (int): Current time step index
            
        Returns:
            float: Flow rate [m³/s]
        """
        try:
            if self.edge_type == "horizontal":
                return self._calculate_horizontal_flow(time_step)
                
            elif self.edge_type == "recharge":
                return self._calculate_recharge_flow(time_step)
                
            elif self.edge_type == "pumping":
                return self._calculate_pumping_flow(time_step)
            
            elif self.edge_type == "sink":
                return self._calculate_sink_flow(time_step)
                
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating flow for edge {self.source.id}->{self.target.id}: {e}")
            return 0.0
    
    def _calculate_horizontal_flow(self, time_step:int) -> float:
        """
        Calculate horizontal flow using Darcy's law: Q = K * A * (h1 - h2) / L
        
        Returns:
            float: Flow rate [m³/s] (positive = flow from source to target)
        """
        # Get source head
        source_head = 0.0
        if hasattr(self.source, 'head_history') and time_step < len(self.source.head_history):
            source_head = self.source.head_history[time_step]
        
        # Get target head
        target_head = 0.0
        if hasattr(self.target, 'current_head'):
            target_head = self.target.current_head
        elif hasattr(self.target, 'water_level') and len(self.target.water_level) > 0:
            # For surface water nodes that might have water levels
            target_head = self.target.water_level[-1]
        
        # Calculate head difference
        head_diff = source_head - target_head
        
        # Apply Darcy's law: Q = K * A * dh / L
        flow = self.conductivity * self.area * head_diff / self.length
        
        # Ensure flow is non-negative (no reverse flow for simplicity)
        return max(0, flow)
    
    def _calculate_recharge_flow(self, time_step: int) -> float:
        """
        Calculate recharge flow as fraction of source discharge.
        
        Args:
            time_step (int): Current time step index
            
        Returns:
            float: Recharge flow rate [m³/s]
        """
        # Get source discharge
        source_discharge = 0.0
        
        if hasattr(self.source, 'supply_rates') and time_step < len(self.source.supply_rates):
            # For SupplyNode
            source_discharge = self.source.supply_rates[time_step]

        elif hasattr(self.source, 'rainfall_discharge') and time_step < len(self.source.rainfall_discharge):
            # For RainfallNode
            source_discharge = self.source.rainfall_discharge[time_step]

        # Calculate recharge as fraction of source discharge
        return source_discharge * self.recharge_fraction
    
    def _calculate_pumping_flow(self, time_step: int) -> float:
        """
        Calculate pumping flow (controlled by target well).
        
        Args:
            time_step (int): Current time step index
            
        Returns:
            float: Pumping flow rate [m³/s]
        """
        if hasattr(self.target, 'pumping_history') and time_step < len(self.target.pumping_history):
            return self.target.pumping_history[time_step]
        elif hasattr(self.target, 'max_pumping_rate'):
            # Default to some fraction of max pumping if no history available
            return self.target.max_pumping_rate * 0.5  # 50% as default
        return 0.0
    
    def _calculate_sink_flow(self, time_step: int) -> float:

        # Get source head
        source_head = 0.0
        if hasattr(self.source, 'head_history') and time_step < len(self.source.head_history):
            source_head = self.source.head_history[time_step]

        # Get target head
        target_head = self.fixed_head
        # Calculate head difference
        head_diff = source_head - target_head
        
        # Apply Darcy's law: Q = K * A * dh / L
        flow = self.conductivity * self.area * head_diff / self.length
        
        # Ensure flow is non-negative (no reverse flow for simplicity)
        return max(0, flow)

    def update(self, flow) -> None:
        """
        Update edge flow for current time step.
        
        Args:
            time_step (int): Current time step index
        """
        self.flow_before_losses.append(flow)
        self.flow_after_losses.append(flow)  # Initially same as before losses
        self.losses.append(0)
        self.flow_history.append(max(0, flow))
    
class WellNode:
    """
    Represents a groundwater extraction well.
    
    Simplified version focusing on pumping rate constraints.
    """
    
    all_ids = []
    
    def __init__(
        self,
        id: str,
        easting: float,
        northing: float,
        max_pumping_rate: float
    ) -> None:
        """
        Initialize a WellNode.
        
        Args:
            id (str): Unique identifier
            easting (float): Easting coordinate [m]
            northing (float): Northing coordinate [m]
            max_pumping_rate (float): Maximum pumping rate [m³/s]
        """
        validate_node_id(id, "WellNode")
        validate_coordinates(easting, northing, id)
        validate_positive_float(max_pumping_rate, "max_pumping_rate")
        
        self.id = id
        WellNode.all_ids.append(id)
        self.easting = easting
        self.northing = northing
        self.max_pumping_rate = max_pumping_rate
        
        # State tracking
        self.pumping_history = []
        self.current_head = 0.0  # Will be updated by connected aquifer
        
        # Connections
        self.inflow_edges = {}  # From aquifer
        self.outflow_edges = {}  # To demand points
    
    def add_inflow_edge(self, edge):
        """Add connection from aquifer."""
        self.inflow_edges[edge.source.id] = edge
    
    def add_outflow_edge(self, edge):
        """Add connection to demand point."""
        self.outflow_edges[edge.target.id] = edge
    
    def update(self, time_step: int, dt: float) -> None:
        """
        Update well state. Pumping is controlled by connected GroundwaterEdges.
        
        Args:
            time_step (int): Current time step
            dt (float): Time step duration [s]
        """
        try:
            # Calculate total inflow (pumping from aquifers)
            total_pumping = 0
            for edge in self.inflow_edges.values():
                if hasattr(edge, 'flow_history') and time_step < len(edge.flow_history):
                    total_pumping += edge.flow_history[time_step]
            
            # Record pumping
            self.pumping_history.append(total_pumping)
            
            # Distribute pumped water to outflow edges
            if self.outflow_edges and total_pumping > 0:
                flow_per_outlet = total_pumping / len(self.outflow_edges)
                for edge in self.outflow_edges.values():
                    edge.update(flow_per_outlet)
            
        except Exception as e:
            print(f"Error updating well node {self.id}: {str(e)}")
            self.pumping_history.append(0)
