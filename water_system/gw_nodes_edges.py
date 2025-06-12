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
            max_thickness (float): Maximum saturated thickness [m]
            porosity (float): Effective porosity [-]
            initial_head (float): Initial hydraulic head above datum [m]
        """
        # Validate inputs using existing validation functions
        validate_node_id(id, "AquiferNode")
        validate_coordinates(easting, northing, id)
        validate_positive_float(area, "area")
        validate_positive_float(max_thickness, "max_thickness")
        validate_positive_float(porosity, "porosity")
        validate_nonnegativity_int_or_float(initial_head, "initial_head")
        
        if porosity >= 1.0:
            raise ValueError(f"Porosity ({porosity}) must be less than 1.0")
        if initial_head > max_thickness:
            raise ValueError(f"Initial head ({initial_head}) cannot exceed max thickness ({max_thickness})")
            
        self.id = id
        AquiferNode.all_ids.append(id)
        self.easting = easting
        self.northing = northing
        self.area = area
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
        effective_head = max(0, min(head, self.max_thickness))
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
        return storage / (self.area * self.porosity)
    
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
                    # For GroundwaterEdge, calculate flow based on current head
                    outflow = edge.calculate_flow(self.current_head)
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
    Groundwater edge using Darcy's law for horizontal flow calculation.
    
    Flow is calculated as: Q = K * A * (h1 - h2) / L
    where:
    - K is hydraulic conductivity [m/s]
    - A is cross-sectional area [m²] 
    - h1, h2 are hydraulic heads [m]
    - L is distance between nodes [m]
    """
    
    def __init__(
        self,
        source,
        target, 
        conductivity: float,
        area: float,
        length: Optional[float] = None
    ) -> None:
        """
        Initialize groundwater edge with Darcy's law parameters.
        
        Args:
            source: Source node (usually AquiferNode)
            target: Target node (AquiferNode, WellNode, or SinkNode)
            conductivity (float): Hydraulic conductivity [m/s]
            area (float): Cross-sectional area for flow [m²]
            length (float, optional): Distance between nodes [m]. If None, calculated from coordinates.
        """
        validate_positive_float(conductivity, "conductivity")
        validate_positive_float(area, "area")
        if length is not None:
            validate_positive_float(length, "length")
        
        self.source = source
        self.target = target
        self.conductivity = conductivity  # K [m/s]
        self.area = area  # A [m²]
        
        # Calculate length if not provided
        if length is not None:
            self.length = length
        else:
            self.length = self._calculate_length()
        
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
    
    def calculate_flow(self, source_head: float) -> float:
        """
        Calculate flow using Darcy's law: Q = K * A * (h1 - h2) / L
        
        Args:
            source_head (float): Hydraulic head at source [m]
            
        Returns:
            float: Flow rate [m³/s] (positive = flow from source to target)
        """
        try:
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
            
        except Exception as e:
            print(f"Error calculating flow for edge {self.source.id}->{self.target.id}: {e}")
            return 0.0
    
    def update(self, flow: Optional[float] = None) -> None:
        """
        Update edge flow for current time step.
        
        Args:
            flow (float, optional): Specified flow rate. If None, uses last calculated flow.
        """
        if flow is None:
            flow = 0.0
        
        self.flow_history.append(max(0, flow))
    
    @property
    def flow_after_losses(self) -> List[float]:
        """Compatibility with existing edge interface."""
        return self.flow_history

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

    """
    Extend existing water system with simplified groundwater components.
    
    Args:
        base_system: Existing WaterSystem instance
        aquifer_config: Dictionary with aquifer parameters
    
    Returns:
        Enhanced system with groundwater
    """
    if aquifer_config is None:
        aquifer_config = {
            'area': 1e8,  # 100 km²
            'max_thickness': 50.0,  # 50 m
            'porosity': 0.2,  # 20%
            'initial_head': 25.0  # 25 m
        }
    
    # Create main aquifer
    main_aquifer = AquiferNode(
        id="MainAquifer",
        easting=300000,
        northing=4400000,
        **aquifer_config
    )
    
    # Add to system (assuming WaterSystem has been extended with groundwater methods)
    if hasattr(base_system, 'add_aquifer_node'):
        base_system.add_aquifer_node(main_aquifer)
    else:
        # Fallback to generic node addition
        base_system.add_node(main_aquifer)
    
    return base_system