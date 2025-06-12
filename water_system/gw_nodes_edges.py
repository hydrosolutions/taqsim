import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from .validation_functions import (validate_node_id, validate_coordinates, 
                                   validate_positive_float, validate_nonnegativity_int_or_float)

class AquiferNode:
    """
    Represents a groundwater aquifer using linear reservoir concept.
    
    Compatible with the existing node-edge framework while providing efficient
    groundwater simulation for optimization algorithms.
    
    Attributes:
        id (str): Unique identifier for the aquifer node
        easting (float): Easting coordinate (UTM)
        northing (float): Northing coordinate (UTM)
        storage_coefficient (float): Aquifer storage coefficient [-]
        area (float): Aquifer area [m²]
        recession_coefficient (float): Linear discharge coefficient [1/s]
        max_depth (float): Maximum saturated thickness [m]
        initial_head (float): Initial hydraulic head [m]
        storage_history (list): Record of storage volumes [m³]
        head_history (list): Record of hydraulic heads [m]
        baseflow_history (list): Record of baseflow discharge [m³/s]
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
        storage_coefficient: float = 0.1,
        recession_coefficient: float = 1e-7,
        max_depth: float = 50.0,
        initial_head: float = 10.0,
        min_head: float = 0.0
    ) -> None:
        """
        Initialize an AquiferNode with linear reservoir characteristics.
        
        Args:
            id (str): Unique identifier
            easting (float): Easting coordinate [m]
            northing (float): Northing coordinate [m] 
            area (float): Aquifer area [m²]
            storage_coefficient (float): Storage coefficient (specific yield) [-]
            recession_coefficient (float): Linear discharge coefficient [1/s]
            max_depth (float): Maximum saturated thickness [m]
            initial_head (float): Initial hydraulic head above datum [m]
            min_head (float): Minimum allowable head [m]
        """
        # Validate inputs using existing validation functions
        validate_node_id(id, "AquiferNode")
        validate_coordinates(easting, northing, id)
        validate_positive_float(area, "area")
        validate_positive_float(storage_coefficient, "storage_coefficient")
        validate_positive_float(recession_coefficient, "recession_coefficient")
        validate_positive_float(max_depth, "max_depth")
        validate_nonnegativity_int_or_float(initial_head, "initial_head")
        validate_nonnegativity_int_or_float(min_head, "min_head")
        
        if initial_head > max_depth:
            raise ValueError(f"Initial head ({initial_head}) cannot exceed max depth ({max_depth})")
        if min_head >= max_depth:
            raise ValueError(f"Min head ({min_head}) must be less than max depth ({max_depth})")
            
        self.id = id
        AquiferNode.all_ids.append(id)
        self.easting = easting
        self.northing = northing
        self.area = area
        self.storage_coefficient = storage_coefficient
        self.recession_coefficient = recession_coefficient
        self.max_depth = max_depth
        self.min_head = min_head
        
        # Initialize state variables
        self.current_head = initial_head
        self.current_storage = self._head_to_storage(initial_head)
        
        # History tracking
        self.storage_history = [self.current_storage]
        self.head_history = [initial_head]
        self.baseflow_history = []
        self.recharge_history = []
        self.pumping_history = []
        
        # Edge connections
        self.inflow_edges = {}  # From surface water or other aquifers
        self.outflow_edges = {}  # To streams, other aquifers, or sinks
        self.pumping_edges = {}  # Extraction wells
    
    def _head_to_storage(self, head: float) -> float:
        """Convert hydraulic head to storage volume."""
        effective_head = max(0, min(head, self.max_depth))
        return effective_head * self.area * self.storage_coefficient
    
    def _storage_to_head(self, storage: float) -> float:
        """Convert storage volume to hydraulic head."""
        max_storage = self.max_depth * self.area * self.storage_coefficient
        storage = max(0, min(storage, max_storage))
        return storage / (self.area * self.storage_coefficient)
    
    def calculate_baseflow(self, head: float) -> float:
        """
        Calculate baseflow using linear reservoir equation.
        
        Q = α * S where α is recession coefficient, S is storage
        """
        if head <= self.min_head:
            return 0.0
        storage = self._head_to_storage(head)
        return self.recession_coefficient * storage
    
    def add_inflow_edge(self, edge):
        """Add an inflow edge (recharge or inter-aquifer flow)."""
        self.inflow_edges[edge.source.id] = edge
    
    def add_outflow_edge(self, edge):
        """Add an outflow edge (discharge to streams or other aquifers)."""
        self.outflow_edges[edge.target.id] = edge
    
    def add_pumping_edge(self, edge):
        """Add a pumping edge (well extraction)."""
        self.pumping_edges[edge.target.id] = edge
    
    def update(self, time_step: int, dt: float) -> None:
        """
        Update aquifer state using water balance equation.
        
        dS/dt = Recharge + Inflow - Baseflow - Pumping
        
        Args:
            time_step (int): Current time step index
            dt (float): Time step duration [s]
        """
        try:
            # Calculate total inflows (recharge + inter-aquifer)
            total_inflow = sum(edge.flow_after_losses[time_step] 
                             for edge in self.inflow_edges.values())
            
            # Calculate total pumping
            total_pumping = sum(edge.flow_after_losses[time_step] 
                              for edge in self.pumping_edges.values())
            
            # Calculate baseflow based on current head
            baseflow = self.calculate_baseflow(self.current_head)
            
            # Water balance: dS/dt = In - Out
            storage_change = (total_inflow - baseflow - total_pumping) * dt
            new_storage = max(0, self.current_storage + storage_change)
            
            # Update state variables
            self.current_storage = new_storage
            self.current_head = self._storage_to_head(new_storage)
            
            # Recalculate baseflow with updated head
            actual_baseflow = self.calculate_baseflow(self.current_head)
            
            # Record history
            self.storage_history.append(new_storage)
            self.head_history.append(self.current_head)
            self.baseflow_history.append(actual_baseflow)
            self.recharge_history.append(total_inflow)
            self.pumping_history.append(total_pumping)
            
            # Update outflow edges with calculated baseflow
            if self.outflow_edges:
                # Distribute baseflow among outflow edges
                n_outlets = len(self.outflow_edges)
                flow_per_outlet = actual_baseflow / n_outlets if n_outlets > 0 else 0
                
                for edge in self.outflow_edges.values():
                    edge.update(flow_per_outlet)
            
        except Exception as e:
            print(f"Error updating aquifer node {self.id}: {str(e)}")
            # Maintain last known state
            self.storage_history.append(self.current_storage)
            self.head_history.append(self.current_head)
            self.baseflow_history.append(0)
            self.recharge_history.append(0)
            self.pumping_history.append(0)

class WellNode:
    """
    Represents a groundwater extraction well.
    
    Connects to AquiferNode through edges and can be optimized for pumping rates.
    Compatible with existing DemandNode structure for water allocation.
    
    Attributes:
        id (str): Unique identifier
        easting (float): Easting coordinate
        northing (float): Northing coordinate
        max_pumping_rate (float): Maximum pumping capacity [m³/s]
        pumping_history (list): Record of actual pumping rates [m³/s]
        efficiency (float): Well efficiency factor [-]
        inflow_edge: Connection to aquifer (single edge)
        outflow_edges (dict): Connections to demand points
    """
    
    all_ids = []
    
    def __init__(
        self,
        id: str,
        easting: float,
        northing: float,
        max_pumping_rate: float,
        efficiency: float = 1.0,
        min_head_constraint: float = 0.0
    ) -> None:
        """
        Initialize a WellNode.
        
        Args:
            id (str): Unique identifier
            easting (float): Easting coordinate [m]
            northing (float): Northing coordinate [m]
            max_pumping_rate (float): Maximum pumping rate [m³/s]
            efficiency (float): Pumping efficiency [-]
            min_head_constraint (float): Minimum head for operation [m]
        """
        validate_node_id(id, "WellNode")
        validate_coordinates(easting, northing, id)
        validate_positive_float(max_pumping_rate, "max_pumping_rate")
        validate_positive_float(efficiency, "efficiency")
        
        self.id = id
        WellNode.all_ids.append(id)
        self.easting = easting
        self.northing = northing
        self.max_pumping_rate = max_pumping_rate
        self.efficiency = efficiency
        self.min_head_constraint = min_head_constraint
        
        # State tracking
        self.pumping_history = []
        self.demand_history = []
        self.constraint_violations = []
        
        # Connections
        self.inflow_edge = None  # From aquifer
        self.outflow_edges = {}  # To demand points
    
    def add_inflow_edge(self, edge):
        """Set connection from aquifer."""
        if self.inflow_edge is not None:
            raise ValueError(f"WellNode {self.id} already has an inflow edge")
        self.inflow_edge = edge
    
    def add_outflow_edge(self, edge):
        """Add connection to demand point."""
        self.outflow_edges[edge.target.id] = edge
    
    def update(self, time_step: int, dt: float, requested_pumping: float = None) -> None:
        """
        Update well pumping based on demand and constraints.
        
        Args:
            time_step (int): Current time step
            dt (float): Time step duration [s]
            requested_pumping (float): Requested pumping rate [m³/s]
        """
        try:
            # Calculate total downstream demand
            total_demand = sum(edge.target.demand_rates[time_step] 
                             for edge in self.outflow_edges.values()
                             if hasattr(edge.target, 'demand_rates'))
            
            # Use requested pumping or demand, whichever is smaller
            if requested_pumping is not None:
                target_pumping = min(requested_pumping, total_demand)
            else:
                target_pumping = total_demand
            
            # Apply capacity and efficiency constraints
            max_available = self.max_pumping_rate * self.efficiency
            constrained_pumping = min(target_pumping, max_available)
            
            # Check head constraints through connected aquifer
            aquifer_head = 0.0
            if (self.inflow_edge and 
                hasattr(self.inflow_edge.source, 'current_head')):
                aquifer_head = self.inflow_edge.source.current_head
            
            # Apply head constraint
            if aquifer_head < self.min_head_constraint:
                constrained_pumping = 0.0
                self.constraint_violations.append(1)
            else:
                self.constraint_violations.append(0)
            
            # Record actual pumping
            self.pumping_history.append(constrained_pumping)
            self.demand_history.append(total_demand)
            
            # Update inflow edge (pumping from aquifer)
            if self.inflow_edge:
                self.inflow_edge.update(constrained_pumping)
            
            # Distribute pumped water to outflow edges
            if self.outflow_edges and constrained_pumping > 0:
                # Proportional allocation based on demand
                for edge in self.outflow_edges.values():
                    if hasattr(edge.target, 'demand_rates') and total_demand > 0:
                        proportion = edge.target.demand_rates[time_step] / total_demand
                        allocated_flow = constrained_pumping * proportion
                        edge.update(allocated_flow)
                    else:
                        edge.update(0)
            
        except Exception as e:
            print(f"Error updating well node {self.id}: {str(e)}")
            self.pumping_history.append(0)
            self.demand_history.append(0)
            self.constraint_violations.append(1)

class GroundwaterEdge:
    """
    Specialized edge for groundwater connections.
    
    Extends the existing Edge class concept for groundwater-specific flows
    including recharge, inter-aquifer flow, and surface-groundwater exchange.
    """
    
    def __init__(
        self,
        source,
        target, 
        connection_type: str,
        conductance: float = 1e-6,
        exchange_coefficient: float = 0.0
    ) -> None:
        """
        Initialize groundwater edge.
        
        Args:
            source: Source node (aquifer, surface water, or recharge)
            target: Target node (aquifer, stream, or well)
            connection_type (str): Type of connection ('recharge', 'baseflow', 'pumping')
            conductance (float): Hydraulic conductance [m²/s]
            exchange_coefficient (float): Exchange coefficient for surface water [1/s]
        """
        self.source = source
        self.target = target
        self.connection_type = connection_type
        self.conductance = conductance
        self.exchange_coefficient = exchange_coefficient
        
        # Flow tracking
        self.flow_history = []
        
        # Connect to nodes
        if hasattr(source, 'add_outflow_edge'):
            source.add_outflow_edge(self)
        if hasattr(target, 'add_inflow_edge'):
            target.add_inflow_edge(self)
    
    def calculate_exchange_flow(self, time_step: int) -> float:
        """
        Calculate flow based on connection type and hydraulic gradients.
        
        Returns:
            float: Flow rate [m³/s]
        """
        if self.connection_type == 'recharge':
            # Direct recharge from surface (e.g., from RunoffNode)
            if hasattr(self.source, 'runoff_history') and time_step < len(self.source.runoff_history):
                return self.source.runoff_history[time_step] * self.exchange_coefficient
            return 0.0
            
        elif self.connection_type == 'baseflow':
            # Baseflow from aquifer to stream
            if hasattr(self.source, 'current_head'):
                return self.source.calculate_baseflow(self.source.current_head)
            return 0.0
            
        elif self.connection_type == 'pumping':
            # Well pumping from aquifer
            if (hasattr(self.target, 'pumping_history') and 
                time_step < len(self.target.pumping_history)):
                return self.target.pumping_history[time_step]
            return 0.0
            
        else:
            # Inter-aquifer flow based on head difference
            if (hasattr(self.source, 'current_head') and 
                hasattr(self.target, 'current_head')):
                head_diff = self.source.current_head - self.target.current_head
                return self.conductance * head_diff
            return 0.0
    
    def update(self, flow: Optional[float] = None) -> None:
        """
        Update edge flow for current time step.
        
        Args:
            flow (float, optional): Specified flow rate. If None, calculates automatically.
        """
        if flow is None:
            # Calculate flow based on current conditions
            current_time_step = len(self.flow_history)
            calculated_flow = self.calculate_exchange_flow(current_time_step)
            self.flow_history.append(max(0, calculated_flow))
        else:
            # Use specified flow
            self.flow_history.append(max(0, flow))
    
    @property
    def flow_after_losses(self) -> List[float]:
        """Compatibility with existing edge interface."""
        return self.flow_history