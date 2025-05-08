"""
water_system

This package provides a framework for simulating and optimizing water flow in a network system.

The package includes classes for representing different types of nodes in a water system
(such as supply sources, storage facilities, and demand points), as well as edges
representing connections between these nodes.

Classes:
    WaterSystem: The main class for creating and managing a water system simulation.
    Node: Base class for all types of nodes in the water system.
    SupplyNode: Represents a water supply source.
    SinkNode: Represents a point where water exits the system.
    DemandNode: Represents a point of water demand (e.g., agricultural, domestic, or industrial use).
    StorageNode: Represents a water storage facility (e.g., a reservoir).
    HydroWorks: Represents a point where water can be redistributed, combining diversion and confluence functionality.
    Edge: Represents a connection between two nodes in the water system. This can either represent a river or a canal.
"""

from .water_system import WaterSystem
from .structure import Node, SupplyNode, SinkNode, DemandNode, StorageNode, HydroWorks, RunoffNode
from .edge import Edge
from .visualization import WaterSystemVisualizer
from .single_objective_ga import SingleObjectiveOptimizer
from .two_objective_ga import TwoObjectiveOptimizer
from .multi_objective_ga import MultiObjectiveOptimizer
from .pareto_dashboard_3d import ParetoFrontDashboard

# Define what should be imported with "from water_system import *"
__all__ = ['WaterSystem', 'Node', 'SupplyNode', 'SinkNode', 'DemandNode', 
           'StorageNode', 'HydroWorks','RunoffNode', 'Edge', 'WaterSystemVisualizer', 'SingleObjectiveOptimizer', 'TwoObjectiveOptimizer', 'MultiObjectiveOptimizer', 'ParetoFrontDashboard']

# Package version
__version__ = '0.2.0'