"""
taqsim

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

from .edge import Edge
from .nodes import DemandNode, HydroWorks, RunoffNode, SinkNode, StorageNode, SupplyNode
from .optimization.optimizer import DeapOptimizer
from .optimization.pareto_visualization import ParetoVisualizer
from .visualization import WaterSystemVisualizer
from .water_system import WaterSystem

# Define what should be imported with "from taqsim import *"
__all__ = [
    "WaterSystem",
    "SupplyNode",
    "SinkNode",
    "DemandNode",
    "StorageNode",
    "HydroWorks",
    "RunoffNode",
    "Edge",
    "WaterSystemVisualizer",
    "ParetoVisualizer",
    "DeapOptimizer",
]

# Package version
__version__ = "0.2.0"
