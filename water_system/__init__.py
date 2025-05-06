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

Usage:
    from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, Edge

    # Create a water system
    system = WaterSystem()

    # Add nodes and edges to the system
    supply = SupplyNode("Supply1", default_supply_rate=10)
    storage = StorageNode("Reservoir1", capacity=1000)
    demand = DemandNode("Demand1", demand_rate=5)

    system.add_node(supply)
    system.add_node(storage)
    system.add_node(demand)

    system.add_edge(Edge(supply, storage, capacity=15))
    system.add_edge(Edge(storage, demand, capacity=10))

    # Run simulation and visualize results
    system.simulate(num_time_steps=12)
    system.visualize()
"""

from .water_system import WaterSystem
from .structure import Node, SupplyNode, SinkNode, DemandNode, StorageNode, HydroWorks
from .edge import Edge
from .visualization import WaterSystemVisualizer
from .single_objective_ga import SingleObjectiveOptimizer
from .two_objective_ga import TwoObjectiveOptimizer
from .multi_objective_ga import MultiObjectiveOptimizer
from .pareto_dashboard_3d import ParetoFrontDashboard

# Define what should be imported with "from water_system import *"
__all__ = ['WaterSystem', 'Node', 'SupplyNode', 'SinkNode', 'DemandNode', 
           'StorageNode', 'HydroWorks', 'Edge', 'WaterSystemVisualizer', 'SingleObjectiveOptimizer', 'TwoObjectiveOptimizer', 'MultiObjectiveOptimizer', 'ParetoFrontDashboard']

# Package version
__version__ = '0.2.0'