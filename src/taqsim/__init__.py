"""
taqsim - Water system simulation framework using event sourcing.

This package provides a DAG-based framework for simulating water flow through
a network of nodes (sources, storage, demand, sinks) connected by edges
(rivers, canals, pipes).

Core modules:
    - taqsim.node: Node types and events
    - taqsim.edge: Edge class and events
    - taqsim.system: WaterSystem orchestrator
    - taqsim.common: Shared types (LossReason)
    - taqsim.objective: Optimization objectives and Trace arithmetics
"""

from .common import (
    CAPACITY_EXCEEDED,
    EVAPORATION,
    INEFFICIENCY,
    OVERFLOW,
    SEEPAGE,
    LossReason,
    ParamSpec,
    Strategy,
    summarize_losses,
)
from .constraints import Constraint, Ordered, SumToOne
from .optimization import make_repair
from .edge import Edge, EdgeEvent, EdgeLossRule, WaterDelivered, WaterLost, WaterReceived
from .node import (
    BaseNode,
    Demand,
    PassThrough,
    Sink,
    Source,
    Splitter,
    Storage,
    TimeSeries,
)
from .objective import (
    Direction,
    HasTimestep,
    Objective,
    ObjectiveRegistry,
    Trace,
    lift,
    maximize,
    minimize,
)
from .system import WaterSystem

__all__ = [
    # Common
    "LossReason",
    "CAPACITY_EXCEEDED",
    "EVAPORATION",
    "INEFFICIENCY",
    "OVERFLOW",
    "SEEPAGE",
    "summarize_losses",
    "Strategy",
    "ParamSpec",
    # Constraints
    "Constraint",
    "Ordered",
    "SumToOne",
    # Optimization
    "make_repair",
    # Nodes
    "BaseNode",
    "Source",
    "Storage",
    "Demand",
    "Splitter",
    "PassThrough",
    "Sink",
    "TimeSeries",
    # Edges
    "Edge",
    "EdgeEvent",
    "EdgeLossRule",
    "WaterReceived",
    "WaterLost",
    "WaterDelivered",
    # System
    "WaterSystem",
    # Objectives
    "Direction",
    "HasTimestep",
    "Objective",
    "ObjectiveRegistry",
    "Trace",
    "lift",
    "maximize",
    "minimize",
]

__version__ = "0.3.0"
