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

from .basin import (
    Basin,
    BasinRun,
    BuiltBasin,
    Presence,
    Reach,
    TimeAxis,
    WaterSeries,
    WaterValue,
    WaterValues,
)
from .common import (
    EVAPORATION,
    INEFFICIENCY,
    OVERFLOW,
    SEEPAGE,
    LossReason,
    ParamSpec,
    Strategy,
    summarize_losses,
)
from .constraints import (
    BoundViolationError,
    Constraint,
    ConstraintViolationError,
    Ordered,
    SumToOne,
)
from .docs import get_docs_path
from .edge import Edge
from .node import (
    BaseNode,
    Demand,
    NoLoss,
    NoReachLoss,
    NoRelease,
    NoRouting,
    NoSplit,
    PassThrough,
    ReachLossRule,
    RoutingModel,
    Sink,
    Source,
    Splitter,
    Storage,
    TimeSeries,
    WaterEnteredReach,
    WaterExitedReach,
    WaterInTransit,
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
from .optimization import OptimizeResult, Solution, make_repair, optimize
from .system import WaterSystem
from .time import Frequency, Timestep, time_index
from .vocabulary import (
    CanalLosses,
    CanalLossRule,
    EFlowSplit,
    EFlowSplitPolicy,
    EvaporationLossRule,
    MonthlyDistribution,
    Parameter,
    PriorityDistribution,
    ReservoirEvaporation,
    ZoneRelease,
    ZoneReleasePolicy,
    monthly_parameters,
)

__all__ = [
    # Basin authoring
    "Basin",
    "BuiltBasin",
    "BasinRun",
    "Reach",
    "TimeAxis",
    "Presence",
    "WaterSeries",
    "WaterValues",
    "WaterValue",
    # Rule vocabulary
    "Parameter",
    "monthly_parameters",
    "ZoneRelease",
    "ZoneReleasePolicy",
    "MonthlyDistribution",
    "PriorityDistribution",
    "EFlowSplit",
    "EFlowSplitPolicy",
    "ReservoirEvaporation",
    "EvaporationLossRule",
    "CanalLosses",
    "CanalLossRule",
    # Common
    "LossReason",
    "EVAPORATION",
    "INEFFICIENCY",
    "OVERFLOW",
    "SEEPAGE",
    "summarize_losses",
    "Strategy",
    "ParamSpec",
    # Constraints
    "BoundViolationError",
    "Constraint",
    "ConstraintViolationError",
    "Ordered",
    "SumToOne",
    # Optimization
    "make_repair",
    "optimize",
    "OptimizeResult",
    "Solution",
    # Nodes
    "BaseNode",
    "NoLoss",
    "NoRelease",
    "NoSplit",
    "Source",
    "Storage",
    "Demand",
    "Splitter",
    "PassThrough",
    "Sink",
    "TimeSeries",
    # Reach Strategies
    "RoutingModel",
    "ReachLossRule",
    "NoReachLoss",
    "NoRouting",
    # Reach Events
    "WaterEnteredReach",
    "WaterExitedReach",
    "WaterInTransit",
    # Edges
    "Edge",
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
    # Time
    "Frequency",
    "Timestep",
    "time_index",
    # Documentation
    "get_docs_path",
]

__version__ = "0.1.4"
