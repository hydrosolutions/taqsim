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
"""

from .common import EVAPORATION, OVERFLOW, SEEPAGE, LossReason, summarize_losses, Strategy, ParamSpec
from .edge import Edge, EdgeEvent, EdgeLossRule, FlowDelivered, FlowLost, FlowReceived
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
from .system import WaterSystem

__all__ = [
    # Common
    "LossReason",
    "EVAPORATION",
    "SEEPAGE",
    "OVERFLOW",
    "summarize_losses",
    "Strategy",
    "ParamSpec",
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
    "FlowReceived",
    "FlowLost",
    "FlowDelivered",
    # System
    "WaterSystem",
]

__version__ = "0.3.0"
