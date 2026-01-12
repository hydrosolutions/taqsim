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

from .common import CAPACITY_EXCEEDED, EVAPORATION, INEFFICIENCY, OVERFLOW, SEEPAGE, LossReason, ParamSpec, Strategy, summarize_losses
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
]

__version__ = "0.3.0"
