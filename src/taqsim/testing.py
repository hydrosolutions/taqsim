from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from taqsim.common import EVAPORATION, SEEPAGE, LossReason, Strategy
from taqsim.edge import Edge
from taqsim.node import Demand, NoLoss, NoReachLoss, NoRouting, PassThrough, Reach, Sink, Source, Splitter, Storage
from taqsim.node.timeseries import TimeSeries
from taqsim.system import WaterSystem
from taqsim.time import Frequency, Timestep

if TYPE_CHECKING:
    from taqsim.node.base import BaseNode


# --- Decision Policies (Strategy subclasses — optimizable) ---


@dataclass(frozen=True)
class FixedRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 1e6)}
    rate: float = 50.0

    def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
        return min(self.rate, node.storage)


@dataclass(frozen=True)
class ProportionalRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("fraction",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"fraction": (0.0, 1.0)}
    fraction: float = 0.5

    def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
        return node.storage * self.fraction


@dataclass(frozen=True)
class EvenSplit(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ()
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {}

    def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


@dataclass(frozen=True)
class FixedSplit(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ()
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {}
    weights: tuple[float, ...] = (0.5, 0.5)

    def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        return {tid: amount * w for tid, w in zip(targets, self.weights, strict=False)}


# --- Physical Model Rules (frozen dataclasses — NOT Strategy) ---


@dataclass(frozen=True)
class ConstantLoss:
    evaporation_rate: float = 0.01
    seepage_rate: float = 0.005

    def calculate(self, node: "Storage", t: Timestep) -> dict[LossReason, float]:
        return {
            EVAPORATION: node.storage * self.evaporation_rate,
            SEEPAGE: node.storage * self.seepage_rate,
        }


@dataclass(frozen=True)
class ProportionalReachLoss:
    loss_fraction: float = 0.1

    def calculate(self, reach: "Reach", flow: float, t: Timestep) -> dict[LossReason, float]:
        return {SEEPAGE: flow * self.loss_fraction}


@dataclass(frozen=True)
class TemperatureDependentLoss:
    required_auxiliary: ClassVar[frozenset[str]] = frozenset({"temperature"})
    base_rate: float = 0.01

    def calculate(self, node: "Storage", t: Timestep) -> dict[LossReason, float]:
        temperatures = node.auxiliary_data["temperature"]
        temp = temperatures[t.index]
        adjusted_rate = self.base_rate * (1.0 + max(0.0, temp - 20.0) / 10.0)
        return {EVAPORATION: node.storage * adjusted_rate}


# --- Re-exports from core ---

__all__ = [
    # Decision policies
    "FixedRelease",
    "ProportionalRelease",
    "EvenSplit",
    "FixedSplit",
    # Physical model rules
    "ConstantLoss",
    "ProportionalReachLoss",
    "TemperatureDependentLoss",
    # Core no-ops (re-exported for convenience)
    "NoLoss",
    "NoReachLoss",
    "NoRouting",
    # Factory functions
    "make_source",
    "make_sink",
    "make_storage",
    "make_demand",
    "make_splitter",
    "make_passthrough",
    "make_reach",
    "make_edge",
    "make_system",
]


# --- Factory Functions ---


def make_source(
    id: str = "source",
    *,
    n_steps: int = 12,
    **overrides: Any,
) -> Source:
    overrides.setdefault("inflow", TimeSeries([100.0] * n_steps))
    return Source(id=id, **overrides)


def make_sink(id: str = "sink", **overrides: Any) -> Sink:
    return Sink(id=id, **overrides)


def make_storage(id: str = "storage", **overrides: Any) -> Storage:
    overrides.setdefault("capacity", 1000.0)
    overrides.setdefault("initial_storage", 500.0)
    overrides.setdefault("release_policy", ProportionalRelease())
    overrides.setdefault("loss_rule", NoLoss())
    return Storage(id=id, **overrides)


def make_demand(
    id: str = "demand",
    *,
    n_steps: int = 12,
    **overrides: Any,
) -> Demand:
    overrides.setdefault("requirement", TimeSeries([50.0] * n_steps))
    return Demand(id=id, **overrides)


def make_splitter(id: str = "splitter", **overrides: Any) -> Splitter:
    overrides.setdefault("split_policy", EvenSplit())
    return Splitter(id=id, **overrides)


def make_passthrough(id: str = "passthrough", **overrides: Any) -> PassThrough:
    return PassThrough(id=id, **overrides)


def make_reach(
    id: str = "reach",
    **overrides: Any,
) -> Reach:
    overrides.setdefault("routing_model", NoRouting())
    overrides.setdefault("loss_rule", NoReachLoss())
    return Reach(id=id, **overrides)


def make_edge(
    id: str,
    source: str,
    target: str,
    **overrides: Any,
) -> Edge:
    return Edge(id=id, source=source, target=target, **overrides)


# --- System Builder ---


def make_system(
    *components: "BaseNode | Edge",
    frequency: Frequency = Frequency.MONTHLY,
    start_date: date | None = None,
    validate: bool = True,
) -> WaterSystem:
    system = WaterSystem(frequency=frequency, start_date=start_date)
    for component in components:
        if isinstance(component, Edge):
            system.add_edge(component)
        else:
            system.add_node(component)
    if validate:
        system.validate()
    return system
