from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim import Edge, Objective, Sink, Source, Storage, Strategy, TimeSeries, WaterSystem

if TYPE_CHECKING:
    from taqsim.common import LossReason


@dataclass(frozen=True)
class FakeReleaseRule(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
    rate: float = 50.0

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.rate * dt, node.storage)


@dataclass(frozen=True)
class FakeLossRule:
    def calculate(self, node: "Storage", t: int, dt: float) -> dict["LossReason", float]:
        return {}


@dataclass(frozen=True)
class FakeEdgeLossRule:
    def calculate(
        self, edge: "Edge", flow: float, t: int, dt: float
    ) -> dict["LossReason", float]:
        return {}


@pytest.fixture
def fake_release_rule() -> FakeReleaseRule:
    return FakeReleaseRule()


@pytest.fixture
def fake_loss_rule() -> FakeLossRule:
    return FakeLossRule()


@pytest.fixture
def fake_edge_loss_rule() -> FakeEdgeLossRule:
    return FakeEdgeLossRule()


@pytest.fixture
def minimal_water_system() -> WaterSystem:
    system = WaterSystem(dt=1.0)

    system.add_node(Source(id="src", inflow=TimeSeries(values=[100.0] * 10)))
    system.add_node(
        Storage(
            id="dam",
            capacity=1000.0,
            initial_storage=500.0,
            release_rule=FakeReleaseRule(),
            loss_rule=FakeLossRule(),
        )
    )
    system.add_node(Sink(id="sink"))

    system.add_edge(
        Edge(id="e1", source="src", target="dam", capacity=1000.0, loss_rule=FakeEdgeLossRule())
    )
    system.add_edge(
        Edge(id="e2", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule())
    )

    system.validate()
    return system


@pytest.fixture
def simple_minimize_objective() -> Objective:
    def evaluate(system: WaterSystem) -> float:
        dam = system.nodes["dam"]
        return dam.release_rule.rate

    return Objective(name="minimize_rate", direction="minimize", evaluate=evaluate)


@pytest.fixture
def simple_maximize_objective() -> Objective:
    def evaluate(system: WaterSystem) -> float:
        dam = system.nodes["dam"]
        return dam.release_rule.rate

    return Objective(name="maximize_rate", direction="maximize", evaluate=evaluate)
