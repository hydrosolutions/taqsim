from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim import Edge, Objective, Sink, Source, Storage, Strategy, TimeSeries, WaterSystem
from taqsim.testing import NoEdgeLoss, NoLoss
from taqsim.time import Frequency, Timestep

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class FakeReleasePolicy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
    rate: float = 50.0

    def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
        return min(self.rate, node.storage)


@pytest.fixture
def fake_release_policy() -> FakeReleasePolicy:
    return FakeReleasePolicy()


@pytest.fixture
def fake_loss_rule() -> NoLoss:
    return NoLoss()


@pytest.fixture
def fake_edge_loss_rule() -> NoEdgeLoss:
    return NoEdgeLoss()


@pytest.fixture
def minimal_water_system() -> WaterSystem:
    system = WaterSystem(frequency=Frequency.MONTHLY)

    system.add_node(Source(id="src", inflow=TimeSeries(values=[100.0] * 10)))
    system.add_node(
        Storage(
            id="dam",
            capacity=1000.0,
            initial_storage=500.0,
            release_policy=FakeReleasePolicy(),
            loss_rule=NoLoss(),
        )
    )
    system.add_node(Sink(id="sink"))

    system.add_edge(Edge(id="e1", source="src", target="dam", capacity=1000.0, loss_rule=NoEdgeLoss()))
    system.add_edge(Edge(id="e2", source="dam", target="sink", capacity=1000.0, loss_rule=NoEdgeLoss()))

    system.validate()
    return system


@pytest.fixture
def simple_minimize_objective() -> Objective:
    def evaluate(system: WaterSystem) -> float:
        dam = system.nodes["dam"]
        return dam.release_policy.rate

    return Objective(name="minimize_rate", direction="minimize", evaluate=evaluate)


@pytest.fixture
def simple_maximize_objective() -> Objective:
    def evaluate(system: WaterSystem) -> float:
        dam = system.nodes["dam"]
        return dam.release_policy.rate

    return Objective(name="maximize_rate", direction="maximize", evaluate=evaluate)
