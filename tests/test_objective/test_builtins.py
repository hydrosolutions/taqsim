from typing import TYPE_CHECKING

import pytest

from taqsim.edge import Edge
from taqsim.node import Sink, Source, Storage
from taqsim.node.events import DeficitRecorded, WaterSpilled
from taqsim.node.timeseries import TimeSeries
from taqsim.objective.builtins import deficit, spill
from taqsim.system import WaterSystem

if TYPE_CHECKING:
    pass


class FakeReleaseRule:
    def release(
        self,
        node: "Storage",
        inflow: float,
        t: int,
        dt: float,
    ) -> float:
        return 0.0


class FakeLossRule:
    def calculate(self, node: "Storage", t: int, dt: float) -> dict[str, float]:
        return {}


class FakeEdgeLossRule:
    def calculate(self, edge: "Edge", flow: float, t: int, dt: float) -> dict[str, float]:
        return {}


@pytest.fixture
def simple_system() -> WaterSystem:
    system = WaterSystem(dt=1.0)

    source = Source(id="source", inflow=TimeSeries([100.0] * 12))
    storage = Storage(
        id="reservoir",
        capacity=1000.0,
        initial_storage=500.0,
        release_rule=FakeReleaseRule(),
        loss_rule=FakeLossRule(),
    )
    sink = Sink(id="outlet")

    edge1 = Edge(
        id="e1",
        source="source",
        target="reservoir",
        capacity=1000.0,
        loss_rule=FakeEdgeLossRule(),
    )
    edge2 = Edge(
        id="e2",
        source="reservoir",
        target="outlet",
        capacity=1000.0,
        loss_rule=FakeEdgeLossRule(),
    )

    system.add_node(source)
    system.add_node(storage)
    system.add_node(sink)
    system.add_edge(edge1)
    system.add_edge(edge2)
    system.validate()

    return system


class TestSpillObjective:
    def test_sums_spill_events(self, simple_system: WaterSystem) -> None:
        node = simple_system.nodes["reservoir"]
        node.record(WaterSpilled(amount=100.0, t=0))
        node.record(WaterSpilled(amount=50.0, t=1))
        node.record(WaterSpilled(amount=25.0, t=2))

        obj = spill("reservoir")
        result = obj.evaluate(simple_system)

        assert result == 175.0

    def test_returns_zero_for_no_spill(self, simple_system: WaterSystem) -> None:
        obj = spill("reservoir")
        result = obj.evaluate(simple_system)

        assert result == 0.0

    def test_raises_for_missing_node(self, simple_system: WaterSystem) -> None:
        obj = spill("nonexistent")

        with pytest.raises(ValueError, match="not found"):
            obj.evaluate(simple_system)

    def test_has_minimize_direction(self) -> None:
        obj = spill("any_node")
        assert obj.direction == "minimize"


class TestDeficitObjective:
    def test_sums_deficit_events(self, simple_system: WaterSystem) -> None:
        node = simple_system.nodes["reservoir"]
        node.record(DeficitRecorded(required=100.0, actual=80.0, deficit=20.0, t=0))
        node.record(DeficitRecorded(required=100.0, actual=70.0, deficit=30.0, t=1))

        obj = deficit("reservoir")
        result = obj.evaluate(simple_system)

        assert result == 50.0

    def test_returns_zero_for_no_deficit(self, simple_system: WaterSystem) -> None:
        obj = deficit("reservoir")
        result = obj.evaluate(simple_system)

        assert result == 0.0

    def test_raises_for_missing_node(self, simple_system: WaterSystem) -> None:
        obj = deficit("nonexistent")

        with pytest.raises(ValueError, match="not found"):
            obj.evaluate(simple_system)
