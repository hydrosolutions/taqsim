from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from taqsim import Edge, Sink, Source, Splitter, Storage, TimeSeries, WaterSystem
from taqsim.common import Strategy
from taqsim.constraints import Ordered, SumToOne
from taqsim.optimization import make_repair

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class OrderedRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("low", "high")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "low": (0.0, 100.0),
        "high": (0.0, 100.0),
    }
    __constraints__: ClassVar[tuple[Ordered, ...]] = (Ordered(low="low", high="high"),)
    low: float = 10.0
    high: float = 50.0

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.high * dt, node.storage)


@dataclass(frozen=True)
class ConstrainedSplit(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("r1", "r2", "r3")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "r1": (0.0, 1.0),
        "r2": (0.0, 1.0),
        "r3": (0.0, 1.0),
    }
    __constraints__: ClassVar[tuple[SumToOne, ...]] = (SumToOne(params=("r1", "r2", "r3")),)
    r1: float = 0.5
    r2: float = 0.3
    r3: float = 0.2

    def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
        ratios = (self.r1, self.r2, self.r3)
        return {t: amount * r for t, r in zip(node.targets, ratios, strict=True)}


@dataclass(frozen=True)
class SimpleLoss:
    """Physical model - NOT a Strategy."""

    rate: float = 0.01

    def calculate(self, node: "Storage", t: int, dt: float) -> dict:
        return {}


@dataclass(frozen=True)
class SimpleEdgeLoss:
    """Physical model - NOT a Strategy."""

    def calculate(self, edge: "Edge", flow: float, t: int, dt: float) -> dict:
        return {}


def build_constrained_system() -> WaterSystem:
    """Build a system with constraints for testing repair."""
    system = WaterSystem(dt=1.0)

    system.add_node(Source(id="river", inflow=TimeSeries(values=[100.0] * 10)))
    system.add_node(
        Storage(
            id="dam",
            capacity=1000.0,
            initial_storage=500.0,
            release_rule=OrderedRelease(low=10.0, high=50.0),
            loss_rule=SimpleLoss(),
        )
    )
    system.add_node(
        Splitter(
            id="junction",
            split_rule=ConstrainedSplit(r1=0.5, r2=0.3, r3=0.2),
        )
    )
    system.add_node(Sink(id="city"))
    system.add_node(Sink(id="farm"))
    system.add_node(Sink(id="env"))

    system.add_edge(Edge(id="e1", source="river", target="dam", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e2", source="dam", target="junction", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e3", source="junction", target="city", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e4", source="junction", target="farm", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e5", source="junction", target="env", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

    system.validate()
    return system


class TestMakeRepair:
    def test_returns_callable(self):
        system = build_constrained_system()
        repair = make_repair(system)
        assert callable(repair)

    def test_clips_to_bounds(self):
        system = build_constrained_system()
        repair = make_repair(system)
        schema = system.param_schema()

        # Build out-of-bounds vector dynamically
        x = np.zeros(len(schema))
        for i, spec in enumerate(schema):
            if "high" in spec.path or "low" in spec.path:
                x[i] = 150.0  # Out of bounds (0, 100)
            else:
                x[i] = 1.5  # Out of bounds for ratios (0, 1)

        repaired = repair(x)

        # Check clipping worked
        bounds = system.param_bounds()
        for i, spec in enumerate(schema):
            lo, hi = bounds[spec.path]
            assert lo <= repaired[i] <= hi

    def test_sum_to_one_constraint(self):
        system = build_constrained_system()
        repair = make_repair(system)
        schema = system.param_schema()

        # Find indices of r1, r2, r3
        ratio_indices = [i for i, s in enumerate(schema) if "split_rule" in s.path]

        # Build vector dynamically with ratios that don't sum to 1
        x = np.zeros(len(schema))
        for i, spec in enumerate(schema):
            if "split_rule" in spec.path:
                x[i] = 0.3  # sum = 0.9 (need to scale to 1.0)
            elif "high" in spec.path:
                x[i] = 40.0
            elif "low" in spec.path:
                x[i] = 20.0

        repaired = repair(x)

        ratio_sum = sum(repaired[i] for i in ratio_indices)
        assert abs(ratio_sum - 1.0) < 1e-6

    def test_ordered_constraint(self):
        system = build_constrained_system()
        repair = make_repair(system)

        # Schema order is: high, low, r1, r2, r3
        # Create vector with low > high (low=60, high=20)
        x = np.array([20.0, 60.0, 0.5, 0.3, 0.2])  # high=20, low=60
        repaired = repair(x)

        # Find indices dynamically
        schema = system.param_schema()
        low_idx = next(i for i, s in enumerate(schema) if s.path.endswith(".low"))
        high_idx = next(i for i, s in enumerate(schema) if s.path.endswith(".high"))

        # After repair, low <= high
        assert repaired[low_idx] <= repaired[high_idx]

    def test_idempotent(self):
        system = build_constrained_system()
        repair = make_repair(system)
        schema = system.param_schema()

        # Build vector with constraint violations dynamically
        x = np.zeros(len(schema))
        for i, spec in enumerate(schema):
            if "split_rule" in spec.path:
                x[i] = 0.5  # Each ratio, sum > 1.0
            elif "high" in spec.path:
                x[i] = 20.0  # Low > High
            elif "low" in spec.path:
                x[i] = 60.0

        repaired_once = repair(x)
        repaired_twice = repair(repaired_once)

        np.testing.assert_array_almost_equal(repaired_once, repaired_twice)


# Time-varying strategy with SumToOne constraint
# Note: We override constraints() instead of using __constraints__ because
# the Strategy.__post_init__ validation doesn't yet handle time-varying params
# with constraints (it tries to sum tuples instead of checking per-timestep).
@dataclass(frozen=True)
class TimeVaryingSplit(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("r1", "r2")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "r1": (0.0, 1.0),
        "r2": (0.0, 1.0),
    }
    __time_varying__: ClassVar[tuple[str, ...]] = ("r1", "r2")
    r1: tuple[float, ...] = (0.6, 0.5, 0.4)
    r2: tuple[float, ...] = (0.4, 0.5, 0.6)

    def split(self, node, amount: float, t: int) -> dict[str, float]:
        return {"a": amount * self.r1[t], "b": amount * self.r2[t]}

    def constraints(self, node) -> tuple:
        return (SumToOne(params=("r1", "r2")),)


class TestMakeRepairTimeVarying:
    """Tests for make_repair with time-varying parameters."""

    def test_clips_each_timestep_independently(self):
        """Each indexed value clipped to bounds independently."""

        # Create simple system with time-varying param (no constraints)
        @dataclass(frozen=True)
        class TVRelease(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (50.0, 50.0, 50.0)

            def release(self, node, inflow: float, t: int, dt: float) -> float:
                return self.rate[t]

        storage = Storage(id="dam", capacity=1000.0, release_rule=TVRelease(), loss_rule=SimpleLoss())
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

        repair = make_repair(system)

        # Values outside bounds
        x = np.array([-10.0, 50.0, 150.0])
        repaired = repair(x)

        assert repaired[0] == 0.0  # Clipped to lower
        assert repaired[1] == 50.0  # Unchanged
        assert repaired[2] == 100.0  # Clipped to upper

    def test_constraint_applied_per_timestep(self):
        """SumToOne applied to r1[t], r2[t] for each t."""
        splitter = Splitter(id="split", split_rule=TimeVaryingSplit())
        splitter._set_targets(["a", "b"])
        sink_a = Sink(id="sink_a")
        sink_b = Sink(id="sink_b")
        source = Source(id="src", inflow=TimeSeries(values=[100.0] * 10))

        system = WaterSystem()
        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink_a)
        system.add_node(sink_b)
        system.add_edge(Edge(id="e1", source="src", target="split", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
        system.add_edge(Edge(id="a", source="split", target="sink_a", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
        system.add_edge(Edge(id="b", source="split", target="sink_b", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

        repair = make_repair(system)

        # Vector: [r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]]
        # Values that don't sum to 1 at each timestep
        x = np.array([0.3, 0.4, 0.5, 0.3, 0.4, 0.5])  # Each pair sums to 0.6
        repaired = repair(x)

        # After repair, each timestep should sum to 1.0
        assert abs(repaired[0] + repaired[3] - 1.0) < 1e-9  # t=0
        assert abs(repaired[1] + repaired[4] - 1.0) < 1e-9  # t=1
        assert abs(repaired[2] + repaired[5] - 1.0) < 1e-9  # t=2

    def test_constant_only_constraints_unchanged(self):
        """Constraints with only constants behave as before."""
        # Use existing OrderedRelease from the file (constant params)
        storage = Storage(id="dam", capacity=1000.0, release_rule=OrderedRelease(), loss_rule=SimpleLoss())
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

        repair = make_repair(system)

        # Schema order: high (index 0), low (index 1)
        # Violates Ordered constraint: low > high (80 > 20)
        x = np.array([20.0, 80.0])  # high=20, low=80 violates low <= high
        repaired = repair(x)

        # After repair: low <= high (repaired[1] <= repaired[0])
        assert repaired[1] <= repaired[0]

    def test_idempotent_with_time_varying(self):
        """repair(repair(x)) == repair(x) for time-varying params."""
        splitter = Splitter(id="split", split_rule=TimeVaryingSplit())
        splitter._set_targets(["a", "b"])
        sink_a = Sink(id="sink_a")
        sink_b = Sink(id="sink_b")
        source = Source(id="src", inflow=TimeSeries(values=[100.0] * 10))

        system = WaterSystem()
        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink_a)
        system.add_node(sink_b)
        system.add_edge(Edge(id="e1", source="src", target="split", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
        system.add_edge(Edge(id="a", source="split", target="sink_a", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
        system.add_edge(Edge(id="b", source="split", target="sink_b", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

        repair = make_repair(system)

        x = np.array([0.3, 0.4, 0.5, 0.3, 0.4, 0.5])
        once = repair(x)
        twice = repair(once)

        np.testing.assert_array_almost_equal(once, twice)
