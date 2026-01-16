from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim import Edge, Sink, Source, Splitter, Storage, TimeSeries, WaterSystem
from taqsim.common import Strategy
from taqsim.constraints import ConstraintSpec, Ordered, SumToOne

from .conftest import FakeEdgeLossRule, FakeLossRule

if TYPE_CHECKING:
    pass


# Create proper strategy classes for testing
@dataclass(frozen=True)
class FixedRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 200.0)}
    rate: float = 50.0

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.rate * dt, node.storage)


@dataclass(frozen=True)
class ProportionalSplit(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("r1", "r2")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "r1": (0.0, 1.0),
        "r2": (0.0, 1.0),
    }
    r1: float = 0.5
    r2: float = 0.5

    def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
        total = self.r1 + self.r2
        ratios = (self.r1 / total, self.r2 / total)
        return {t: amount * r for t, r in zip(node.targets, ratios, strict=True)}


@dataclass(frozen=True)
class UnboundedStrategy(Strategy):
    """Strategy without __bounds__ for testing error handling."""

    __params__: ClassVar[tuple[str, ...]] = ("value",)
    value: float = 1.0

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return self.value


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


def build_test_system() -> WaterSystem:
    """Build a simple test system with strategies."""
    system = WaterSystem(dt=1.0)

    system.add_node(Source(id="river", inflow=TimeSeries(values=[100.0] * 10)))
    system.add_node(
        Storage(
            id="dam",
            capacity=1000.0,
            initial_storage=500.0,
            release_rule=FixedRelease(rate=50.0),
            loss_rule=SimpleLoss(),
        )
    )
    system.add_node(
        Splitter(
            id="junction",
            split_rule=ProportionalSplit(r1=0.6, r2=0.4),
        )
    )
    system.add_node(Sink(id="city"))
    system.add_node(Sink(id="farm"))

    system.add_edge(Edge(id="e1", source="river", target="dam", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e2", source="dam", target="junction", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e3", source="junction", target="city", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
    system.add_edge(Edge(id="e4", source="junction", target="farm", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

    system.validate()
    return system


class TestParamSchema:
    """Tests for WaterSystem.param_schema()."""

    def test_discovers_strategy_params(self):
        system = build_test_system()
        schema = system.param_schema()

        paths = [s.path for s in schema]

        # Should include release_rule params
        assert "dam.release_rule.rate" in paths
        # Should include split_rule params
        assert "junction.split_rule.r1" in paths
        assert "junction.split_rule.r2" in paths
        # Should NOT include loss_rule (not a Strategy)
        assert not any("loss_rule" in p for p in paths)

    def test_empty_system_returns_empty_schema(self):
        system = WaterSystem()
        assert system.param_schema() == []

    def test_scalar_params_are_collected(self):
        system = build_test_system()
        schema = system.param_schema()

        r1_specs = [s for s in schema if "r1" in s.path]
        r2_specs = [s for s in schema if "r2" in s.path]
        assert len(r1_specs) == 1
        assert len(r2_specs) == 1


class TestToVector:
    """Tests for WaterSystem.to_vector()."""

    def test_returns_flat_vector(self):
        system = build_test_system()
        vector = system.to_vector()

        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)
        # rate=50.0, ratios=(0.6, 0.4) -> 3 values
        assert len(vector) == 3

    def test_vector_values_match_schema(self):
        system = build_test_system()
        schema = system.param_schema()
        vector = system.to_vector()

        for spec, val in zip(schema, vector, strict=True):
            assert spec.value == val


class TestWithVector:
    """Tests for WaterSystem.with_vector()."""

    def test_creates_new_system(self):
        system = build_test_system()
        vector = system.to_vector()
        new_system = system.with_vector(vector)

        assert new_system is not system

    def test_original_unchanged(self):
        system = build_test_system()
        original_vector = system.to_vector()

        # Modify values (using valid values within bounds)
        # rate bounds: (0.0, 200.0), r1/r2 bounds: (0.0, 1.0)
        new_vector = [100.0, 0.7, 0.3]
        system.with_vector(new_vector)

        # Original should be unchanged
        assert system.to_vector() == original_vector

    def test_new_system_has_updated_params(self):
        system = build_test_system()

        # New values: rate, r1, r2
        new_vector = [100.0, 0.7, 0.3]

        new_system = system.with_vector(new_vector)

        assert new_system.to_vector() == new_vector

    def test_rejects_wrong_length_vector(self):
        system = build_test_system()

        with pytest.raises(ValueError, match="does not match"):
            system.with_vector([1.0, 2.0])  # wrong length

    def test_roundtrip(self):
        system = build_test_system()
        vector = system.to_vector()
        new_system = system.with_vector(vector)

        assert new_system.to_vector() == vector


class TestSystemReset:
    """Tests for WaterSystem.reset()."""

    def test_reset_clears_all_events(self):
        system = build_test_system()
        system.simulate(5)

        # Events accumulated
        total_events = sum(len(n.events) for n in system.nodes.values())
        assert total_events > 0

        system.reset()

        total_events = sum(len(n.events) for n in system.nodes.values())
        assert total_events == 0

    def test_reset_restores_storage(self):
        system = build_test_system()
        initial = system.nodes["dam"].storage

        system.simulate(5)
        assert system.nodes["dam"].storage != initial

        system.reset()
        assert system.nodes["dam"].storage == initial


class TestOptimizationLoopPattern:
    """Integration test for the GA optimization loop pattern."""

    def test_optimization_loop_pattern(self):
        system = build_test_system()
        base_vector = system.to_vector()

        # Simulate multiple "generations"
        for _ in range(3):
            # Create variant
            modified = [v * 1.1 for v in base_vector]
            variant = system.with_vector(modified)

            # Run simulation
            variant.simulate(5)

            # Could evaluate fitness here

            # Original should be unchanged
            assert system.to_vector() == base_vector


class TestParamBounds:
    """Tests for WaterSystem.param_bounds()."""

    def test_empty_system_returns_empty_dict(self):
        """Empty system has no bounds."""
        system = WaterSystem()
        assert system.param_bounds() == {}

    def test_collects_bounds_from_strategies(self):
        """param_bounds() returns bounds from all strategies."""
        system = build_test_system()
        bounds = system.param_bounds()

        assert "dam.release_rule.rate" in bounds
        assert bounds["dam.release_rule.rate"] == (0.0, 200.0)

        assert "junction.split_rule.r1" in bounds
        assert "junction.split_rule.r2" in bounds
        assert bounds["junction.split_rule.r1"] == (0.0, 1.0)
        assert bounds["junction.split_rule.r2"] == (0.0, 1.0)

    def test_raises_for_missing_bounds(self):
        """Raises ValueError when strategy params lack bounds."""
        system = WaterSystem(dt=1.0)
        system.add_node(Source(id="src", inflow=TimeSeries(values=[100.0] * 10)))
        system.add_node(
            Storage(
                id="tank",
                capacity=1000.0,
                initial_storage=500.0,
                release_rule=UnboundedStrategy(value=10.0),
                loss_rule=SimpleLoss(),
            )
        )
        system.add_edge(Edge(id="e1", source="src", target="tank", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

        with pytest.raises(ValueError, match="Missing bounds"):
            system.param_bounds()

    def test_scalar_params_have_simple_keys(self):
        """Scalar parameters have simple dot-separated keys."""
        system = build_test_system()
        bounds = system.param_bounds()

        split_keys = [k for k in bounds if "split_rule" in k]
        assert len(split_keys) == 2
        assert "junction.split_rule.r1" in split_keys
        assert "junction.split_rule.r2" in split_keys


class TestBoundsVector:
    """Tests for WaterSystem.bounds_vector()."""

    def test_length_matches_to_vector(self):
        """bounds_vector() length equals to_vector() length."""
        system = build_test_system()

        bounds_vec = system.bounds_vector()
        param_vec = system.to_vector()

        assert len(bounds_vec) == len(param_vec)

    def test_order_matches_param_schema(self):
        """Bounds align with param_schema() order."""
        system = build_test_system()

        schema = system.param_schema()
        bounds_vec = system.bounds_vector()

        for spec, bound in zip(schema, bounds_vec, strict=True):
            if "release_rule.rate" in spec.path:
                assert bound == (0.0, 200.0)
            elif "r1" in spec.path or "r2" in spec.path:
                assert bound == (0.0, 1.0)


class TestConstraintSpecs:
    """Tests for WaterSystem.constraint_specs()."""

    def test_constraint_specs_returns_empty_for_no_constraints(self):
        """System with no constrained strategies returns empty list."""
        system = build_test_system()
        specs = system.constraint_specs()

        assert specs == []

    def test_constraint_specs_discovers_from_strategies(self):
        """constraint_specs() discovers constraints from node strategies."""
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

        specs = system.constraint_specs()

        assert len(specs) == 2

        constraint_types = [type(spec.constraint).__name__ for spec in specs]
        assert "Ordered" in constraint_types
        assert "SumToOne" in constraint_types

    def test_constraint_specs_returns_fully_resolved_spec(self):
        """ConstraintSpec contains prefix, param_paths, and param_bounds."""
        system = WaterSystem(dt=1.0)

        system.add_node(Source(id="river", inflow=TimeSeries(values=[100.0] * 10)))
        system.add_node(
            Storage(
                id="tank",
                capacity=1000.0,
                initial_storage=500.0,
                release_rule=OrderedRelease(low=10.0, high=50.0),
                loss_rule=SimpleLoss(),
            )
        )
        system.add_node(Sink(id="sink"))

        system.add_edge(Edge(id="e1", source="river", target="tank", capacity=1000.0, loss_rule=SimpleEdgeLoss()))
        system.add_edge(Edge(id="e2", source="tank", target="sink", capacity=1000.0, loss_rule=SimpleEdgeLoss()))

        specs = system.constraint_specs()

        assert len(specs) == 1
        spec = specs[0]

        assert isinstance(spec, ConstraintSpec)
        assert spec.prefix == "tank.release_rule"
        assert isinstance(spec.constraint, Ordered)
        assert spec.constraint.params == ("low", "high")

        # Check param_paths
        assert spec.param_paths == {
            "low": "tank.release_rule.low",
            "high": "tank.release_rule.high",
        }

        # Check param_bounds
        assert spec.param_bounds == {
            "low": (0.0, 100.0),
            "high": (0.0, 100.0),
        }


# Test strategy with time-varying param
@dataclass(frozen=True)
class TimeVaryingRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
    __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
    rate: tuple[float, ...] = (10.0, 20.0, 30.0)

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.rate[t] * dt, node.storage)


# Mixed strategy: constant + time-varying
@dataclass(frozen=True)
class MixedParamRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "base": (0.0, 50.0),
        "multiplier": (0.5, 2.0),
    }
    __time_varying__: ClassVar[tuple[str, ...]] = ("multiplier",)
    base: float = 10.0
    multiplier: tuple[float, ...] = (1.0, 1.5, 2.0)

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.base * self.multiplier[t] * dt, node.storage)


class TestParamSchemaTimeVarying:
    """Tests for param_schema() with time-varying parameters."""

    def test_expands_time_varying_to_indexed_paths(self):
        """Time-varying param 'rate' with 3 values becomes rate[0], rate[1], rate[2]."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        schema = system.param_schema()
        paths = [s.path for s in schema]

        assert "dam.release_rule.rate[0]" in paths
        assert "dam.release_rule.rate[1]" in paths
        assert "dam.release_rule.rate[2]" in paths
        assert "dam.release_rule.rate" not in paths  # No base path

    def test_values_match_tuple_elements(self):
        """ParamSpec.value for rate[i] equals strategy.rate[i]."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        schema = system.param_schema()
        schema_dict = {s.path: s.value for s in schema}

        assert schema_dict["dam.release_rule.rate[0]"] == 10.0
        assert schema_dict["dam.release_rule.rate[1]"] == 20.0
        assert schema_dict["dam.release_rule.rate[2]"] == 30.0


class TestToVectorTimeVarying:
    """Tests for to_vector() with time-varying parameters."""

    def test_flattens_tuple_to_consecutive_elements(self):
        """Tuple (10, 20, 30) flattens to consecutive vector elements."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        vector = system.to_vector()
        assert vector == [10.0, 20.0, 30.0]

    def test_vector_length_accounts_for_expanded_tuples(self):
        """Vector length = sum of tuple lengths."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=MixedParamRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        vector = system.to_vector()
        # base (1) + multiplier (3) = 4
        assert len(vector) == 4


class TestWithVectorTimeVarying:
    """Tests for with_vector() reconstructing tuples."""

    def test_reconstructs_tuple_from_indexed_values(self):
        """Indexed vector elements reconstructed as tuple."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        new_system = system.with_vector([40.0, 50.0, 60.0])
        new_strategy = new_system.nodes["dam"].release_rule
        assert new_strategy.rate == (40.0, 50.0, 60.0)

    def test_roundtrip_preserves_values(self):
        """system.with_vector(system.to_vector()) preserves all values."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        vector = system.to_vector()
        new_system = system.with_vector(vector)
        assert new_system.to_vector() == vector


class TestBoundsVectorTimeVarying:
    """Tests for bounds_vector() with time-varying parameters."""

    def test_bounds_expanded_for_each_timestep(self):
        """Bounds repeated for each index."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        bounds = system.bounds_vector()
        assert bounds == [(0.0, 100.0), (0.0, 100.0), (0.0, 100.0)]

    def test_bounds_length_matches_vector(self):
        """bounds_vector() length equals to_vector() length."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=MixedParamRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))

        assert len(system.bounds_vector()) == len(system.to_vector())


class TestValidateTimeVaryingLengths:
    """Tests for _validate_time_varying_lengths."""

    def test_sufficient_length_passes(self):
        """Time-varying param with length >= timesteps passes."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        # Should not raise - 3 values, 3 timesteps
        system.simulate(3)

    def test_insufficient_length_raises(self):
        """Time-varying param with length < timesteps raises."""
        from taqsim.system.validation import InsufficientLengthError

        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=TimeVaryingRelease(),
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        # Should raise - 3 values, 5 timesteps
        with pytest.raises(InsufficientLengthError, match="length 3"):
            system.simulate(5)


# Cyclical time-varying strategy
@dataclass(frozen=True)
class CyclicalRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
    __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
    __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
    rate: tuple[float, ...] = (10.0, 20.0, 30.0)

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.rate[t % len(self.rate)] * dt, node.storage)


# Non-cyclical time-varying strategy (same as TimeVaryingRelease but explicit)
@dataclass(frozen=True)
class NonCyclicalRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
    __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
    rate: tuple[float, ...] = (10.0, 20.0, 30.0)

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return min(self.rate[t] * dt, node.storage)


# Mixed strategy: one cyclical, one non-cyclical
@dataclass(frozen=True)
class MixedCyclicalRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "base": (0.0, 50.0),
        "multiplier": (0.5, 2.0),
    }
    __time_varying__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
    __cyclical__: ClassVar[tuple[str, ...]] = ("base",)  # base is cyclical, multiplier is not
    base: tuple[float, ...] = (10.0, 20.0, 30.0)  # 3 values, cyclical
    multiplier: tuple[float, ...] = (1.0,) * 10  # 10 values, non-cyclical

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        base_val = self.base[t % len(self.base)]
        mult_val = self.multiplier[t]
        return min(base_val * mult_val * dt, node.storage)


# All-cyclical strategy with multiple params
@dataclass(frozen=True)
class AllCyclicalRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "base": (0.0, 50.0),
        "multiplier": (0.5, 2.0),
    }
    __time_varying__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
    __cyclical__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
    base: tuple[float, ...] = (10.0, 20.0)  # 2 values
    multiplier: tuple[float, ...] = (1.0, 1.5, 2.0)  # 3 values

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        base_val = self.base[t % len(self.base)]
        mult_val = self.multiplier[t % len(self.multiplier)]
        return min(base_val * mult_val * dt, node.storage)


class TestValidateTimeVaryingLengthsCyclical:
    """Tests for cyclical parameter validation in _validate_time_varying_lengths."""

    def test_cyclical_param_skips_length_validation(self):
        """Cyclical param with 3 values works for 10 timestep simulation."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=CyclicalRelease(),  # 3 values, cyclical
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        # Should not raise - cyclical params are not length-validated
        system.simulate(10)

    def test_non_cyclical_param_still_requires_sufficient_length(self):
        """Non-cyclical param with 3 values raises InsufficientLengthError for 10 timestep simulation."""
        from taqsim.system.validation import InsufficientLengthError

        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=NonCyclicalRelease(),  # 3 values, non-cyclical
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        # Should raise - 3 values, 10 timesteps needed
        with pytest.raises(InsufficientLengthError, match="length 3"):
            system.simulate(10)

    def test_mixed_strategy_only_non_cyclical_validated(self):
        """Strategy with both cyclical (3 values) and non-cyclical (10 values) params works for 10 timesteps."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=MixedCyclicalRelease(),  # base=3 cyclical, multiplier=10 non-cyclical
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        # Should not raise - base is cyclical (skips validation), multiplier has 10 values
        system.simulate(10)

    def test_all_cyclical_strategy_never_raises_length_error(self):
        """Strategy where all time-varying params are cyclical never raises length error."""
        storage = Storage(
            id="dam",
            capacity=1000.0,
            release_rule=AllCyclicalRelease(),  # base=2, multiplier=3, both cyclical
            loss_rule=FakeLossRule(),
        )
        sink = Sink(id="sink")
        system = WaterSystem()
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(Edge(id="e1", source="dam", target="sink", capacity=1000.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        # Should not raise regardless of simulation length - all params are cyclical
        system.simulate(100)
