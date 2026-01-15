from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim import Edge, Sink, Source, Splitter, Storage, TimeSeries, WaterSystem
from taqsim.common import Strategy
from taqsim.constraints import ConstraintSpec, Ordered, SumToOne

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

        # Modify values
        new_vector = [v * 2 for v in original_vector]
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
