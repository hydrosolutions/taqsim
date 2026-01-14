from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim import Edge, Sink, Source, Splitter, Storage, TimeSeries, WaterSystem
from taqsim.common import Strategy
from taqsim.constraints import Ordered, SumToOne

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
    __params__: ClassVar[tuple[str, ...]] = ("ratios",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"ratios": (0.0, 1.0)}
    ratios: tuple[float, ...] = (0.5, 0.5)

    def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
        total = sum(self.ratios)
        return {t: amount * r / total for t, r in zip(node.targets, self.ratios)}


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
        return {t: amount * r for t, r in zip(node.targets, ratios)}


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
            split_rule=ProportionalSplit(ratios=(0.6, 0.4)),
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
        # Should include split_rule params (tuple flattened)
        assert any("junction.split_rule.ratios" in p for p in paths)
        # Should NOT include loss_rule (not a Strategy)
        assert not any("loss_rule" in p for p in paths)

    def test_empty_system_returns_empty_schema(self):
        system = WaterSystem()
        assert system.param_schema() == []

    def test_tuple_params_are_flattened(self):
        system = build_test_system()
        schema = system.param_schema()

        ratio_specs = [s for s in schema if "ratios" in s.path]
        assert len(ratio_specs) == 2
        assert ratio_specs[0].index == 0
        assert ratio_specs[1].index == 1


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

        for spec, val in zip(schema, vector):
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

        # Double all values
        original_vector = system.to_vector()
        new_vector = [100.0, 0.7, 0.3]  # new rate and ratios

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

        assert "junction.split_rule.ratios[0]" in bounds
        assert "junction.split_rule.ratios[1]" in bounds
        assert bounds["junction.split_rule.ratios[0]"] == (0.0, 1.0)
        assert bounds["junction.split_rule.ratios[1]"] == (0.0, 1.0)

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

    def test_tuple_params_get_indexed_keys(self):
        """Tuple parameters get separate indexed keys."""
        system = build_test_system()
        bounds = system.param_bounds()

        ratio_keys = [k for k in bounds if "ratios" in k]
        assert len(ratio_keys) == 2
        assert "junction.split_rule.ratios[0]" in ratio_keys
        assert "junction.split_rule.ratios[1]" in ratio_keys


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

        for spec, bound in zip(schema, bounds_vec):
            if "release_rule.rate" in spec.path:
                assert bound == (0.0, 200.0)
            elif "ratios" in spec.path:
                assert bound == (0.0, 1.0)


class TestConstraints:
    """Tests for WaterSystem.constraints()."""

    def test_constraints_returns_empty_for_no_constraints(self):
        """System with no constrained strategies returns empty list."""
        system = build_test_system()
        constraints = system.constraints()

        assert constraints == []

    def test_constraints_discovers_from_strategies(self):
        """constraints() discovers constraints from node strategies."""
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

        constraints = system.constraints()

        assert len(constraints) == 2

        constraint_types = [type(c).__name__ for _, c in constraints]
        assert "Ordered" in constraint_types
        assert "SumToOne" in constraint_types

    def test_constraints_returns_prefix_for_path_mapping(self):
        """Constraints are returned with correct prefix for path remapping."""
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

        constraints = system.constraints()

        assert len(constraints) == 1
        prefix, constraint = constraints[0]

        assert prefix == "tank.release_rule"
        assert isinstance(constraint, Ordered)
        assert constraint.params == ("low", "high")
