from dataclasses import dataclass
from typing import ClassVar

import pytest

from taqsim.common import ParamSpec, Strategy


class TestStrategy:
    """Tests for Strategy mixin class."""

    def test_params_returns_declared_params(self):
        @dataclass(frozen=True)
        class TestStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
            rate: float = 1.0
            threshold: float = 0.5
            name: str = "test"  # not in __params__

        strategy = TestStrategy()
        params = strategy.params()

        assert params == {"rate": 1.0, "threshold": 0.5}
        assert "name" not in params

    def test_params_handles_tuple_values(self):
        @dataclass(frozen=True)
        class SplitStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("ratios",)
            ratios: tuple[float, ...] = (0.6, 0.4)

        strategy = SplitStrategy()
        params = strategy.params()

        assert params == {"ratios": (0.6, 0.4)}

    def test_with_params_creates_new_instance(self):
        @dataclass(frozen=True)
        class TestStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: float = 1.0

        original = TestStrategy(rate=1.0)
        modified = original.with_params(rate=2.0)

        assert original.rate == 1.0  # unchanged
        assert modified.rate == 2.0
        assert original is not modified

    def test_with_params_rejects_unknown_params(self):
        @dataclass(frozen=True)
        class TestStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: float = 1.0

        strategy = TestStrategy()

        with pytest.raises(ValueError, match="Unknown parameters"):
            strategy.with_params(unknown=5.0)

    def test_empty_params(self):
        @dataclass(frozen=True)
        class NoParamStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()
            fixed_value: float = 42.0

        strategy = NoParamStrategy()
        assert strategy.params() == {}


class TestParamSpec:
    """Tests for ParamSpec dataclass."""

    def test_scalar_param_spec(self):
        spec = ParamSpec(path="dam.release_rule.rate", value=50.0)
        assert spec.path == "dam.release_rule.rate"
        assert spec.value == 50.0
        assert spec.index is None

    def test_tuple_param_spec(self):
        spec = ParamSpec(path="splitter.strategy.ratios", value=0.6, index=0)
        assert spec.path == "splitter.strategy.ratios"
        assert spec.value == 0.6
        assert spec.index == 0

    def test_param_spec_is_frozen(self):
        spec = ParamSpec(path="test", value=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.value = 2.0


class TestStrategyBounds:
    """Tests for Strategy bounds functionality."""

    def test_default_bounds_is_empty_dict(self):
        """Strategy with no __bounds__ returns empty dict."""

        @dataclass(frozen=True)
        class MinimalStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("value",)
            value: float = 1.0

        strategy = MinimalStrategy()
        assert strategy.bounds(None) == {}

    def test_fixed_bounds_from_class_variable(self):
        """Strategy returns __bounds__ from class variable."""

        @dataclass(frozen=True)
        class BoundedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "rate": (0.0, 100.0),
                "threshold": (0.0, 1.0),
            }
            rate: float = 50.0
            threshold: float = 0.5

        strategy = BoundedStrategy()
        bounds = strategy.bounds(None)

        assert bounds == {"rate": (0.0, 100.0), "threshold": (0.0, 1.0)}

    def test_bounds_method_receives_node(self):
        """bounds() method receives the node parameter."""

        @dataclass(frozen=True)
        class NodeAwareStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: float = 50.0

            def bounds(self, node) -> dict[str, tuple[float, float]]:
                if node is not None and hasattr(node, "capacity"):
                    return {"rate": (0.0, node.capacity)}
                return {"rate": (0.0, 1000.0)}

        class FakeNode:
            capacity = 500.0

        strategy = NodeAwareStrategy()

        assert strategy.bounds(None) == {"rate": (0.0, 1000.0)}
        assert strategy.bounds(FakeNode()) == {"rate": (0.0, 500.0)}

    def test_bounds_returns_copy_not_reference(self):
        """bounds() returns a new dict, not the class variable."""

        @dataclass(frozen=True)
        class BoundedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            rate: float = 50.0

        strategy = BoundedStrategy()
        bounds1 = strategy.bounds(None)
        bounds2 = strategy.bounds(None)

        assert bounds1 == bounds2
        assert bounds1 is not bounds2  # Different dict objects
