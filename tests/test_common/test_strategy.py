from dataclasses import dataclass
from typing import ClassVar

import pytest

from taqsim.common import ParamSpec, Strategy
from taqsim.constraints import (
    BoundViolationError,
    ConstraintViolationError,
    Ordered,
    SumToOne,
)
from taqsim.time import Frequency, Timestep


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
        class FakeSplitStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("ratios",)
            ratios: tuple[float, ...] = (0.6, 0.4)

        strategy = FakeSplitStrategy()
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
        spec = ParamSpec(path="dam.release_policy.rate", value=50.0)
        assert spec.path == "dam.release_policy.rate"
        assert spec.value == 50.0

    def test_param_spec_is_frozen(self):
        from dataclasses import FrozenInstanceError

        spec = ParamSpec(path="test", value=1.0)
        with pytest.raises(FrozenInstanceError):
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


class TestStrategyConstraints:
    """Tests for Strategy constraints functionality."""

    def test_default_constraints_is_empty_tuple(self):
        """Strategy with no __constraints__ returns empty tuple."""

        @dataclass(frozen=True)
        class MinimalStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("value",)
            value: float = 1.0

        strategy = MinimalStrategy()
        assert strategy.constraints(None) == ()

    def test_init_subclass_raises_for_unknown_param(self):
        """Strategy raises TypeError when constraint references unknown param."""

        class FakeConstraint:
            params = ("unknown_param",)

            def satisfied(self, values: dict[str, float], tol: float = 1e-9) -> bool:
                return True

        with pytest.raises(TypeError, match="constraint references unknown params"):

            @dataclass(frozen=True)
            class BadStrategy(Strategy):
                __params__: ClassVar[tuple[str, ...]] = ("rate",)
                __constraints__: ClassVar[tuple] = (FakeConstraint(),)
                rate: float = 1.0

    def test_init_subclass_accepts_valid_constraints(self):
        """Strategy accepts constraints that reference valid params."""

        class FakeConstraint:
            params = ("rate", "threshold")

            def satisfied(self, values: dict[str, float], tol: float = 1e-9) -> bool:
                return True

        @dataclass(frozen=True)
        class ValidStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
            __constraints__: ClassVar[tuple] = (FakeConstraint(),)
            rate: float = 1.0
            threshold: float = 0.5

        strategy = ValidStrategy()
        assert len(strategy.__constraints__) == 1

    def test_constraints_method_returns_class_constraints(self):
        """constraints() method returns the __constraints__ class variable."""

        class FakeConstraint:
            params = ("rate",)

            def satisfied(self, values: dict[str, float], tol: float = 1e-9) -> bool:
                return True

        @dataclass(frozen=True)
        class ConstrainedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __constraints__: ClassVar[tuple] = (FakeConstraint(),)
            rate: float = 1.0

        strategy = ConstrainedStrategy()
        constraints = strategy.constraints(None)

        assert len(constraints) == 1
        assert isinstance(constraints[0], FakeConstraint)


# Test strategy classes for __post_init__ validation tests


@dataclass(frozen=True)
class BoundedStrategy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
    rate: float = 50.0


@dataclass(frozen=True)
class ConstrainedStrategy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("r1", "r2", "r3")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "r1": (0.0, 1.0),
        "r2": (0.0, 1.0),
        "r3": (0.0, 1.0),
    }
    __constraints__: ClassVar[tuple] = (SumToOne(params=("r1", "r2", "r3")),)
    r1: float = 0.5
    r2: float = 0.3
    r3: float = 0.2


@dataclass(frozen=True)
class OrderedBoundedStrategy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("low", "high")
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
        "low": (0.0, 100.0),
        "high": (0.0, 100.0),
    }
    __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
    low: float = 10.0
    high: float = 50.0


class TestStrategyPostInitBoundValidation:
    """Tests for Strategy __post_init__ bounds validation."""

    def test_valid_params_within_bounds_succeeds(self):
        """Strategy with valid params constructs without error."""
        strategy = BoundedStrategy(rate=50.0)
        assert strategy.rate == 50.0

    def test_param_below_lower_bound_raises(self):
        """Param below lower bound raises BoundViolationError."""
        with pytest.raises(BoundViolationError, match="rate"):
            BoundedStrategy(rate=-1.0)

    def test_param_above_upper_bound_raises(self):
        """Param above upper bound raises BoundViolationError."""
        with pytest.raises(BoundViolationError, match="rate"):
            BoundedStrategy(rate=150.0)

    def test_param_at_lower_bound_succeeds(self):
        """Param exactly at lower bound is valid."""
        strategy = BoundedStrategy(rate=0.0)
        assert strategy.rate == 0.0

    def test_param_at_upper_bound_succeeds(self):
        """Param exactly at upper bound is valid."""
        strategy = BoundedStrategy(rate=100.0)
        assert strategy.rate == 100.0

    def test_error_contains_param_name_value_and_bounds(self):
        """BoundViolationError message includes param name, value, and bounds."""
        with pytest.raises(BoundViolationError) as exc_info:
            BoundedStrategy(rate=150.0)
        assert exc_info.value.param == "rate"
        assert exc_info.value.value == 150.0
        assert exc_info.value.bounds == (0.0, 100.0)


class TestStrategyPostInitConstraintValidation:
    """Tests for Strategy __post_init__ constraint validation."""

    def test_satisfied_constraint_succeeds(self):
        """Strategy with satisfied constraints constructs without error."""
        strategy = ConstrainedStrategy(r1=0.5, r2=0.3, r3=0.2)
        assert strategy.r1 == 0.5

    def test_violated_sum_to_one_raises(self):
        """Violated SumToOne constraint raises ConstraintViolationError."""
        with pytest.raises(ConstraintViolationError, match="SumToOne"):
            ConstrainedStrategy(r1=0.5, r2=0.5, r3=0.5)  # sum = 1.5

    def test_violated_ordered_raises(self):
        """Violated Ordered constraint raises ConstraintViolationError."""
        with pytest.raises(ConstraintViolationError, match="Ordered"):
            OrderedBoundedStrategy(low=60.0, high=40.0)  # low > high

    def test_error_contains_constraint_and_values(self):
        """ConstraintViolationError includes constraint type and values."""
        with pytest.raises(ConstraintViolationError) as exc_info:
            ConstrainedStrategy(r1=0.5, r2=0.5, r3=0.5)
        assert isinstance(exc_info.value.constraint, SumToOne)
        assert "r1" in exc_info.value.values

    def test_constraint_tolerance_respected(self):
        """Constraint satisfied() uses appropriate tolerance."""
        # Values that sum to 1.0 within floating point tolerance
        strategy = ConstrainedStrategy(r1=0.33333333333, r2=0.33333333333, r3=0.33333333334)
        assert strategy is not None


class TestStrategyPostInitValidationOrder:
    """Tests for validation order in Strategy __post_init__."""

    def test_bounds_validated_before_constraints(self):
        """Bound violation raised even if constraints also violated."""
        # rate=150.0 is out of bounds AND would violate constraints
        with pytest.raises(BoundViolationError):
            ConstrainedStrategy(r1=1.5, r2=0.3, r3=0.2)  # r1 out of bounds

    def test_with_params_validates_new_instance(self):
        """Strategy.with_params() validates the new instance."""
        strategy = BoundedStrategy(rate=50.0)
        with pytest.raises(BoundViolationError):
            strategy.with_params(rate=150.0)


class TestTimeVaryingDeclaration:
    """Tests for __time_varying__ class variable declaration and validation."""

    def test_time_varying_default_is_empty_tuple(self):
        """Strategy without __time_varying__ returns empty tuple."""

        @dataclass(frozen=True)
        class NoTimeVarying(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            rate: float = 50.0

        s = NoTimeVarying()
        assert s.time_varying() == ()

    def test_init_subclass_accepts_valid_time_varying(self):
        """Strategy with valid __time_varying__ constructs without error."""

        @dataclass(frozen=True)
        class ValidTimeVarying(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (50.0, 60.0, 70.0)

        s = ValidTimeVarying()
        assert s.time_varying() == ("rate",)

    def test_init_subclass_raises_for_unknown_time_varying_param(self):
        """Raises TypeError if __time_varying__ references param not in __params__."""
        with pytest.raises(TypeError, match="unknown params"):

            @dataclass(frozen=True)
            class InvalidTimeVarying(Strategy):
                __params__: ClassVar[tuple[str, ...]] = ("rate",)
                __time_varying__: ClassVar[tuple[str, ...]] = ("unknown",)
                rate: float = 50.0


class TestTimeVaryingBoundsValidation:
    """Tests for __post_init__ bounds validation with tuple values."""

    def test_valid_tuple_within_bounds(self):
        """Valid tuple where all elements satisfy bounds passes validation."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, 50.0, 90.0)

        s = TVStrategy()
        assert s.rate == (10.0, 50.0, 90.0)

    def test_tuple_element_violates_lower_bound(self):
        """Raises BoundViolationError if tuple element below lower bound."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, -5.0, 90.0)

        with pytest.raises(BoundViolationError, match=r"rate\[1\]"):
            TVStrategy()

    def test_tuple_element_violates_upper_bound(self):
        """Raises BoundViolationError if tuple element exceeds upper bound."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, 50.0, 150.0)

        with pytest.raises(BoundViolationError, match=r"rate\[2\]"):
            TVStrategy()

    def test_time_varying_param_must_be_tuple(self):
        """Raises TypeError if time-varying param is not a tuple."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: float = 50.0  # Wrong type!

        with pytest.raises(TypeError, match="must be tuple"):
            TVStrategy()


class TestTimeVaryingParams:
    """Tests for params() with time-varying values."""

    def test_params_returns_tuple_for_time_varying(self):
        """Time-varying param value is returned as tuple."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = TVStrategy()
        assert s.params() == {"rate": (10.0, 20.0, 30.0)}

    def test_params_mixed_constant_and_time_varying(self):
        """Strategy with both constant and time-varying params."""

        @dataclass(frozen=True)
        class MixedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("base", "multiplier")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "base": (0.0, 100.0),
                "multiplier": (0.5, 2.0),
            }
            __time_varying__: ClassVar[tuple[str, ...]] = ("multiplier",)
            base: float = 10.0
            multiplier: tuple[float, ...] = (1.0, 1.5, 2.0)

        s = MixedStrategy()
        params = s.params()
        assert params["base"] == 10.0
        assert params["multiplier"] == (1.0, 1.5, 2.0)


class TestTimeVaryingWithParams:
    """Tests for with_params() with time-varying parameters."""

    def test_with_params_accepts_tuple(self):
        """with_params() accepts tuple for time-varying param."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = TVStrategy()
        s2 = s.with_params(rate=(40.0, 50.0, 60.0))
        assert s2.rate == (40.0, 50.0, 60.0)
        assert s.rate == (10.0, 20.0, 30.0)  # Original unchanged


class TestTimeVaryingPostInitConstraintValidation:
    """Tests for __post_init__ constraint validation with time-varying parameters."""

    def test_ordered_validated_per_timestep(self):
        """Ordered constraint is validated at each timestep."""

        @dataclass(frozen=True)
        class TVOrderedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("low", "high")
            low: tuple[float, ...] = (10.0, 20.0, 30.0)
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        # All timesteps valid: low < high at t=0,1,2
        strategy = TVOrderedStrategy(low=(10.0, 20.0, 30.0), high=(50.0, 60.0, 70.0))
        assert strategy.low == (10.0, 20.0, 30.0)
        assert strategy.high == (50.0, 60.0, 70.0)

    def test_ordered_violation_at_specific_timestep_raises(self):
        """Ordered constraint violation at specific timestep raises error."""

        @dataclass(frozen=True)
        class TVOrderedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("low", "high")
            low: tuple[float, ...] = (10.0, 20.0, 30.0)
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        # Violation at t=1: low[1]=60 > high[1]=40
        with pytest.raises(ConstraintViolationError, match=r"low\[1\].*high\[1\]"):
            TVOrderedStrategy(low=(10.0, 60.0, 30.0), high=(50.0, 40.0, 70.0))

    def test_sum_to_one_validated_per_timestep(self):
        """SumToOne constraint is validated at each timestep."""

        @dataclass(frozen=True)
        class TVSumStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "r1": (0.0, 1.0),
                "r2": (0.0, 1.0),
            }
            __constraints__: ClassVar[tuple] = (SumToOne(params=("r1", "r2")),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            r1: tuple[float, ...] = (0.6, 0.5, 0.4)
            r2: tuple[float, ...] = (0.4, 0.5, 0.6)

        # All timesteps valid: sum = 1.0 at t=0,1,2
        strategy = TVSumStrategy(r1=(0.6, 0.5, 0.4), r2=(0.4, 0.5, 0.6))
        assert strategy.r1 == (0.6, 0.5, 0.4)
        assert strategy.r2 == (0.4, 0.5, 0.6)

    def test_sum_to_one_violation_at_specific_timestep_raises(self):
        """SumToOne constraint violation at specific timestep raises error."""

        @dataclass(frozen=True)
        class TVSumStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "r1": (0.0, 1.0),
                "r2": (0.0, 1.0),
            }
            __constraints__: ClassVar[tuple] = (SumToOne(params=("r1", "r2")),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            r1: tuple[float, ...] = (0.6, 0.5, 0.4)
            r2: tuple[float, ...] = (0.4, 0.5, 0.6)

        # Violation at t=2: r1[2]=0.8 + r2[2]=0.4 = 1.2
        with pytest.raises(ConstraintViolationError, match=r"r1\[2\].*r2\[2\]"):
            TVSumStrategy(r1=(0.6, 0.5, 0.8), r2=(0.4, 0.5, 0.4))

    def test_mixed_scalar_and_tv_constraint(self):
        """Constraint with mixed scalar and time-varying params validates correctly."""

        @dataclass(frozen=True)
        class MixedOrderedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("high",)
            low: float = 20.0
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        # Scalar low=20, time-varying high=(50, 60, 70) - all valid
        strategy = MixedOrderedStrategy(low=20.0, high=(50.0, 60.0, 70.0))
        assert strategy.low == 20.0
        assert strategy.high == (50.0, 60.0, 70.0)

        # Scalar low=65, time-varying high=(50, 60, 70) - violation at t=0, t=1
        with pytest.raises(ConstraintViolationError):
            MixedOrderedStrategy(low=65.0, high=(50.0, 60.0, 70.0))

    def test_error_includes_timestep_index(self):
        """Error message includes the timestep index where violation occurred."""

        @dataclass(frozen=True)
        class TVOrderedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("low", "high")
            low: tuple[float, ...] = (10.0, 20.0, 30.0)
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        # Violation at t=2: low[2]=80 > high[2]=70
        with pytest.raises(ConstraintViolationError) as exc_info:
            TVOrderedStrategy(low=(10.0, 20.0, 80.0), high=(50.0, 60.0, 70.0))

        error_msg = str(exc_info.value)
        # Error message should indicate timestep 2 via indexed param names
        assert "low[2]" in error_msg and "high[2]" in error_msg


class TestCyclicalDeclaration:
    """Tests for __cyclical__ class variable declaration and validation."""

    def test_cyclical_default_is_empty_tuple(self):
        """Strategy without __cyclical__ returns empty tuple."""

        @dataclass(frozen=True)
        class NoCyclical(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            rate: float = 50.0

        s = NoCyclical()
        assert s.cyclical() == ()

    def test_init_subclass_accepts_valid_cyclical(self):
        """Strategy with valid __cyclical__ (subset of __time_varying__) constructs without error."""

        @dataclass(frozen=True)
        class ValidCyclical(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (50.0, 60.0, 70.0)

        s = ValidCyclical()
        assert s.cyclical() == ("rate",)

    def test_init_subclass_raises_for_cyclical_not_in_time_varying(self):
        """Raises TypeError if __cyclical__ references param not in __time_varying__."""
        with pytest.raises(TypeError, match="__cyclical__ params must be in __time_varying__"):

            @dataclass(frozen=True)
            class InvalidCyclical(Strategy):
                __params__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
                __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
                __cyclical__: ClassVar[tuple[str, ...]] = ("threshold",)  # not in __time_varying__
                rate: tuple[float, ...] = (50.0, 60.0)
                threshold: float = 0.5

    def test_cyclical_method_returns_declared_params(self):
        """cyclical() method returns the declared __cyclical__ tuple."""

        @dataclass(frozen=True)
        class MultiCyclical(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __time_varying__: ClassVar[tuple[str, ...]] = ("low", "high")
            __cyclical__: ClassVar[tuple[str, ...]] = ("low", "high")
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"low": Frequency.MONTHLY, "high": Frequency.MONTHLY}
            low: tuple[float, ...] = (10.0, 20.0, 30.0)
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        s = MultiCyclical()
        assert s.cyclical() == ("low", "high")


class TestCyclicalValuesAtTimestep:
    """Tests for _values_at_timestep() with cyclical parameters."""

    def test_cyclical_param_uses_modulo_indexing(self):
        """For cyclical param with 3 values, _values_at_timestep(5) returns value[5 % 3] = value[2]."""

        @dataclass(frozen=True)
        class CyclicalStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = CyclicalStrategy()
        values = s._values_at_timestep(5)
        # 5 % 3 = 2, so should return rate[2] = 30.0
        assert values["rate"] == 30.0

    def test_non_cyclical_param_uses_direct_indexing(self):
        """For non-cyclical param, _values_at_timestep(t) returns value[t] directly."""

        @dataclass(frozen=True)
        class NonCyclicalStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

        s = NonCyclicalStrategy()
        values = s._values_at_timestep(5)
        # Direct indexing: rate[5] = 60.0
        assert values["rate"] == 60.0

    def test_mixed_cyclical_and_non_cyclical(self):
        """Strategy with both cyclical and non-cyclical params uses correct indexing for each."""

        @dataclass(frozen=True)
        class MixedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("cyclical_rate", "linear_rate")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "cyclical_rate": (0.0, 100.0),
                "linear_rate": (0.0, 100.0),
            }
            __time_varying__: ClassVar[tuple[str, ...]] = ("cyclical_rate", "linear_rate")
            __cyclical__: ClassVar[tuple[str, ...]] = ("cyclical_rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"cyclical_rate": Frequency.MONTHLY}
            cyclical_rate: tuple[float, ...] = (10.0, 20.0, 30.0)  # 3 values, cycles
            linear_rate: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)  # 6 values, direct

        s = MixedStrategy()
        values = s._values_at_timestep(4)
        # cyclical_rate: 4 % 3 = 1 -> 20.0
        assert values["cyclical_rate"] == 20.0
        # linear_rate: direct index 4 -> 5.0
        assert values["linear_rate"] == 5.0

    def test_cyclical_wraparound_at_exact_multiple(self):
        """_values_at_timestep(6) for 3-value cyclical returns value[0]."""

        @dataclass(frozen=True)
        class CyclicalStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = CyclicalStrategy()
        values = s._values_at_timestep(6)
        # 6 % 3 = 0, so should return rate[0] = 10.0
        assert values["rate"] == 10.0


class TestCyclicalConstraintValidation:
    """Tests for __post_init__ constraint validation with cyclical parameters."""

    def test_cyclical_constraint_validated_per_cycle_length(self):
        """For cyclical-only strategy, constraint validated for each position in cycle."""

        @dataclass(frozen=True)
        class CyclicalSumStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "r1": (0.0, 1.0),
                "r2": (0.0, 1.0),
            }
            __constraints__: ClassVar[tuple] = (SumToOne(params=("r1", "r2")),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            __cyclical__: ClassVar[tuple[str, ...]] = ("r1", "r2")
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"r1": Frequency.MONTHLY, "r2": Frequency.MONTHLY}
            r1: tuple[float, ...] = (0.6, 0.5, 0.4)
            r2: tuple[float, ...] = (0.4, 0.5, 0.6)

        # All 3 cycle positions valid: sum = 1.0 at positions 0, 1, 2
        strategy = CyclicalSumStrategy()
        assert strategy.r1 == (0.6, 0.5, 0.4)
        assert strategy.r2 == (0.4, 0.5, 0.6)

    def test_constraint_satisfied_at_all_positions_passes(self):
        """Valid cyclical strategy constructs without error."""

        @dataclass(frozen=True)
        class CyclicalOrderedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("low", "high")
            __cyclical__: ClassVar[tuple[str, ...]] = ("low", "high")
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"low": Frequency.MONTHLY, "high": Frequency.MONTHLY}
            low: tuple[float, ...] = (10.0, 20.0, 30.0)
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        # All positions valid: low < high at each cycle position
        strategy = CyclicalOrderedStrategy()
        assert strategy.low == (10.0, 20.0, 30.0)
        assert strategy.high == (50.0, 60.0, 70.0)

    def test_constraint_violated_at_specific_position_raises(self):
        """ConstraintViolationError raised with position info when constraint violated at specific cycle position."""

        @dataclass(frozen=True)
        class CyclicalOrderedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("low", "high")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "low": (0.0, 100.0),
                "high": (0.0, 100.0),
            }
            __constraints__: ClassVar[tuple] = (Ordered(low="low", high="high"),)
            __time_varying__: ClassVar[tuple[str, ...]] = ("low", "high")
            __cyclical__: ClassVar[tuple[str, ...]] = ("low", "high")
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"low": Frequency.MONTHLY, "high": Frequency.MONTHLY}
            low: tuple[float, ...] = (10.0, 20.0, 30.0)
            high: tuple[float, ...] = (50.0, 60.0, 70.0)

        # Violation at position 1: low[1]=70 > high[1]=40
        with pytest.raises(ConstraintViolationError) as exc_info:
            CyclicalOrderedStrategy(low=(10.0, 70.0, 30.0), high=(50.0, 40.0, 70.0))

        error_msg = str(exc_info.value)
        # Error message should indicate cycle position 1
        assert "low[1]" in error_msg and "high[1]" in error_msg


class TestStrategyTags:
    """Tests for Strategy __tags__ ClassVar and tags() method."""

    def test_default_tags_is_empty_frozenset(self):
        """Strategy without __tags__ returns empty frozenset."""

        @dataclass(frozen=True)
        class NoTagsStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()

        strategy = NoTagsStrategy()
        assert strategy.tags() == frozenset()

    def test_strategy_with_class_tags(self):
        """Strategy with __tags__ returns declared tags."""

        @dataclass(frozen=True)
        class TaggedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()
            __tags__: ClassVar[frozenset[str]] = frozenset({"operational", "seasonal"})

        strategy = TaggedStrategy()
        assert strategy.tags() == frozenset({"operational", "seasonal"})

    def test_tags_returns_frozenset_type(self):
        """tags() returns a frozenset."""

        @dataclass(frozen=True)
        class TaggedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()
            __tags__: ClassVar[frozenset[str]] = frozenset({"test"})

        strategy = TaggedStrategy()
        assert isinstance(strategy.tags(), frozenset)


class TestStrategyMetadata:
    """Tests for Strategy __metadata__ ClassVar and metadata() method."""

    def test_default_metadata_is_empty_mapping(self):
        """Strategy without __metadata__ returns empty mapping."""

        @dataclass(frozen=True)
        class NoMetadataStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()

        strategy = NoMetadataStrategy()
        assert strategy.metadata() == {}

    def test_strategy_with_class_metadata(self):
        """Strategy with __metadata__ returns declared metadata."""
        from collections.abc import Mapping

        @dataclass(frozen=True)
        class MetadataStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()
            __metadata__: ClassVar[Mapping[str, object]] = {"version": 1, "author": "test"}

        strategy = MetadataStrategy()
        assert strategy.metadata() == {"version": 1, "author": "test"}

    def test_metadata_returns_mapping_type(self):
        """metadata() returns a Mapping."""
        from collections.abc import Mapping

        @dataclass(frozen=True)
        class MetadataStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ()
            __metadata__: ClassVar[Mapping[str, object]] = {"key": "value"}

        strategy = MetadataStrategy()
        assert isinstance(strategy.metadata(), Mapping)


class TestCyclicalFreqDeclaration:
    """Tests for __cyclical_freq__ class variable declaration and validation."""

    def test_cyclical_without_cyclical_freq_raises(self):
        """Strategy with __cyclical__ but no __cyclical_freq__ raises TypeError."""
        with pytest.raises(TypeError, match="__cyclical_freq__ is required"):

            @dataclass(frozen=True)
            class MissingFreq(Strategy):
                __params__: ClassVar[tuple[str, ...]] = ("rate",)
                __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
                __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
                rate: tuple[float, ...] = (10.0, 20.0, 30.0)

    def test_cyclical_freq_keys_not_matching_cyclical_raises(self):
        """Strategy with __cyclical_freq__ keys not matching __cyclical__ raises TypeError."""
        with pytest.raises(TypeError, match="__cyclical_freq__ keys must match __cyclical__"):

            @dataclass(frozen=True)
            class MismatchedFreq(Strategy):
                __params__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
                __time_varying__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
                __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
                __cyclical_freq__: ClassVar[dict[str, Frequency]] = {
                    "rate": Frequency.MONTHLY,
                    "threshold": Frequency.MONTHLY,
                }
                rate: tuple[float, ...] = (10.0, 20.0, 30.0)
                threshold: tuple[float, ...] = (0.1, 0.2, 0.3)

    def test_non_frequency_value_in_cyclical_freq_raises(self):
        """Strategy with non-Frequency value in __cyclical_freq__ raises TypeError."""
        with pytest.raises(TypeError, match="must be a Frequency"):

            @dataclass(frozen=True)
            class BadFreqType(Strategy):
                __params__: ClassVar[tuple[str, ...]] = ("rate",)
                __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
                __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
                __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": 12}  # type: ignore[dict-item]
                rate: tuple[float, ...] = (10.0, 20.0, 30.0)

    def test_valid_cyclical_freq_constructs_fine(self):
        """Strategy with valid __cyclical_freq__ constructs without error."""

        @dataclass(frozen=True)
        class ValidFreq(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = ValidFreq()
        assert s.rate == (10.0, 20.0, 30.0)

    def test_cyclical_freq_method_returns_declared_dict(self):
        """cyclical_freq() method returns the declared __cyclical_freq__ dict."""

        @dataclass(frozen=True)
        class FreqStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = FreqStrategy()
        assert s.cyclical_freq() == {"rate": Frequency.MONTHLY}


class TestParamAt:
    """Tests for param_at() method with Timestep-aware parameter access."""

    def test_scalar_param_returns_value_regardless_of_timestep(self):
        """Scalar param returns its value regardless of timestep."""

        @dataclass(frozen=True)
        class ScalarStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            rate: float = 42.0

        s = ScalarStrategy()
        assert s.param_at("rate", Timestep(0, Frequency.MONTHLY)) == 42.0
        assert s.param_at("rate", Timestep(5, Frequency.DAILY)) == 42.0
        assert s.param_at("rate", Timestep(100, Frequency.YEARLY)) == 42.0

    def test_non_cyclical_time_varying_uses_direct_index(self):
        """Non-cyclical time-varying param returns value[t.index]."""

        @dataclass(frozen=True)
        class TVStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            rate: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

        s = TVStrategy()
        assert s.param_at("rate", Timestep(0, Frequency.MONTHLY)) == 10.0
        assert s.param_at("rate", Timestep(3, Frequency.MONTHLY)) == 40.0
        assert s.param_at("rate", Timestep(5, Frequency.MONTHLY)) == 60.0

    def test_cyclical_same_frequency_uses_modulo(self):
        """Cyclical param with same frequency as timestep uses modulo indexing."""

        @dataclass(frozen=True)
        class CyclicalStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = CyclicalStrategy()
        # param_freq=MONTHLY=12, t.frequency=MONTHLY=12
        # idx = (5 * 12 // 12) % 3 = 5 % 3 = 2
        assert s.param_at("rate", Timestep(5, Frequency.MONTHLY)) == 30.0

    def test_cross_frequency_daily_timestep_monthly_param(self):
        """Daily timestep with monthly cyclical param uses cross-frequency conversion."""

        @dataclass(frozen=True)
        class MonthlyCyclical(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("rate",)
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"rate": (0.0, 100.0)}
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            rate: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

        s = MonthlyCyclical()
        # param_freq=MONTHLY=12, t.frequency=DAILY=365
        # idx = (30 * 12 // 365) % 12 = (360 // 365) % 12 = 0 % 12 = 0
        assert s.param_at("rate", Timestep(30, Frequency.DAILY)) == 1.0

    def test_param_at_mixed_scalar_and_cyclical(self):
        """param_at works for strategy with both scalar and cyclical params."""

        @dataclass(frozen=True)
        class MixedStrategy(Strategy):
            __params__: ClassVar[tuple[str, ...]] = ("base", "rate")
            __bounds__: ClassVar[dict[str, tuple[float, float]]] = {
                "base": (0.0, 100.0),
                "rate": (0.0, 100.0),
            }
            __time_varying__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical__: ClassVar[tuple[str, ...]] = ("rate",)
            __cyclical_freq__: ClassVar[dict[str, Frequency]] = {"rate": Frequency.MONTHLY}
            base: float = 50.0
            rate: tuple[float, ...] = (10.0, 20.0, 30.0)

        s = MixedStrategy()
        t = Timestep(5, Frequency.MONTHLY)
        # Scalar param always returns its value
        assert s.param_at("base", t) == 50.0
        # Cyclical: idx = (5 * 12 // 12) % 3 = 5 % 3 = 2
        assert s.param_at("rate", t) == 30.0
