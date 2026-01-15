import pytest

from taqsim.constraints import Constraint, Ordered, SumToOne


class TestSumToOne:
    def test_normalizes_to_sum_one(self):
        constraint = SumToOne(params=("a", "b", "c"))
        values = {"a": 2.0, "b": 3.0, "c": 5.0}

        result = constraint.repair(values)

        assert result["a"] == pytest.approx(0.2)
        assert result["b"] == pytest.approx(0.3)
        assert result["c"] == pytest.approx(0.5)
        assert sum(result[p] for p in constraint.params) == pytest.approx(1.0)

    def test_preserves_already_valid(self):
        constraint = SumToOne(params=("x", "y"))
        values = {"x": 0.4, "y": 0.6}

        result = constraint.repair(values)

        assert result["x"] == pytest.approx(0.4)
        assert result["y"] == pytest.approx(0.6)

    def test_handles_all_zeros(self):
        constraint = SumToOne(params=("a", "b", "c"))
        values = {"a": 0.0, "b": 0.0, "c": 0.0}

        result = constraint.repair(values)

        assert result["a"] == pytest.approx(1.0 / 3.0)
        assert result["b"] == pytest.approx(1.0 / 3.0)
        assert result["c"] == pytest.approx(1.0 / 3.0)

    def test_clamps_negatives(self):
        constraint = SumToOne(params=("a", "b", "c"))
        values = {"a": -1.0, "b": 2.0, "c": 3.0}

        result = constraint.repair(values)

        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.4)
        assert result["c"] == pytest.approx(0.6)
        assert sum(result[p] for p in constraint.params) == pytest.approx(1.0)

    def test_preserves_other_keys(self):
        constraint = SumToOne(params=("a", "b"))
        values = {"a": 1.0, "b": 3.0, "other": 99.0, "another": -5.0}

        result = constraint.repair(values)

        assert result["other"] == 99.0
        assert result["another"] == -5.0
        assert result["a"] == pytest.approx(0.25)
        assert result["b"] == pytest.approx(0.75)

    def test_single_param(self):
        constraint = SumToOne(params=("only",))
        values = {"only": 5.0}

        result = constraint.repair(values)

        assert result["only"] == pytest.approx(1.0)

    def test_satisfies_constraint_protocol(self):
        constraint = SumToOne(params=("a", "b"))
        assert isinstance(constraint, Constraint)


class TestOrdered:
    def test_swaps_when_out_of_order(self):
        constraint = Ordered(low="min", high="max")
        values = {"min": 10.0, "max": 5.0}

        result = constraint.repair(values)

        assert result["min"] == 5.0
        assert result["max"] == 10.0

    def test_preserves_when_valid(self):
        constraint = Ordered(low="min", high="max")
        values = {"min": 3.0, "max": 7.0}

        result = constraint.repair(values)

        assert result["min"] == 3.0
        assert result["max"] == 7.0

    def test_preserves_other_keys(self):
        constraint = Ordered(low="lo", high="hi")
        values = {"lo": 100.0, "hi": 50.0, "unrelated": 42.0}

        result = constraint.repair(values)

        assert result["lo"] == 50.0
        assert result["hi"] == 100.0
        assert result["unrelated"] == 42.0

    def test_params_property(self):
        constraint = Ordered(low="a", high="b")
        assert constraint.params == ("a", "b")

    def test_preserves_when_equal(self):
        constraint = Ordered(low="x", high="y")
        values = {"x": 5.0, "y": 5.0}

        result = constraint.repair(values)

        assert result["x"] == 5.0
        assert result["y"] == 5.0

    def test_satisfies_constraint_protocol(self):
        constraint = Ordered(low="a", high="b")
        assert isinstance(constraint, Constraint)


class TestSumToOneBoundsAware:
    """Tests for bounds-aware SumToOne repair."""

    def test_respects_upper_bounds_when_reducing(self):
        """When sum > target, don't push any value above its upper bound."""
        constraint = SumToOne(params=("a", "b", "c"))
        values = {"a": 0.5, "b": 0.5, "c": 0.5}  # sum = 1.5
        bounds = {"a": (0, 0.3), "b": (0, 0.5), "c": (0, 1.0)}

        result = constraint.repair(values, bounds)

        assert result["a"] <= 0.3 + 1e-9
        assert result["b"] <= 0.5 + 1e-9
        assert result["c"] <= 1.0 + 1e-9
        assert abs(sum(result[p] for p in constraint.params) - 1.0) < 1e-6

    def test_respects_lower_bounds_when_increasing(self):
        """When sum < target, don't push any value below its lower bound."""
        constraint = SumToOne(params=("a", "b", "c"))
        values = {"a": 0.1, "b": 0.1, "c": 0.1}  # sum = 0.3
        bounds = {"a": (0.1, 0.5), "b": (0.1, 0.5), "c": (0.1, 0.8)}

        result = constraint.repair(values, bounds)

        assert result["a"] >= 0.1 - 1e-9
        assert result["b"] >= 0.1 - 1e-9
        assert result["c"] >= 0.1 - 1e-9
        assert abs(sum(result[p] for p in constraint.params) - 1.0) < 1e-6

    def test_distributes_proportionally_to_headroom(self):
        """Increase goes to params with most headroom."""
        constraint = SumToOne(params=("a", "b", "c"))
        values = {"a": 0.3, "b": 0.5, "c": 0.0}  # sum = 0.8, need +0.2
        bounds = {"a": (0, 0.3), "b": (0, 0.5), "c": (0, 1.0)}
        # a has 0 headroom, b has 0 headroom, c has 1.0 headroom

        result = constraint.repair(values, bounds)

        # Only c should increase
        assert result["a"] == pytest.approx(0.3)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(0.2)

    def test_configurable_target(self):
        """SumToOne can target values other than 1.0."""
        constraint = SumToOne(params=("a", "b"), target=0.5)
        values = {"a": 0.3, "b": 0.3}  # sum = 0.6
        bounds = {"a": (0, 1.0), "b": (0, 1.0)}

        result = constraint.repair(values, bounds)

        assert abs(sum(result[p] for p in constraint.params) - 0.5) < 1e-6

    def test_is_feasible_true(self):
        """is_feasible returns True when bounds allow target."""
        constraint = SumToOne(params=("a", "b", "c"))
        bounds = {"a": (0, 0.5), "b": (0, 0.5), "c": (0, 0.5)}  # max = 1.5, min = 0
        assert constraint.is_feasible(bounds)

    def test_is_feasible_false_target_too_high(self):
        """is_feasible returns False when max sum < target."""
        constraint = SumToOne(params=("a", "b"))
        bounds = {"a": (0, 0.3), "b": (0, 0.3)}  # max = 0.6 < 1.0
        assert not constraint.is_feasible(bounds)

    def test_is_feasible_false_target_too_low(self):
        """is_feasible returns False when min sum > target."""
        constraint = SumToOne(params=("a", "b"), target=0.5)
        bounds = {"a": (0.3, 1.0), "b": (0.3, 1.0)}  # min = 0.6 > 0.5
        assert not constraint.is_feasible(bounds)


class TestOrderedBoundsAware:
    """Tests for bounds-aware Ordered repair."""

    def test_simple_swap_within_bounds(self):
        """When swap keeps both in bounds, use it."""
        constraint = Ordered(low="min", high="max")
        values = {"min": 70, "max": 30}
        bounds = {"min": (0, 100), "max": (0, 100)}

        result = constraint.repair(values, bounds)

        assert result["min"] == 30
        assert result["max"] == 70

    def test_swap_violates_bounds_use_overlap(self):
        """When swap violates bounds, use overlap region."""
        constraint = Ordered(low="min", high="max")
        # swap would give min=20, max=80, but 20 < min bounds (25,50), 80 > max bounds (40,75)
        values = {"min": 80, "max": 20}
        bounds = {"min": (25, 50), "max": (40, 75)}  # overlap [40, 50]

        result = constraint.repair(values, bounds)

        # Both should be in overlap region, and min <= max
        assert 40 <= result["min"] <= 50
        assert 40 <= result["max"] <= 75
        assert result["min"] <= result["max"]

    def test_no_overlap_push_to_gap(self):
        """When no overlap but valid gap exists, push to boundaries."""
        constraint = Ordered(low="min", high="max")
        # swap would give min=35, max=80, but 35 is in (0,40), 80 > max bounds (60,75)
        values = {"min": 80, "max": 35}
        bounds = {"min": (0, 40), "max": (60, 75)}  # gap between 40 and 60

        result = constraint.repair(values, bounds)

        assert result["min"] == 40
        assert result["max"] == 60

    def test_impossible_bounds_best_effort(self):
        """When bounds make constraint impossible, return best effort."""
        constraint = Ordered(low="min", high="max")
        values = {"min": 70, "max": 30}
        bounds = {"min": (60, 100), "max": (0, 50)}  # min always > max

        result = constraint.repair(values, bounds)

        # Best effort: minimize violation
        assert result["min"] == 60  # lowest possible
        assert result["max"] == 50  # highest possible

    def test_is_feasible_true(self):
        """is_feasible returns True when constraint can be satisfied."""
        constraint = Ordered(low="min", high="max")
        bounds = {"min": (0, 50), "max": (40, 100)}
        assert constraint.is_feasible(bounds)

    def test_is_feasible_false(self):
        """is_feasible returns False when min bounds > max bounds."""
        constraint = Ordered(low="min", high="max")
        bounds = {"min": (60, 100), "max": (0, 50)}
        assert not constraint.is_feasible(bounds)
