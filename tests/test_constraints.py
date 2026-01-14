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
