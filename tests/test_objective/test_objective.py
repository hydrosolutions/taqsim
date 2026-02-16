from dataclasses import FrozenInstanceError

import pytest

from taqsim.objective.objective import Objective


class TestObjective:
    def test_creates_with_required_fields(self) -> None:
        def eval_fn(system) -> float:
            return 42.0

        obj = Objective(
            name="test_objective",
            direction="minimize",
            evaluate=eval_fn,
        )

        assert obj.name == "test_objective"
        assert obj.direction == "minimize"
        assert obj.evaluate is eval_fn

    def test_is_frozen(self) -> None:
        obj = Objective(
            name="frozen_test",
            direction="maximize",
            evaluate=lambda s: 0.0,
        )

        with pytest.raises(FrozenInstanceError):
            obj.name = "new_name"

    def test_evaluate_calls_function(self) -> None:
        call_count = []

        def tracker(system) -> float:
            call_count.append(1)
            return 99.0

        obj = Objective(
            name="tracker",
            direction="maximize",
            evaluate=tracker,
        )

        result = obj.evaluate(None)
        assert result == 99.0
        assert len(call_count) == 1
