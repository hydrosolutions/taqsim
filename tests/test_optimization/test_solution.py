from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from taqsim import WaterSystem
from taqsim.optimization.result import Solution


class TestSolutionInit:
    def test_creates_with_required_fields(self, minimal_water_system: WaterSystem) -> None:
        vector = np.array([50.0])
        solution = Solution(
            scores={"obj1": 1.0},
            parameters={"dam.release_rule.rate": 50.0},
            _vector=vector,
            _template=minimal_water_system,
        )

        assert solution.scores == {"obj1": 1.0}
        assert solution.parameters == {"dam.release_rule.rate": 50.0}

    def test_stores_scores_dict(self, minimal_water_system: WaterSystem) -> None:
        scores = {"objective_a": 10.5, "objective_b": 20.3}
        vector = np.array([50.0])
        solution = Solution(
            scores=scores,
            parameters={"dam.release_rule.rate": 50.0},
            _vector=vector,
            _template=minimal_water_system,
        )

        assert solution.scores == scores
        assert solution.scores["objective_a"] == 10.5
        assert solution.scores["objective_b"] == 20.3

    def test_stores_parameters_dict(self, minimal_water_system: WaterSystem) -> None:
        parameters = {"dam.release_rule.rate": 75.0, "other.param": 25.0}
        vector = np.array([75.0])
        solution = Solution(
            scores={"obj": 1.0},
            parameters=parameters,
            _vector=vector,
            _template=minimal_water_system,
        )

        assert solution.parameters == parameters
        assert solution.parameters["dam.release_rule.rate"] == 75.0
        assert solution.parameters["other.param"] == 25.0


class TestSolutionImmutability:
    def test_is_frozen(self, minimal_water_system: WaterSystem) -> None:
        vector = np.array([50.0])
        solution = Solution(
            scores={"obj": 1.0},
            parameters={"dam.release_rule.rate": 50.0},
            _vector=vector,
            _template=minimal_water_system,
        )

        with pytest.raises(FrozenInstanceError):
            solution.scores = {"new": 2.0}


class TestSolutionToSystem:
    def test_returns_water_system(self, minimal_water_system: WaterSystem) -> None:
        vector = np.array([50.0])
        solution = Solution(
            scores={"obj": 1.0},
            parameters={"dam.release_rule.rate": 50.0},
            _vector=vector,
            _template=minimal_water_system,
        )

        result = solution.to_system()

        assert isinstance(result, WaterSystem)

    def test_applies_parameters(self, minimal_water_system: WaterSystem) -> None:
        new_rate = 75.0
        vector = np.array([new_rate])
        solution = Solution(
            scores={"obj": 1.0},
            parameters={"dam.release_rule.rate": new_rate},
            _vector=vector,
            _template=minimal_water_system,
        )

        result = solution.to_system()
        dam = result.nodes["dam"]

        assert dam.release_rule.rate == new_rate

    def test_returns_fresh_instance(self, minimal_water_system: WaterSystem) -> None:
        vector = np.array([50.0])
        solution = Solution(
            scores={"obj": 1.0},
            parameters={"dam.release_rule.rate": 50.0},
            _vector=vector,
            _template=minimal_water_system,
        )

        result1 = solution.to_system()
        result2 = solution.to_system()

        assert result1 is not result2

    def test_preserves_original_template(self, minimal_water_system: WaterSystem) -> None:
        original_rate = minimal_water_system.nodes["dam"].release_rule.rate
        new_rate = 99.0
        vector = np.array([new_rate])
        solution = Solution(
            scores={"obj": 1.0},
            parameters={"dam.release_rule.rate": new_rate},
            _vector=vector,
            _template=minimal_water_system,
        )

        _ = solution.to_system()

        assert minimal_water_system.nodes["dam"].release_rule.rate == original_rate
