from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from taqsim.optimization.result import OptimizeResult, Solution


@dataclass
class FakePopulation:
    size: int = 100


@dataclass(frozen=True)
class FakeWaterSystem:
    def with_vector(self, vector: list[float]) -> "FakeWaterSystem":
        return self


class TestOptimizeResultInit:
    def test_creates_with_solutions_list(self) -> None:
        solutions = [
            Solution(
                scores={"obj1": 1.0},
                parameters={"p1": 0.5},
                _vector=np.array([0.5]),
                _template=FakeWaterSystem(),
            ),
            Solution(
                scores={"obj1": 2.0},
                parameters={"p1": 0.7},
                _vector=np.array([0.7]),
                _template=FakeWaterSystem(),
            ),
        ]
        population = FakePopulation()

        result = OptimizeResult(solutions=solutions, population=population)

        assert result.solutions == solutions

    def test_stores_population(self) -> None:
        solutions = [
            Solution(
                scores={"obj1": 1.0},
                parameters={"p1": 0.5},
                _vector=np.array([0.5]),
                _template=FakeWaterSystem(),
            ),
        ]
        population = FakePopulation(size=200)

        result = OptimizeResult(solutions=solutions, population=population)

        assert result.population is population
        assert result.population.size == 200


class TestOptimizeResultImmutability:
    def test_is_frozen(self) -> None:
        solutions = [
            Solution(
                scores={"obj1": 1.0},
                parameters={"p1": 0.5},
                _vector=np.array([0.5]),
                _template=FakeWaterSystem(),
            ),
        ]
        population = FakePopulation()

        result = OptimizeResult(solutions=solutions, population=population)

        with pytest.raises(FrozenInstanceError):
            result.solutions = []


class TestOptimizeResultIteration:
    def test_len_returns_solution_count(self) -> None:
        template = FakeWaterSystem()
        solutions = [
            Solution(scores={"a": 1.0}, parameters={"p": 0.1}, _vector=np.array([0.1]), _template=template),
            Solution(scores={"a": 2.0}, parameters={"p": 0.2}, _vector=np.array([0.2]), _template=template),
            Solution(scores={"a": 3.0}, parameters={"p": 0.3}, _vector=np.array([0.3]), _template=template),
        ]
        result = OptimizeResult(solutions=solutions, population=FakePopulation())

        assert len(result) == 3

    def test_indexing_returns_solution(self) -> None:
        template = FakeWaterSystem()
        solutions = [
            Solution(scores={"a": 1.0}, parameters={"p": 0.1}, _vector=np.array([0.1]), _template=template),
            Solution(scores={"a": 2.0}, parameters={"p": 0.2}, _vector=np.array([0.2]), _template=template),
        ]
        result = OptimizeResult(solutions=solutions, population=FakePopulation())

        assert result[0] is solutions[0]
        assert result[1] is solutions[1]

    def test_negative_indexing_works(self) -> None:
        template = FakeWaterSystem()
        solutions = [
            Solution(scores={"a": 1.0}, parameters={"p": 0.1}, _vector=np.array([0.1]), _template=template),
            Solution(scores={"a": 2.0}, parameters={"p": 0.2}, _vector=np.array([0.2]), _template=template),
            Solution(scores={"a": 3.0}, parameters={"p": 0.3}, _vector=np.array([0.3]), _template=template),
        ]
        result = OptimizeResult(solutions=solutions, population=FakePopulation())

        assert result[-1] is solutions[2]
        assert result[-2] is solutions[1]

    def test_iteration_over_solutions(self) -> None:
        template = FakeWaterSystem()
        solutions = [
            Solution(scores={"a": 1.0}, parameters={"p": 0.1}, _vector=np.array([0.1]), _template=template),
            Solution(scores={"a": 2.0}, parameters={"p": 0.2}, _vector=np.array([0.2]), _template=template),
        ]
        result = OptimizeResult(solutions=solutions, population=FakePopulation())

        iterated = list(result)

        assert iterated == solutions

    def test_index_out_of_bounds_raises(self) -> None:
        template = FakeWaterSystem()
        solutions = [
            Solution(scores={"a": 1.0}, parameters={"p": 0.1}, _vector=np.array([0.1]), _template=template),
        ]
        result = OptimizeResult(solutions=solutions, population=FakePopulation())

        with pytest.raises(IndexError):
            _ = result[5]
