"""optimize_water_system : BuiltWaterSystem × Objectives × SearchConfig → ParetoSolutions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from ctrl_freak import NSGA2Result, nsga2, polynomial_mutation, sbx_crossover
from numpy.typing import NDArray

from taqsim.water_system import BuiltWaterSystem, RuleParameter, RunId, WaterSystemRun

Direction = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class WaterSystemObjective:
    """A named scalar reading of an immutable WaterSystem run."""

    name: str
    direction: Direction
    evaluate: Callable[[WaterSystemRun], float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("objective name must not be empty")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError(f"unsupported objective direction {self.direction!r}")


@dataclass(frozen=True)
class WaterSystemSolution:
    """One Pareto solution that can execute against the held model."""

    scores: Mapping[str, float]
    parameters: Mapping[str, float]
    _model: BuiltWaterSystem

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def run(self, run_id: RunId) -> WaterSystemRun:
        """Execute this solution without rebuilding or mutating its model."""
        return self._model.run(run_id, self.parameters)


@dataclass(frozen=True)
class WaterSystemOptimizeResult:
    """Pareto solutions and the underlying ctrl-freak result."""

    solutions: tuple[WaterSystemSolution, ...]
    ctrl_freak_result: NSGA2Result


def optimize_water_system(
    model: BuiltWaterSystem,
    objectives: Sequence[WaterSystemObjective],
    *,
    pop_size: int = 100,
    generations: int = 200,
    seed: int | None = None,
    n_workers: int = 1,
) -> WaterSystemOptimizeResult:
    """Optimize bounded rule parameters while holding one compiled model."""
    if not objectives:
        raise ValueError("at least one WaterSystem objective is required")
    if len({objective.name for objective in objectives}) != len(objectives):
        raise ValueError("WaterSystem objective names must be unique")
    if n_workers != 1:
        raise ValueError("held WaterSystem optimization currently requires n_workers=1")
    parameters = tuple(parameter for parameter in model.parameters if parameter.bounds is not None)
    if not parameters:
        raise ValueError("built WaterSystem has no bounded rule parameters")

    lower = np.array([_bounds(parameter)[0] for parameter in parameters], dtype=float)
    upper = np.array([_bounds(parameter)[1] for parameter in parameters], dtype=float)
    crossover = sbx_crossover(eta=15.0, bounds=(lower, upper))
    mutate = polynomial_mutation(eta=20.0, bounds=(lower, upper))

    def init(rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.uniform(lower, upper)

    def evaluate(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        values = {parameter.path: float(vector[index]) for index, parameter in enumerate(parameters)}
        run = model.run(_candidate_run_id(model.model_digest, parameters, vector), values)
        scores = np.array([float(objective.evaluate(run)) for objective in objectives], dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("WaterSystem objectives must return finite values")
        for index, objective in enumerate(objectives):
            if objective.direction == "maximize":
                scores[index] = -scores[index]
        return scores

    result = nsga2(
        init=init,
        evaluate=evaluate,
        crossover=crossover,
        mutate=mutate,
        pop_size=pop_size,
        n_generations=generations,
        seed=seed,
        n_workers=n_workers,
    )
    pareto = result.rank == 0
    vectors = result.population.x[pareto]
    raw_scores = result.population.objectives
    assert raw_scores is not None
    solution_scores = raw_scores[pareto]
    solutions = []
    for vector, scores in zip(vectors, solution_scores, strict=True):
        displayed = {
            objective.name: float(-scores[index] if objective.direction == "maximize" else scores[index])
            for index, objective in enumerate(objectives)
        }
        values = {parameter.path: float(vector[index]) for index, parameter in enumerate(parameters)}
        solutions.append(WaterSystemSolution(displayed, values, model))
    return WaterSystemOptimizeResult(tuple(solutions), result)


def _bounds(parameter: RuleParameter) -> tuple[float, float]:
    assert parameter.bounds is not None
    return parameter.bounds


def _candidate_run_id(
    model_digest: str,
    parameters: tuple[RuleParameter, ...],
    vector: NDArray[np.float64],
) -> bytes:
    digest = hashlib.sha256()
    digest.update(model_digest.encode())
    for parameter, value in zip(parameters, vector, strict=True):
        digest.update(parameter.path.encode())
        digest.update(np.float64(value).tobytes())
    return digest.digest()[:16]
