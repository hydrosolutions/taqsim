from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from ctrl_freak import Population, nsga2, polynomial_mutation, sbx_crossover

from taqsim.objective import Objective
from taqsim.system import ValidationError, WaterSystem

from .repair import make_repair
from .result import OptimizeResult, Solution

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _derive_seeds(seed: int | None, n: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 2**31, size=n)]


def _make_verbose_callback(
    objectives: list[Objective],
    user_callback: Callable[[Population, int], bool] | None,
) -> Callable[[Population, int], bool]:
    def callback(pop: Population, gen: int) -> bool:
        n_pareto = int(np.sum(pop.rank == 0)) if pop.rank is not None else 0
        pareto_obj = pop.objectives[pop.rank == 0] if pop.rank is not None else pop.objectives
        if pareto_obj is not None and len(pareto_obj) > 0:
            # Reverse transformation for maximize objectives
            display_vals = []
            for j, obj in enumerate(objectives):
                val = pareto_obj[0, j]
                if obj.direction == "maximize":
                    val = -val
                display_vals.append(f"{obj.name}={val:.4f}")
            obj_str = ", ".join(display_vals)
        else:
            obj_str = "N/A"
        print(f"Gen {gen:4d} | Pareto: {n_pareto:3d} | {obj_str}")
        if user_callback is not None:
            return user_callback(pop, gen)
        return False

    return callback


def optimize(
    system: WaterSystem,
    objectives: list[Objective],
    timesteps: int,
    *,
    pop_size: int = 100,
    generations: int = 200,
    seed: int | None = None,
    warm_start: bool = False,
    verbose: bool = False,
    callback: Callable[[Population, int], bool] | None = None,
) -> OptimizeResult:
    """Run multi-objective optimization on a water system.

    Uses NSGA-II to find Pareto-optimal parameter configurations
    that balance multiple objectives.
    """
    # Validation
    if not system._validated:
        raise ValidationError("System must be validated before optimization")
    if not system.param_schema():
        raise ValueError("System has no tunable parameters")
    if not objectives:
        raise ValueError("At least one objective is required")
    if pop_size < 4:
        raise ValueError("pop_size must be at least 4")

    # Extract bounds
    bounds = system.bounds_vector()
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    # Create repair function
    repair = make_repair(system)

    # Derive sub-seeds for reproducibility
    seeds = _derive_seeds(seed, 3)
    init_seed, crossover_seed, mutate_seed = seeds

    # Operators
    def init(rng: np.random.Generator) -> NDArray[np.float64]:
        x = rng.uniform(lower, upper)
        return repair(x)

    def evaluate(x: NDArray[np.float64]) -> NDArray[np.float64]:
        candidate = system.with_vector(x.tolist())
        candidate.simulate(timesteps)
        scores = []
        for obj in objectives:
            val = obj.evaluate(candidate)
            # Negate maximize objectives (NSGA-II minimizes by default)
            if obj.direction == "maximize":
                val = -val
            scores.append(val)
        return np.array(scores)

    # Create crossover and mutation with repair composition
    sbx = sbx_crossover(eta=15.0, bounds=(float(lower.min()), float(upper.max())), seed=crossover_seed)
    poly_mut = polynomial_mutation(eta=20.0, bounds=(float(lower.min()), float(upper.max())), seed=mutate_seed)

    def crossover(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> NDArray[np.float64]:
        child = sbx(p1, p2)
        return repair(child)

    def mutate(x: NDArray[np.float64]) -> NDArray[np.float64]:
        mutated = poly_mut(x)
        return repair(mutated)

    # Build callback
    final_callback = _make_verbose_callback(objectives, callback) if verbose else callback

    # Run NSGA-II
    final_pop = nsga2(
        init=init,
        evaluate=evaluate,
        crossover=crossover,
        mutate=mutate,
        pop_size=pop_size,
        n_generations=generations,
        seed=init_seed,
        callback=final_callback,
    )

    # Build OptimizeResult
    # Extract Pareto front (rank == 0)
    pareto_mask = final_pop.rank == 0
    pareto_x = final_pop.x[pareto_mask]
    pareto_obj = final_pop.objectives[pareto_mask]

    # Reverse direction transform for scores
    solutions: list[Solution] = []
    schema = system.param_schema()
    param_keys = [spec.path for spec in schema]

    for i in range(len(pareto_x)):
        vec = pareto_x[i]
        raw_scores = pareto_obj[i]

        # Reverse negation for maximize objectives
        scores: dict[str, float] = {}
        for j, obj in enumerate(objectives):
            val = float(raw_scores[j])
            if obj.direction == "maximize":
                val = -val
            scores[obj.name] = val

        # Build parameters dict
        parameters: dict[str, float] = {key: float(vec[k]) for k, key in enumerate(param_keys)}

        solutions.append(
            Solution(
                scores=scores,
                parameters=parameters,
                _vector=vec,
                _template=system,
            )
        )

    return OptimizeResult(solutions=solutions, population=final_pop)
