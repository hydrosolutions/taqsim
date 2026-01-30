import pytest

from taqsim import Objective, WaterSystem
from taqsim.optimization import optimize
from taqsim.optimization.result import OptimizeResult
from taqsim.system import ValidationError


class TestOptimizeBasic:
    def test_returns_optimize_result(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        assert isinstance(result, OptimizeResult)

    def test_returns_at_least_one_solution(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        assert len(result.solutions) >= 1

    def test_solution_parameters_within_bounds(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        bounds = minimal_water_system.param_bounds()

        for solution in result.solutions:
            for param_path, value in solution.parameters.items():
                lo, hi = bounds[param_path]
                assert lo <= value <= hi, f"{param_path}={value} not in [{lo}, {hi}]"


class TestOptimizeObjectives:
    def test_single_objective_optimization(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        assert len(result.solutions) >= 1
        for solution in result.solutions:
            assert simple_minimize_objective.name in solution.scores

    def test_two_objective_optimization(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
        simple_maximize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective, simple_maximize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        assert len(result.solutions) >= 1
        for solution in result.solutions:
            assert simple_minimize_objective.name in solution.scores
            assert simple_maximize_objective.name in solution.scores

    def test_respects_minimize_direction(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=10,
            seed=42,
        )
        bounds = minimal_water_system.param_bounds()
        rate_lo, _ = bounds["dam.release_rule.rate"]

        best_solution = min(
            result.solutions, key=lambda s: s.scores[simple_minimize_objective.name]
        )
        assert best_solution.scores[simple_minimize_objective.name] <= 50.0
        assert best_solution.parameters["dam.release_rule.rate"] >= rate_lo

    def test_respects_maximize_direction(
        self,
        minimal_water_system: WaterSystem,
        simple_maximize_objective: Objective,
    ) -> None:
        result = optimize(
            minimal_water_system,
            [simple_maximize_objective],
            timesteps=5,
            pop_size=10,
            generations=10,
            seed=42,
        )
        bounds = minimal_water_system.param_bounds()
        _, rate_hi = bounds["dam.release_rule.rate"]

        best_solution = max(
            result.solutions, key=lambda s: s.scores[simple_maximize_objective.name]
        )
        assert best_solution.scores[simple_maximize_objective.name] >= 50.0
        assert best_solution.parameters["dam.release_rule.rate"] <= rate_hi


class TestOptimizeDeterminism:
    def test_same_seed_produces_same_result(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result1 = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        result2 = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        assert len(result1.solutions) == len(result2.solutions)
        for s1, s2 in zip(result1.solutions, result2.solutions, strict=True):
            assert s1.parameters == s2.parameters
            assert s1.scores == s2.scores

    def test_different_seed_produces_different_result(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        result1 = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        result2 = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=99,
        )
        params1 = [s.parameters for s in result1.solutions]
        params2 = [s.parameters for s in result2.solutions]
        assert params1 != params2


class TestOptimizeValidation:
    def test_raises_with_empty_objectives(
        self,
        minimal_water_system: WaterSystem,
    ) -> None:
        with pytest.raises(ValueError, match="At least one objective is required"):
            optimize(
                minimal_water_system,
                [],
                timesteps=5,
                pop_size=10,
                generations=3,
                seed=42,
            )

    def test_raises_with_unvalidated_system(
        self,
        simple_minimize_objective: Objective,
    ) -> None:
        system = WaterSystem(dt=1.0)
        with pytest.raises(ValidationError, match="must be validated"):
            optimize(
                system,
                [simple_minimize_objective],
                timesteps=5,
                pop_size=10,
                generations=3,
                seed=42,
            )

    def test_raises_with_small_pop_size(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        with pytest.raises(ValueError, match="pop_size must be at least 4"):
            optimize(
                minimal_water_system,
                [simple_minimize_objective],
                timesteps=5,
                pop_size=2,
                generations=3,
                seed=42,
            )


class TestOptimizeNWorkers:
    def test_default_n_workers_backwards_compatible(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """Ensure existing code without n_workers continues to work."""
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
        )
        assert len(result.solutions) >= 1

    def test_n_workers_one_produces_valid_result(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """Explicit n_workers=1 should work the same as default."""
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
            n_workers=1,
        )
        assert len(result.solutions) >= 1

    def test_n_workers_two_parallel_execution(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """Parallel execution with 2 workers should produce valid results."""
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
            n_workers=2,
        )
        assert len(result.solutions) >= 1

    def test_n_workers_minus_one_all_cores(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """n_workers=-1 should use all CPU cores."""
        result = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
            n_workers=-1,
        )
        assert len(result.solutions) >= 1

    def test_n_workers_zero_raises_value_error(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """n_workers=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="n_workers must be positive or -1"):
            optimize(
                minimal_water_system,
                [simple_minimize_objective],
                timesteps=5,
                pop_size=10,
                generations=3,
                seed=42,
                n_workers=0,
            )

    def test_n_workers_negative_two_raises_value_error(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """n_workers=-2 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="n_workers must be positive or -1"):
            optimize(
                minimal_water_system,
                [simple_minimize_objective],
                timesteps=5,
                pop_size=10,
                generations=3,
                seed=42,
                n_workers=-2,
            )

    def test_parallel_determinism_same_seed(
        self,
        minimal_water_system: WaterSystem,
        simple_minimize_objective: Objective,
    ) -> None:
        """Same seed with parallel workers should produce same results."""
        result1 = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
            n_workers=2,
        )
        result2 = optimize(
            minimal_water_system,
            [simple_minimize_objective],
            timesteps=5,
            pop_size=10,
            generations=3,
            seed=42,
            n_workers=2,
        )
        assert len(result1.solutions) == len(result2.solutions)
        for s1, s2 in zip(result1.solutions, result2.solutions, strict=True):
            assert s1.parameters == s2.parameters
            assert s1.scores == s2.scores
