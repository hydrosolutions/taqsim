import operator

import pytest

from taqsim import Basin, BasinObjective, Parameter, ZoneRelease, optimize_basin


def bounded_basin():
    basin = Basin(start_date="2020-01-01", timesteps=1, resolution="1 mL")
    basin.source("river", [100.0])
    basin.reach(
        "reservoir",
        "river",
        "downstream",
        rule=ZoneRelease(0.0, 0.0, 1_000.0, Parameter("release-rate", 1.0, (0.0, 2.0))),
    )
    return basin.build()


def test_optimizer_evaluates_basin_runs_and_returns_rerunnable_solutions() -> None:
    model = bounded_basin()
    objective = BasinObjective(
        "release",
        "maximize",
        lambda run: float(run.flow("reservoir").values[0] or 0.0),
    )

    result = optimize_basin(model, [objective], pop_size=4, generations=0, seed=7)

    assert result.ctrl_freak_result.evaluations == 4
    assert result.solutions
    assert all(0.0 <= solution.parameters["reservoir.release-rate"] <= 2.0 for solution in result.solutions)
    result.solutions[0].run(bytes.fromhex("01" * 16))


def test_optimizer_solution_parameters_cannot_be_changed_after_evaluation() -> None:
    model = bounded_basin()
    objective = BasinObjective(
        "release",
        "maximize",
        lambda run: float(run.flow("reservoir").values[0] or 0.0),
    )

    result = optimize_basin(model, [objective], pop_size=4, generations=0, seed=7)
    solution = result.solutions[0]

    with pytest.raises(TypeError):
        operator.setitem(solution.parameters, "reservoir.release-rate", 1_000.0)


def test_unknown_and_non_finite_substitutions_fail_before_execution() -> None:
    model = bounded_basin()

    with pytest.raises(ValueError, match="unknown rule parameter"):
        model.run(bytes(16), {"reservoir.stranger": 1.0})
    with pytest.raises(ValueError, match="must be finite"):
        model.run(bytes(16), {"reservoir.release-rate": float("nan")})
