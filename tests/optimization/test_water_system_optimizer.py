import operator

import pytest

from taqsim import CanalLosses, Parameter, WaterSystemObjective, optimize_water_system
from tests import interval_volume, make_water_system


def bounded_water_system():
    water_system = make_water_system(1, "1 mL")
    water_system.source("river", interval_volume([100.0]))
    water_system.reach(
        "reservoir",
        "river",
        "downstream",
        rule=CanalLosses(Parameter("seepage", 1.0, (0.0, 2.0)), 1.0),
    )
    return water_system.build()


def test_optimizer_evaluates_water_system_runs_and_returns_rerunnable_solutions() -> None:
    model = bounded_water_system()
    objective = WaterSystemObjective(
        "release",
        "maximize",
        lambda run: float(run.flow("reservoir").values[0] or 0.0),
    )

    result = optimize_water_system(model, [objective], pop_size=4, generations=0, seed=7)

    assert result.ctrl_freak_result.evaluations == 4
    assert result.solutions
    assert all(0.0 <= solution.parameters["reservoir.seepage"] <= 2.0 for solution in result.solutions)
    result.solutions[0].run(bytes.fromhex("01" * 16))


def test_optimizer_solution_parameters_cannot_be_changed_after_evaluation() -> None:
    model = bounded_water_system()
    objective = WaterSystemObjective(
        "release",
        "maximize",
        lambda run: float(run.flow("reservoir").values[0] or 0.0),
    )

    result = optimize_water_system(model, [objective], pop_size=4, generations=0, seed=7)
    solution = result.solutions[0]

    with pytest.raises(TypeError):
        operator.setitem(  # ty: ignore[no-matching-overload]
            solution.parameters, "reservoir.seepage", 1_000.0
        )


def test_unknown_and_non_finite_substitutions_fail_before_execution() -> None:
    model = bounded_water_system()

    with pytest.raises(ValueError, match="unknown rule parameter"):
        model.run(bytes(16), {"reservoir.stranger": 1.0})
    with pytest.raises(ValueError, match="must be finite"):
        model.run(bytes(16), {"reservoir.seepage": float("nan")})
