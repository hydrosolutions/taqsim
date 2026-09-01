import operator

import pytest

from taqsim import (
    Parameter,
    VolumetricRate,
    WaterSystemObjective,
    WaterVolume,
    ZoneRelease,
    optimize_water_system,
)
from tests import interval_volume, make_water_system


def bounded_water_system():
    water_system = make_water_system(1, "1 mL")
    water_system.source("river", interval_volume([100.0]))
    water_system.reach(
        "reservoir",
        "river",
        "downstream",
        rule=ZoneRelease(
            WaterVolume(0.0, "m3"),
            WaterVolume(0.0, "m3"),
            WaterVolume(1_000.0, "m3"),
            VolumetricRate(Parameter("release-rate", 1.0, (0.0, 2.0)), "m3/s"),
        ),
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
    for solution in result.solutions:
        rate = solution.parameters["reservoir.release-rate"]
        assert isinstance(rate, VolumetricRate)
        assert isinstance(rate.value, float)
        assert 0.0 <= rate.value <= 2.0
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
            solution.parameters, "reservoir.release-rate", 1_000.0
        )


def test_unknown_and_non_finite_substitutions_fail_before_execution() -> None:
    model = bounded_water_system()

    with pytest.raises(ValueError, match="unknown rule parameter"):
        model.run(bytes(16), {"reservoir.stranger": 1.0})
    with pytest.raises(TypeError, match="must be a VolumetricRate"):
        model.run(bytes(16), {"reservoir.release-rate": 1.0})
    model.run(bytes(16), {"reservoir.release-rate": VolumetricRate(1_000.0, "L/s")})


def test_physical_parameter_units_survive_compilation_and_substitution() -> None:
    system = make_water_system(1, "1 L")
    system.source("river", interval_volume([100.0]))
    system.reach(
        "reservoir",
        "river",
        "downstream",
        rule=ZoneRelease(
            WaterVolume(0.0, "m3"),
            WaterVolume(0.0, "m3"),
            WaterVolume(1_000.0, "m3"),
            VolumetricRate(Parameter("release-rate", 1_000.0, (0.0, 2_000.0)), "L/s"),
        ),
    )
    model = system.build()
    parameter = model.parameters[0]
    assert parameter.value == VolumetricRate(1.0, "m3/s")
    assert parameter.bounds == (VolumetricRate(0.0, "m3/s"), VolumetricRate(2.0, "m3/s"))
    model.run(bytes(16), {parameter.path: VolumetricRate(1_000.0, "L/s")})
    with pytest.raises(TypeError, match="VolumetricRate"):
        model.run(bytes(16), {parameter.path: 1.0})
