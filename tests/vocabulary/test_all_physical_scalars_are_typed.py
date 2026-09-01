"""Public physical scalar declarations refuse unitless values at construction."""

import inspect
from typing import Any, cast

import pytest

from taqsim import (
    CanalLosses,
    CanalSeepageCoefficient,
    IntervalVolume,
    Length,
    Parameter,
    ReservoirEvaporation,
    SourceProvenance,
    SurfaceArea,
    VolumetricRate,
    WaterDepth,
    WaterVolume,
)
from tests import interval_volume, make_water_system


def test_reservoir_evaporation_refuses_unitless_depths_and_surface_areas() -> None:
    with pytest.raises(TypeError, match="WaterDepth"):
        cast(Any, ReservoirEvaporation)(
            (5.0,),
            ((WaterVolume(0.0, "m3"), 100.0), (WaterVolume(1_000.0, "m3"), 200.0)),
        )


def test_canal_losses_refuses_every_unitless_physical_scalar() -> None:
    with pytest.raises(TypeError, match="CanalSeepageCoefficient"):
        cast(Any, CanalLosses)(0.0014, 12.0)


def test_direct_interval_constructor_has_no_provenance_injection_hook() -> None:
    assert "_source_provenance" not in inspect.signature(IntervalVolume).parameters
    forged = SourceProvenance(
        "interval_mean_rate",
        "L/s",
        "1h",
        VolumetricRate(0.1, "L/s"),
    )
    direct = interval_volume([1.0])
    with pytest.raises(TypeError, match="_source_provenance"):
        cast(Any, IntervalVolume)(
            direct.data,
            "m3",
            "1d",
            "1 m3",
            _source_provenance=forged,
        )


def test_reservoir_evaporation_requires_typed_surface_areas() -> None:
    with pytest.raises(TypeError, match="SurfaceArea"):
        cast(Any, ReservoirEvaporation)(
            (WaterDepth(5.0, "mm"),),
            ((WaterVolume(0.0, "m3"), 100.0), (WaterVolume(1_000.0, "m3"), 200.0)),
        )


def test_canal_losses_requires_typed_length_depth_and_width() -> None:
    coefficient = CanalSeepageCoefficient(0.0014, "sqrt(m3/s)/km")
    length = Length(12.0, "km")
    with pytest.raises(TypeError, match="Length"):
        cast(Any, CanalLosses)(coefficient, 12.0)
    with pytest.raises(TypeError, match="WaterDepth"):
        cast(Any, CanalLosses)(coefficient, length, evaporation_depth=(5.0,))
    with pytest.raises(TypeError, match="Length"):
        cast(Any, CanalLosses)(coefficient, length, width=2.0)


def test_all_typed_evaporation_and_canal_scalars_compile() -> None:
    reservoir = ReservoirEvaporation(
        (WaterDepth(5.0, "mm"),),
        (
            (WaterVolume(0.0, "m3"), SurfaceArea(100.0, "m2")),
            (WaterVolume(1_000.0, "m3"), SurfaceArea(2.0, "ha")),
        ),
    )
    canal = CanalLosses(
        CanalSeepageCoefficient(0.0014, "sqrt(m3/s)/km"),
        Length(12.0, "km"),
        evaporation_depth=(WaterDepth(5.0, "mm"),),
        width=Length(2.0, "m"),
    )
    assert reservoir.evaporation_depths[0].to("m") == WaterDepth(0.005, "m")
    assert reservoir.volume_surface_area[1][1].to("m2") == SurfaceArea(20_000.0, "m2")
    assert canal.length.to("m") == Length(12_000.0, "m")


def test_depth_parameters_retain_units_through_compilation_and_substitution() -> None:
    system = make_water_system(1, "1 L")
    system.source("river", interval_volume([100.0]))
    system.reach(
        "reservoir",
        "river",
        "downstream",
        rule=ReservoirEvaporation(
            (WaterDepth(Parameter("evaporation-depth", 5.0, (0.0, 10.0)), "mm"),),
            (
                (WaterVolume(0.0, "m3"), SurfaceArea(100.0, "m2")),
                (WaterVolume(1_000.0, "m3"), SurfaceArea(200.0, "m2")),
            ),
        ),
    )
    model = system.build()
    parameter = model.parameters[0]
    assert parameter.value == WaterDepth(0.005, "m")
    assert parameter.bounds == (WaterDepth(0.0, "m"), WaterDepth(0.01, "m"))
    model.run(bytes(16), {parameter.path: WaterDepth(7.0, "mm")})
    with pytest.raises(TypeError, match="WaterDepth"):
        model.run(bytes(16), {parameter.path: 0.007})
