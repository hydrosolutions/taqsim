"""Public physical scalar declarations refuse unitless values at construction."""

import inspect

import pytest

from taqsim import (
    CanalLosses,
    IntervalVolume,
    ReservoirEvaporation,
    SourceProvenance,
    VolumetricRate,
    WaterVolume,
)
from tests import interval_volume


def test_reservoir_evaporation_refuses_unitless_depths_and_surface_areas() -> None:
    with pytest.raises(TypeError, match="WaterDepth"):
        ReservoirEvaporation(
            (5.0,),
            ((WaterVolume(0.0, "m3"), 100.0), (WaterVolume(1_000.0, "m3"), 200.0)),
        )


def test_canal_losses_refuses_every_unitless_physical_scalar() -> None:
    with pytest.raises(TypeError, match="CanalSeepageCoefficient"):
        CanalLosses(0.0014, 12.0, evaporation_mm=(5.0,), width_m=2.0)


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
        IntervalVolume(
            direct.data,
            "m3",
            "1d",
            "1 m3",
            _source_provenance=forged,
        )
