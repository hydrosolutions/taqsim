from taqsim import Parameter, VolumetricRate, WaterVolume, ZoneRelease
from tests import interval_volume, make_water_system


def build_with_bounds(bounds: tuple[float, float]):
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
            VolumetricRate(Parameter("release-rate", 1.0, bounds), "m3/s"),
        ),
    )
    return water_system.build()


def test_bounds_are_not_model_identity_and_are_offered_to_the_optimizer() -> None:
    narrow = build_with_bounds((0.5, 1.5))
    wide = build_with_bounds((0.0, 2.0))

    assert narrow.model_digest == wide.model_digest
    assert narrow.document == wide.document
    assert narrow.parameter_bounds == {
        "reservoir.release-rate": (VolumetricRate(0.5, "m3/s"), VolumetricRate(1.5, "m3/s"))
    }
    assert wide.parameter_bounds == {
        "reservoir.release-rate": (VolumetricRate(0.0, "m3/s"), VolumetricRate(2.0, "m3/s"))
    }
