from taqsim import CanalLosses, Parameter
from tests import interval_volume, make_water_system


def build_with_bounds(bounds: tuple[float, float]):
    water_system = make_water_system(1, "1 mL")
    water_system.source("river", interval_volume([100.0]))
    water_system.reach(
        "reservoir",
        "river",
        "downstream",
        rule=CanalLosses(Parameter("seepage", 1.0, bounds), 1.0),
    )
    return water_system.build()


def test_bounds_are_not_model_identity_and_are_offered_to_the_optimizer() -> None:
    narrow = build_with_bounds((0.5, 1.5))
    wide = build_with_bounds((0.0, 2.0))

    assert narrow.model_digest == wide.model_digest
    assert narrow.document == wide.document
    assert narrow.parameter_bounds == {"reservoir.seepage": (0.5, 1.5)}
    assert wide.parameter_bounds == {"reservoir.seepage": (0.0, 2.0)}
