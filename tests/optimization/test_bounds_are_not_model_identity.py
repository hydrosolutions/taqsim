from taqsim import Basin, Parameter, ZoneRelease


def build_with_bounds(bounds: tuple[float, float]):
    basin = Basin(start_date="2020-01-01", timesteps=1)
    basin.source("river", [100.0])
    basin.reach(
        "reservoir",
        "river",
        "downstream",
        rule=ZoneRelease(0.0, 0.0, 1_000.0, Parameter("release-rate", 1.0, bounds)),
    )
    return basin.build()


def test_bounds_are_not_model_identity_and_are_offered_to_the_optimizer() -> None:
    narrow = build_with_bounds((0.5, 1.5))
    wide = build_with_bounds((0.0, 2.0))

    assert narrow.model_digest == wide.model_digest
    assert narrow.document == wide.document
    assert narrow.parameter_bounds == {"reservoir.release-rate": (0.5, 1.5)}
    assert wide.parameter_bounds == {"reservoir.release-rate": (0.0, 2.0)}
