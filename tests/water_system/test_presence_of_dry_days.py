from __future__ import annotations

from taqsim import Presence
from tests import interval_volume, make_water_system


def test_zero_inside_the_horizon_is_not_missing() -> None:
    water_system = make_water_system(3, "1 mL")
    water_system.source("river", interval_volume([5.0, 0.0, 0.0]))
    water_system.sink("farm")
    water_system.add_reach("dry-reach", "river", "farm")

    run = water_system.build().run(bytes(16))
    flow = run.flow("dry-reach", start="2019-12-31", end="2020-01-04")

    assert flow.values == (None, 5.0, 0.0, 0.0, None)
    assert flow.values.presence == flow.presence
    assert flow.presence == (
        Presence.NOT_MODELLED,
        Presence.PRESENT,
        Presence.PRESENT,
        Presence.PRESENT,
        Presence.NOT_MODELLED,
    )
