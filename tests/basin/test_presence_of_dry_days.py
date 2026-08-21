from __future__ import annotations

from taqsim import Basin, Presence


def test_zero_inside_the_horizon_is_not_missing() -> None:
    basin = Basin(start_date="2020-01-01", timesteps=3)
    basin.source("river", flow=[5.0, 0.0, 0.0])
    basin.sink("farm")
    basin.add_reach("dry-reach", "river", "farm")

    run = basin.build().run(bytes(16))
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
