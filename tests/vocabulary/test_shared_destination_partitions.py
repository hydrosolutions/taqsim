from __future__ import annotations

import pytest

from taqsim import EFlowSplit, PriorityDistribution, WaterVolume
from tests import interval_volume, make_water_system


@pytest.mark.parametrize(
    ("rule", "run_id"),
    [
        (PriorityDistribution("shared", WaterVolume(20.0, "m3"), {"shared": 1.0}), bytes([31]) * 16),
        (EFlowSplit({"shared": 1.0}, {"shared": 1.0}), bytes([32]) * 16),
    ],
)
def test_staged_partition_can_send_both_stages_to_one_destination(rule: object, run_id: bytes) -> None:
    water_system = make_water_system(1, "1 mL")
    water_system.source("source", interval_volume([100.0]))
    water_system.reach("split", "source", "unused", rule=rule)

    run = water_system.build().run(run_id)

    assert run.arrivals("shared").values[0] == pytest.approx(100.0)
