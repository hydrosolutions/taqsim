from __future__ import annotations

import pytest

from taqsim import Basin, EFlowSplit, PriorityDistribution


@pytest.mark.parametrize(
    ("rule", "run_id"),
    [
        (PriorityDistribution("shared", 20.0, {"shared": 1.0}), bytes([31]) * 16),
        (EFlowSplit({"shared": 1.0}, {"shared": 1.0}), bytes([32]) * 16),
    ],
)
def test_staged_partition_can_send_both_stages_to_one_destination(rule: object, run_id: bytes) -> None:
    basin = Basin(start_date="2020-01-01", timesteps=1, resolution="1 mL")
    basin.source("source", [100.0])
    basin.reach("split", "source", "unused", rule=rule)

    run = basin.build().run(run_id)

    assert run.arrivals("shared").values[0] == pytest.approx(100.0)
