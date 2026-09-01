from __future__ import annotations

from taqsim import (
    CanalLosses,
    EFlowSplit,
    MonthlyDistribution,
    Parameter,
    PriorityDistribution,
    ReservoirEvaporation,
    VolumetricRate,
    WaterVolume,
    ZoneRelease,
)
from tests import interval_volume, make_water_system

RUN_IDS = [bytes([item]) * 16 for item in range(1, 7)]


def _run(rule: object, amount: float = 100.0):
    water_system = make_water_system(1, "1 mL")
    water_system.source("source", interval_volume([amount]))
    water_system.reach("structure", "source", "downstream", rule=rule)
    return water_system.build().run(RUN_IDS.pop(0))


def test_the_six_real_rule_shapes_compile_and_run() -> None:
    shapes = [
        ZoneRelease(
            WaterVolume(0.0, "m3"),
            WaterVolume(30.0, "m3"),
            WaterVolume(80.0, "m3"),
            (VolumetricRate(1.0, "m3/s"),) * 12,
        ),
        MonthlyDistribution({"left": (0.6,) * 12, "right": (0.4,) * 12}),
        PriorityDistribution("powerplant", (WaterVolume(20.0, "m3"),) * 12, {"irrigation": 1.0}),
        EFlowSplit({"river": 1.0}, {"canal": 1.0}, Parameter("eflow-fraction", 0.2, (0.0, 1.0))),
        ReservoirEvaporation((10.0,) * 12, ((WaterVolume(0.0, "m3"), 0.0), (WaterVolume(100.0, "m3"), 1_000.0))),
        CanalLosses(0.01, 1.0, operational_fraction=0.01),
    ]

    for shape in shapes:
        run = _run(shape)
        assert run.authoritative_log_digest
