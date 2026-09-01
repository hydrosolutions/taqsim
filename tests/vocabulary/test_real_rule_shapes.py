from __future__ import annotations

from taqsim import (
    Basin,
    CanalLosses,
    EFlowSplit,
    MonthlyDistribution,
    Parameter,
    PriorityDistribution,
    ReservoirEvaporation,
    ZoneRelease,
    monthly_parameters,
)

RUN_IDS = [bytes([item]) * 16 for item in range(1, 7)]


def _run(rule: object, amount: float = 100.0):
    basin = Basin(start_date="2020-01-01", timesteps=1, resolution="1 mL")
    basin.source("source", [amount])
    basin.reach("structure", "source", "downstream", rule=rule)
    return basin.build().run(RUN_IDS.pop(0))


def test_the_six_real_rule_shapes_compile_and_run() -> None:
    monthly_release = monthly_parameters("seasonal-release", (1.0,) * 12, (0.0, 500.0))
    shapes = [
        ZoneRelease(0.0, 30.0, 80.0, monthly_release),
        MonthlyDistribution({"left": (0.6,) * 12, "right": (0.4,) * 12}),
        PriorityDistribution("powerplant", (20.0,) * 12, {"irrigation": 1.0}),
        EFlowSplit({"river": 1.0}, {"canal": 1.0}, Parameter("eflow-fraction", 0.2, (0.0, 1.0))),
        ReservoirEvaporation((10.0,) * 12, ((0.0, 0.0), (100.0, 1_000.0))),
        CanalLosses(0.01, 1.0, operational_fraction=0.01),
    ]

    for shape in shapes:
        run = _run(shape)
        assert run.authoritative_log_digest
