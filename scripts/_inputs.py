"""Typed input constructors for executable repository checks."""

from collections.abc import Iterable
from datetime import datetime, timedelta

import polars as pl

from taqsim import ConservationQuantum, IntervalVolume, TimeAxis, WaterSystem


def interval_volume(values: Iterable[float], *, cadence: str = "1d") -> IntervalVolume:
    materialized = list(values)
    seconds = {"1d": 86400}[cadence]
    start = datetime(2020, 1, 1)
    return IntervalVolume(
        pl.DataFrame(
            {
                "time": [start + timedelta(seconds=index * seconds) for index in range(len(materialized))],
                "value": materialized,
            },
            schema={"time": pl.Datetime("us"), "value": pl.Float64},
        ),
        "m3",
        cadence,
        "1 mm3",
    )


def make_water_system(periods: int, quantum: str | ConservationQuantum) -> WaterSystem:
    return WaterSystem(time=TimeAxis("2020-01-01", periods=periods, frequency="1d"), quantum=quantum)
