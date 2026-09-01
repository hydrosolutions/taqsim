"""Shared typed-input constructors for contract tests."""

from collections.abc import Iterable
from datetime import datetime, timedelta

import polars as pl

from taqsim import ConservationQuantum, IntervalVolume, TimeAxis, WaterSystem


def interval_volume(
    values: Iterable[float], *, start: str | datetime = "2020-01-01", cadence: str = "1d"
) -> IntervalVolume:
    materialized = [float(value) for value in values]
    origin = datetime.fromisoformat(start) if isinstance(start, str) else start.replace(tzinfo=None)
    seconds = {"1h": 3600, "1d": 86400, "31d": 31 * 86400}[cadence]
    timestamps = [origin + timedelta(seconds=seconds * index) for index in range(len(materialized))]
    data = pl.DataFrame(
        {"time": timestamps, "value": materialized},
        schema={"time": pl.Datetime("us"), "value": pl.Float64},
    )
    return IntervalVolume(data, "m3", cadence, "1 mm3")


def make_water_system(
    periods: int,
    quantum: str | ConservationQuantum,
    *,
    start: str | datetime = "2020-01-01",
    frequency: str = "1d",
) -> WaterSystem:
    return WaterSystem(time=TimeAxis(start, periods=periods, frequency=frequency), quantum=quantum)
