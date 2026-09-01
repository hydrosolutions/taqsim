"""Public contract evidence for physically and temporally explicit WaterSystem inputs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from taqsim import (
    ConservationQuantum,
    IntervalMeanRate,
    IntervalVolume,
    TimeAxis,
    VolumetricRate,
    WaterSystem,
    WaterVolume,
)


def frame(start: str, count: int, interval: str, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": pl.datetime_range(start=start, end=None, interval=interval, eager=True).head(count),
            "value": values,
        },
        schema={"time": pl.Datetime("us"), "value": pl.Float64},
    )


def test_daily_interval_volume_builds_runs_and_keeps_meaning() -> None:
    axis = TimeAxis(start="2020-01-01", periods=2, frequency="1d")
    inflow = IntervalVolume(
        data=frame("2020-01-01", 2, "1d", [100.0, 80.0]),
        unit="m3",
        cadence="1d",
        data_resolution="0.1 m3",
    )
    system = WaterSystem(time=axis, quantum=ConservationQuantum.MILLILITRE)
    system.source("river", inflow)
    system.sink("farm")
    system.reach("canal", "river", "farm")

    run = system.build().run(bytes(16))

    assert list(run.arrivals("farm").values) == [100.0, 80.0]
    assert inflow.unit == "m3"
    assert inflow.data_resolution == WaterVolume(0.1, "m3")
    assert inflow.source_provenance.data_resolution == WaterVolume(0.1, "m3")
    assert run.quantum is ConservationQuantum.MILLILITRE


def test_hourly_rate_is_explicitly_integrated_and_aggregated() -> None:
    axis = TimeAxis(start="2020-01-01", periods=1, frequency="1d")
    hourly = IntervalMeanRate(
        data=frame("2020-01-01", 24, "1h", [1.0] * 24),
        unit="m3/s",
        cadence="1h",
        data_resolution="0.01 m3/s",
    )

    daily = hourly.aggregate_to(axis)

    assert daily.data["value"].to_list() == [86_400.0]
    assert daily.unit == "m3"
    assert daily.cadence == "1d"
    assert daily.data_resolution == WaterVolume(36.0, "m3")
    assert daily.source_provenance.kind == "interval_mean_rate"
    assert daily.source_provenance.data_resolution == VolumetricRate(0.01, "m3/s")
    with pytest.raises(FrozenInstanceError):
        daily.source_provenance.unit = "L/s"  # type: ignore[misc]


def test_finer_interval_volumes_sum_only_when_they_exactly_partition_axis() -> None:
    hourly = IntervalVolume(
        data=frame("2020-01-01", 24, "1h", [1.0] * 24),
        unit="m3",
        cadence="1h",
        data_resolution="0.25 m3",
    )
    daily = hourly.aggregate_to(TimeAxis("2020-01-01", periods=1, frequency="1d"))
    assert daily.data["value"].to_list() == [24.0]
    assert daily.data_resolution == WaterVolume(0.25, "m3")


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (pl.DataFrame({"time": [None], "value": [1.0]}, schema={"time": pl.Datetime, "value": pl.Float64}), "null"),
        (frame("2020-01-01", 2, "1d", [1.0, float("nan")]), "finite"),
        (frame("2020-01-01", 2, "1d", [1.0, -1.0]), "non-negative"),
        (pl.DataFrame({"time": [1], "value": [1.0]}), "schema"),
        (pl.DataFrame({"time": [pl.datetime(2020, 1, 1)], "value": [1.0], "extra": [1.0]}), "columns"),
    ],
)
def test_interval_input_refuses_invalid_frames(data: pl.DataFrame, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        IntervalVolume(data=data, unit="m3", cadence="1d", data_resolution="1 m3")


def test_temporal_refusals_are_loud() -> None:
    with pytest.raises(ValueError, match="cadence"):
        IntervalVolume(
            data=frame("2020-01-01", 2, "2d", [1.0, 1.0]),
            unit="m3",
            cadence="1d",
            data_resolution="1 m3",
        )
    coarse = IntervalVolume(
        data=frame("2020-01-01", 1, "1d", [1.0]),
        unit="m3",
        cadence="1d",
        data_resolution="1 m3",
    )
    with pytest.raises(ValueError, match="disaggregate"):
        coarse.aggregate_to(TimeAxis("2020-01-01", periods=24, frequency="1h"))
    partial = IntervalMeanRate(
        data=frame("2020-01-01", 23, "1h", [1.0] * 23),
        unit="m3/s",
        cadence="1h",
        data_resolution="1 m3/s",
    )
    with pytest.raises(ValueError, match="cover"):
        partial.aggregate_to(TimeAxis("2020-01-01", periods=1, frequency="1d"))


def test_units_and_scalar_water_values_are_typed() -> None:
    assert WaterVolume(1.0, "L").to("m3") == WaterVolume(0.001, "m3")
    assert VolumetricRate(1.0, "L/s").to("m3/day") == VolumetricRate(86.4, "m3/day")
    with pytest.raises(ValueError, match="volume"):
        WaterVolume(1.0, "m3/s")
    with pytest.raises(ValueError, match="rate"):
        VolumetricRate(1.0, "m3")
    with pytest.raises((TypeError, ValueError), match="WaterVolume"):
        WaterSystem(
            time=TimeAxis("2020-01-01", periods=1, frequency="1d"),
            quantum=ConservationQuantum.LITRE,
        ).reach("canal", "river", "farm", initial_water=1.0)


def test_unaligned_input_is_not_prepared_implicitly() -> None:
    system = WaterSystem(
        time=TimeAxis("2020-01-01", periods=1, frequency="1d"),
        quantum=ConservationQuantum.LITRE,
    )
    system.source(
        "river",
        IntervalVolume(
            data=frame("2020-01-01", 24, "1h", [1.0] * 24),
            unit="m3",
            cadence="1h",
            data_resolution="1 m3",
        ),
    )
    system.reach("canal", "river", "farm")
    with pytest.raises(ValueError, match="aggregate_to"):
        system.build()
