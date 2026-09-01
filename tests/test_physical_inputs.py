"""Public contract evidence for physically and temporally explicit WaterSystem inputs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

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
    origin = datetime.fromisoformat(start)
    seconds = {"1h": 3600, "1d": 86400, "2d": 172800}[interval]
    return pl.DataFrame(
        {
            "time": [origin + timedelta(seconds=seconds * index) for index in range(count)],
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
        daily.source_provenance.unit = "L/s"  # ty: ignore[invalid-assignment]


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
        (pl.DataFrame({"time": [datetime(2020, 1, 1)], "value": [1.0], "extra": [1.0]}), "columns"),
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


def test_common_units_convert_case_sensitively_and_preserve_direct_declaration() -> None:
    declared = IntervalVolume(
        data=frame("2020-01-01", 1, "1d", [1.0]),
        unit="m3",
        cadence="1d",
        data_resolution="100 L",
    )
    assert declared.data_resolution == WaterVolume(100.0, "L")
    assert declared.source_provenance.data_resolution == WaterVolume(100.0, "L")
    assert WaterVolume(1.0, "ML").to("m3") == WaterVolume(1_000.0, "m3")
    assert WaterVolume(1.0, "mL").to("m3") == WaterVolume(1e-6, "m3")
    assert WaterVolume(1.0, "mm3").to("m3") == WaterVolume(1e-9, "m3")
    with pytest.raises(ValueError, match="unknown"):
        WaterVolume(1.0, "l")


def test_same_cadence_conversion_is_explicit_and_preserves_provenance() -> None:
    litres = IntervalVolume(
        data=frame("2020-01-01", 1, "1d", [1_000.0]),
        unit="L",
        cadence="1d",
        data_resolution="1 L",
    )
    converted = litres.aggregate_to(TimeAxis("2020-01-01", periods=1, frequency="1d"), unit="m3")
    assert converted.data["value"].to_list() == [1.0]
    assert converted.data_resolution == WaterVolume(0.001, "m3")
    assert converted.source_provenance.unit == "L"


def test_timezone_and_datetime_precision_are_refused() -> None:
    aware = frame("2020-01-01", 1, "1d", [1.0]).with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    with pytest.raises(ValueError, match="timezone-naive"):
        IntervalVolume(aware, "m3", "1d", "1 m3")
    nanosecond = frame("2020-01-01", 1, "1d", [1.0]).with_columns(pl.col("time").cast(pl.Datetime("ns")))
    with pytest.raises(ValueError, match="schema"):
        IntervalVolume(nanosecond, "m3", "1d", "1 m3")


def test_source_frame_is_detached_from_caller_changes() -> None:
    original = frame("2020-01-01", 1, "1d", [1.0])
    typed = IntervalVolume(original, "m3", "1d", "1 m3")
    original[0, "value"] = 9.0
    detached = typed.data
    detached[0, "value"] = 8.0
    assert typed.data["value"].to_list() == [1.0]


def test_physical_rules_reject_bare_water_amounts_and_rates() -> None:
    from typing import Any, cast

    from taqsim import PriorityDistribution, ZoneRelease

    untyped = cast(Any, 1.0)
    with pytest.raises(TypeError, match="WaterVolume"):
        PriorityDistribution("farm", untyped, {"other": 1.0})
    with pytest.raises(TypeError, match="WaterVolume"):
        ZoneRelease(
            untyped,
            WaterVolume(1.0, "m3"),
            WaterVolume(2.0, "m3"),
            VolumetricRate(1.0, "m3/s"),
        )


def test_decimal_grid_aggregation_remains_quantum_representable() -> None:
    hourly = IntervalVolume(
        data=frame("2020-01-01", 2, "1h", [0.1, 0.2]),
        unit="m3",
        cadence="1h",
        data_resolution="0.001 m3",
    )
    axis = TimeAxis("2020-01-01", periods=1, frequency="2h")
    prepared = hourly.aggregate_to(axis)
    assert prepared.data["value"].to_list() == [0.3]
    system = WaterSystem(time=axis, quantum=ConservationQuantum.LITRE)
    system.source("river", prepared)
    system.reach("canal", "river", "farm")
    assert system.build().run(bytes(16)).arrivals("farm").values[0] == 0.3


def test_common_rate_unit_aggregation_defaults_to_model_ready_cubic_metres() -> None:
    litres_per_second = IntervalMeanRate(
        data=frame("2020-01-01", 24, "1h", [1.0] * 24),
        unit="L/s",
        cadence="1h",
        data_resolution="0.1 L/s",
    )
    daily = litres_per_second.aggregate_to(TimeAxis("2020-01-01", periods=1, frequency="1d"))
    assert daily.unit == "m3"
    assert daily.data["value"].to_list() == [86.4]
    assert daily.data_resolution == WaterVolume(0.36, "m3")
