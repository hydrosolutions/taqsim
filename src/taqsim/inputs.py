"""prepare_input : IntervalWaterInput × TimeAxis × VolumeUnit → IntervalVolume   (pure)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from .water_system import TimeAxis

_VOLUME_FACTORS: dict[str, Decimal] = {
    "km3": Decimal("1e9"),
    "Mm3": Decimal("1e6"),
    "hm3": Decimal("1e6"),
    "ML": Decimal("1e3"),
    "dam3": Decimal("1e3"),
    "m3": Decimal("1"),
    "kL": Decimal("1"),
    "L": Decimal("1e-3"),
    "dm3": Decimal("1e-3"),
    "dL": Decimal("1e-4"),
    "cL": Decimal("1e-5"),
    "cm3": Decimal("1e-6"),
    "mL": Decimal("1e-6"),
    "uL": Decimal("1e-9"),
    "µL": Decimal("1e-9"),
    "mm3": Decimal("1e-9"),
    "nL": Decimal("1e-12"),
}
_TIME_UNITS: dict[str, tuple[str, Decimal]] = {
    "s": ("s", Decimal(1)),
    "sec": ("s", Decimal(1)),
    "secs": ("s", Decimal(1)),
    "second": ("s", Decimal(1)),
    "seconds": ("s", Decimal(1)),
    "min": ("min", Decimal(60)),
    "mins": ("min", Decimal(60)),
    "minute": ("min", Decimal(60)),
    "minutes": ("min", Decimal(60)),
    "h": ("h", Decimal(3600)),
    "hr": ("h", Decimal(3600)),
    "hrs": ("h", Decimal(3600)),
    "hour": ("h", Decimal(3600)),
    "hours": ("h", Decimal(3600)),
    "d": ("day", Decimal(86400)),
    "day": ("day", Decimal(86400)),
    "days": ("day", Decimal(86400)),
}
_VOLUME_ALIASES = {
    "m³": "m3",
    "m^3": "m3",
    "cubic metre": "m3",
    "cubic metres": "m3",
    "cubic meter": "m3",
    "cubic meters": "m3",
    "liter": "L",
    "litre": "L",
    "liters": "L",
    "litres": "L",
}


def _volume_unit(unit: str) -> tuple[str, Decimal]:
    normalized = _VOLUME_ALIASES.get(unit.strip(), unit.strip())
    try:
        return normalized, _VOLUME_FACTORS[normalized]
    except KeyError as error:
        raise ValueError(f"unknown or non-volume unit {unit!r}") from error


def _rate_unit(unit: str) -> tuple[str, Decimal, Decimal]:
    text = unit.strip().replace(" per ", "/")
    if text.count("/") != 1:
        raise ValueError(f"unknown or non-rate unit {unit!r}")
    volume, period = text.split("/")
    canonical_volume, factor = _volume_unit(volume)
    try:
        canonical_period, seconds = _TIME_UNITS[period]
    except KeyError as error:
        raise ValueError(f"unknown or non-rate unit {unit!r}") from error
    return f"{canonical_volume}/{canonical_period}", factor, seconds


def _converted(value: float, factor: Decimal) -> float:
    return float(Decimal(str(value)) * factor)


@dataclass(frozen=True, init=False)
class WaterVolume:
    """One finite non-negative scalar water amount with an explicit volume unit."""

    value: float
    unit: str

    def __init__(self, value: float, unit: str) -> None:
        if isinstance(value, bool):
            raise TypeError("WaterVolume value must be a number, not bool")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("WaterVolume value must be finite and non-negative")
        canonical, _ = _volume_unit(unit)
        object.__setattr__(self, "value", numeric)
        object.__setattr__(self, "unit", canonical)

    def to(self, unit: str) -> WaterVolume:
        """Convert this volume to an explicitly selected compatible unit."""
        target, target_factor = _volume_unit(unit)
        _, source_factor = _volume_unit(self.unit)
        return WaterVolume(_converted(self.value, source_factor / target_factor), target)

    @property
    def m3(self) -> float:
        """Return the amount expressed in cubic metres."""
        return self.to("m3").value


@dataclass(frozen=True, init=False)
class VolumetricRate:
    """One finite non-negative scalar mean volumetric rate with an explicit rate unit."""

    value: float
    unit: str

    def __init__(self, value: float, unit: str) -> None:
        if isinstance(value, bool):
            raise TypeError("VolumetricRate value must be a number, not bool")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("VolumetricRate value must be finite and non-negative")
        canonical, _, _ = _rate_unit(unit)
        object.__setattr__(self, "value", numeric)
        object.__setattr__(self, "unit", canonical)

    def to(self, unit: str) -> VolumetricRate:
        """Convert this rate to an explicitly selected compatible unit."""
        target, target_volume, target_period = _rate_unit(unit)
        _, source_volume, source_period = _rate_unit(self.unit)
        factor = (source_volume / source_period) / (target_volume / target_period)
        return VolumetricRate(_converted(self.value, factor), target)

    @property
    def m3_per_second(self) -> float:
        """Return the rate expressed in cubic metres per second."""
        return self.to("m3/s").value


def _quantity(text: str, kind: Literal["volume", "rate"]) -> WaterVolume | VolumetricRate:
    match = re.fullmatch(r"\s*([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+(.+?)\s*", text)
    if match is None:
        raise ValueError(f"data_resolution must contain a number and a {kind} unit")
    cls = WaterVolume if kind == "volume" else VolumetricRate
    return cls(float(match.group(1)), match.group(2))


@dataclass(frozen=True)
class SourceProvenance:
    """The immutable original physical and temporal declaration of an interval input."""

    kind: Literal["interval_volume", "interval_mean_rate"]
    unit: str
    cadence: str
    data_resolution: WaterVolume | VolumetricRate


def _frequency_seconds(frequency: str) -> int:
    match = re.fullmatch(r"([1-9]\d*)\s*(s|min|h|d)", frequency.strip())
    if match is None:
        raise ValueError(f"unsupported frequency {frequency!r}; use a positive integer followed by s, min, h, or d")
    multiplier = {"s": 1, "min": 60, "h": 3600, "d": 86400}[match.group(2)]
    return int(match.group(1)) * multiplier


def _validated_frame(data: pl.DataFrame, cadence: str) -> pl.DataFrame:
    if not isinstance(data, pl.DataFrame):
        raise TypeError("interval data must be a polars.DataFrame")
    if data.columns != ["time", "value"]:
        raise ValueError("interval data columns must be exactly ['time', 'value'] in that order")
    time_dtype = data.schema["time"]
    if (
        not isinstance(time_dtype, pl.datatypes.Datetime)
        or time_dtype.time_zone is not None
        or time_dtype.time_unit != "us"
    ):
        raise ValueError("interval data schema requires a timezone-naive Polars Datetime 'time' column")
    if data.schema["value"] != pl.Float64:
        raise ValueError("interval data schema requires a Float64 'value' column")
    if data.height < 1:
        raise ValueError("interval data must not be empty")
    if data.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("interval data must not contain null values")
    values = data["value"].to_list()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("interval values must be finite")
    if any(value < 0 for value in values):
        raise ValueError("interval values must be non-negative")
    timestamps: list[datetime] = data["time"].to_list()
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("interval timestamps must be ordered and unique")
    expected = timedelta(seconds=_frequency_seconds(cadence))
    if any(later - earlier != expected for earlier, later in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError(f"interval timestamps contain a gap or overlap relative to declared cadence {cadence!r}")
    return data.clone()


@dataclass(frozen=True, init=False)
class IntervalVolume:
    """A strict regular Polars series whose values are interval-total water volumes."""

    _data: pl.DataFrame
    unit: str
    cadence: str
    data_resolution: WaterVolume
    source_provenance: SourceProvenance

    def __init__(
        self,
        data: pl.DataFrame,
        unit: str,
        cadence: str,
        data_resolution: str | WaterVolume,
        *,
        source_provenance: SourceProvenance | None = None,
    ) -> None:
        canonical, _ = _volume_unit(unit)
        current = _quantity(data_resolution, "volume") if isinstance(data_resolution, str) else data_resolution
        if not isinstance(current, WaterVolume):
            raise ValueError("IntervalVolume data_resolution must be a WaterVolume")
        if current.value <= 0:
            raise ValueError("IntervalVolume data_resolution must be positive")
        validated = _validated_frame(data, cadence)
        provenance = source_provenance or SourceProvenance("interval_volume", canonical, cadence, current)
        object.__setattr__(self, "_data", validated)
        object.__setattr__(self, "unit", canonical)
        object.__setattr__(self, "cadence", cadence)
        object.__setattr__(self, "data_resolution", current)
        object.__setattr__(self, "source_provenance", provenance)

    @property
    def data(self) -> pl.DataFrame:
        """Return a detached Polars frame so the typed value cannot be mutated through it."""
        return self._data.clone()

    def aggregate_to(self, axis: TimeAxis, *, unit: str | None = None) -> IntervalVolume:
        """Convert and sum exact source intervals into an explicitly supplied target axis."""
        target_unit = self.unit if unit is None else unit
        return _aggregate(self, axis, target_unit)

    def to_unit(self, unit: str) -> IntervalVolume:
        """Explicitly convert volume values without changing their interval cadence."""
        from .water_system import TimeAxis

        timestamps: list[datetime] = self._data["time"].to_list()
        return self.aggregate_to(TimeAxis(timestamps[0], periods=self._data.height, frequency=self.cadence), unit=unit)


@dataclass(frozen=True, init=False)
class IntervalMeanRate:
    """A strict regular Polars series whose values are interval-mean volumetric rates."""

    _data: pl.DataFrame
    unit: str
    cadence: str
    data_resolution: VolumetricRate
    source_provenance: SourceProvenance

    def __init__(self, data: pl.DataFrame, unit: str, cadence: str, data_resolution: str | VolumetricRate) -> None:
        canonical, _, _ = _rate_unit(unit)
        current = _quantity(data_resolution, "rate") if isinstance(data_resolution, str) else data_resolution
        if not isinstance(current, VolumetricRate):
            raise ValueError("IntervalMeanRate data_resolution must be a VolumetricRate")
        if current.value <= 0:
            raise ValueError("IntervalMeanRate data_resolution must be positive")
        object.__setattr__(self, "_data", _validated_frame(data, cadence))
        object.__setattr__(self, "unit", canonical)
        object.__setattr__(self, "cadence", cadence)
        object.__setattr__(self, "data_resolution", current)
        object.__setattr__(
            self, "source_provenance", SourceProvenance("interval_mean_rate", canonical, cadence, current)
        )

    @property
    def data(self) -> pl.DataFrame:
        """Return a detached Polars frame so the typed value cannot be mutated through it."""
        return self._data.clone()

    def aggregate_to(self, axis: TimeAxis, *, unit: str | None = None) -> IntervalVolume:
        """Integrate rates and sum exact source intervals into an explicitly supplied target axis."""
        source_volume = self.unit.split("/", 1)[0]
        return _aggregate(self, axis, source_volume if unit is None else unit)


def _aggregate(source: IntervalVolume | IntervalMeanRate, axis: TimeAxis, target_unit: str) -> IntervalVolume:
    canonical_target, target_factor = _volume_unit(target_unit)
    source_seconds = _frequency_seconds(source.cadence)
    target_seconds = int(axis.timestep.total_seconds())
    if source_seconds > target_seconds:
        raise ValueError("aggregate_to cannot disaggregate coarser input into finer target intervals")
    if target_seconds % source_seconds != 0:
        raise ValueError("source cadence does not exactly partition the target frequency")
    per_target = target_seconds // source_seconds
    required = axis.periods * per_target
    timestamps: list[datetime] = source._data["time"].to_list()
    start = timestamps[0].replace(tzinfo=UTC)
    if start != axis.start or source._data.height != required:
        raise ValueError("input intervals do not exactly cover the target TimeAxis")
    expected_end = axis.start + axis.timestep * axis.periods
    source_end = start + timedelta(seconds=source_seconds * source._data.height)
    if source_end != expected_end:
        raise ValueError("input intervals do not exactly cover the target TimeAxis")

    values = source._data["value"].to_list()
    if isinstance(source, IntervalMeanRate):
        _, source_volume_factor, source_period = _rate_unit(source.unit)
        factor = (source_volume_factor / source_period) * Decimal(source_seconds) / target_factor
        prepared = [_converted(value, factor) for value in values]
        resolution_magnitude = _converted(source.data_resolution.to(source.unit).value, factor)
    else:
        _, source_volume_factor = _volume_unit(source.unit)
        factor = source_volume_factor / target_factor
        prepared = [_converted(value, factor) for value in values]
        resolution_magnitude = _converted(source.data_resolution.to(source.unit).value, factor)
    totals = [
        float(sum((Decimal(str(value)) for value in prepared[index : index + per_target]), start=Decimal(0)))
        for index in range(0, len(prepared), per_target)
    ]
    output_times = [(axis.start + axis.timestep * index).replace(tzinfo=None) for index in range(axis.periods)]
    frame = pl.DataFrame(
        {"time": output_times, "value": totals},
        schema={"time": pl.Datetime("us"), "value": pl.Float64},
    )
    return IntervalVolume(
        frame,
        canonical_target,
        axis.frequency,
        WaterVolume(resolution_magnitude, canonical_target),
        source_provenance=source.source_provenance,
    )
