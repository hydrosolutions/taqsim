"""WaterSystem.build : WaterSystem → BuiltWaterSystem; BuiltWaterSystem.run : BuiltWaterSystem × RunId × RuleSubstitutions → WaterSystemRun."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from os import PathLike
from types import MappingProxyType
from typing import Any, NewType, overload
from uuid import UUID

import incidence

from .inputs import (
    CanalSeepageCoefficient,
    IntervalVolume,
    Length,
    Parameter,
    SurfaceArea,
    VolumetricRate,
    WaterDepth,
    WaterVolume,
    _frequency_seconds,
)
from .vocabulary import RuleContext, RulePlan, WaterRule, subtract


class Presence(StrEnum):
    """The three possible states of a time-indexed water reading."""

    PRESENT = "present"
    ABSENT = "absent"
    NOT_MODELLED = "not_modelled"


class ConservationQuantum(StrEnum):
    """One accepted arithmetic quantum for water amounts expressed in m3."""

    CUBIC_METRE = "1 m3"
    LITRE = "1 L"
    MILLILITRE = "1 mL"
    CUBIC_MILLIMETRE = "1 mm3"

    @property
    def quantum_m3(self) -> float:
        """Return one quantum unit expressed in cubic metres."""
        return {
            ConservationQuantum.CUBIC_METRE: 1.0,
            ConservationQuantum.LITRE: 1e-3,
            ConservationQuantum.MILLILITRE: 1e-6,
            ConservationQuantum.CUBIC_MILLIMETRE: 1e-9,
        }[self]


@dataclass(frozen=True, init=False)
class TimeAxis:
    """A finite sequence of equally spaced Gregorian model intervals."""

    start: datetime
    periods: int
    frequency: str
    timestep: timedelta

    def __init__(self, start: date | datetime | str, periods: int, frequency: str = "1d") -> None:
        if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
            raise ValueError("time axis periods must be a positive integer")
        seconds = _frequency_seconds(frequency)
        parsed_start = _parse_start(start)
        if parsed_start.microsecond != 0:
            raise ValueError("time axis start must align to a whole second")
        normalized = _normalized_frequency(seconds)
        object.__setattr__(self, "start", parsed_start)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "frequency", normalized)
        object.__setattr__(self, "timestep", timedelta(seconds=seconds))

    @property
    def steps(self) -> int:
        """Return the internal engine horizon length."""
        return self.periods

    @property
    def end(self) -> datetime:
        """Return the datetime of the final modelled interval start."""
        return self.start + self.timestep * (self.periods - 1)

    @property
    def exclusive_end(self) -> datetime:
        """Return the exclusive end boundary of the final modelled interval."""
        return self.start + self.timestep * self.periods

    def datetime_at(self, timestep: int) -> datetime:
        """Convert an engine timestep to its declared datetime."""
        return self.start + self.timestep * timestep

    def timestep_at(self, when: date | datetime | str) -> int:
        """Convert an aligned datetime to its engine timestep."""
        moment = _parse_start(when)
        elapsed = (moment - self.start).total_seconds()
        step_seconds = self.timestep.total_seconds()
        index, remainder = divmod(elapsed, step_seconds)
        if remainder != 0:
            raise ValueError(f"{moment.isoformat()} is not aligned to the declared timestep")
        return int(index)


# The engine needs bytes, while callers should be free to use the normal UUID value type.
RunId = UUID | bytes | bytearray | str
type RuleScalar = float | WaterVolume | VolumetricRate | WaterDepth | Length | SurfaceArea | CanalSeepageCoefficient
RuleSubstitutions = Mapping[str, RuleScalar]
WaterQuantumCount = NewType("WaterQuantumCount", int)


@dataclass(frozen=True)
class RuleParameter:
    """One addressed rule scalar whose physical descriptor remains attached."""

    compartment: str
    name: str
    value: RuleScalar
    bounds: tuple[RuleScalar, RuleScalar] | None

    @property
    def path(self) -> str:
        """Return the stable user-facing address of this rule-local parameter."""
        return f"{self.compartment}.{self.name}"

    @property
    def numeric_value(self) -> float:
        """Return the canonical engine-unit magnitude for internal optimization."""
        return _rule_scalar_magnitude(self.value)

    @property
    def numeric_bounds(self) -> tuple[float, float] | None:
        """Return canonical engine-unit bounds for internal optimization."""
        if self.bounds is None:
            return None
        return (_rule_scalar_magnitude(self.bounds[0]), _rule_scalar_magnitude(self.bounds[1]))

    def from_numeric(self, value: float) -> RuleScalar:
        """Attach this parameter's physical descriptor to an internal numeric candidate."""
        if isinstance(self.value, WaterVolume):
            return WaterVolume(value, self.value.unit)
        if isinstance(self.value, VolumetricRate):
            return VolumetricRate(value, self.value.unit)
        if isinstance(self.value, WaterDepth):
            return WaterDepth(value, self.value.unit)
        if isinstance(self.value, Length):
            return Length(value, self.value.unit)
        if isinstance(self.value, SurfaceArea):
            return SurfaceArea(value, self.value.unit)
        if isinstance(self.value, CanalSeepageCoefficient):
            return CanalSeepageCoefficient(value, self.value.unit)
        return value

    def engine_value(self, value: RuleScalar) -> float:
        """Convert one matching typed public substitution to the canonical engine magnitude."""
        if isinstance(self.value, WaterVolume):
            if not isinstance(value, WaterVolume):
                raise TypeError(f"rule parameter {self.path!r} substitution must be a WaterVolume")
            return _fixed_rule_quantity(value.to(self.value.unit))
        if isinstance(self.value, VolumetricRate):
            if not isinstance(value, VolumetricRate):
                raise TypeError(f"rule parameter {self.path!r} substitution must be a VolumetricRate")
            return _fixed_rule_quantity(value.to(self.value.unit))
        if isinstance(self.value, WaterDepth):
            if not isinstance(value, WaterDepth):
                raise TypeError(f"rule parameter {self.path!r} substitution must be a WaterDepth")
            return _fixed_rule_quantity(value.to(self.value.unit))
        if isinstance(self.value, Length):
            if not isinstance(value, Length):
                raise TypeError(f"rule parameter {self.path!r} substitution must be a Length")
            return _fixed_rule_quantity(value.to(self.value.unit))
        if isinstance(self.value, SurfaceArea):
            if not isinstance(value, SurfaceArea):
                raise TypeError(f"rule parameter {self.path!r} substitution must be a SurfaceArea")
            return _fixed_rule_quantity(value.to(self.value.unit))
        if isinstance(self.value, CanalSeepageCoefficient):
            if not isinstance(value, CanalSeepageCoefficient):
                raise TypeError(f"rule parameter {self.path!r} substitution must be a CanalSeepageCoefficient")
            return _fixed_rule_quantity(value)
        if isinstance(
            value,
            (WaterVolume, VolumetricRate, WaterDepth, Length, SurfaceArea, CanalSeepageCoefficient),
        ):
            raise TypeError(f"dimensionless rule parameter {self.path!r} substitution must be a number")
        return float(value)


@dataclass(frozen=True, init=False)
class Reach:
    """A named water connection and its closed-world typed capacity declaration."""

    name: str
    source: str
    destination: str
    capacity: WaterVolume | None
    overflow_destination: str | None
    initial_water: WaterVolume
    rule: WaterRule | None

    def __init__(
        self,
        name: str,
        source: str | None = None,
        destination: str | None = None,
        *,
        upstream: str | None = None,
        downstream: str | None = None,
        capacity: WaterVolume | None = None,
        overflow_destination: str | None = None,
        initial_water: WaterVolume | None = None,
        rule: WaterRule | None = None,
    ) -> None:
        if source is not None and upstream is not None:
            raise ValueError(f"reach {name!r} declares both source and upstream")
        if destination is not None and downstream is not None:
            raise ValueError(f"reach {name!r} declares both destination and downstream")
        source = source if source is not None else upstream
        destination = destination if destination is not None else downstream
        if not name:
            raise ValueError("reach name must not be empty")
        if not source or not destination:
            raise ValueError(f"reach {name!r} must name both source and destination")
        if capacity is not None and not isinstance(capacity, WaterVolume):
            raise TypeError(f"reach {name!r} capacity must be a WaterVolume")
        if capacity is not None and isinstance(capacity.value, Parameter):
            raise TypeError(f"reach {name!r} capacity must be a fixed WaterVolume")
        if initial_water is not None and not isinstance(initial_water, WaterVolume):
            raise TypeError(f"reach {name!r} initial_water must be a WaterVolume")
        if initial_water is not None and isinstance(initial_water.value, Parameter):
            raise TypeError(f"reach {name!r} initial_water must be a fixed WaterVolume")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "overflow_destination", overflow_destination)
        object.__setattr__(self, "initial_water", initial_water or WaterVolume(0.0, "m3"))
        object.__setattr__(self, "rule", rule)


@dataclass(frozen=True)
class Source:
    """A named boundary supply whose frame remains bound to interval-volume meaning."""

    name: str
    data: IntervalVolume

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source name must not be empty")
        if not isinstance(self.data, IntervalVolume):
            raise TypeError(f"source {self.name!r} data must be an IntervalVolume")

    @property
    def flow(self) -> tuple[float, ...]:
        """Return model-ready current values for exact count conversion."""
        return tuple(self.data.data["value"].to_list())


@dataclass(frozen=True)
class Sink:
    """A named boundary destination for water leaving the water system."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sink name must not be empty")


@dataclass(frozen=True)
class WaterValue:
    """One water value with an explicit presence state."""

    when: datetime
    value: float | None
    presence: Presence


@dataclass(frozen=True)
class WaterValues(Sequence[float | None]):
    """Numerical readings that cannot be detached from their presence states."""

    _items: tuple[float | None, ...]
    presence: tuple[Presence, ...]

    def __post_init__(self) -> None:
        if len(self._items) != len(self.presence):
            raise ValueError("values and presence must have equal lengths")

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> float | None: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float | None, ...]: ...

    def __getitem__(self, index: int | slice) -> float | None | tuple[float | None, ...]:
        return self._items[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WaterValues):
            return self._items == other._items and self.presence == other.presence
        if isinstance(other, Sequence):
            return self._items == tuple(other)
        return False


@dataclass(frozen=True, init=False)
class WaterSeries(Sequence[WaterValue]):
    """A date-indexed projection whose every value carries presence."""

    dates: tuple[datetime, ...]
    values: WaterValues

    def __init__(
        self,
        dates: tuple[datetime, ...],
        values: tuple[float | None, ...],
        presence: tuple[Presence, ...],
    ) -> None:
        if len(dates) != len(values) or len(values) != len(presence):
            raise ValueError("dates, values, and presence must have equal lengths")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "values", WaterValues(values, presence))

    @property
    def presence(self) -> tuple[Presence, ...]:
        """Return the explicit state corresponding to each value."""
        return self.values.presence

    @property
    def timesteps(self) -> tuple[datetime, ...]:
        """Return the real dates corresponding to the engine timesteps."""
        return self.dates

    @property
    def index(self) -> tuple[datetime, ...]:
        """Alias for the real-date index."""
        return self.dates

    def __len__(self) -> int:
        return len(self.dates)

    def __iter__(self) -> Iterator[WaterValue]:
        for when, value, state in zip(self.dates, self.values, self.presence, strict=True):
            yield WaterValue(when, value, state)

    @overload
    def __getitem__(self, index: int) -> WaterValue: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[WaterValue, ...]: ...

    def __getitem__(self, index: int | slice) -> WaterValue | tuple[WaterValue, ...]:
        if isinstance(index, slice):
            return tuple(
                WaterValue(when, value, state)
                for when, value, state in zip(self.dates[index], self.values[index], self.presence[index], strict=True)
            )
        return WaterValue(self.dates[index], self.values[index], self.presence[index])

    def at(self, when: date | datetime | str) -> WaterValue:
        """Read one exactly indexed value."""
        moment = _parse_start(when)
        try:
            index = self.dates.index(moment)
        except ValueError as error:
            raise KeyError(moment) from error
        return WaterValue(self.dates[index], self.values[index], self.presence[index])


@dataclass(init=False)
class WaterSystem:
    """A typed water-system declaration compiled exactly once per build."""

    name: str
    time: TimeAxis
    quantum: ConservationQuantum
    _reaches: list[Reach]
    _sources: list[Source]
    _sinks: list[Sink]

    def __init__(
        self,
        *,
        time: TimeAxis,
        quantum: str | ConservationQuantum,
        name: str = "water-system",
    ) -> None:
        if not isinstance(time, TimeAxis):
            raise TypeError("WaterSystem time must be a TimeAxis")
        try:
            declared_quantum = ConservationQuantum(quantum)
        except ValueError as error:
            accepted = ", ".join(repr(item.value) for item in ConservationQuantum)
            raise ValueError(f"unknown conservation quantum {quantum!r}; accepted values are {accepted}") from error
        if not name:
            raise ValueError("water system name must not be empty")
        self.name = name
        self.time = time
        self.quantum = declared_quantum
        self._reaches = []
        self._sources = []
        self._sinks = []

    @property
    def reaches(self) -> tuple[Reach, ...]:
        """Return declared reaches without exposing mutable system state."""
        return tuple(self._reaches)

    def source(self, name: str, data: IntervalVolume) -> Source:
        """Attach one already prepared interval-volume input without transforming it."""
        declared = Source(name, data)
        if any(item.name == name for item in self._sources):
            raise ValueError(f"source {name!r} is already declared")
        self._sources.append(declared)
        return declared

    def sink(self, name: str) -> Sink:
        """Declare a named boundary destination."""
        declared = Sink(name)
        if any(item.name == name for item in self._sinks):
            raise ValueError(f"sink {name!r} is already declared")
        self._sinks.append(declared)
        return declared

    def add_reach(
        self,
        reach: Reach | str,
        source: str | None = None,
        destination: str | None = None,
        **options: Any,
    ) -> Reach:
        """Declare one reach from a Reach or its narrow constructor fields."""
        if isinstance(reach, Reach):
            if source is not None or destination is not None or options:
                raise TypeError("a Reach instance cannot be combined with reach constructor arguments")
            declared = reach
        else:
            declared = Reach(reach, source, destination, **options)
        if any(existing.name == declared.name for existing in self._reaches):
            raise ValueError(f"reach {declared.name!r} is already declared")
        self._reaches.append(declared)
        return declared

    def reach(self, name: str, source: str | None = None, destination: str | None = None, **options: Any) -> Reach:
        """Fluent alias for add_reach."""
        return self.add_reach(name, source, destination, **options)

    def build(self) -> BuiltWaterSystem:
        """Validate typed declarations, then compile one incidence model."""
        _require_closed_capacity(self._reaches)
        _require_source_horizons(self._sources, self.time)
        source_counts, initial_counts = _external_water_counts(self._reaches, self._sources, self.time, self.quantum)
        _require_countable_initial_total(source_counts, initial_counts, self.quantum)
        document, parameters = _model_document(
            tuple(self._reaches),
            tuple(self._sources),
            tuple(self._sinks),
            self.time,
            self.quantum,
            source_counts,
            initial_counts,
        )
        return BuiltWaterSystem(
            document,
            incidence.compile_model(document),
            self.time,
            self.quantum,
            frozenset(reach.name for reach in self._reaches),
            initial_counts,
            parameters,
        )


@dataclass(frozen=True, init=False)
class BuiltWaterSystem:
    """One validated model document held with its compiled incidence model."""

    _document: dict[str, Any]
    _compiled: incidence.CompiledModel
    time: TimeAxis
    quantum: ConservationQuantum
    reaches: frozenset[str]
    _initial_counts: Mapping[str, WaterQuantumCount]
    parameters: tuple[RuleParameter, ...]

    def __init__(
        self,
        document: dict[str, Any],
        compiled: incidence.CompiledModel,
        time: TimeAxis,
        quantum: ConservationQuantum,
        reaches: frozenset[str],
        initial_counts: Mapping[str, WaterQuantumCount],
        parameters: tuple[RuleParameter, ...],
    ):
        paths = [parameter.path for parameter in parameters]
        if len(paths) != len(set(paths)):
            raise ValueError("rule parameter paths must be unique")
        object.__setattr__(self, "_document", deepcopy(document))
        object.__setattr__(self, "_compiled", compiled)
        if compiled.quantum("water") != quantum.quantum_m3:
            raise ValueError("compiled water quantum differs from the declared WaterSystem quantum")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "quantum", quantum)
        object.__setattr__(self, "reaches", reaches)
        object.__setattr__(self, "_initial_counts", MappingProxyType(dict(initial_counts)))
        object.__setattr__(self, "parameters", parameters)

    @property
    def document(self) -> Mapping[str, Any]:
        """Return a detached view of the emitted incidence model document."""
        return deepcopy(self._document)

    @property
    def model_document(self) -> Mapping[str, Any]:
        """Alias naming the emitted model document explicitly."""
        return self.document

    @property
    def model_digest(self) -> str:
        """Return the content address assigned by incidence."""
        return self._compiled.model_digest

    @property
    def parameter_bounds(self) -> Mapping[str, tuple[RuleScalar, RuleScalar]]:
        """Return optimizer bounds without adding them to model identity."""
        return {parameter.path: parameter.bounds for parameter in self.parameters if parameter.bounds is not None}

    def run(self, run_id: RunId, parameters: RuleSubstitutions | None = None) -> WaterSystemRun:
        """Execute one run, optionally substituting addressed rule parameters."""
        substitutions = self._substitutions(parameters or {})
        completed = self._compiled.run(_engine_run_id(run_id), substitutions=substitutions)
        return WaterSystemRun(completed, self.time, self.quantum, self.reaches, self._initial_counts)

    def sweep(self, parameter: str | RuleParameter, values: Iterable[RuleScalar]) -> tuple[WaterSystemRun, ...]:
        """Run a one-dimensional parameter sweep while holding the compiled model."""
        path = parameter.path if isinstance(parameter, RuleParameter) else parameter
        return tuple(self.run(index.to_bytes(16, byteorder="big"), {path: value}) for index, value in enumerate(values))

    def _substitutions(self, values: RuleSubstitutions) -> list[dict[str, object]]:
        declared = {parameter.path: parameter for parameter in self.parameters}
        unknown = sorted(set(values) - declared.keys())
        if unknown:
            raise ValueError(f"unknown rule parameter(s): {', '.join(unknown)}")
        substitutions: list[dict[str, object]] = []
        for path, value in values.items():
            parameter = declared[path]
            numeric = parameter.engine_value(value)
            if not math.isfinite(numeric):
                raise ValueError(f"rule parameter {path!r} substitution must be finite")
            substitutions.append(
                {
                    "compartment": parameter.compartment,
                    "substance": "water",
                    "parameter": parameter.name,
                    "value": numeric,
                }
            )
        return substitutions


@dataclass(frozen=True, init=False)
class WaterSystemRun:
    """An immutable live or cached run interpreted through WaterSystem dates and water names."""

    _completed: incidence.CompletedRun | None
    _cached_flows: Mapping[str, WaterSeries]
    _cached_retained: Mapping[str, WaterSeries]
    _cached_log: tuple[bytes, str] | None
    _cached_model_digest: str | None
    _initial_counts: Mapping[str, WaterQuantumCount]
    time: TimeAxis
    quantum: ConservationQuantum
    reaches: frozenset[str]

    def __init__(
        self,
        completed: incidence.CompletedRun,
        time: TimeAxis,
        quantum: ConservationQuantum,
        reaches: frozenset[str],
        initial_counts: Mapping[str, WaterQuantumCount],
    ):
        if completed.quantum("water") != quantum.quantum_m3:
            raise ValueError("completed-run water quantum differs from the model quantum")
        object.__setattr__(self, "_completed", completed)
        object.__setattr__(self, "_cached_flows", MappingProxyType({}))
        object.__setattr__(self, "_cached_retained", MappingProxyType({}))
        object.__setattr__(self, "_cached_log", None)
        object.__setattr__(self, "_cached_model_digest", None)
        object.__setattr__(self, "_initial_counts", MappingProxyType(dict(initial_counts)))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "quantum", quantum)
        object.__setattr__(self, "reaches", reaches)

    @classmethod
    def _from_cache(
        cls,
        *,
        model_digest: str,
        authoritative_log: tuple[bytes, str],
        time: TimeAxis,
        quantum: ConservationQuantum,
        flows: Mapping[str, WaterSeries],
        retained: Mapping[str, WaterSeries],
    ) -> WaterSystemRun:
        run = object.__new__(cls)
        object.__setattr__(run, "_completed", None)
        object.__setattr__(run, "_cached_flows", MappingProxyType(dict(flows)))
        object.__setattr__(run, "_cached_retained", MappingProxyType(dict(retained)))
        object.__setattr__(run, "_cached_log", authoritative_log)
        object.__setattr__(run, "_cached_model_digest", model_digest)
        object.__setattr__(run, "_initial_counts", MappingProxyType({}))
        object.__setattr__(run, "time", time)
        object.__setattr__(run, "quantum", quantum)
        object.__setattr__(run, "reaches", frozenset(flows))
        return run

    @classmethod
    def load(cls, path: str | PathLike[str]) -> WaterSystemRun:
        """Load a compatible saved-run cache without compiling or running a model."""
        from .persistence import load_run

        return load_run(path)

    @property
    def model_digest(self) -> str:
        if self._completed is not None:
            return self._completed.model_digest
        if self._cached_model_digest is None:
            raise RuntimeError("cached run is missing its model digest")
        return self._cached_model_digest

    @property
    def authoritative_log_digest(self) -> str:
        """Return the verified digest of the authoritative incidence log."""
        return self.authoritative_log()[1]

    @property
    def log_digest(self) -> str:
        """Short alias for authoritative_log_digest."""
        return self.authoritative_log_digest

    def authoritative_log(self) -> tuple[bytes, str]:
        """Return incidence's canonical log bytes and verified digest."""
        if self._completed is not None:
            return self._completed.authoritative_log()
        if self._cached_log is None:
            raise RuntimeError("cached run is missing its authoritative log")
        return self._cached_log

    def verify_replay(self, other: WaterSystemRun) -> None:
        """Verify this live run by replaying its exact log against the other run's model."""
        if self._completed is None or other._completed is None:
            raise ValueError("replay verification requires two live runs")
        self._completed.replay_against(other._completed)

    def save(self, path: str | PathLike[str]) -> None:
        """Save this completed run as a version-stamped cache."""
        from .persistence import save_run

        save_run(self, path)

    def flow(
        self,
        reach: str | Reach,
        *,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None,
    ) -> WaterSeries:
        """Read outgoing reach flow over an inclusive real-date interval."""
        name = reach.name if isinstance(reach, Reach) else reach
        if name not in self.reaches:
            raise KeyError(f"unknown reach {name!r}")
        first = 0 if start is None else self.time.timestep_at(start)
        last = self.time.steps - 1 if end is None else self.time.timestep_at(end)
        if last < first:
            raise ValueError("flow end precedes flow start")
        modelled_first = max(first, 0)
        modelled_last = min(last, self.time.steps - 1)
        engine_values: dict[int, float | None] = {}
        engine_states: dict[int, Presence] = {}
        if modelled_first <= modelled_last:
            modelled = self._modelled_flow(name, modelled_first, modelled_last)
            for offset, step in enumerate(range(modelled_first, modelled_last + 1)):
                engine_values[step] = modelled.values[offset]
                engine_states[step] = modelled.presence[offset]

        dates = tuple(self.time.datetime_at(step) for step in range(first, last + 1))
        values: list[float | None] = []
        states: list[Presence] = []
        for step in range(first, last + 1):
            if step in engine_values:
                values.append(engine_values[step])
                states.append(engine_states[step])
            else:
                values.append(None)
                states.append(Presence.NOT_MODELLED)
        return WaterSeries(dates, tuple(values), tuple(states))

    def _modelled_flow(self, name: str, first: int, last: int) -> WaterSeries:
        dates = tuple(self.time.datetime_at(step) for step in range(first, last + 1))
        if self._completed is not None:
            engine_series: incidence.PresenceSeries = self._completed.transfer_series(
                name,
                "water",
                direction="outgoing",
                first=first,
                last=last,
            )
            return WaterSeries(
                dates,
                tuple(engine_series.values),
                tuple(Presence(state) for state in engine_series.presence),
            )
        cached = self._cached_flows[name]
        return WaterSeries(dates, cached.values[first : last + 1], cached.presence[first : last + 1])

    def retained(
        self,
        reach: str | Reach,
        *,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None,
    ) -> WaterSeries:
        """Read represented water retained in one reach after each timestep."""
        name = reach.name if isinstance(reach, Reach) else reach
        if name not in self.reaches:
            raise KeyError(f"unknown reach {name!r}")
        first = 0 if start is None else self.time.timestep_at(start)
        last = self.time.steps - 1 if end is None else self.time.timestep_at(end)
        if first < 0 or last >= self.time.steps:
            raise ValueError("retained-stock reads must lie inside the modelled horizon")
        if last < first:
            raise ValueError("retained-stock end precedes start")
        if self._completed is None:
            cached = self._cached_retained[name]
            dates = tuple(self.time.datetime_at(step) for step in range(first, last + 1))
            return WaterSeries(dates, cached.values[first : last + 1], cached.presence[first : last + 1])
        incoming: incidence.PresenceCountSeries = self._completed.transfer_count_series(
            name, "water", direction="incoming", first=0, last=self.time.steps - 1
        )
        outgoing: incidence.PresenceCountSeries = self._completed.transfer_count_series(
            name, "water", direction="outgoing", first=0, last=self.time.steps - 1
        )
        stock_count = int(self._initial_counts[name])
        values: list[float] = []
        for step, (incoming_count, outgoing_count) in enumerate(zip(incoming.values, outgoing.values, strict=True)):
            if incoming_count is None or outgoing_count is None:
                raise RuntimeError(f"incidence omitted retained-stock input for reach {name!r} at timestep {step}")
            stock_count += incoming_count
            stock_count -= outgoing_count
            if stock_count < 0:
                raise RuntimeError(f"incidence returned negative retained count for reach {name!r}")
            values.append(_amount_from_count(stock_count, self.quantum))
        dates = tuple(self.time.datetime_at(step) for step in range(first, last + 1))
        selected = tuple(values[first : last + 1])
        return WaterSeries(dates, selected, (Presence.PRESENT,) * len(selected))

    def reach_flow(self, reach: str | Reach, **interval: Any) -> WaterSeries:
        """Explicit alias for flow."""
        return self.flow(reach, **interval)

    def arrivals(
        self,
        endpoint: str,
        *,
        start: date | datetime | str | None = None,
        end: date | datetime | str | None = None,
    ) -> WaterSeries:
        """Read all water arriving at one named endpoint."""
        first = 0 if start is None else self.time.timestep_at(start)
        last = self.time.steps - 1 if end is None else self.time.timestep_at(end)
        if first < 0 or last >= self.time.steps:
            raise ValueError("arrival reads must lie inside the modelled horizon")
        if last < first:
            raise ValueError("arrival end precedes arrival start")
        if self._completed is None:
            raise ValueError("arrival reads are unavailable from saved-run caches")
        series: incidence.PresenceSeries = self._completed.transfer_series(
            endpoint, "water", direction="incoming", first=first, last=last
        )
        return WaterSeries(
            tuple(self.time.datetime_at(step) for step in range(first, last + 1)),
            tuple(series.values),
            tuple(Presence(state) for state in series.presence),
        )


def _normalized_frequency(seconds: int) -> str:
    for suffix, divisor in (("d", 86400), ("h", 3600), ("min", 60), ("s", 1)):
        if seconds % divisor == 0:
            return f"{seconds // divisor}{suffix}"
    raise AssertionError("whole seconds always normalize")


def _parse_start(value: date | datetime | str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"invalid start date {value!r}") from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.combine(value, datetime.min.time())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _require_closed_capacity(reaches: Sequence[Reach]) -> None:
    for reach in reaches:
        if reach.capacity is not None and reach.overflow_destination is None:
            raise ValueError(f"capacity-limited reach {reach.name!r} must declare an overflow destination")


def _require_source_horizons(sources: Sequence[Source], time: TimeAxis) -> None:
    """require_alignment : Sources × TimeAxis → None or refusal."""
    for source in sources:
        data = source.data
        timestamps: list[datetime] = data.data["time"].to_list()
        aligned = (
            data.unit == "m3"
            and _frequency_seconds(data.cadence) == int(time.timestep.total_seconds())
            and len(timestamps) == time.periods
            and all(
                timestamp.replace(tzinfo=UTC) == time.datetime_at(index) for index, timestamp in enumerate(timestamps)
            )
        )
        if not aligned:
            raise ValueError(
                f"source {source.name!r} is not aligned in m3 to the model TimeAxis; "
                "call aggregate_to(system.time, unit='m3') explicitly"
            )


def _fixed_volume_m3(volume: WaterVolume) -> float:
    value = volume.m3
    if isinstance(value, Parameter):
        raise TypeError("expected a fixed WaterVolume")
    return value


def _external_water_counts(
    reaches: Sequence[Reach],
    sources: Sequence[Source],
    time: TimeAxis,
    quantum: ConservationQuantum,
) -> tuple[dict[str, tuple[WaterQuantumCount, ...]], dict[str, WaterQuantumCount]]:
    """quantize_external : Sources × InitialStocks × ConservationQuantum → WaterQuantumCounts."""
    source_counts: dict[str, tuple[WaterQuantumCount, ...]] = {}
    for source_position, source in enumerate(sources):
        counts = tuple(
            _external_count(
                value,
                quantum,
                f"source {source.name!r} at timestep {position} ({time.datetime_at(position).isoformat()})",
            )
            for position, value in enumerate(source.flow)
        )
        source_counts[source.name] = counts
        total_count = sum(int(count) for count in counts)
        if total_count <= 2**53:
            _require_unambiguous_projection(
                total_count,
                quantum,
                f"source {source.name!r} aggregate initial stock at position {source_position}",
            )
    initial_counts = {
        reach.name: _external_count(
            _fixed_volume_m3(reach.initial_water),
            quantum,
            f"reach {reach.name!r} initial stock at position {position}",
        )
        for position, reach in enumerate(reaches)
    }
    return source_counts, initial_counts


def _external_count(value: float, quantum: ConservationQuantum, carrier: str) -> WaterQuantumCount:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{carrier} has invalid water value {value!r}; expected a finite non-negative multiple "
            f"of quantum {quantum.quantum_m3!r}"
        )
    quotient = round(value / quantum.quantum_m3)
    if quotient > 2**53:
        raise ValueError(
            f"{carrier} has water value {value!r}, whose quantum count {quotient} exceeds the "
            f"exactly countable ceiling 2^53 at quantum {quantum.quantum_m3!r}"
        )
    projected = _amount_from_count(quotient, quantum)
    collision_below = quotient > 0 and _float_bits(_amount_from_count(quotient - 1, quantum)) == _float_bits(projected)
    collision_above = quotient < 2**53 and _float_bits(_amount_from_count(quotient + 1, quantum)) == _float_bits(
        projected
    )
    if _float_bits(projected) != _float_bits(value) or collision_below or collision_above:
        raise ValueError(
            f"{carrier} has water value {value!r}, which is not an exact multiple of quantum {quantum.quantum_m3!r}"
        )
    return WaterQuantumCount(quotient)


def _amount_from_count(count: WaterQuantumCount | int, quantum: ConservationQuantum) -> float:
    """project_water : WaterQuantumCount × ConservationQuantum → PublicAmount."""
    return float(float(count) * quantum.quantum_m3)


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _require_unambiguous_projection(count: int, quantum: ConservationQuantum, carrier: str) -> None:
    projected = _amount_from_count(count, quantum)
    collision_below = count > 0 and _float_bits(_amount_from_count(count - 1, quantum)) == _float_bits(projected)
    collision_above = count < 2**53 and _float_bits(_amount_from_count(count + 1, quantum)) == _float_bits(projected)
    decoded_count = round(projected / quantum.quantum_m3)
    if decoded_count != count or collision_below or collision_above:
        raise ValueError(
            f"{carrier} has quantum count {count}, whose public value {projected!r} does not identify one count "
            f"at quantum {quantum.quantum_m3!r}"
        )


def _require_countable_initial_total(
    source_counts: Mapping[str, Sequence[WaterQuantumCount]],
    initial_counts: Mapping[str, WaterQuantumCount],
    quantum: ConservationQuantum,
) -> None:
    total_count = sum(int(count) for counts in source_counts.values() for count in counts) + sum(
        int(count) for count in initial_counts.values()
    )
    if total_count <= 2**53:
        return
    total = Decimal(total_count) * Decimal(str(quantum.quantum_m3))
    ceiling = Decimal(2**53) * Decimal(str(quantum.quantum_m3))
    raise ValueError(f"substance 'water' declared total {total} exceeds the exactly countable ceiling {ceiling}")


def _model_document(
    reaches: tuple[Reach, ...],
    sources: tuple[Source, ...],
    sinks: tuple[Sink, ...],
    time: TimeAxis,
    quantum: ConservationQuantum,
    source_counts: Mapping[str, Sequence[WaterQuantumCount]],
    initial_counts: Mapping[str, WaterQuantumCount],
) -> tuple[dict[str, Any], tuple[RuleParameter, ...]]:
    reach_names = {reach.name for reach in reaches}
    source_names = {source.name for source in sources}
    finite_names = reach_names | source_names
    outgoing_by_source = {
        source.name: [reach.name for reach in reaches if reach.source == source.name] for source in sources
    }
    for source_name, destinations in outgoing_by_source.items():
        if len(destinations) != 1:
            raise ValueError(f"source {source_name!r} must feed exactly one reach; found {len(destinations)}")

    contexts: dict[str, RuleContext] = {}
    plans: dict[str, RulePlan] = {}
    for reach in reaches:
        context = RuleContext(reach.name, time.start, time.steps, time.timestep)
        contexts[reach.name] = context
        if reach.rule is not None:
            if reach.capacity is not None:
                raise ValueError(f"reach {reach.name!r} cannot combine a capacity with a custom water rule")
            plans[reach.name] = reach.rule.compile(context, reach.destination)
        elif reach.capacity is not None:
            available = context.available
            normal = incidence.min(available, incidence.literal(_fixed_volume_m3(reach.capacity)))
            overflow = incidence.max(incidence.literal(0.0), subtract(available, normal))
            plans[reach.name] = RulePlan(
                (
                    ("out", reach.destination, normal),
                    ("overflow", reach.overflow_destination or "", overflow),
                )
            )
        else:
            plans[reach.name] = RulePlan((("out", reach.destination, context.available),))

    rule_destinations = {destination for plan in plans.values() for _, destination, _ in plan.branches}
    declared_endpoints = {reach.source for reach in reaches} | rule_destinations
    boundaries = sorted(
        {sink.name for sink in sinks} | {endpoint for endpoint in declared_endpoints if endpoint not in finite_names}
    )
    connections = [{"source": reach.source, "target": reach.name} for reach in reaches] + [
        {"source": reach.name, "target": destination}
        for reach in reaches
        for _, destination, _ in plans[reach.name].branches
    ]

    source_rules = [
        incidence.rule(
            source.name,
            "water",
            incidence.literal(0.0),
            _exogenous_series(f"{source.name}-out", f"{source.name}-flow"),
        )
        for source in sources
    ]
    reach_rules = []
    for reach in reaches:
        plan = plans[reach.name]
        disposition = {
            "rule_ir_version": "v1",
            "numerical_semantics_version": "v1",
            "partition": {
                "kind": "expression_partition",
                "branches": [{"branch": branch, "expression": expression} for branch, _, expression in plan.branches],
            },
        }
        reach_rules.append(
            incidence.rule(
                reach.name,
                "water",
                incidence.literal(0.0),
                disposition,
                contexts[reach.name].parameters,
            )
        )

    forcing_values = {
        identifier: values for context in contexts.values() for identifier, values in context.forcings.items()
    }
    tables = [table for context in contexts.values() for table in context.tables]
    document = incidence.model_document(
        finite_compartments=sorted(finite_names),
        boundary_accounts=boundaries,
        connections=_unique_connections(connections),
        substances=["water"],
        initial_stocks=[
            {
                "compartment": source.name,
                "amounts": [
                    {"substance": "water", "amount": _amount_from_count(sum(source_counts[source.name]), quantum)}
                ],
            }
            for source in sorted(sources, key=lambda item: item.name)
        ]
        + [
            {
                "compartment": reach.name,
                "amounts": [{"substance": "water", "amount": _amount_from_count(initial_counts[reach.name], quantum)}],
            }
            for reach in sorted(reaches, key=lambda item: item.name)
        ],
        calendar={
            "origin_unix_seconds": int(time.start.timestamp()),
            "timestep_seconds": int(time.timestep.total_seconds()),
        },
        horizon={"first": 0, "last": time.steps - 1},
        projections={
            "specifications": [_available_projection(reach) for reach in reaches],
            "initial_states": [
                {
                    "projection": f"{reach.name}-available",
                    "values": [
                        {
                            "kind": "extensive",
                            "value": _amount_from_count(initial_counts[reach.name], quantum),
                        }
                    ],
                }
                for reach in reaches
            ],
        },
        forcings=[
            {
                "id": f"{source.name}-flow",
                "horizon": {"first": 0, "last": time.steps - 1},
                "values": [_amount_from_count(count, quantum) for count in source_counts[source.name]],
            }
            for source in sources
        ]
        + [
            {
                "id": identifier,
                "horizon": {"first": 0, "last": time.steps - 1},
                "values": values,
            }
            for identifier, values in sorted(forcing_values.items())
        ],
        interpolation_tables=tables,
        rules=source_rules + reach_rules,
        transfer_bindings=[
            {
                "compartment": source.name,
                "substance": "water",
                "branch": f"{source.name}-out",
                "destination": outgoing_by_source[source.name][0],
            }
            for source in sources
        ]
        + [
            {
                "compartment": reach.name,
                "substance": "water",
                "branch": branch,
                "destination": destination,
            }
            for reach in reaches
            for branch, destination, _ in plans[reach.name].branches
        ],
        input_bindings=[],
        units=[
            {
                "substance": "water",
                "unit": "m3",
                "quantum": quantum.quantum_m3,
            }
        ],
    )
    parameters = tuple(
        _rule_parameter(reach.name, contexts[reach.name].parameter_descriptors[name])
        for reach in reaches
        for name in contexts[reach.name].parameters
    )
    return document, parameters


def _fixed_rule_quantity(
    quantity: WaterVolume | VolumetricRate | WaterDepth | Length | SurfaceArea | CanalSeepageCoefficient,
) -> float:
    if isinstance(quantity.value, Parameter):
        raise TypeError("rule substitution quantity must contain a fixed value")
    return quantity.value


def _rule_scalar_magnitude(value: RuleScalar) -> float:
    if isinstance(value, (WaterVolume, VolumetricRate, WaterDepth, Length, SurfaceArea, CanalSeepageCoefficient)):
        return _fixed_rule_quantity(value)
    return value


def _quantity_scalar(parameter: Parameter, value: float) -> RuleScalar:
    if parameter.physical_kind == "volume":
        assert parameter.unit is not None
        return WaterVolume(value, parameter.unit)
    if parameter.physical_kind == "rate":
        assert parameter.unit is not None
        return VolumetricRate(value, parameter.unit)
    if parameter.physical_kind == "depth":
        assert parameter.unit is not None
        return WaterDepth(value, parameter.unit)
    if parameter.physical_kind == "length":
        assert parameter.unit is not None
        return Length(value, parameter.unit)
    if parameter.physical_kind == "area":
        assert parameter.unit is not None
        return SurfaceArea(value, parameter.unit)
    if parameter.physical_kind == "seepage_coefficient":
        assert parameter.unit is not None
        return CanalSeepageCoefficient(value, parameter.unit)
    return value


def _rule_parameter(compartment: str, parameter: Parameter) -> RuleParameter:
    bounds = None
    if parameter.bounds is not None:
        bounds = (
            _quantity_scalar(parameter, parameter.bounds[0]),
            _quantity_scalar(parameter, parameter.bounds[1]),
        )
    return RuleParameter(compartment, parameter.name, _quantity_scalar(parameter, parameter.value), bounds)


def _available_projection(reach: Reach) -> dict[str, Any]:
    previous = incidence.input(f"{reach.name}-previous-stock")
    incoming = incidence.input(f"{reach.name}-current-incoming")
    outgoing = incidence.input(f"{reach.name}-current-outgoing")
    update = incidence.add(previous, subtract(incoming, outgoing))
    return {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "id": f"{reach.name}-available",
        "value_kind": "extensive",
        "spec": {
            "kind": "finite_recurrence",
            "state_kinds": ["extensive"],
            "inputs": [
                {
                    "reference": {"id": f"{reach.name}-previous-stock", "value_kind": "scalar"},
                    "source": {"kind": "previous_state", "index": 0, "value_kind": "extensive"},
                },
                {
                    "reference": {"id": f"{reach.name}-current-incoming", "value_kind": "scalar"},
                    "source": {
                        "kind": "authoritative_fact",
                        "selector": {
                            "kind": "incoming_transfer_amount",
                            "compartment": reach.name,
                            "substance": "water",
                        },
                    },
                },
                {
                    "reference": {"id": f"{reach.name}-current-outgoing", "value_kind": "scalar"},
                    "source": {
                        "kind": "authoritative_fact",
                        "selector": {
                            "kind": "outgoing_transfer_amount",
                            "compartment": reach.name,
                            "substance": "water",
                        },
                    },
                },
            ],
            "parameters": [],
            "updates": [update],
            "output_index": 0,
        },
    }


def _exogenous_series(branch: str, forcing: str) -> dict[str, Any]:
    return {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "partition": {
            "kind": "exogenous_series",
            "branch": branch,
            "series": {"id": forcing},
        },
    }


def _unique_connections(connections: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for connection in connections:
        edge = (connection["source"], connection["target"])
        if edge not in seen:
            seen.add(edge)
            result.append(connection)
    return result


def _engine_run_id(run_id: RunId) -> bytes | str:
    if isinstance(run_id, UUID):
        return run_id.bytes
    if isinstance(run_id, bytearray):
        return bytes(run_id)
    return run_id
