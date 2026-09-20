"""TransportResult = time-indexed PhysicalSample projections × TransferRecord ledger × Balance accounts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .constituents import Constituent
    from .water_system import TimeAxis


class QualityState(StrEnum):
    """Physical support, independent of engine presence."""

    SUPPORTED = "supported"
    MISSING = "missing"
    DRY = "dry"
    UNSUPPORTED = "unsupported"
    ABSENT = "absent"
    OUTSIDE_HORIZON = "outside_horizon"


class BalanceState(StrEnum):
    """An independently evaluated conservation equation."""

    SUPPORTED = "supported"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


def _count(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("physical counts must be nonnegative integers or unknown")


@dataclass(frozen=True)
class PhysicalSample:
    """Exact extensive amounts and independently declared concentration support."""

    location: str
    step: int
    water_count: int | None
    mass_counts: Mapping[str, int | None]
    quality: Mapping[str, QualityState]
    reasons: Mapping[str, str]
    water_quality: QualityState = QualityState.SUPPORTED
    water_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.location or type(self.step) is not int:
            raise ValueError("sample requires a location and integer step")
        _count(self.water_count)
        if not isinstance(self.water_quality, QualityState):
            raise TypeError("water quality must be a QualityState")
        for value in self.mass_counts.values():
            _count(value)
        if set(self.mass_counts) != set(self.quality):
            raise ValueError("mass counts and quality must identify the same constituents")
        for name, state in self.quality.items():
            if not isinstance(state, QualityState):
                raise TypeError("quality must contain QualityState values")
            if state is QualityState.SUPPORTED and self.mass_counts[name] is None:
                raise ValueError("supported chemistry requires a mass count")
        object.__setattr__(self, "mass_counts", MappingProxyType(dict(self.mass_counts)))
        object.__setattr__(self, "quality", MappingProxyType(dict(self.quality)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))


@dataclass(frozen=True)
class TransferRecord:
    """One attributed realised physical exchange, never source apportionment after mixing."""

    source: str
    destination: str
    branch: str
    step: int
    kind: str
    sample: PhysicalSample


@dataclass(frozen=True)
class Balance:
    """residual = start + incoming - outgoing - end, without invented closing terms."""

    location: str
    step: int
    substance: str
    start: int | None
    end: int | None
    incoming: int | None
    outgoing: int | None

    def __post_init__(self) -> None:
        for value in (self.start, self.end, self.incoming, self.outgoing):
            _count(value)

    @property
    def residual(self) -> int | None:
        if self.start is None or self.end is None or self.incoming is None or self.outgoing is None:
            return None
        return self.start + self.incoming - self.outgoing - self.end

    @property
    def status(self) -> BalanceState:
        residual = self.residual
        if residual is None:
            return BalanceState.INCOMPLETE
        return BalanceState.SUPPORTED if residual == 0 else BalanceState.FAILED


type PhysicalView = Literal["incoming", "outgoing", "storage"]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        json.dumps(value, allow_nan=False)
        return value
    raise TypeError("metadata must be JSON compatible")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class TransportResult:
    """Immutable supported projections, exact accounts and attributable metadata; not a checkpoint."""

    time: TimeAxis
    water_quantum: Decimal
    constituents: tuple[Constituent, ...]
    incoming: Mapping[str, tuple[PhysicalSample, ...]]
    outgoing: Mapping[str, tuple[PhysicalSample, ...]]
    storage: Mapping[str, tuple[PhysicalSample, ...]]
    transfers: tuple[TransferRecord, ...]
    balances: tuple[Balance, ...]
    basin_balances: tuple[Balance, ...]
    metadata: Mapping[str, Any]
    process_balances: tuple[Balance, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.water_quantum, Decimal) or not self.water_quantum.is_finite() or self.water_quantum <= 0:
            raise ValueError("water quantum must be a positive finite Decimal")
        names = {item.id for item in self.constituents}
        if len(names) != len(self.constituents):
            raise ValueError("constituent names must be unique")
        object.__setattr__(self, "constituents", tuple(self.constituents))
        object.__setattr__(self, "balances", tuple(self.balances))
        object.__setattr__(self, "basin_balances", tuple(self.basin_balances))
        object.__setattr__(self, "process_balances", tuple(self.process_balances))
        for view in ("incoming", "outgoing", "storage"):
            rows = getattr(self, view)
            frozen = {}
            for location, samples in rows.items():
                if len(samples) != self.time.steps:
                    raise ValueError("physical series must cover the declared horizon")
                for step, sample in enumerate(samples):
                    if sample.location != location or sample.step != step or set(sample.mass_counts) != names:
                        raise ValueError("physical series location, step or constituents differ")
                frozen[location] = tuple(self._checked(sample) for sample in samples)
            object.__setattr__(self, view, MappingProxyType(frozen))
        object.__setattr__(
            self, "transfers", tuple(replace(item, sample=self._checked(item.sample)) for item in self.transfers)
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def _checked(self, sample: PhysicalSample) -> PhysicalSample:
        quality = dict(sample.quality)
        reasons = dict(sample.reasons)
        water_quality = sample.water_quality
        water_reason = sample.water_reason
        for balance in (*self.balances, *self.basin_balances, *self.process_balances):
            if balance.step != sample.step or balance.status is not BalanceState.FAILED:
                continue
            if balance not in self.basin_balances and balance.location != sample.location:
                continue
            if balance.substance == "water":
                water_quality = QualityState.UNSUPPORTED
                water_reason = f"failed water balance at {balance.location}"
            affected = quality if balance.substance == "water" else {balance.substance: quality[balance.substance]}
            for name in affected:
                quality[name] = QualityState.UNSUPPORTED
                reasons[name] = f"failed {balance.substance} balance at {balance.location}"
        return replace(sample, quality=quality, reasons=reasons, water_quality=water_quality, water_reason=water_reason)

    def sample(self, location: str, step: int, *, view: PhysicalView = "outgoing") -> PhysicalSample:
        """Read one model interval; out-of-horizon reads remain explicitly unsupported."""
        if view not in ("incoming", "outgoing", "storage"):
            raise ValueError(f"unknown physical view {view!r}")
        if type(step) is not int:
            raise TypeError("step must be an integer model interval, not a finer-resolution inference")
        rows = getattr(self, view)
        if location not in rows:
            raise KeyError(location)
        if not 0 <= step < self.time.steps:
            names = [item.id for item in self.constituents]
            return PhysicalSample(
                location,
                step,
                None,
                dict.fromkeys(names),
                dict.fromkeys(names, QualityState.OUTSIDE_HORIZON),
                dict.fromkeys(names, "outside model horizon"),
                QualityState.OUTSIDE_HORIZON,
                "outside model horizon",
            )
        return rows[location][step]

    def amount(
        self, location: str, step: int, substance: str = "water", *, view: PhysicalView = "outgoing"
    ) -> Decimal | None:
        """Project counts once to m3 or kg; never feed this display projection back to transport."""
        sample = self.sample(location, step, view=view)
        if not 0 <= step < self.time.steps:
            return None
        if substance == "water":
            return None if sample.water_count is None else _amount(sample.water_count, self.water_quantum)
        count = sample.mass_counts[substance]
        constituent = next(item for item in self.constituents if item.id == substance)
        return None if count is None else _amount(count, constituent.quantum_kg)

    def quality_state(
        self, location: str, step: int, constituent: str, *, view: PhysicalView = "outgoing"
    ) -> QualityState:
        """Distinguish unregistered chemistry from dry, missing and out-of-horizon support.

        Unknown locations and invalid interval indices are refused, even when the
        requested constituent is absent from this result's registry.
        """
        sample = self.sample(location, step, view=view)
        if constituent not in sample.quality:
            return QualityState.ABSENT
        return sample.quality[constituent]

    def concentration(
        self, location: str, step: int, constituent: str, *, view: PhysicalView = "outgoing"
    ) -> Fraction | None:
        """Return supported kg/m3, or None for absent or unsupported chemistry."""
        return self.sample_concentration(self.sample(location, step, view=view), constituent)

    def sample_concentration(self, sample: PhysicalSample, constituent: str) -> Fraction | None:
        """Derive an exact ratio, including for a mass/volume-first aggregate."""
        if constituent not in sample.quality:
            return None
        if (
            sample.quality[constituent] is not QualityState.SUPPORTED
            or sample.water_quality not in (QualityState.SUPPORTED, QualityState.DRY)
            or sample.water_count in (None, 0)
        ):
            return None
        count = sample.mass_counts[constituent]
        if count is None:
            return None
        definition = next(item for item in self.constituents if item.id == constituent)
        return Fraction(count) * Fraction(definition.quantum_kg) / (sample.water_count * Fraction(self.water_quantum))

    def aggregate(self, location: str, start: int, stop: int, *, view: PhysicalView = "outgoing") -> PhysicalSample:
        """Sum contiguous [start, stop) transport amounts before deriving concentration."""
        if view == "storage":
            raise ValueError("terminal storage inventories cannot be summed as interval transport")
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.time.steps:
            raise ValueError("aggregation requires whole contiguous model intervals inside the horizon")
        samples = [self.sample(location, step, view=view) for step in range(start, stop)]
        masses: dict[str, int | None] = {}
        quality: dict[str, QualityState] = {}
        reasons: dict[str, str] = {}
        water = (
            None
            if any(sample.water_count is None for sample in samples)
            else sum(sample.water_count for sample in samples if sample.water_count is not None)
        )
        for definition in self.constituents:
            name = definition.id
            values = [sample.mass_counts[name] for sample in samples]
            masses[name] = (
                None if any(value is None for value in values) else sum(value for value in values if value is not None)
            )
            states = {sample.quality[name] for sample in samples}
            unsupported = states - {QualityState.SUPPORTED, QualityState.DRY}
            if unsupported:
                quality[name] = next(
                    state
                    for state in (
                        QualityState.UNSUPPORTED,
                        QualityState.MISSING,
                        QualityState.OUTSIDE_HORIZON,
                        QualityState.ABSENT,
                    )
                    if state in unsupported
                )
                reasons[name] = "; ".join(
                    dict.fromkeys(
                        sample.reasons.get(name, sample.quality[name].value)
                        for sample in samples
                        if sample.quality[name] in unsupported
                    )
                )
            else:
                quality[name] = QualityState.SUPPORTED if water else QualityState.DRY
        water_states = {sample.water_quality for sample in samples}
        water_quality = next(
            state
            for state in (
                QualityState.UNSUPPORTED,
                QualityState.MISSING,
                QualityState.OUTSIDE_HORIZON,
                QualityState.ABSENT,
                QualityState.SUPPORTED,
                QualityState.DRY,
            )
            if state in water_states
        )
        water_reasons = "; ".join(dict.fromkeys(sample.water_reason for sample in samples if sample.water_reason))
        return PhysicalSample(location, start, water, masses, quality, reasons, water_quality, water_reasons or None)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible physical-output document."""
        return {
            "time": {"start": self.time.start.isoformat(), "steps": self.time.steps, "frequency": self.time.frequency},
            "water_quantum": str(self.water_quantum),
            "constituents": [_constituent_document(item) for item in self.constituents],
            "incoming": {name: [_sample_document(item) for item in samples] for name, samples in self.incoming.items()},
            "outgoing": {name: [_sample_document(item) for item in samples] for name, samples in self.outgoing.items()},
            "storage": {name: [_sample_document(item) for item in samples] for name, samples in self.storage.items()},
            "transfers": [
                {
                    "source": item.source,
                    "destination": item.destination,
                    "branch": item.branch,
                    "step": item.step,
                    "kind": item.kind,
                    "sample": _sample_document(item.sample),
                }
                for item in self.transfers
            ],
            "balances": [_balance_document(item) for item in self.balances],
            "basin_balances": [_balance_document(item) for item in self.basin_balances],
            "process_balances": [_balance_document(item) for item in self.process_balances],
            "metadata": _thaw(self.metadata),
        }

    @property
    def digest(self) -> str:
        """Content address of exact projections and their support/provenance metadata."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> TransportResult:
        """Restore an output projection, not executable state or a restart checkpoint."""
        from .constituents import Constituent
        from .water_system import TimeAxis

        time = document["time"]
        return cls(
            time=TimeAxis(time["start"], periods=time["steps"], frequency=time["frequency"]),
            water_quantum=Decimal(document["water_quantum"]),
            constituents=tuple(Constituent(**item) for item in document["constituents"]),
            incoming={name: tuple(_read_sample(item) for item in rows) for name, rows in document["incoming"].items()},
            outgoing={name: tuple(_read_sample(item) for item in rows) for name, rows in document["outgoing"].items()},
            storage={name: tuple(_read_sample(item) for item in rows) for name, rows in document["storage"].items()},
            transfers=tuple(
                TransferRecord(
                    item["source"],
                    item["destination"],
                    item["branch"],
                    item["step"],
                    item["kind"],
                    _read_sample(item["sample"]),
                )
                for item in document["transfers"]
            ),
            balances=tuple(Balance(**item) for item in document["balances"]),
            basin_balances=tuple(Balance(**item) for item in document["basin_balances"]),
            metadata=document["metadata"],
            process_balances=tuple(Balance(**item) for item in document["process_balances"]),
        )


def _amount(count: int, quantum: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = max(28, len(str(count)) + len(quantum.as_tuple().digits))
        return Decimal(count) * quantum


def _sample_document(sample: PhysicalSample) -> dict[str, Any]:
    return {
        "location": sample.location,
        "step": sample.step,
        "water_count": sample.water_count,
        "water_quality": sample.water_quality.value,
        "water_reason": sample.water_reason,
        "mass_counts": dict(sample.mass_counts),
        "quality": {name: state.value for name, state in sample.quality.items()},
        "reasons": dict(sample.reasons),
    }


def _read_sample(item: Mapping[str, Any]) -> PhysicalSample:
    return PhysicalSample(
        item["location"],
        item["step"],
        item["water_count"],
        item["mass_counts"],
        {name: QualityState(state) for name, state in item["quality"].items()},
        item["reasons"],
        QualityState(item["water_quality"]),
        item["water_reason"],
    )


def _balance_document(item: Balance) -> dict[str, Any]:
    return {
        "location": item.location,
        "step": item.step,
        "substance": item.substance,
        "start": item.start,
        "end": item.end,
        "incoming": item.incoming,
        "outgoing": item.outgoing,
    }


def _constituent_document(item: Constituent) -> dict[str, Any]:
    return {
        "id": item.id,
        "chemical_form": item.chemical_form,
        "reporting_basis": item.reporting_basis,
        "quantum_kg": str(item.quantum_kg),
    }
