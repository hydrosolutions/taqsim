"""ConservativeTransport = constituent-indexed Composition declarations × Remobilisation assumptions × RunMetadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType


def _decimal(value: Decimal | str | int | float, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be boolean")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a decimal number") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


class InputStatus(StrEnum):
    ORIGINAL = "original"
    CORRECTED = "corrected"
    INFILLED = "infilled"
    MISSING = "missing"
    MODELLED = "modelled"


class DomainSupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Uncertainty:
    """Attributed scientific uncertainty, separate from arithmetic precision."""

    description: str
    source: str

    def __post_init__(self) -> None:
        _name(self.description, "uncertainty description")
        _name(self.source, "uncertainty source")


@dataclass(frozen=True)
class InputMetadata:
    source: str | None = None
    version: str | None = None
    status: InputStatus = InputStatus.ORIGINAL
    support: DomainSupport = DomainSupport.SUPPORTED
    provenance: tuple[str, ...] = ()
    uncertainty: Uncertainty | None = None
    reason: str | None = None
    predecessor: str | None = None
    owner: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, InputStatus) or not isinstance(self.support, DomainSupport):
            raise TypeError("input status and support must be typed enums")
        object.__setattr__(self, "provenance", tuple(self.provenance))
        for value in self.provenance:
            _name(value, "provenance")
        if self.support != DomainSupport.SUPPORTED and not self.reason:
            raise ValueError("unsupported or unresolved input requires an attributable reason")


@dataclass(frozen=True)
class SoftwareIdentity:
    """Supplied exact software or physical-model identity, not inferred provenance."""

    component: str
    version: str
    revision: str

    def __post_init__(self) -> None:
        for attribute in ("component", "version", "revision"):
            _name(getattr(self, attribute), attribute)


@dataclass(frozen=True)
class RunMetadata:
    scenario: str | None = None
    reference: str | None = None
    version: str = "1"
    provenance: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    uncertainty: Uncertainty | None = None
    support: DomainSupport = DomainSupport.SUPPORTED
    software_identities: tuple[SoftwareIdentity, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "software_identities", tuple(self.software_identities))
        if not all(isinstance(item, SoftwareIdentity) for item in self.software_identities):
            raise TypeError("software identities require SoftwareIdentity records")
        components = [item.component for item in self.software_identities]
        if len(components) != len(set(components)):
            raise ValueError("duplicate software component identity")
        _name(self.version, "configuration version")
        if not isinstance(self.support, DomainSupport):
            raise TypeError("domain support must be a DomainSupport")
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True, init=False)
class Mass:
    value: Decimal
    unit: str

    def __init__(self, value: Decimal | str | int | float, unit: str = "kg") -> None:
        if unit not in ("kg", "g", "mg"):
            raise ValueError(f"unsupported mass unit {unit!r}")
        object.__setattr__(self, "value", _decimal(value, "mass"))
        object.__setattr__(self, "unit", unit)

    @property
    def kg(self) -> Decimal:
        return self.value * {"kg": Decimal(1), "g": Decimal("0.001"), "mg": Decimal("0.000001")}[self.unit]


@dataclass(frozen=True, init=False)
class Concentration:
    value: Decimal
    unit: str

    def __init__(self, value: Decimal | str | int | float, unit: str = "kg/m3") -> None:
        if unit not in ("kg/m3", "kg/m³", "mg/l", "mg/L", "g/m3", "g/L", "g/l"):
            raise ValueError(f"unsupported concentration unit {unit!r}")
        object.__setattr__(self, "value", _decimal(value, "concentration"))
        object.__setattr__(self, "unit", unit)

    @property
    def kg_per_m3(self) -> Decimal:
        return self.value * (Decimal("0.001") if self.unit in ("mg/l", "mg/L", "g/m3") else Decimal(1))


@dataclass(frozen=True, init=False)
class MassRate:
    value: Decimal
    unit: str

    def __init__(self, value: Decimal | str | int | float, unit: str = "kg/s") -> None:
        if unit not in ("kg/s", "g/s", "mg/s"):
            raise ValueError(f"unsupported mass rate unit {unit!r}")
        object.__setattr__(self, "value", _decimal(value, "mass rate"))
        object.__setattr__(self, "unit", unit)

    @property
    def kg_per_second(self) -> Decimal:
        return Mass(self.value, self.unit[:-2]).kg


@dataclass(frozen=True)
class MassCounts:
    """Authoritative mass counts on the identified constituent quantum."""

    count: int

    def __post_init__(self) -> None:
        _count(self.count, "mass count")


@dataclass(frozen=True)
class Constituent:
    id: str
    chemical_form: str
    reporting_basis: str
    quantum_kg: Decimal = Decimal("0.000001")

    def __post_init__(self) -> None:
        for attribute in ("id", "chemical_form", "reporting_basis"):
            _name(getattr(self, attribute), attribute)
        quantum = _decimal(self.quantum_kg, "mass quantum")
        if quantum == 0:
            raise ValueError("mass quantum must be positive")
        object.__setattr__(self, "quantum_kg", quantum)


@dataclass(frozen=True)
class ConstituentInput:
    """One chemical declaration; no value denotes explicit missing chemistry."""

    constituent: str
    mass: Mass | MassRate | MassCounts | None = None
    concentration: Concentration | None = None
    chemical_form: str | None = None
    reporting_basis: str | None = None
    metadata: InputMetadata = InputMetadata()

    def __post_init__(self) -> None:
        _name(self.constituent, "constituent")
        if self.mass is not None and self.concentration is not None:
            raise ValueError("conflicting mass and concentration declarations")
        if self.mass is not None and not isinstance(self.mass, (Mass, MassRate, MassCounts)):
            raise TypeError("mass requires Mass, MassRate or MassCounts")
        if self.concentration is not None and not isinstance(self.concentration, Concentration):
            raise TypeError("concentration requires Concentration")
        if self.metadata.status == InputStatus.MISSING and (self.mass is not None or self.concentration is not None):
            raise ValueError("missing chemistry cannot declare a numerical value")


@dataclass(frozen=True)
class Composition:
    entries: tuple[ConstituentInput, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if not all(isinstance(entry, ConstituentInput) for entry in self.entries):
            raise TypeError("composition requires ConstituentInput entries")
        identities = [entry.constituent for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate constituent declaration in composition")


class Remobilisation(StrEnum):
    UNRESOLVED = "unresolved"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ExchangeReplacement:
    """Replace both incoming and outgoing water/chemistry of a process owner."""

    owner: str
    replacement_owner: str

    def __post_init__(self) -> None:
        _name(self.owner, "replaced owner")
        _name(self.replacement_owner, "replacement owner")
        if self.owner == self.replacement_owner:
            raise ValueError("replacement owner must differ from replaced owner")


@dataclass(frozen=True)
class ProcessState:
    """Attributed specialist state, not computed by conservative mixing.

    Signed values are permitted for datum-relative stage or directional velocity.
    The declared domain and unit belong to the supplied specialist model.
    """

    variable: str
    location: str
    step: int
    value: Decimal | None
    unit: str
    domain: str
    metadata: InputMetadata
    chemical_form: str | None = None
    reporting_basis: str | None = None

    def __post_init__(self) -> None:
        for attribute in ("variable", "location", "unit", "domain"):
            _name(getattr(self, attribute), attribute)
        if type(self.step) is not int or self.step < -1:
            raise ValueError("process state interval must be an integer at or after predecessor -1")
        if not self.metadata.source or not self.metadata.version:
            raise ValueError("imported process state requires model source and version")
        if self.value is not None:
            if isinstance(self.value, bool):
                raise TypeError("process state must not be boolean")
            try:
                value = Decimal(str(self.value))
            except InvalidOperation as error:
                raise ValueError("process state must be decimal") from error
            if not value.is_finite():
                raise ValueError("process state must be finite")
            if self.metadata.status == InputStatus.MISSING:
                raise ValueError("missing process state cannot declare a value")
            object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ProcessAccount:
    """Supplied specialist balance terms on registered water or mass count quanta.

    These terms are independent diagnostic evidence, never additional transfers
    or basin inputs. Missing terms stay unknown; residuals are computed downstream.
    """

    owner: str
    location: str
    step: int
    substance: str
    start: int | None
    end: int | None
    incoming: int | None
    outgoing: int | None
    metadata: InputMetadata

    def __post_init__(self) -> None:
        for attribute in ("owner", "location", "substance"):
            _name(getattr(self, attribute), attribute)
        _count(self.step, "process account interval")
        for attribute in ("start", "end", "incoming", "outgoing"):
            value = getattr(self, attribute)
            if value is not None:
                _count(value, f"process account {attribute}")
        if not isinstance(self.metadata, InputMetadata):
            raise TypeError("process account requires InputMetadata")
        if not self.metadata.source or not self.metadata.version:
            raise ValueError("process account requires model source and version")
        if self.metadata.owner is not None and self.metadata.owner != self.owner:
            raise ValueError("process account metadata has conflicting owner")


@dataclass(frozen=True)
class LocationMapping:
    """Versioned supplied mapping from a physical model location to assessment identities."""

    location: str
    physical_water_body: str
    calculation_point: str
    mapping_version: str
    plan_unit: str | None = None

    def __post_init__(self) -> None:
        for attribute in ("location", "physical_water_body", "calculation_point", "mapping_version"):
            _name(getattr(self, attribute), attribute)
        if self.plan_unit is not None:
            _name(self.plan_unit, "plan unit")


@dataclass(frozen=True)
class ConservativeTransport:
    """Chemistry attached to existing WaterSystem reach and source identities.

    WaterSystem rules alone control water movement and operation order. Omitted
    initial chemistry is unknown for a wet reach and zero for an empty reach.
    Omitted source chemistry is unknown, never an inferred clean-water input.
    Each constituent has an independent account; no implicit TDS/ion sum exists.
    """

    constituents: tuple[Constituent, ...]
    initial: Mapping[str, Composition] = field(default_factory=dict)
    boundaries: Mapping[str, tuple[Composition, ...]] = field(default_factory=dict)
    remobilisation: Mapping[str, Remobilisation] = field(default_factory=dict)
    metadata: RunMetadata = RunMetadata()
    replacements: tuple[ExchangeReplacement, ...] = ()
    process_states: tuple[ProcessState, ...] = ()
    predecessor_states: tuple[ProcessState, ...] = ()
    location_mappings: tuple[LocationMapping, ...] = ()
    process_accounts: tuple[ProcessAccount, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "constituents", tuple(self.constituents))
        object.__setattr__(self, "initial", MappingProxyType(dict(self.initial)))
        object.__setattr__(
            self, "boundaries", MappingProxyType({key: tuple(value) for key, value in self.boundaries.items()})
        )
        object.__setattr__(self, "remobilisation", MappingProxyType(dict(self.remobilisation)))
        object.__setattr__(self, "replacements", tuple(self.replacements))
        object.__setattr__(self, "process_states", tuple(self.process_states))
        object.__setattr__(self, "predecessor_states", tuple(self.predecessor_states))
        object.__setattr__(self, "location_mappings", tuple(self.location_mappings))
        object.__setattr__(self, "process_accounts", tuple(self.process_accounts))
        if not all(isinstance(item, ProcessAccount) for item in self.process_accounts):
            raise TypeError("process accounts require ProcessAccount records")
        account_keys = [(item.owner, item.location, item.step, item.substance) for item in self.process_accounts]
        if len(account_keys) != len(set(account_keys)):
            raise ValueError("duplicate process account")
        if not all(isinstance(item, LocationMapping) for item in self.location_mappings):
            raise TypeError("location mappings require LocationMapping records")
        mapped = [item.location for item in self.location_mappings]
        if len(mapped) != len(set(mapped)):
            raise ValueError("duplicate mapped location")
        if any(state.step != -1 for state in self.predecessor_states):
            raise ValueError("predecessor states require interval -1")
        if any(state.step < 0 for state in self.process_states):
            raise ValueError("current process states require non-negative intervals")
        state_keys = [
            (state.variable, state.location, state.step) for state in (*self.process_states, *self.predecessor_states)
        ]
        if len(state_keys) != len(set(state_keys)):
            raise ValueError("duplicate imported process state")
        if not self.constituents or not all(isinstance(item, Constituent) for item in self.constituents):
            raise ValueError("transport requires identified constituents")
        constituents = {item.id: item for item in self.constituents}
        if len(constituents) != len(self.constituents):
            raise ValueError("duplicate constituent identity")
        for state in (*self.process_states, *self.predecessor_states):
            if state.variable in constituents:
                chemical = constituents[state.variable]
                if state.chemical_form != chemical.chemical_form or state.reporting_basis != chemical.reporting_basis:
                    raise ValueError(
                        "imported constituent concentration has incompatible chemical form or reporting basis"
                    )
                if state.unit not in ("kg/m3", "mg/l", "mg/L", "g/m3", "g/l", "g/L"):
                    raise ValueError("imported constituent concentration requires a supported concentration unit")
                if state.value is not None and state.value < 0:
                    raise ValueError("imported conservative concentration must be non-negative")
        for account in self.process_accounts:
            if account.substance != "water" and account.substance not in constituents:
                raise ValueError(f"unregistered process account substance {account.substance!r}")
        for location in (*self.initial, *self.boundaries, *self.remobilisation):
            _name(location, "chemistry location")
        if not all(isinstance(value, Remobilisation) for value in self.remobilisation.values()):
            raise TypeError("remobilisation requires Remobilisation enums")
        compositions = list(self.initial.values()) + [value for series in self.boundaries.values() for value in series]
        for composition in compositions:
            if not isinstance(composition, Composition):
                raise TypeError("chemistry declarations require Composition")
            for entry in composition.entries:
                if entry.constituent not in constituents:
                    raise ValueError(f"unregistered constituent {entry.constituent!r}")
                species = constituents[entry.constituent]
                if entry.chemical_form is not None and entry.chemical_form != species.chemical_form:
                    raise ValueError(f"incompatible chemical form for {entry.constituent!r}")
                if entry.reporting_basis is not None and entry.reporting_basis != species.reporting_basis:
                    raise ValueError(f"incompatible reporting basis for {entry.constituent!r}")
        for composition in self.initial.values():
            if any(
                entry.concentration is not None or isinstance(entry.mass, MassRate) for entry in composition.entries
            ):
                raise ValueError("initial inventory requires mass totals or counts, not concentrations or rates")
        replaced = [item.owner for item in self.replacements]
        if len(replaced) != len(set(replaced)):
            raise ValueError("duplicate replacement owner")
        if any(item.replacement_owner in replaced for item in self.replacements):
            raise ValueError("chained or cyclic exchange replacements are ambiguous")

    def validate_network(
        self, reaches: tuple[str, ...], sources: tuple[str, ...], periods: int, endpoints: tuple[str, ...] = ()
    ) -> None:
        """Bind chemistry to existing physical locations and exact source intervals."""
        _count(periods, "horizon")
        if periods == 0:
            raise ValueError("horizon must be positive")
        locations = set(reaches) | set(sources) | set(endpoints)
        for account in self.process_accounts:
            if account.location not in locations or account.step >= periods:
                raise ValueError("process account has incompatible location or interval")
        for mapping in self.location_mappings:
            if mapping.location not in locations:
                raise ValueError(f"unknown mapped location {mapping.location!r}")
        for state in self.predecessor_states:
            if state.location not in locations:
                raise ValueError(f"unknown predecessor location {state.location!r}")
        for state in self.process_states:
            if state.location not in reaches or state.step >= periods:
                raise ValueError("imported process state has incompatible location or interval")
        for location in (*self.initial, *self.remobilisation):
            if location not in reaches:
                raise ValueError(f"unknown chemistry reach {location!r}")
        for location, series in self.boundaries.items():
            if location not in sources:
                raise ValueError(f"unknown chemistry source {location!r}")
            if len(series) != periods:
                raise ValueError(f"source {location!r} chemistry must match the exact interval horizon")


def _exact_mass_counts(
    composition: Composition,
    constituent: Constituent,
    water_count: int,
    water_quantum: Decimal,
    seconds: int,
) -> Fraction | None:
    """Compute exact supplied interval mass on the declared mass quantum.

    Count-native declarations bypass conversion. Missing chemistry stays unknown.
    Fraction arithmetic prevents ambient Decimal precision from rounding large
    authoritative counts. The caller discloses the discarded fractional quantum.
    """
    _count(water_count, "water count")
    _count(seconds, "interval duration")
    quantum = _decimal(water_quantum, "water quantum")
    if seconds == 0 or quantum == 0:
        raise ValueError("interval duration and water quantum must be positive")
    entry = next((item for item in composition.entries if item.constituent == constituent.id), None)
    if entry is None:
        return Fraction(0) if water_count == 0 else None
    if entry.chemical_form is not None and entry.chemical_form != constituent.chemical_form:
        raise ValueError("incompatible chemical form")
    if entry.reporting_basis is not None and entry.reporting_basis != constituent.reporting_basis:
        raise ValueError("incompatible reporting basis")
    if isinstance(entry.mass, MassCounts):
        return Fraction(entry.mass.count)
    if isinstance(entry.mass, Mass):
        mass = (
            Fraction(entry.mass.value)
            * {"kg": Fraction(1), "g": Fraction(1, 1000), "mg": Fraction(1, 1000000)}[entry.mass.unit]
        )
    elif isinstance(entry.mass, MassRate):
        mass = (
            Fraction(entry.mass.value)
            * {"kg/s": Fraction(1), "g/s": Fraction(1, 1000), "mg/s": Fraction(1, 1000000)}[entry.mass.unit]
            * seconds
        )
    elif entry.concentration is not None:
        factor = Fraction(1, 1000) if entry.concentration.unit in ("mg/l", "mg/L", "g/m3") else Fraction(1)
        mass = Fraction(entry.concentration.value) * factor * water_count * Fraction(quantum)
    else:
        return Fraction(0) if water_count == 0 else None
    return mass / Fraction(constituent.quantum_kg)


def mass_counts(
    composition: Composition,
    constituent: Constituent,
    water_count: int,
    water_quantum: Decimal,
    seconds: int,
) -> int | None:
    """Floor exact mass to counts; pair with mass_quantisation_remainder.

    Missing chemistry on an actual-zero water input contributes zero; an explicit
    dry mass load still contributes its declared mass.
    """
    exact = _exact_mass_counts(composition, constituent, water_count, water_quantum, seconds)
    return None if exact is None else exact.numerator // exact.denominator


def mass_quantisation_remainder(
    composition: Composition,
    constituent: Constituent,
    water_count: int,
    water_quantum: Decimal,
    seconds: int,
) -> Fraction | None:
    """Return the discarded fraction of one mass count, not scientific uncertainty."""
    exact = _exact_mass_counts(composition, constituent, water_count, water_quantum, seconds)
    return None if exact is None else exact - exact.numerator // exact.denominator
