"""compute_transport : CompletedRun × TimeAxis × ConservationQuantum × TransportTopology × ConservativeTransport → TransportResult.

Incidence alone allocates water. This component advects conservative mass using
its realised integer branch counts. It never runs or adjusts a water rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Literal

from .constituents import (
    Composition,
    ConservativeTransport,
    DomainSupport,
    Remobilisation,
    mass_counts,
    mass_quantisation_remainder,
)
from .physical_results import Balance, PhysicalSample, QualityState, TransferRecord, TransportResult

if TYPE_CHECKING:
    import incidence

    from .water_system import ConservationQuantum, TimeAxis, TransportTopology


def _sum(values: list[int | None]) -> int | None:
    return None if any(value is None for value in values) else sum(value for value in values if value is not None)


def _difference(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left - right


def _allocate(mass: int | None, water: int, denominator: int, dry_mass: int) -> int | None:
    return None if mass is None else (mass - dry_mass) * water // denominator


def _json(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "items"):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def compute_transport(
    completed: incidence.CompletedRun,
    time: TimeAxis,
    quantum: ConservationQuantum,
    topology: TransportTopology,
    config: ConservativeTransport,
) -> TransportResult:
    """Interpret one authoritative water execution, retaining unknown and dry mass."""
    species = tuple(c.id for c in config.constituents)
    replaced_owners = {owner for owner, _ in topology.replacements}
    active_accounts = tuple(account for account in config.process_accounts if account.owner not in replaced_owners)
    process_balances = tuple(
        Balance(
            account.location,
            account.step,
            account.substance,
            account.start,
            account.end,
            account.incoming,
            account.outgoing,
        )
        for account in active_accounts
    )
    process_support: dict[tuple[str, int, str], DomainSupport] = {}
    for account, balance in zip(active_accounts, process_balances, strict=True):
        residual = balance.residual
        state = (
            DomainSupport.UNRESOLVED
            if residual is None
            else DomainSupport.UNSUPPORTED
            if residual != 0
            else account.metadata.support
        )
        key = (account.location, account.step, account.substance)
        previous = process_support.get(key, DomainSupport.SUPPORTED)
        process_support[key] = (
            DomainSupport.UNSUPPORTED
            if DomainSupport.UNSUPPORTED in (previous, state)
            else DomainSupport.UNRESOLVED
            if DomainSupport.UNRESOLVED in (previous, state)
            else DomainSupport.SUPPORTED
        )

    def combined_support(left: DomainSupport, right: DomainSupport) -> DomainSupport:
        if DomainSupport.UNSUPPORTED in (left, right):
            return DomainSupport.UNSUPPORTED
        if DomainSupport.UNRESOLVED in (left, right):
            return DomainSupport.UNRESOLVED
        return DomainSupport.SUPPORTED

    def account_support(location: str, step: int, constituent: str) -> DomainSupport:
        return process_support.get((location, step, constituent), DomainSupport.SUPPORTED)

    exchange_checks: list[dict[str, Any]] = []

    def check_exchanges(location: str, step: int, actual_terms: Mapping[str, Mapping[str, int | None]]) -> None:
        for account in active_accounts:
            if account.location != location or account.step != step:
                continue
            for term, actual in actual_terms.items():
                supplied = getattr(account, term)
                observed = actual[account.substance]
                difference = _difference(supplied, observed)
                state = (
                    DomainSupport.UNRESOLVED
                    if difference is None
                    else DomainSupport.UNSUPPORTED
                    if difference
                    else DomainSupport.SUPPORTED
                )
                key = (location, step, account.substance)
                process_support[key] = combined_support(process_support[key], state)
                exchange_checks.append(
                    {
                        "owner": account.owner,
                        "location": location,
                        "step": step,
                        "substance": account.substance,
                        "term": term,
                        "supplied_count": supplied,
                        "realised_count": observed,
                        "difference_counts": difference,
                        "support": state.value,
                    }
                )

    constituents = {c.id: c for c in config.constituents}
    water_quantum = Decimal(str(quantum.quantum_m3))
    seconds = int(time.timestep.total_seconds())
    names = topology.order
    if set(config.initial) - set(names) or set(config.remobilisation) - set(names):
        raise ValueError("initial chemistry/remobilisation must name declared physical reaches")
    if set(config.boundaries) - set(topology.sources):
        raise ValueError("boundary chemistry must name declared water sources")
    for name, chemistry in config.boundaries.items():
        if len(chemistry) != time.steps:
            raise ValueError(f"source {name!r} chemistry interval count differs from water time axis")
    for state in config.process_states:
        if state.location not in names or not 0 <= state.step < time.steps:
            raise ValueError("imported process state location/interval is outside the model")

    def water_counts(name: str, direction: Literal["incoming", "outgoing"]) -> tuple[int, ...]:
        series = completed.transfer_count_series(name, "water", direction=direction, first=0, last=time.steps - 1)
        if any(v is None or v < 0 for v in series.values):
            raise ValueError(f"missing or negative authoritative water counts at {name!r}")
        return tuple(int(v) for v in series.values if v is not None)

    water_in = {n: water_counts(n, "incoming") for n in names}
    water_out = {n: water_counts(n, "outgoing") for n in names}
    sources = {n: water_counts(n, "outgoing") for n in topology.sources}
    branch_water = {b.observer: water_counts(b.observer, "incoming") for bs in topology.branches.values() for b in bs}
    endpoints = tuple(sorted(set(names) | {b.destination for bs in topology.branches.values() for b in bs}))
    incoming: dict[str, list[PhysicalSample]] = {n: [] for n in endpoints}
    outgoing: dict[str, list[PhysicalSample]] = {n: [] for n in names}
    storage: dict[str, list[PhysicalSample]] = {n: [] for n in names}
    mass_stock: dict[str, dict[str, int | None]] = {}
    support = {n: dict.fromkeys(species, DomainSupport.SUPPORTED) for n in names}
    dry_inventory = {n: dict.fromkeys(species, 0) for n in names}
    water_support = dict.fromkeys(names, config.metadata.support)
    water_stock = dict(topology.initial_counts)
    for n in names:
        composition = config.initial[n] if n in config.initial else Composition()
        mass_stock[n] = {
            c: (
                0
                if n not in config.initial and water_stock[n] == 0
                else mass_counts(composition, constituents[c], water_stock[n], water_quantum, seconds)
            )
            for c in species
        }
        if n in topology.delays and any(value != 0 for value in mass_stock[n].values()):
            raise ValueError("TravelDelay initial constituent inventory needs dated arrival cohorts; none are declared")
        for entry in composition.entries:
            support[n][entry.constituent] = entry.metadata.support
        if config.metadata.support != DomainSupport.SUPPORTED:
            support[n] = dict.fromkeys(species, config.metadata.support)
    transfers: list[TransferRecord] = []
    balances: list[Balance] = []
    basin_balances: list[Balance] = []
    parcels: dict[str, dict[int, PhysicalSample]] = {n: {} for n in topology.delays}
    transit_deposits: dict[str, dict[str, int | None]] = {n: dict.fromkeys(species, 0) for n in topology.delays}
    deposit_support = {n: dict.fromkeys(species, DomainSupport.SUPPORTED) for n in topology.delays}
    quantisation: list[dict[str, Any]] = []
    for n, composition in config.initial.items():
        for c in species:
            remainder = mass_quantisation_remainder(
                composition, constituents[c], topology.initial_counts[n], water_quantum, seconds
            )
            quantisation.append(
                {
                    "location": n,
                    "step": "initial",
                    "constituent": c,
                    "discarded_fraction_of_mass_quantum": None if remainder is None else str(remainder),
                }
            )
    dry_stocks: dict[str, list[dict[str, int | None]]] = {n: [] for n in names}
    deficits: dict[str, list[str]] = {n: [] for n in topology.requested}

    def sample(
        n: str,
        step: int,
        water: int,
        mass: Mapping[str, int | None],
        supportedness: Mapping[str, DomainSupport],
        water_state: DomainSupport = DomainSupport.SUPPORTED,
    ) -> PhysicalSample:
        water_state = combined_support(water_state, account_support(n, step, "water"))
        quality = {}
        reasons = {}
        for c in species:
            if mass[c] is None:
                quality[c] = QualityState.MISSING
                reasons[c] = "missing constituent input in connected physical inventory"
            elif (
                combined_support(combined_support(supportedness[c], account_support(n, step, c)), water_state)
                != DomainSupport.SUPPORTED
            ):
                quality[c] = QualityState.UNSUPPORTED
                reasons[c] = "dry inventory remobilisation or upstream physical support is unresolved"
            elif water == 0:
                quality[c] = QualityState.DRY
                reasons[c] = "zero water denominator; aqueous concentration undefined"
            else:
                quality[c] = QualityState.SUPPORTED
        return PhysicalSample(
            n,
            step,
            water,
            mass,
            quality,
            reasons,
            QualityState.SUPPORTED if water_state == DomainSupport.SUPPORTED else QualityState.UNSUPPORTED,
            None
            if water_state == DomainSupport.SUPPORTED
            else "supplied process water account or upstream water support is unresolved or failed",
        )

    def merge(n: str, step: int, parts: list[PhysicalSample]) -> PhysicalSample:
        water = sum(p.water_count for p in parts if p.water_count is not None)
        mass = {c: _sum([p.mass_counts[c] for p in parts]) for c in species}
        supportedness = {
            c: DomainSupport.UNSUPPORTED
            if any(
                p.quality[c] == QualityState.UNSUPPORTED and (p.water_count != 0 or p.mass_counts[c] != 0)
                for p in parts
            )
            else DomainSupport.SUPPORTED
            for c in species
        }
        water_state = (
            DomainSupport.UNSUPPORTED
            if any(p.water_quality == QualityState.UNSUPPORTED and p.water_count != 0 for p in parts)
            else DomainSupport.SUPPORTED
        )
        return sample(n, step, water, mass, supportedness, water_state)

    def apply_account_checks(part: PhysicalSample, owner: str, step: int) -> PhysicalSample:
        """Narrow current account checks onto this parcel without replacing its support."""
        if part.water_count is None:
            raise RuntimeError("physical transfer requires an authoritative water count")
        constituent_support = {}
        for c in species:
            existing = (
                DomainSupport.UNSUPPORTED if part.quality[c] == QualityState.UNSUPPORTED else DomainSupport.SUPPORTED
            )
            current = (
                account_support(owner, step, c)
                if part.water_count != 0 or part.mass_counts[c] != 0
                else DomainSupport.SUPPORTED
            )
            constituent_support[c] = combined_support(existing, current)
        existing_water = (
            DomainSupport.UNSUPPORTED if part.water_quality == QualityState.UNSUPPORTED else DomainSupport.SUPPORTED
        )
        current_water = account_support(owner, step, "water") if part.water_count != 0 else DomainSupport.SUPPORTED
        return sample(
            part.location,
            step,
            part.water_count,
            part.mass_counts,
            constituent_support,
            combined_support(existing_water, current_water),
        )

    for step in range(time.steps):
        incoming_parts: dict[str, list[PhysicalSample]] = {n: [] for n in endpoints}
        start_water = dict(water_stock)
        start_mass = {n: dict(mass_stock[n]) for n in names}
        external_inputs: list[PhysicalSample] = []
        external_outputs: list[PhysicalSample] = []
        for source, destination in topology.sources.items():
            water = sources[source][step]
            composition = config.boundaries[source][step] if source in config.boundaries else Composition()
            disabled = source in topology.disabled_sources
            masses = {
                c: 0 if disabled else mass_counts(composition, constituents[c], water, water_quantum, seconds)
                for c in species
            }
            for c in species:
                remainder = (
                    None
                    if disabled
                    else mass_quantisation_remainder(composition, constituents[c], water, water_quantum, seconds)
                )
                quantisation.append(
                    {
                        "location": source,
                        "step": step,
                        "constituent": c,
                        "discarded_fraction_of_mass_quantum": None if remainder is None else str(remainder),
                    }
                )
            check_exchanges(source, step, {"outgoing": {"water": water, **masses}})
            supportedness = dict.fromkeys(species, config.metadata.support)
            for entry in composition.entries:
                if entry.metadata.support != DomainSupport.SUPPORTED:
                    supportedness[entry.constituent] = entry.metadata.support
            for c in species:
                if water == 0 and masses[c] == 0:
                    supportedness[c] = DomainSupport.SUPPORTED
            for c in species:
                if water != 0 or masses[c] != 0:
                    supportedness[c] = combined_support(supportedness[c], account_support(source, step, c))
            source_water_support = account_support(source, step, "water") if water != 0 else DomainSupport.SUPPORTED
            part = sample(destination, step, water, masses, supportedness, source_water_support)
            incoming_parts[destination].append(part)
            external_inputs.append(part)
            transfers.append(TransferRecord(source, destination, "boundary", step, "boundary_input", part))

        for n in topology.order:
            arrival = merge(n, step, incoming_parts[n])
            if arrival.water_count != water_in[n][step]:
                raise ValueError(f"physical water routing disagrees with engine incoming count at {n!r}, step {step}")
            incoming[n].append(arrival)
            if arrival.water_quality == QualityState.UNSUPPORTED:
                water_support[n] = DomainSupport.UNSUPPORTED
            water_support[n] = combined_support(water_support[n], account_support(n, step, "water"))
            available_water = water_stock[n] + water_in[n][step]
            available_mass = {c: _sum([mass_stock[n][c], arrival.mass_counts[c]]) for c in species}
            for c in species:
                support[n][c] = combined_support(support[n][c], account_support(n, step, c))
                if arrival.quality[c] == QualityState.UNSUPPORTED:
                    support[n][c] = DomainSupport.UNSUPPORTED
                old_mass = mass_stock[n][c]
                if (
                    n not in topology.delays
                    and water_stock[n] == 0
                    and old_mass is not None
                    and old_mass > 0
                    and available_water > 0
                    and config.remobilisation.get(n, Remobilisation.UNRESOLVED) != Remobilisation.COMPLETE
                ):
                    support[n][c] = DomainSupport.UNRESOLVED
                    dry_inventory[n][c] = old_mass
            original_water = available_water
            original_mass = dict(available_mass)
            released_parts: list[PhysicalSample] = []
            if n in topology.delays:
                if arrival.water_count == 0:
                    for c in species:
                        transit_deposits[n][c] = _sum([transit_deposits[n][c], arrival.mass_counts[c]])
                        if arrival.quality[c] == QualityState.UNSUPPORTED and arrival.mass_counts[c] != 0:
                            deposit_support[n][c] = DomainSupport.UNSUPPORTED
                else:
                    cohort_mass = dict(arrival.mass_counts)
                    cohort_support = {
                        c: DomainSupport.UNSUPPORTED
                        if arrival.quality[c] == QualityState.UNSUPPORTED
                        else DomainSupport.SUPPORTED
                        for c in species
                    }
                    for c in species:
                        if transit_deposits[n][c] != 0:
                            if config.remobilisation.get(n, Remobilisation.UNRESOLVED) == Remobilisation.COMPLETE:
                                cohort_mass[c] = _sum([cohort_mass[c], transit_deposits[n][c]])
                                cohort_support[c] = combined_support(cohort_support[c], deposit_support[n][c])
                                transit_deposits[n][c] = 0
                                deposit_support[n][c] = DomainSupport.SUPPORTED
                            else:
                                cohort_support[c] = combined_support(cohort_support[c], DomainSupport.UNRESOLVED)
                    if arrival.water_count is None:
                        raise RuntimeError("delay arrival requires an authoritative water count")
                    parcels[n][step] = sample(
                        n,
                        step,
                        arrival.water_count,
                        cohort_mass,
                        cohort_support,
                        DomainSupport.UNSUPPORTED
                        if arrival.water_quality == QualityState.UNSUPPORTED
                        else DomainSupport.SUPPORTED,
                    )
            for branch in topology.branches[n]:
                old: PhysicalSample | None = None
                water = branch_water[branch.observer][step]
                if water > available_water:
                    raise ValueError(f"branch water exceeds physical inventory at {n!r}")
                if n in topology.delays:
                    old = parcels[n].pop(step - topology.delays[n], None)
                    if old is None:
                        if water:
                            raise ValueError("travel delay released water without a due parcel")
                        masses = dict.fromkeys(species, 0)
                    else:
                        if old.water_count != water:
                            raise ValueError("travel delay water differs from due parcel")
                        masses = dict(old.mass_counts)
                elif branch.kind == "evaporation" or water == 0:
                    masses = dict.fromkeys(species, 0)
                else:
                    denominator = original_water if topology.mixing[n] == "simultaneous" else available_water
                    pool = original_mass if topology.mixing[n] == "simultaneous" else available_mass
                    masses = {c: _allocate(pool[c], water, denominator, dry_inventory[n][c]) for c in species}
                part = sample(
                    branch.destination,
                    step,
                    water,
                    masses,
                    {
                        c: support[n][c] if water > 0 and branch.kind != "evaporation" else DomainSupport.SUPPORTED
                        for c in species
                    },
                    water_support[n] if water > 0 else DomainSupport.SUPPORTED,
                )
                if n in topology.delays and old is not None:
                    part = PhysicalSample(
                        branch.destination,
                        step,
                        water,
                        masses,
                        old.quality,
                        old.reasons,
                        old.water_quality,
                        old.water_reason,
                    )
                released_parts.append(part)
                available_water -= water
                available_mass = {c: _difference(available_mass[c], masses[c]) for c in species}
                if any(v is not None and v < 0 for v in available_mass.values()):
                    raise ValueError(f"negative constituent inventory at {n!r}")
            if n in topology.delays:
                remaining_parcels = list(parcels[n].values())
                available_mass = {
                    c: _sum([transit_deposits[n][c], *[parcel.mass_counts[c] for parcel in remaining_parcels]])
                    for c in species
                }
                support[n] = {
                    c: DomainSupport.UNSUPPORTED
                    if any(parcel.quality[c] == QualityState.UNSUPPORTED for parcel in remaining_parcels)
                    else combined_support(config.metadata.support, deposit_support[n][c])
                    for c in species
                }
                water_support[n] = (
                    DomainSupport.UNSUPPORTED
                    if any(parcel.water_quality == QualityState.UNSUPPORTED for parcel in remaining_parcels)
                    else config.metadata.support
                )
            departure = merge(n, step, released_parts)
            check_exchanges(
                n,
                step,
                {
                    "start": {"water": start_water[n], **start_mass[n]},
                    "incoming": {"water": arrival.water_count, **arrival.mass_counts},
                    "outgoing": {"water": departure.water_count, **departure.mass_counts},
                    "end": {"water": available_water, **available_mass},
                },
            )
            water_support[n] = combined_support(water_support[n], account_support(n, step, "water"))
            for c in species:
                support[n][c] = combined_support(support[n][c], account_support(n, step, c))
            if n in topology.delays:
                for entry_step, retained_parcel in tuple(parcels[n].items()):
                    parcels[n][entry_step] = apply_account_checks(retained_parcel, n, step)
                for c in species:
                    if transit_deposits[n][c] != 0:
                        current_check = combined_support(account_support(n, step, c), account_support(n, step, "water"))
                        deposit_support[n][c] = combined_support(deposit_support[n][c], current_check)
            checked_parts = []
            for branch, part in zip(topology.branches[n], released_parts, strict=True):
                if part.water_count is None:
                    raise RuntimeError("realised branch lost its authoritative water count")
                checked = apply_account_checks(part, n, step)
                checked_parts.append(checked)
                incoming_parts[branch.destination].append(checked)
                transfers.append(TransferRecord(n, branch.destination, branch.label, step, branch.kind, checked))
                if branch.destination not in names:
                    external_outputs.append(checked)
            departure = merge(n, step, checked_parts)
            if departure.water_count != water_out[n][step]:
                raise ValueError(f"physical water routing disagrees with engine outgoing count at {n!r}")
            outgoing[n].append(departure)
            water_stock[n] = start_water[n] + water_in[n][step] - water_out[n][step]
            if water_stock[n] < 0:
                raise ValueError(f"negative authoritative physical water inventory at {n!r}")
            mass_stock[n] = available_mass
            dry_stocks[n].append(
                dict(transit_deposits[n])
                if n in topology.delays
                else dict(available_mass)
                if water_stock[n] == 0
                else dict(dry_inventory[n])
            )
            storage_support = dict(support[n])
            if n in topology.delays and water_stock[n] > 0:
                for c in species:
                    if transit_deposits[n][c] != 0:
                        storage_support[c] = combined_support(storage_support[c], DomainSupport.UNRESOLVED)
            storage[n].append(sample(n, step, water_stock[n], available_mass, storage_support, water_support[n]))
            balances.append(
                Balance(n, step, "water", start_water[n], water_stock[n], arrival.water_count, departure.water_count)
            )
            for c in species:
                balances.append(
                    Balance(
                        n, step, c, start_mass[n][c], mass_stock[n][c], arrival.mass_counts[c], departure.mass_counts[c]
                    )
                )
            if n in deficits:
                requested = Decimal(str(topology.requested[n][step]))
                delivery = next(
                    branch for branch in topology.branches[n] if branch.label == topology.requested_branches[n]
                )
                delivered_count = branch_water[delivery.observer][step]
                deficits[n].append(str(max(Decimal(0), requested - Decimal(delivered_count) * water_quantum)))
        for n in sorted(set(endpoints) - set(names)):
            arrival = merge(n, step, incoming_parts[n])
            check_exchanges(n, step, {"incoming": {"water": arrival.water_count, **arrival.mass_counts}})
            incoming[n].append(merge(n, step, incoming_parts[n]))
        basin_balances.append(
            Balance(
                "basin",
                step,
                "water",
                sum(start_water.values()),
                sum(water_stock.values()),
                sum(p.water_count for p in external_inputs if p.water_count is not None),
                sum(p.water_count for p in external_outputs if p.water_count is not None),
            )
        )
        for c in species:
            basin_balances.append(
                Balance(
                    "basin",
                    step,
                    c,
                    _sum([start_mass[n][c] for n in names]),
                    _sum([mass_stock[n][c] for n in names]),
                    _sum([p.mass_counts[c] for p in external_inputs]),
                    _sum([p.mass_counts[c] for p in external_outputs]),
                )
            )

    metadata = {
        "process_account_exchange_checks": exchange_checks,
        "dry_inventory_counts": dry_stocks,
        "input_quantisation": quantisation,
        "schema_version": "1",
        "transport_method": "realised-count conservative compartment transport",
        "water_authority": "incidence",
        "water_model_digest": completed.model_digest,
        "water_execution_digest": completed.authoritative_log()[1],
        "transport_semantics_version": "1",
        "incidence_version": version("incidence"),
        "taqsim_version": version("taqsim"),
        "inputs": _json(config),
        "topology": _json(topology),
        "water_deficit_m3": deficits,
        "allocation_precision": "integer floors per declared branch; unallocated mass remains inventory",
        "scientific_scope": "declared complete mixing and parcel delays; no reactive/hydraulic/site validity inference",
        "source_attribution": "input identity preserved; no source apportionment after mixing",
        "storage_boundary": "physical reaches only; future forcing and branch observers excluded",
    }
    return TransportResult(
        time,
        water_quantum,
        config.constituents,
        {n: tuple(v) for n, v in incoming.items()},
        {n: tuple(v) for n, v in outgoing.items()},
        {n: tuple(v) for n, v in storage.items()},
        tuple(transfers),
        tuple(balances),
        tuple(basin_balances),
        metadata,
        process_balances=process_balances,
    )
