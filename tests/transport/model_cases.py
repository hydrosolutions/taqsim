"""network : tuple[Quantity, ...] × tuple[Inventory, ...] × Horizon → ModelDocument (test data).

Independent local, engine and physical count balances use only public transfer readback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import incidence as inc
import numpy as np

RUN_ID = bytes(range(16))


@dataclass(frozen=True)
class Inventory:
    """An initial whole-quantum inventory at a named finite compartment."""

    compartment: str
    substance: str
    count: int


@dataclass(frozen=True)
class Quantity:
    """A conserved substance's declared reporting unit and arithmetic quantum."""

    name: str
    unit: str
    quantum: float


WATER = Quantity("water", "m3", 1.0)
SALT = Quantity("salt", "kg", 0.1)


def network(quantities: tuple[Quantity, ...], stocks: tuple[Inventory, ...], steps: int = 1) -> dict[str, Any]:
    """network : ConservedQuantities × InitialInventories × Horizon → ModelDocument."""
    quanta = {q.name: q.quantum for q in quantities}
    compartments = sorted({stock.compartment for stock in stocks})
    return inc.model_document(
        finite_compartments=compartments,
        boundary_accounts=[],
        connections=[],
        substances=[q.name for q in quantities],
        initial_stocks=[
            {
                "compartment": node,
                "amounts": [
                    {"substance": s.substance, "amount": s.count * quanta[s.substance]}
                    for s in stocks
                    if s.compartment == node
                ],
            }
            for node in compartments
        ],
        calendar={"origin_unix_seconds": 1577836800, "timestep_seconds": 86400},
        horizon={"first": 0, "last": steps - 1},
        projections={"specifications": [], "initial_states": []},
        forcings=[],
        interpolation_tables=[],
        rules=[],
        transfer_bindings=[],
        input_bindings=[],
        units=[{"substance": q.name, "unit": q.unit, "quantum": q.quantum} for q in quantities],
    )


def partition(branches: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "partition": {
            "kind": "expression_partition",
            "branches": [{"branch": b, "expression": e} for b, e in branches.items()],
        },
    }


def bind(doc: dict[str, Any], node: str, substance: str, destinations: dict[str, str]) -> None:
    for branch, target in destinations.items():
        connection = {"source": node, "target": target}
        if connection not in doc["connections"]:
            doc["connections"].append(connection)
        if target not in doc["finite_compartments"] and target not in doc["boundary_accounts"]:
            doc["boundary_accounts"].append(target)
        doc["transfer_bindings"].append(
            {"compartment": node, "substance": substance, "branch": branch, "destination": target}
        )


def rule(doc: dict[str, Any], node: str, substance: str, branches: dict[str, tuple[str, Any]]) -> None:
    bind(doc, node, substance, {b: target for b, (target, _) in branches.items()})
    doc["rules"].append(
        inc.rule(node, substance, inc.literal(0), partition({b: expr for b, (_, expr) in branches.items()}))
    )


def scheduled(
    doc: dict[str, Any], node: str, substance: str, branches: dict[str, tuple[str, tuple[float, ...]]]
) -> None:
    expressions = {}
    for branch, (target, amounts) in branches.items():
        identifier = f"{node}-{substance}-{branch}"
        assert len(amounts) == doc["horizon"]["last"] + 1
        doc["forcings"].append({"id": identifier, "horizon": doc["horizon"].copy(), "values": list(amounts)})
        expressions[branch] = (target, inc.forcing(identifier))
    rule(doc, node, substance, expressions)


def coupled(
    doc: dict[str, Any], node: str, substance: str, mappings: dict[str, tuple[str, str]], carrier: str = "water"
) -> None:
    """Map dependent branch names to (carrier branch, destination), without physical defaults."""
    bind(doc, node, substance, {b: target for b, (_, target) in mappings.items()})
    disposition = {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "partition": {
            "kind": "carrier_proportional",
            "carrier": carrier,
            "branches": [{"branch": b, "carrier_branch": cb} for b, (cb, _) in mappings.items()],
        },
    }
    doc["rules"].append(inc.rule(node, substance, inc.literal(0), disposition))


def available(doc: dict[str, Any], node: str, substance: str, initial_count: int) -> Any:
    identifier = f"{node}-{substance}-available"
    quantum = next(q["quantum"] for q in doc["units"] if q["substance"] == substance)
    previous = inc.input(identifier + "-previous")
    incoming = inc.input(identifier + "-incoming")
    outgoing = inc.input(identifier + "-outgoing")
    difference = {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "expression": {"kind": "subtract", "lhs": incoming["expression"], "rhs": outgoing["expression"]},
    }
    inputs = [
        {
            "reference": {"id": identifier + "-previous", "value_kind": "scalar"},
            "source": {"kind": "previous_state", "index": 0, "value_kind": "extensive"},
        }
    ]
    inputs.extend(
        {
            "reference": {"id": identifier + "-" + direction, "value_kind": "scalar"},
            "source": fact(node, substance, direction),
        }
        for direction in ("incoming", "outgoing")
    )
    spec = {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "id": identifier,
        "value_kind": "extensive",
        "spec": {
            "kind": "finite_recurrence",
            "state_kinds": ["extensive"],
            "inputs": inputs,
            "parameters": [],
            "updates": [inc.add(previous, difference)],
            "output_index": 0,
        },
    }
    doc["projections"]["specifications"].append(spec)
    doc["projections"]["initial_states"].append(
        {"projection": identifier, "values": [{"kind": "extensive", "value": initial_count * quantum}]}
    )
    return inc.projection(identifier, "extensive")


def fact(node: str, substance: str, direction: str = "incoming") -> dict[str, Any]:
    return {
        "kind": "authoritative_fact",
        "selector": {"kind": direction + "_transfer_amount", "compartment": node, "substance": substance},
    }


def lag(doc: dict[str, Any], node: str, substance: str) -> Any:
    identifier = f"{node}-{substance}-lag"
    doc["projections"]["specifications"].append(
        {
            "rule_ir_version": "v1",
            "numerical_semantics_version": "v1",
            "id": identifier,
            "value_kind": "extensive",
            "spec": {"kind": "bounded_lag", "source": fact(node, substance), "steps": 1},
        }
    )
    doc["projections"]["initial_states"].append(
        {"projection": identifier, "values": [{"kind": "extensive", "value": 0}]}
    )
    return inc.projection(identifier, "extensive")


def counts(run: Any, node: str, substance: str, direction: Literal["incoming", "outgoing"] = "incoming") -> list[int]:
    series = run.transfer_count_series(node, substance, direction=direction)
    assert list(series.presence) == ["present"] * len(series.values)
    result = list(series.values)
    assert all(isinstance(value, int) and value >= 0 for value in result)
    return result


def stock(run: Any, initial: Inventory) -> list[int]:
    incoming = counts(run, initial.compartment, initial.substance)
    outgoing = counts(run, initial.compartment, initial.substance, "outgoing")
    remaining = initial.count
    values = []
    for incoming_count, outgoing_count in zip(incoming, outgoing, strict=True):
        remaining += incoming_count - outgoing_count
        assert remaining >= 0
        values.append(remaining)
    return values


def assert_balances(
    run: Any,
    quantities: tuple[Quantity, ...],
    stocks: tuple[Inventory, ...],
    boundaries: tuple[str, ...],
    physical: frozenset[str],
    entries: tuple[str, ...],
) -> None:
    """Independent balances from public counts; future forcing is outside the physical store set."""
    for q in quantities:
        selected = [s for s in stocks if s.substance == q.name]
        inventories = {s.compartment: stock(run, s) for s in selected}
        steps = len(next(iter(inventories.values())))
        exports = np.array([sum(counts(run, b, q.name)[t] for b in boundaries) for t in range(steps)], dtype=object)
        engine_stock = np.array([sum(v[t] for v in inventories.values()) for t in range(steps)], dtype=object)
        np.testing.assert_array_equal(engine_stock + np.cumsum(exports), [sum(s.count for s in selected)] * steps)
        physical_stock = np.array([sum(inventories[n][t] for n in physical) for t in range(steps)], dtype=object)
        physical_entries = np.array(
            [sum(counts(run, n, q.name, "outgoing")[t] for n in entries) for t in range(steps)], dtype=object
        )
        initial = sum(s.count for s in selected if s.compartment in physical)
        np.testing.assert_array_equal(physical_stock + np.cumsum(exports), initial + np.cumsum(physical_entries))


def mixed_pool(
    initial: tuple[int, int] = (0, 0), inputs: tuple[tuple[int, int], ...] = ((10, 80), (5, 10))
) -> tuple[dict[str, Any], tuple[Inventory, ...]]:
    """One completely mixed pool and explicit current-interval boundary supplies, in counts."""
    stocks = (Inventory("pool", "water", initial[0]), Inventory("pool", "salt", initial[1])) + tuple(
        Inventory(f"supply-{i}", s, count)
        for i, pair in enumerate(inputs)
        for s, count in zip(("water", "salt"), pair, strict=True)
    )
    doc = network((WATER, SALT), stocks)
    for i, (water, salt) in enumerate(inputs):
        scheduled(doc, f"supply-{i}", "water", {"entry": ("pool", (water * WATER.quantum,))})
        scheduled(doc, f"supply-{i}", "salt", {"entry": ("pool", (salt * SALT.quantum,))})
    return doc, stocks


def assert_pool_balances(run: Any, doc: dict[str, Any], stocks: tuple[Inventory, ...]) -> None:
    assert_balances(
        run,
        (WATER, SALT),
        stocks,
        tuple(doc["boundary_accounts"]),
        frozenset({"pool"}),
        tuple(n for n in doc["finite_compartments"] if n != "pool"),
    )
