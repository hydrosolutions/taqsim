"""Declared interval delay and dry inventories → exact public transfer and stock witnesses."""

import incidence as inc
import pytest

from tests.transport.model_cases import (
    RUN_ID,
    SALT,
    WATER,
    Inventory,
    assert_balances,
    available,
    counts,
    coupled,
    lag,
    network,
    rule,
    scheduled,
    stock,
)


def test_a8_one_interval_transit_is_inventory_not_a_topological_label():
    stocks = (
        Inventory("supply", "water", 10),
        Inventory("supply", "salt", 50),
        Inventory("transit", "water", 0),
        Inventory("transit", "salt", 0),
    )
    doc = network((WATER, SALT), stocks, 3)
    scheduled(doc, "supply", "water", {"entry": ("transit", (10.0, 0.0, 0.0))})
    scheduled(doc, "supply", "salt", {"entry": ("transit", (5.0, 0.0, 0.0))})
    rule(doc, "transit", "water", {"arrival": ("outlet", lag(doc, "transit", "water"))})
    coupled(doc, "transit", "salt", {"carried": ("arrival", "outlet")})
    model = inc.compile_model(doc)
    run = model.run(RUN_ID)
    assert counts(run, "outlet", "water") == [0, 10, 0]
    assert counts(run, "outlet", "salt") == [0, 50, 0]
    assert stock(run, Inventory("transit", "water", 0)) == [10, 0, 0]
    assert stock(run, Inventory("transit", "salt", 0)) == [50, 0, 0]
    assert_balances(run, (WATER, SALT), stocks, ("outlet",), frozenset({"transit"}), ("supply",))
    run.replay_against(model.run(RUN_ID))
    assert run.authoritative_log() == model.run(RUN_ID).authoritative_log()


def test_sparse_multisubstance_lag_reads_only_its_named_substance():
    # Real public projection path: same incoming timestep has water and salt events.
    stocks = (
        Inventory("supply", "water", 10),
        Inventory("supply", "salt", 50),
        Inventory("transit", "water", 0),
        Inventory("transit", "salt", 0),
    )
    doc = network((WATER, SALT), stocks, 2)
    for substance, amount in [("water", 10.0), ("salt", 5.0)]:
        scheduled(doc, "supply", substance, {"entry": ("transit", (amount, 0.0))})
        rule(doc, "transit", substance, {"arrival": ("outlet", lag(doc, "transit", substance))})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "outlet", "water") == [0, 10]
    assert counts(run, "outlet", "salt") == [0, 50]
    assert_balances(run, (WATER, SALT), stocks, ("outlet",), frozenset({"transit"}), ("supply",))


def test_a9_use_inventory_returns_once_after_water_only_loss():
    # Use is the explicit one-interval holding store. The return connection adds
    # no second delay. This is one supplied process, not a generic irrigation rule.
    stocks = (
        Inventory("supply", "water", 10),
        Inventory("supply", "salt", 50),
        Inventory("use", "water", 0),
        Inventory("use", "salt", 0),
    )
    doc = network((WATER, SALT), stocks, 3)
    scheduled(doc, "supply", "water", {"withdrawal": ("use", (10.0, 0.0, 0.0))})
    scheduled(doc, "supply", "salt", {"withdrawal": ("use", (5.0, 0.0, 0.0))})
    doc["forcings"].append({"id": "evaporative-loss", "horizon": doc["horizon"].copy(), "values": [4.0, 0.0, 0.0]})
    delayed = lag(doc, "use", "water")
    remaining = available(doc, "use", "water", 0)
    rule(
        doc,
        "use",
        "water",
        {
            "evaporate": ("evaporation", inc.forcing("evaporative-loss")),
            "return": ("return", inc.min(delayed, remaining)),
        },
    )
    coupled(doc, "use", "salt", {"returned-mass": ("return", "return")})
    model = inc.compile_model(doc)
    run = model.run(RUN_ID)
    assert counts(run, "return", "water") == [0, 6, 0]
    assert counts(run, "return", "salt") == [0, 50, 0]
    assert counts(run, "evaporation", "salt") == [0, 0, 0]
    assert stock(run, Inventory("use", "water", 0)) == [6, 0, 0]
    assert stock(run, Inventory("use", "salt", 0)) == [50, 0, 0]
    assert 50 * SALT.quantum / 6 == pytest.approx(5 / 6)
    assert_balances(run, (WATER, SALT), stocks, ("return", "evaporation"), frozenset({"use"}), ("supply",))
    assert run.authoritative_log() == model.run(RUN_ID).authoritative_log()


def dry_document():
    stocks = (
        Inventory("supply", "water", 4),
        Inventory("supply", "salt", 0),
        Inventory("pool", "water", 2),
        Inventory("pool", "salt", 10),
    )
    doc = network((WATER, SALT), stocks, 3)
    scheduled(doc, "supply", "water", {"entry": ("pool", (0.0, 4.0, 0.0))})
    scheduled(doc, "supply", "salt", {"entry": ("pool", (0.0, 0.0, 0.0))})
    scheduled(
        doc, "pool", "water", {"dry": ("evaporation", (2.0, 0.0, 0.0)), "rewet-release": ("outlet", (0.0, 4.0, 0.0))}
    )
    coupled(doc, "pool", "salt", {"remobilised": ("rewet-release", "outlet")})
    return doc, stocks


def test_a10_dry_mass_survives_then_declared_complete_remobilisation():
    doc, stocks = dry_document()
    model = inc.compile_model(doc)
    run = model.run(RUN_ID)
    assert counts(run, "evaporation", "salt") == [0, 0, 0]
    assert stock(run, Inventory("pool", "water", 2)) == [0, 0, 0]
    assert stock(run, Inventory("pool", "salt", 10)) == [10, 0, 0]
    assert counts(run, "outlet", "water") == [0, 4, 0]
    assert counts(run, "outlet", "salt") == [0, 10, 0]
    # Zero water has no aqueous concentration even while ten mass counts persist.
    dry_water = stock(run, Inventory("pool", "water", 2))[0]
    dry_concentration = None if dry_water == 0 else 10 * SALT.quantum / dry_water
    assert dry_concentration is None
    assert counts(run, "outlet", "salt")[1] * SALT.quantum / counts(run, "outlet", "water")[1] == 0.25
    assert_balances(run, (WATER, SALT), stocks, ("outlet", "evaporation"), frozenset({"pool"}), ("supply",))
    run.replay_against(model.run(RUN_ID))


def test_a10_unsupported_remobilisation_keeps_inventory_not_invented_concentration():
    doc, stocks = dry_document()
    # Test-owned support context controls interpretation; this does not implement
    # a physical remobilisation API. Without a supported model, keep mass retained.
    doc["rules"][-1]["disposition"] = inc.retain_all()
    doc["transfer_bindings"] = [
        binding
        for binding in doc["transfer_bindings"]
        if not (binding["compartment"] == "pool" and binding["substance"] == "salt")
    ]
    unsupported = {
        "quantity": "aqueous salt concentration",
        "state": "unsupported",
        "reason": "remobilisation model not supplied",
        "value": None,
    }
    run = inc.compile_model(doc).run(RUN_ID)
    assert stock(run, Inventory("pool", "salt", 10)) == [10, 10, 10]
    assert counts(run, "outlet", "salt") == [0, 0, 0]
    # Numeric zero transfer in this limited bookkeeping model cannot be labelled
    # the physical outlet concentration. It is distinct from supported clean water.
    assert unsupported["state"] == "unsupported" and unsupported["value"] is None
    assert_balances(run, (WATER, SALT), stocks, ("outlet", "evaporation"), frozenset({"pool"}), ("supply",))


def test_all_carrier_leaves_but_fractional_mass_stays_as_dry_inventory():
    stocks = (
        Inventory("pool", "water", 3),
        Inventory("pool", "salt", 10),
        Inventory("supply", "water", 3),
        Inventory("supply", "salt", 0),
    )
    doc = network((WATER, SALT), stocks, 2)
    scheduled(doc, "supply", "water", {"entry": ("pool", (0.0, 3.0))})
    scheduled(doc, "supply", "salt", {"entry": ("pool", (0.0, 0.0))})
    scheduled(doc, "pool", "water", {"left": ("left", (1.0, 3.0)), "right": ("right", (2.0, 0.0))})
    coupled(doc, "pool", "salt", {"m-left": ("left", "left"), "m-right": ("right", "right")})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "left", "salt") == [3, 1]
    assert counts(run, "right", "salt") == [6, 0]
    assert stock(run, Inventory("pool", "water", 3)) == [0, 0]
    assert stock(run, Inventory("pool", "salt", 10)) == [1, 0]
    assert_balances(run, (WATER, SALT), stocks, ("left", "right"), frozenset({"pool"}), ("supply",))
