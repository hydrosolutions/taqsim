"""Realised carrier transfers → exact conserved dependent counts, through the public binding."""

from copy import deepcopy

import incidence as inc
import numpy as np
import pytest

from taqsim.vocabulary import subtract
from tests.transport.model_cases import (
    RUN_ID,
    WATER,
    Inventory,
    Quantity,
    assert_balances,
    assert_pool_balances,
    available,
    counts,
    coupled,
    mixed_pool,
    network,
    rule,
    scheduled,
    stock,
)


@pytest.mark.parametrize(
    "initial,inputs,requests,expected",
    [
        pytest.param((0, 0), ((10, 80), (5, 10)), (15,), ((15, 90),), id="A1-mixing"),
        pytest.param((0, 0), ((10, 80), (5, 10)), (6, 9), ((6, 36), (9, 54)), id="A2-branches"),
        pytest.param((20, 40), ((10, 40),), (40,), ((30, 80),), id="A3-actual-not-requested"),
        pytest.param((20, 40), ((10, 40),), (15,), ((15, 40),), id="A4-retention"),
        pytest.param((0, 0), ((15, 90),), (10, 5), ((10, 60), (5, 30)), id="A7-capacity-overflow"),
    ],
)
def test_realised_mixed_pool(initial, inputs, requests, expected):
    doc, stocks = mixed_pool(initial, inputs)
    pool = available(doc, "pool", "water", initial[0])
    branches = {
        f"branch-{i}": (f"outlet-{i}", inc.min(inc.literal(request), pool)) for i, request in enumerate(requests)
    }
    rule(doc, "pool", "water", branches)
    coupled(doc, "pool", "salt", {f"mass-{i}": (f"branch-{i}", f"outlet-{i}") for i in range(len(requests))})
    run = inc.compile_model(doc).run(RUN_ID)
    actual = [
        (counts(run, f"outlet-{i}", "water")[0], counts(run, f"outlet-{i}", "salt")[0]) for i in range(len(requests))
    ]
    np.testing.assert_array_equal(actual, expected)
    assert_pool_balances(run, doc, stocks)
    assert stock(run, Inventory("pool", "water", initial[0])) == [
        initial[0] + sum(p[0] for p in inputs) - sum(v[0] for v in expected)
    ]
    assert stock(run, Inventory("pool", "salt", initial[1])) == [
        initial[1] + sum(p[1] for p in inputs) - sum(v[1] for v in expected)
    ]


def zero_branch_document():
    doc, stocks = mixed_pool((1, 10), ())
    rule(doc, "pool", "water", {"left": ("left", inc.literal(0.5)), "right": ("right", inc.literal(0.5))})
    coupled(doc, "pool", "salt", {"salt-left": ("left", "left"), "salt-right": ("right", "right")})
    return doc, stocks


def test_zero_realised_carrier_cannot_advect_mass():
    doc, stocks = zero_branch_document()
    run = inc.compile_model(doc).run(RUN_ID)
    for target in ("left", "right"):
        assert counts(run, target, "water") == [0]
        assert counts(run, target, "salt") == [0]
        assert list(run.transfer_series(target, "salt", direction="incoming").presence) == ["present"]
    assert stock(run, Inventory("pool", "salt", 10)) == [10]
    assert_pool_balances(run, doc, stocks)


def test_independent_partitions_are_not_a_coupling_declaration():
    # Old-code discriminant: the same legacy public model still conserves separately,
    # but is not a supported declaration of advective transport.
    doc, stocks = zero_branch_document()
    doc["rules"].pop()
    doc["transfer_bindings"] = [b for b in doc["transfer_bindings"] if b["substance"] != "salt"]
    rule(doc, "pool", "salt", {"left": ("left", inc.literal(0.5)), "right": ("right", inc.literal(0.5))})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "left", "water") == [0]
    assert counts(run, "left", "salt") == [5]
    assert_pool_balances(run, doc, stocks)


def test_nonround_repeated_allocations_keep_remainders_and_independent_constituents():
    tracer = Quantity("tracer", "kg", 1.0)
    quantities = (WATER, Quantity("salt", "kg", 1.0), tracer)
    stocks = tuple(
        Inventory(node, s, count)
        for node, values in [("pool", (2, 4, 11)), ("supply", (5, 9, 12))]
        for s, count in zip(("water", "salt", "tracer"), values, strict=True)
    )
    doc = network(quantities, stocks, 3)
    for s, values in [("water", (3.0, 2.0, 0.0)), ("salt", (6.0, 3.0, 0.0)), ("tracer", (7.0, 5.0, 0.0))]:
        scheduled(doc, "supply", s, {"entry": ("pool", values)})
    scheduled(doc, "pool", "water", {"left": ("left", (1.5, 2.4, 1.0)), "right": ("right", (2.5, 1.6, 0.0))})
    for s in ("salt", "tracer"):
        coupled(doc, "pool", s, {"mass-left": ("left", "left"), "mass-right": ("right", "right")})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "left", "water") == [1, 2, 1]
    assert counts(run, "right", "water") == [2, 1, 0]
    assert counts(run, "left", "salt") == [2, 3, 3]
    assert counts(run, "right", "salt") == [4, 1, 0]
    assert counts(run, "left", "tracer") == [3, 6, 4]
    assert counts(run, "right", "tracer") == [7, 3, 0]
    assert stock(run, Inventory("pool", "water", 2)) == [2, 1, 0]
    assert stock(run, Inventory("pool", "salt", 4)) == [4, 3, 0]
    assert stock(run, Inventory("pool", "tracer", 11)) == [8, 4, 0]
    assert_balances(run, quantities, stocks, ("left", "right"), frozenset({"pool"}), ("supply",))


def test_same_destination_preserves_sum_of_distinct_branch_allocations():
    doc, stocks = mixed_pool((3, 5), ())
    rule(doc, "pool", "water", {"first": ("shared", inc.literal(1.5)), "second": ("shared", inc.literal(1.5))})
    coupled(doc, "pool", "salt", {"salt-first": ("first", "shared"), "salt-second": ("second", "shared")})
    run = inc.compile_model(doc).run(RUN_ID)
    # Per-branch floors are 5*1//3=1 each; a merged floor would incorrectly give 3.
    # This case checks aggregate readback only; it does not claim branch readback.
    assert counts(run, "shared", "water") == [2]
    assert counts(run, "shared", "salt") == [2]
    assert stock(run, Inventory("pool", "salt", 5)) == [3]
    assert_pool_balances(run, doc, stocks)


def test_subset_mapping_retains_unadvected_mass_instead_of_removing_it_with_water():
    doc, stocks = mixed_pool((10, 50), ())
    rule(doc, "pool", "water", {"water-only": ("evaporation", inc.literal(2)), "release": ("outlet", inc.literal(4))})
    coupled(doc, "pool", "salt", {"advected": ("release", "outlet")})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "evaporation", "salt") == [0]
    assert counts(run, "outlet", "salt") == [20]
    assert stock(run, Inventory("pool", "water", 10)) == [4]
    assert stock(run, Inventory("pool", "salt", 50)) == [30]
    assert_pool_balances(run, doc, stocks)


@pytest.mark.parametrize(
    "mutation",
    [
        "negative",
        "nonfinite",
        "overallocation",
        "missing-destination",
        "duplicate-carrier",
        "self-carrier",
        "wrong-destination",
    ],
)
def test_coupled_invalid_models_or_runs_refuse(mutation):
    doc, _ = zero_branch_document()
    if mutation in {"negative", "nonfinite"}:
        doc["initial_stocks"][0]["amounts"][0]["amount"] = -1 if mutation == "negative" else float("nan")
    elif mutation == "overallocation":
        doc["rules"][0]["disposition"]["partition"]["branches"][0]["expression"] = inc.literal(2)
    elif mutation == "missing-destination":
        doc["transfer_bindings"].pop()
    elif mutation == "duplicate-carrier":
        doc["rules"][-1]["disposition"]["partition"]["branches"][1]["carrier_branch"] = "left"
    elif mutation == "self-carrier":
        doc["rules"][-1]["disposition"]["partition"]["carrier"] = "salt"
    else:
        doc["transfer_bindings"][-1]["destination"] = "left"
    with pytest.raises(ValueError):
        inc.compile_model(doc).run(RUN_ID)


def test_model_identity_binds_coupling_mapping_precision_and_initial_input():
    doc, _ = zero_branch_document()
    model = inc.compile_model(doc)
    run = model.run(RUN_ID)
    assert run.authoritative_log() == model.run(RUN_ID).authoritative_log()
    run.replay_against(model.run(RUN_ID))
    for change in ("mapping", "precision", "initial"):
        changed = deepcopy(doc)
        if change == "mapping":
            changed["rules"][-1]["disposition"]["partition"]["branches"].pop()
            changed["transfer_bindings"] = [
                binding for binding in changed["transfer_bindings"] if binding["branch"] != "salt-right"
            ]
            # Different coupling assumption, same zero-transfer observations.
        elif change == "precision":
            changed["units"][1]["quantum"] = 0.05
        else:
            changed["initial_stocks"][0]["amounts"][1]["amount"] = 2
        other = inc.compile_model(changed).run(RUN_ID)
        assert other.model_digest != run.model_digest
        with pytest.raises(ValueError):
            run.replay_against(other)


def test_large_merged_dependent_counts_never_reenter_through_float_amounts():
    fine_mass = Quantity("salt", "kg", 1e-6)
    each = 2_197_111_022_144_419
    total = 4 * each
    sources = tuple(f"supply-{index}" for index in range(4))
    stocks = tuple(Inventory(node, s, value) for node in sources for s, value in (("water", 1), ("salt", each))) + (
        Inventory("pool", "water", 0),
        Inventory("pool", "salt", 0),
    )
    doc = network((WATER, fine_mass), stocks)
    for node in sources:
        scheduled(doc, node, "water", {"entry": ("pool", (1.0,))})
        scheduled(doc, node, "salt", {"entry": ("pool", (each * fine_mass.quantum,))})
    rule(doc, "pool", "water", {"first": ("left", inc.literal(1)), "rest": ("right", inc.literal(3))})
    coupled(doc, "pool", "salt", {"mass-first": ("first", "left"), "mass-rest": ("rest", "right")})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "pool", "salt") == [total]
    assert counts(run, "left", "salt") == [each]
    assert counts(run, "right", "salt") == [3 * each]
    public = run.transfer_series("pool", "salt", direction="incoming")
    assert int(public.values[0] / fine_mass.quantum) != total
    assert_balances(run, (WATER, fine_mass), stocks, ("left", "right"), frozenset({"pool"}), sources)


@pytest.mark.parametrize("amount,quantum", [(float(2**53 + 2), 1.0), (1.0, 0.0)])
def test_unrepresentable_generic_count_domain_refuses(amount, quantum):
    doc, _ = zero_branch_document()
    doc["initial_stocks"][0]["amounts"][0]["amount"] = amount
    doc["units"][0]["quantum"] = quantum
    with pytest.raises(ValueError):
        inc.compile_model(doc).run(RUN_ID)


def test_capacity_overflow_is_the_residual_of_the_available_mixed_pool():
    doc, stocks = mixed_pool((3, 30), ((12, 60),))
    pool = available(doc, "pool", "water", 3)
    delivery = inc.min(pool, inc.literal(10))
    overflow = inc.max(inc.literal(0), subtract(pool, delivery))
    rule(doc, "pool", "water", {"delivery": ("outlet", delivery), "overflow": ("spill", overflow)})
    coupled(doc, "pool", "salt", {"delivered": ("delivery", "outlet"), "spilled": ("overflow", "spill")})
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "outlet", "water") == [10]
    assert counts(run, "outlet", "salt") == [60]
    assert counts(run, "spill", "water") == [5]
    assert counts(run, "spill", "salt") == [30]
    assert_pool_balances(run, doc, stocks)
