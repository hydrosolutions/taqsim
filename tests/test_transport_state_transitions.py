"""Delayed inventory transitions preserve carrier, mass, and evidence identity."""

from decimal import Decimal
from enum import StrEnum

import pytest

from taqsim import WaterVolume
from taqsim.constituents import (
    Composition,
    ConservativeTransport,
    ConstituentInput,
    DomainSupport,
    ExchangeReplacement,
    InputMetadata,
    Mass,
    ProcessAccount,
    Remobilisation,
)
from taqsim.physical_results import BalanceState, QualityState
from taqsim.vocabulary import EvaporateThenRelease, Release, TravelDelay
from tests import interval_volume, make_water_system
from tests.test_transport_support import CHLORIDE
from tests.test_transport_water_authority import RULES


class Evidence(StrEnum):
    KNOWN = "known"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


class AccountEvidence(StrEnum):
    MATCHED = "matched"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


@pytest.mark.parametrize("layout", ["dry_first", "late_dry", "wet_only"])
@pytest.mark.parametrize("remobilisation", list(Remobilisation))
@pytest.mark.parametrize("evidence", list(Evidence))
@pytest.mark.parametrize("account_evidence", list(AccountEvidence))
def test_delayed_inventory_and_evidence_transition_matrix(layout, remobilisation, evidence, account_evidence):
    water, mass, first_wet, terms = {
        "dry_first": ([0.0, 4.0, 0.0, 0.0, 0.0], [1, 2, 0, 0, 0], 1, (1000, 3000, 2000, 0)),
        "late_dry": ([4.0, 0.0, 4.0, 0.0, 0.0], [2, 1, 2, 0, 0], 0, (2000, 3000, 1000, 0)),
        "wet_only": ([4.0, 0.0, 0.0, 0.0, 0.0], [2, 0, 0, 0, 0], 0, (2000, 2000, 0, 0)),
    }[layout]
    chemistry = []
    for step, value in enumerate(mass):
        item = ConstituentInput("chloride", mass=Mass(value))
        if step == first_wet:
            if evidence is Evidence.MISSING:
                item = ConstituentInput("chloride")
            elif evidence is Evidence.UNSUPPORTED:
                item = ConstituentInput(
                    "chloride",
                    mass=Mass(value),
                    metadata=InputMetadata(
                        source="boundary-model",
                        version="1",
                        support=DomainSupport.UNSUPPORTED,
                        reason="outside supplied model domain",
                    ),
                )
        chemistry.append(Composition((item,)))
    start, end, incoming, outgoing = terms
    if account_evidence is AccountEvidence.FAILED:
        end += 1000
    if account_evidence is AccountEvidence.INCOMPLETE:
        start = None
    account = ProcessAccount(
        "transit-model",
        "transit",
        1,
        "chloride",
        start,
        end,
        incoming,
        outgoing,
        InputMetadata(source="transit-model", version="1"),
    )
    model = make_water_system(5, "1 m3")
    model.source("supply", interval_volume(water))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(2))
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={"supply": tuple(chemistry)},
            remobilisation={"transit": remobilisation},
            process_accounts=(account,),
        )
    )
    result = model.build().run(bytes(16)).physical
    for step in range(5):
        delivered = result.sample("outlet", step, view="incoming")
        assert delivered.water_count == (int(water[step - 2]) if step >= 2 else 0)
        if delivered.water_count == 0:
            assert delivered.mass_counts["chloride"] == 0
        stock = result.sample("transit", step, view="storage")
        dry = result.metadata["dry_inventory_counts"]["transit"][step]["chloride"]
        if stock.water_count and dry:
            assert result.concentration("transit", step, "chloride", view="storage") is None
    departure = first_wet + 2
    expected_mass = 3 if layout == "dry_first" and remobilisation is Remobilisation.COMPLETE else 2
    if evidence is Evidence.MISSING:
        assert result.amount("outlet", departure, "chloride", view="incoming") is None
        expected_support = QualityState.MISSING
    else:
        assert result.amount("outlet", departure, "chloride", view="incoming") == expected_mass
        expected_support = QualityState.SUPPORTED
        if evidence is Evidence.UNSUPPORTED or account_evidence is not AccountEvidence.MATCHED:
            expected_support = QualityState.UNSUPPORTED
        if layout == "dry_first" and remobilisation is Remobilisation.UNRESOLVED:
            expected_support = QualityState.UNSUPPORTED
    assert result.quality_state("outlet", departure, "chloride", view="incoming") is expected_support
    if layout == "late_dry":
        assert result.amount("outlet", 4, "chloride", view="incoming") == (
            3 if remobilisation is Remobilisation.COMPLETE else 2
        )
        expected = (
            QualityState.SUPPORTED
            if (
                remobilisation is Remobilisation.COMPLETE
                and account_evidence is AccountEvidence.MATCHED
                and evidence is not Evidence.MISSING
            )
            else QualityState.UNSUPPORTED
        )
        assert result.quality_state("outlet", 4, "chloride", view="incoming") is expected
    expected_dry = 1 if layout != "wet_only" and remobilisation is Remobilisation.UNRESOLVED else 0
    assert result.amount("transit", 4, "chloride", view="storage") == expected_dry
    if evidence is Evidence.MISSING:
        assert any(item.status is BalanceState.INCOMPLETE for item in result.basin_balances)
    else:
        assert all(item.residual == 0 for item in (*result.balances, *result.basin_balances))


@pytest.mark.parametrize(
    "rule", [Release(WaterVolume(4, "m3")), EvaporateThenRelease(WaterVolume(1, "m3"), WaterVolume(4, "m3"))]
)
def test_replacement_preserves_requested_branch_and_other_rule_plan_fields(rule):
    model = make_water_system(1, "1 m3")
    for name in ("old", "replacement"):
        model.source(name, interval_volume([4.0]))
        model.reach(name + "-store", name, name + "-outlet", rule=rule)
        branches = ((name + "-store", "release"),)
        if isinstance(rule, EvaporateThenRelease):
            branches += ((name + "-store", "evaporation"),)
        model.process_exchange(name, sources=(name,), branches=branches)
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={
                name: (Composition((ConstituentInput("chloride", mass=Mass(2)),)),) for name in ("old", "replacement")
            },
            replacements=(ExchangeReplacement("old", "replacement"),),
        )
    )
    result = model.build().run(bytes(16)).physical
    assert result.amount("old-outlet", 0, view="incoming") == 0
    expected = 3 if isinstance(rule, EvaporateThenRelease) else 4
    assert result.amount("replacement-outlet", 0, view="incoming") == expected
    assert Decimal(result.metadata["water_deficit_m3"]["replacement-store"][0]) == 4 - expected


@pytest.mark.parametrize("rule", RULES)
def test_replacement_retains_every_native_rule_physical_plan_field(rule):
    model = make_water_system(3, "1 L")
    for name in ("old", "replacement"):
        model.source(name, interval_volume([100.0, 100.0, 100.0]))
        model.reach(name + "-store", name, name + "-outlet", rule=rule)
    chemistry = {
        name: (Composition((ConstituentInput("chloride", mass=Mass(2)),)),) * 3 for name in ("old", "replacement")
    }
    model.configure_transport(ConservativeTransport((CHLORIDE,), boundaries=chemistry))
    original = model.build().run(bytes(16)).physical
    before = original.metadata["topology"]
    for name in ("old", "replacement"):
        model.process_exchange(
            name,
            sources=(name,),
            branches=tuple((name + "-store", b["label"]) for b in before["branches"][name + "-store"]),
        )
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,), boundaries=chemistry, replacements=(ExchangeReplacement("old", "replacement"),)
        )
    )
    changed = model.build().run(bytes(16)).physical
    after = changed.metadata["topology"]
    for field in ("branches", "mixing", "requested", "requested_branches", "delays"):
        assert after[field] == before[field]
    assert all(changed.amount("old-store", step, view="outgoing") == 0 for step in range(3))
