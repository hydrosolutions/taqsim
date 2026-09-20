"""Repair-edge public-path review regressions."""

from taqsim import WaterVolume
from taqsim.constituents import (
    ConservativeTransport,
    ExchangeReplacement,
    InputMetadata,
    ProcessAccount,
    Remobilisation,
)
from taqsim.physical_results import QualityState
from taqsim.vocabulary import Release, TravelDelay
from tests import interval_volume, make_water_system
from tests.test_review_regressions import chemistry
from tests.test_transport_support import CHLORIDE


def test_replace_requested_release_preserves_delivery_identity():
    model = make_water_system(1, "1 m3")
    for name in ("old", "new"):
        model.source(name, interval_volume([4.0]))
        model.reach(name + "-reach", name, name + "-outlet", rule=Release(WaterVolume(4, "m3")))
        model.process_exchange(name + "-process", sources=(name,), branches=((name + "-reach", "release"),))
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={name: (chemistry(2),) for name in ("old", "new")},
            replacements=(ExchangeReplacement("old-process", "new-process"),),
        )
    )
    result = model.build().run(bytes(16)).physical
    assert result.amount("old-outlet", 0, view="incoming") == 0
    assert result.amount("new-outlet", 0, view="incoming") == 4


def test_salt_only_deposit_does_not_become_supported_aqueous_storage():
    model = make_water_system(3, "1 m3")
    model.source("supply", interval_volume([4.0, 0.0, 0.0]))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(2))
    model.configure_transport(
        ConservativeTransport((CHLORIDE,), boundaries={"supply": (chemistry(2), chemistry(1), chemistry(0))})
    )
    result = model.build().run(bytes(16)).physical
    assert result.amount("transit", 1, view="storage") == 4
    assert result.amount("transit", 1, "chloride", view="storage") == 3
    assert result.metadata["dry_inventory_counts"]["transit"][1]["chloride"] == 1000
    assert result.concentration("transit", 1, "chloride", view="storage") is None


def test_failed_deposit_account_remains_invalid_after_remobilisation():
    model = make_water_system(3, "1 m3")
    model.source("supply", interval_volume([0.0, 4.0, 0.0]))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    account = ProcessAccount(
        "model", "transit", 0, "chloride", 0, 2000, 2000, 0, InputMetadata(source="specialist", version="1")
    )
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={"supply": (chemistry(1), chemistry(2), chemistry(0))},
            process_accounts=(account,),
            remobilisation={"transit": Remobilisation.COMPLETE},
        )
    )
    result = model.build().run(bytes(16)).physical
    assert result.sample("transit", 0, view="storage").quality["chloride"] is QualityState.UNSUPPORTED
    assert result.amount("outlet", 2, "chloride", view="incoming") == 3
    assert result.sample("outlet", 2, view="incoming").quality["chloride"] is QualityState.UNSUPPORTED


def test_failed_account_during_hold_persists_on_older_wet_cohort():
    model = make_water_system(3, "1 m3")
    model.source("supply", interval_volume([4.0, 0.0, 0.0]))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(2))
    account = ProcessAccount(
        "model", "transit", 1, "chloride", 2000, 3000, 0, 0, InputMetadata(source="specialist", version="1")
    )
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,), boundaries={"supply": (chemistry(2), chemistry(0), chemistry(0))}, process_accounts=(account,)
        )
    )
    result = model.build().run(bytes(16)).physical
    assert result.sample("transit", 1, view="storage").quality["chloride"] is QualityState.UNSUPPORTED
    assert result.amount("outlet", 2, "chloride", view="incoming") == 2
    assert result.sample("outlet", 2, view="incoming").quality["chloride"] is QualityState.UNSUPPORTED
