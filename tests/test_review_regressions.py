"""Independent review regressions through the public physical path."""

from decimal import Decimal

import pytest

from taqsim import WaterVolume
from taqsim.constituents import (
    Composition,
    ConservativeTransport,
    ConstituentInput,
    DomainSupport,
    InputMetadata,
    Mass,
    ProcessAccount,
)
from taqsim.physical_results import QualityState
from taqsim.vocabulary import EvaporateThenRelease, TravelDelay
from tests import interval_volume, make_water_system
from tests.test_transport_support import CHLORIDE


def chemistry(mass, unsupported=False):
    metadata = (
        InputMetadata(source="model", version="1", support=DomainSupport.UNSUPPORTED, reason="outside domain")
        if unsupported
        else InputMetadata()
    )
    return Composition((ConstituentInput("chloride", mass=Mass(mass), metadata=metadata),))


def test_zero_water_delayed_load_is_retained_not_advected():
    model = make_water_system(2, "1 m3")
    model.source("supply", interval_volume([0.0, 0.0]))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    model.configure_transport(ConservativeTransport((CHLORIDE,), boundaries={"supply": (chemistry(1), chemistry(0))}))
    result = model.build().run(bytes(16)).physical
    assert result.amount("outlet", 1, view="incoming") == 0
    assert result.amount("outlet", 1, "chloride", view="incoming") == 0
    assert result.amount("transit", 1, "chloride", view="storage") == 1


@pytest.mark.parametrize("unsupported_step", [0, 1])
def test_matching_delay_account_preserves_departing_cohort_support(unsupported_step):
    model = make_water_system(2, "1 m3")
    model.source("supply", interval_volume([4.0, 4.0]))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    account = ProcessAccount("process", "transit", 1, "water", 4, 4, 4, 4, InputMetadata(source="model", version="1"))
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={"supply": tuple(chemistry(2, i == unsupported_step) for i in range(2))},
            process_accounts=(account,),
        )
    )
    result = model.build().run(bytes(16)).physical
    assert all(item["difference_counts"] == 0 for item in result.metadata["process_account_exchange_checks"])
    expected = QualityState.UNSUPPORTED if unsupported_step == 0 else QualityState.SUPPORTED
    assert result.sample("outlet", 1, view="incoming").quality["chloride"] is expected


def test_release_deficit_does_not_count_evaporation_as_delivery():
    model = make_water_system(1, "1 m3")
    model.source("supply", interval_volume([10.0]))
    model.reach("river", "supply", "outlet", rule=EvaporateThenRelease(WaterVolume(4, "m3"), WaterVolume(10, "m3")))
    model.configure_transport(ConservativeTransport((CHLORIDE,), boundaries={"supply": (chemistry(5),)}))
    result = model.build().run(bytes(16)).physical
    assert result.amount("outlet", 0, view="incoming") == 6
    assert Decimal(result.metadata["water_deficit_m3"]["river"][0]) == 4


def test_water_cannot_be_registered_as_constituent_identity():
    from taqsim.constituents import Constituent

    model = make_water_system(1, "1 m3")
    model.source("supply", interval_volume([10.0]))
    model.reach("river", "supply", "outlet")
    with pytest.raises(ValueError, match="(?i)(water|reserved)"):
        model.configure_transport(
            ConservativeTransport(
                (Constituent("water", "NaCl", "as NaCl", Decimal(1)),),
                boundaries={"supply": (Composition((ConstituentInput("water", mass=Mass(5)),)),)},
            )
        )
        model.build().run(bytes(16))


@pytest.mark.parametrize(
    "remobilisation,expected_mass,expected_state",
    [
        ("complete", 3, QualityState.SUPPORTED),
        ("unresolved", 2, QualityState.UNSUPPORTED),
    ],
)
def test_delayed_dry_load_rewetting_keeps_explicit_mobile_and_dry_inventory(
    remobilisation, expected_mass, expected_state
):
    from taqsim.constituents import Remobilisation

    model = make_water_system(3, "1 m3")
    model.source("supply", interval_volume([0.0, 4.0, 0.0]))
    model.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    model.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={"supply": (chemistry(1), chemistry(2), chemistry(0))},
            remobilisation={"transit": Remobilisation(remobilisation)},
        )
    )
    result = model.build().run(bytes(16)).physical
    assert result.amount("outlet", 1, "chloride", view="incoming") == 0
    assert result.amount("outlet", 2, "chloride", view="incoming") == expected_mass
    assert result.amount("transit", 2, "chloride", view="storage") == 3 - expected_mass
    assert result.sample("outlet", 2, view="incoming").quality["chloride"] is expected_state
    assert all(balance.residual == 0 for balance in (*result.balances, *result.basin_balances))
