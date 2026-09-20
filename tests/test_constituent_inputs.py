"""Typed chemistry declarations preserve units, support and exact mass counts."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from typing import Any, cast

import pytest

from taqsim.constituents import (
    Composition,
    Concentration,
    ConservativeTransport,
    Constituent,
    ConstituentInput,
    DomainSupport,
    ExchangeReplacement,
    InputMetadata,
    InputStatus,
    LocationMapping,
    Mass,
    MassCounts,
    MassRate,
    ProcessAccount,
    ProcessState,
    Remobilisation,
    RunMetadata,
    SoftwareIdentity,
    Uncertainty,
    mass_counts,
    mass_quantisation_remainder,
)

SALT = Constituent("salt", "dissolved salt", "as salt", Decimal("0.001"))


def test_units_and_rate_integrate_once():
    assert Mass("1200", "g").kg == Decimal("1.2")
    assert Concentration("500", "mg/l").kg_per_m3 == Decimal("0.5")
    assert MassRate("1000", "g/s").kg_per_second == 1
    concentration = Composition((ConstituentInput("salt", concentration=Concentration("500", "mg/l")),))
    rate = Composition((ConstituentInput("salt", mass=MassRate("5")),))
    total = Composition((ConstituentInput("salt", mass=Mass("432000")),))
    for value in (concentration, rate, total):
        assert mass_counts(value, SALT, 864000, Decimal(1), 86400) == 432000000


def test_authoritative_counts_and_decimal_inputs_never_pass_float():
    count = 10**80 + 17
    direct = Composition((ConstituentInput("salt", mass=MassCounts(count)),))
    concentration = Composition((ConstituentInput("salt", concentration=Concentration("0.001")),))
    with localcontext() as context:
        context.prec = 6
        assert mass_counts(direct, SALT, 1, Decimal(1), 1) == count
        assert mass_counts(concentration, SALT, count, Decimal(1), 1) == count
    assert Mass("0.123456789123456789").value == Decimal("0.123456789123456789")


def test_missing_is_not_zero_and_quantisation_is_floor():
    assert mass_counts(Composition(), SALT, 10, Decimal(1), 1) is None
    assert mass_counts(Composition((ConstituentInput("salt"),)), SALT, 10, Decimal(1), 1) is None
    assert mass_counts(Composition((ConstituentInput("salt", mass=Mass(0)),)), SALT, 10, Decimal(1), 1) == 0
    assert mass_counts(Composition((ConstituentInput("salt", mass=Mass("0.0019")),)), SALT, 10, Decimal(1), 1) == 1


@pytest.mark.parametrize("factory", [Mass, Concentration, MassRate])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-0.1", "-Infinity"])
def test_nonnegative_finite_domain(factory, value):
    with pytest.raises(ValueError):
        factory(value)


@pytest.mark.parametrize("factory,unit", [(Mass, "m3"), (Concentration, "pH"), (MassRate, "kg")])
def test_invalid_units(factory, unit):
    with pytest.raises(ValueError):
        factory(1, unit)


def test_conflicts_duplicate_and_missing_status():
    with pytest.raises(ValueError, match="conflicting"):
        ConstituentInput("salt", mass=Mass(1), concentration=Concentration(1))
    with pytest.raises(ValueError, match="duplicate"):
        Composition((ConstituentInput("salt"), ConstituentInput("salt")))
    with pytest.raises(ValueError, match="missing"):
        ConstituentInput("salt", mass=Mass(1), metadata=InputMetadata(status=InputStatus.MISSING))
    with pytest.raises(ValueError, match="duplicate"):
        ConservativeTransport((SALT, SALT))


@pytest.mark.parametrize(
    "entry",
    [
        ConstituentInput("other"),
        ConstituentInput("salt", chemical_form="Na"),
        ConstituentInput("salt", reporting_basis="as sodium"),
    ],
)
def test_invalid_chemical_identity_or_basis(entry):
    with pytest.raises(ValueError):
        ConservativeTransport((SALT,), boundaries={"source": (Composition((entry,)),)})


def test_configuration_snapshots_mutable_inputs_and_metadata():
    provenance = ["survey"]
    metadata = InputMetadata(
        source="observation",
        version="2",
        status=InputStatus.CORRECTED,
        provenance=cast(Any, provenance),
        uncertainty=Uncertainty("laboratory interval", "lab"),
        owner="groundwater",
    )
    composition = Composition((ConstituentInput("salt", mass=Mass(1), metadata=metadata),))
    series = [composition]
    initial = {"reach": composition}
    boundaries = {"source": series}
    config = ConservativeTransport(
        (SALT,),
        initial,
        cast(Any, boundaries),
        {"reach": Remobilisation.COMPLETE},
        RunMetadata(scenario="synthetic", reference="baseline", assumptions=("complete mixing",)),
    )
    initial.clear()
    series.clear()
    provenance.clear()
    assert config.initial["reach"] == composition
    assert config.boundaries["source"] == (composition,)
    assert metadata.provenance == ("survey",)
    assert metadata.owner == "groundwater"
    with pytest.raises(TypeError):
        cast(Any, config.initial)["other"] = composition
    with pytest.raises(FrozenInstanceError):
        cast(Any, config).metadata = RunMetadata()


def test_network_and_time_binding():
    config = ConservativeTransport((SALT,), initial={"reach": Composition()}, boundaries={"source": (Composition(),)})
    config.validate_network(("reach",), ("source",), 1)
    for reaches, sources, periods in [((), ("source",), 1), (("reach",), (), 1), (("reach",), ("source",), 2)]:
        with pytest.raises(ValueError):
            config.validate_network(reaches, sources, periods)


def test_initial_inventory_cannot_be_a_rate_or_concentration():
    for entry in (ConstituentInput("salt", mass=MassRate(1)), ConstituentInput("salt", concentration=Concentration(1))):
        with pytest.raises(ValueError, match="initial inventory"):
            ConservativeTransport((SALT,), initial={"reach": Composition((entry,))})


def test_support_requires_reason_and_replacements_are_unambiguous():
    with pytest.raises(ValueError, match="reason"):
        InputMetadata(support=DomainSupport.UNSUPPORTED)
    with pytest.raises(ValueError):
        ExchangeReplacement("same", "same")
    with pytest.raises(ValueError, match="cyclic"):
        ConservativeTransport(
            (SALT,), replacements=(ExchangeReplacement("old", "new"), ExchangeReplacement("new", "old"))
        )


def test_imported_signed_states_are_not_conservative_constituents():
    metadata = InputMetadata(source="hydraulic-model", version="3", status=InputStatus.MODELLED)
    state = ProcessState("stage", "reach", 0, Decimal("-1.25"), "m", "local datum", metadata)
    config = ConservativeTransport((SALT,), process_states=(state,))
    config.validate_network(("reach",), (), 1)
    assert config.process_states[0].value == Decimal("-1.25")
    with pytest.raises(ValueError, match="finite"):
        ProcessState("stage", "reach", 0, Decimal("NaN"), "m", "local datum", metadata)
    with pytest.raises(ValueError, match="source and version"):
        ProcessState("stage", "reach", 0, Decimal(0), "m", "local datum", InputMetadata())
    with pytest.raises(ValueError, match="duplicate"):
        ConservativeTransport((SALT,), process_states=(state, state))


def test_zero_water_unknown_chemistry_cannot_advect_unknown_mass():
    assert mass_counts(Composition(), SALT, 0, Decimal(1), 1) == 0
    assert mass_counts(Composition((ConstituentInput("salt"),)), SALT, 0, Decimal(1), 1) == 0
    explicit = Composition((ConstituentInput("salt", mass=Mass(1)),))
    assert mass_counts(explicit, SALT, 0, Decimal(1), 1) == 1000


def test_input_quantisation_remainder_is_disclosed_exactly():
    from fractions import Fraction

    composition = Composition((ConstituentInput("salt", mass=Mass("0.0019")),))
    assert mass_quantisation_remainder(composition, SALT, 1, Decimal(1), 1) == Fraction(9, 10)
    assert mass_quantisation_remainder(Composition(), SALT, 1, Decimal(1), 1) is None
    assert mass_quantisation_remainder(Composition(), SALT, 0, Decimal(1), 1) == 0


def test_supplied_location_mapping_and_revision_identity():
    mapping = LocationMapping("outlet", "river-body", "compliance-point", "map-v2")
    identities = (SoftwareIdentity("groundwater", "1.4", "abc123"),)
    config = ConservativeTransport(
        (SALT,), location_mappings=(mapping,), metadata=RunMetadata(software_identities=identities)
    )
    config.validate_network(("reach",), ("source",), 1, ("outlet",))
    with pytest.raises(ValueError, match="unknown mapped"):
        config.validate_network(("reach",), ("source",), 1)
    with pytest.raises(ValueError, match="duplicate mapped"):
        ConservativeTransport((SALT,), location_mappings=(mapping, mapping))
    with pytest.raises(ValueError, match="duplicate software"):
        RunMetadata(software_identities=identities + identities)


def test_predecessor_state_has_separate_exact_interval_contract():
    metadata = InputMetadata(source="survey", version="1")
    state = ProcessState("stage", "source", -1, Decimal("-0.2"), "m", "datum", metadata)
    config = ConservativeTransport((SALT,), predecessor_states=(state,))
    config.validate_network(("reach",), ("source",), 1)
    with pytest.raises(ValueError, match="unknown predecessor"):
        config.validate_network(("reach",), (), 1)
    with pytest.raises(ValueError, match="non-negative"):
        ConservativeTransport((SALT,), process_states=(state,))
    with pytest.raises(ValueError, match="interval -1"):
        ConservativeTransport(
            (SALT,), predecessor_states=(ProcessState("stage", "source", 0, Decimal(0), "m", "datum", metadata),)
        )
    with pytest.raises(ValueError, match="duplicate imported"):
        ConservativeTransport((SALT,), predecessor_states=(state, state))
    with pytest.raises(ValueError, match="predecessor -1"):
        ProcessState("stage", "source", -2, Decimal(0), "m", "datum", metadata)


def test_supplied_process_accounts_are_exact_independent_diagnostics():
    metadata = InputMetadata(source="groundwater-model", version="2")
    account = ProcessAccount("groundwater", "source", 0, "salt", 10**30 + 1, None, 2, 3, metadata)
    config = ConservativeTransport((SALT,), process_accounts=(account,))
    config.validate_network(("reach",), ("source",), 1)
    assert config.process_accounts[0].start == 10**30 + 1
    assert config.process_accounts[0].end is None
    with pytest.raises(ValueError, match="duplicate process"):
        ConservativeTransport((SALT,), process_accounts=(account, account))
    with pytest.raises(ValueError, match="incompatible"):
        config.validate_network(("reach",), (), 1)
    with pytest.raises(ValueError, match="unregistered process"):
        ConservativeTransport(
            (SALT,), process_accounts=(ProcessAccount("owner", "source", 0, "unknown", 0, 0, 0, 0, metadata),)
        )
    with pytest.raises(ValueError, match="source and version"):
        ProcessAccount("owner", "source", 0, "salt", 0, 0, 0, 0, InputMetadata())
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="integer"):
            ProcessAccount("owner", "source", 0, "salt", cast(Any, invalid), 0, 0, 0, metadata)
