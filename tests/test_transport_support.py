"""Input support and imported specialist evidence through the real physical path."""

from decimal import Decimal
from fractions import Fraction

import pytest

from taqsim import WaterVolume
from taqsim.constituents import (
    Composition,
    Concentration,
    ConservativeTransport,
    Constituent,
    ConstituentInput,
    DomainSupport,
    InputMetadata,
    InputStatus,
    LocationMapping,
    Mass,
    ProcessAccount,
    ProcessState,
    RunMetadata,
    SoftwareIdentity,
    Uncertainty,
)
from taqsim.physical_results import BalanceState, QualityState
from taqsim.vocabulary import Hold, TravelDelay
from taqsim.water_system import WaterSystemRun
from tests import interval_volume, make_water_system

CHLORIDE = Constituent("chloride", "Cl-", "as chloride", Decimal("0.001"))
SULPHATE = Constituent("sulphate", "SO4--", "as sulphate", Decimal("0.001"))


def _system(initial_water=0, *, rule=None):
    system = make_water_system(1, "1 m3")
    system.source("supply", interval_volume([10.0]))
    system.reach("river", "supply", "outlet", initial_water=WaterVolume(initial_water, "m3"), rule=rule)
    return system


def test_unsupported_input_retains_numeric_mass_and_independent_constituent():
    system = _system()
    evidence = InputMetadata(
        source="external-process",
        version="2",
        support=DomainSupport.UNSUPPORTED,
        reason="outside calibrated salinity range",
        uncertainty=Uncertainty("unbounded outside range", "model-v2"),
    )
    composition = Composition(
        (
            ConstituentInput("chloride", concentration=Concentration("0.5"), metadata=evidence),
            ConstituentInput("sulphate", concentration=Concentration("0.2")),
        )
    )
    system.configure_transport(ConservativeTransport((CHLORIDE, SULPHATE), boundaries={"supply": (composition,)}))
    result = system.build().run(bytes(16)).physical
    sample = result.sample("outlet", 0, view="incoming")
    assert sample.mass_counts == {"chloride": 5000, "sulphate": 2000}
    assert sample.quality == {"chloride": QualityState.UNSUPPORTED, "sulphate": QualityState.SUPPORTED}
    assert result.amount("outlet", 0, view="incoming") == 10
    assert result.concentration("outlet", 0, "chloride", view="incoming") is None
    assert result.concentration("outlet", 0, "sulphate", view="incoming") == Fraction(1, 5)
    retained = result.metadata["inputs"]["boundaries"]["supply"][0]["entries"][0]["metadata"]
    assert retained["reason"] == evidence.reason
    assert evidence.uncertainty is not None
    assert retained["uncertainty"]["description"] == evidence.uncertainty.description


@pytest.mark.parametrize(
    "initial_water,expected_mass,quality",
    [
        (0, 5000, QualityState.SUPPORTED),
        (2, None, QualityState.MISSING),
    ],
)
def test_unspecified_initial_chemistry_depends_on_real_initial_water(initial_water, expected_mass, quality):
    system = _system(initial_water)
    composition = Composition((ConstituentInput("chloride", mass=Mass(5)),))
    system.configure_transport(ConservativeTransport((CHLORIDE,), boundaries={"supply": (composition,)}))
    result = system.build().run(bytes(16)).physical
    sample = result.sample("outlet", 0, view="incoming")
    assert sample.water_count == initial_water + 10
    assert sample.mass_counts["chloride"] == expected_mass
    assert sample.quality["chloride"] is quality


def test_imported_signed_stage_and_temperature_survive_saved_physical_result(tmp_path):
    system = _system(rule=Hold())
    metadata = InputMetadata(
        source="specialist-hydraulics-and-thermal",
        version="7",
        status=InputStatus.MODELLED,
        provenance=("calibrated-withdrawal-level",),
        uncertainty=Uncertainty("plus/minus 0.2", "specialist-report"),
    )
    states = (
        ProcessState("stage", "river", 0, Decimal("-1.25"), "m", "local vertical datum", metadata),
        ProcessState("withdrawal_temperature", "river", 0, Decimal("16.4"), "degC", "withdrawal layer", metadata),
    )
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            process_states=states,
            metadata=RunMetadata(scenario="operation-a", reference="observed-reference", version="4"),
        )
    )
    run = system.build().run(bytes(16))
    path = tmp_path / "physical-run"
    run.save(path)
    loaded = WaterSystemRun.load(path)
    assert loaded.physical.to_dict() == run.physical.to_dict()
    records = loaded.physical.metadata["inputs"]["process_states"]
    assert [(record["variable"], record["value"], record["unit"], record["domain"]) for record in records] == [
        ("stage", "-1.25", "m", "local vertical datum"),
        ("withdrawal_temperature", "16.4", "degC", "withdrawal layer"),
    ]
    assert records[1]["metadata"]["version"] == "7"
    assert records[1]["metadata"]["uncertainty"]["source"] == "specialist-report"
    assert loaded.physical.metadata["inputs"]["metadata"]["scenario"] == "operation-a"


def test_physical_output_discloses_input_quantisation_separately():
    system = _system()
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,), boundaries={"supply": (Composition((ConstituentInput("chloride", mass=Mass("0.0019")),)),)}
        )
    )
    result = system.build().run(bytes(16)).physical
    assert result.sample("outlet", 0, view="incoming").mass_counts["chloride"] == 1
    record = next(item for item in result.metadata["input_quantisation"] if item["location"] == "supply")
    assert record["discarded_fraction_of_mass_quantum"] == "9/10"
    assert record["constituent"] == "chloride"
    assert record["step"] == 0
    assert all(balance.residual == 0 for balance in result.basin_balances)


@pytest.mark.parametrize(
    "config",
    [
        ConservativeTransport((CHLORIDE,), initial={"unknown": Composition()}),
        ConservativeTransport((CHLORIDE,), boundaries={"unknown": (Composition(),)}),
        ConservativeTransport((CHLORIDE,), boundaries={"supply": (Composition(), Composition())}),
        ConservativeTransport(
            (CHLORIDE,),
            process_states=(
                ProcessState("stage", "river", 1, Decimal(0), "m", "datum", InputMetadata(source="model", version="1")),
            ),
        ),
    ],
)
def test_invalid_chemistry_bindings_fail_at_build_not_during_execution(config):
    system = _system()
    system.configure_transport(config)
    with pytest.raises(ValueError):
        system.build()


def test_actual_zero_water_unsupported_boundary_cannot_taint_supported_stream():
    system = make_water_system(1, "1 m3")
    system.source("dry_source", interval_volume([0.0]))
    system.source("wet_source", interval_volume([4.0]))
    system.reach("wet_branch", "wet_source", "mix")
    system.reach("mix", "dry_source", "outlet")
    unsupported = InputMetadata(
        source="unavailable-model",
        version="1",
        support=DomainSupport.UNSUPPORTED,
        reason="source outside supported domain",
    )
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={
                "dry_source": (Composition((ConstituentInput("chloride", metadata=unsupported),)),),
                "wet_source": (Composition((ConstituentInput("chloride", mass=Mass(2)),)),),
            },
        )
    )
    result = system.build().run(bytes(16)).physical
    sample = result.sample("outlet", 0, view="incoming")
    assert sample.water_count == 4
    assert sample.mass_counts["chloride"] == 2000
    assert sample.quality["chloride"] is QualityState.SUPPORTED
    assert result.concentration("outlet", 0, "chloride", view="incoming") == Fraction(1, 2)


def test_unresolved_old_dry_mass_does_not_immobilise_fresh_advected_mass():
    system = make_water_system(1, "1 m3")
    system.source("supply", interval_volume([4.0]))
    system.reach("river", "supply", "outlet")
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            initial={"river": Composition((ConstituentInput("chloride", mass=Mass(1)),))},
            boundaries={"supply": (Composition((ConstituentInput("chloride", mass=Mass(2)),)),)},
        )
    )
    result = system.build().run(bytes(16)).physical
    outgoing = result.sample("outlet", 0, view="incoming")
    retained = result.sample("river", 0, view="storage")
    assert outgoing.water_count == 4
    assert outgoing.mass_counts["chloride"] == 2000
    assert retained.mass_counts["chloride"] == 1000
    assert outgoing.quality["chloride"] is QualityState.UNSUPPORTED
    assert result.concentration("outlet", 0, "chloride", view="incoming") is None
    assert all(balance.residual == 0 for balance in result.basin_balances)


def test_supplied_mapping_predecessor_and_exact_software_revision_round_trip(tmp_path):
    system = _system()
    model = InputMetadata(source="hydraulic-model", version="1.4", status=InputStatus.MODELLED)
    mapping = LocationMapping("supply", "aquifer-body", "boundary-observation", "mapping-2026")
    predecessor = ProcessState(
        "directional_velocity", "river", -1, Decimal("-0.07"), "m/s", "survey cross-section", model
    )
    software = SoftwareIdentity("hydraulic-model", "1.4", "b79a58b47f0123")
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            location_mappings=(mapping,),
            predecessor_states=(predecessor,),
            metadata=RunMetadata(software_identities=(software,)),
        )
    )
    run = system.build().run(bytes(16))
    path = tmp_path / "attributed-run"
    run.save(path)
    restored = WaterSystemRun.load(path).physical
    inputs = restored.metadata["inputs"]
    assert inputs["location_mappings"][0]["physical_water_body"] == "aquifer-body"
    assert inputs["location_mappings"][0]["plan_unit"] is None
    assert inputs["location_mappings"][0]["mapping_version"] == "mapping-2026"
    assert inputs["predecessor_states"][0]["step"] == -1
    assert inputs["predecessor_states"][0]["value"] == "-0.07"
    assert inputs["metadata"]["software_identities"][0]["revision"] == "b79a58b47f0123"
    assert restored.to_dict() == run.physical.to_dict()


@pytest.mark.parametrize("include_unknown_account", [False, True])
def test_failed_supplied_process_account_invalidates_only_dependent_species(include_unknown_account):
    system = _system()
    metadata = InputMetadata(source="specialist-salt-model", version="3")
    failed = ProcessAccount("salt-component", "supply", 0, "chloride", 0, 0, 5000, 4000, metadata)
    accounts = (failed,)
    if include_unknown_account:
        accounts += (ProcessAccount("unresolved-component", "supply", 0, "chloride", None, 0, 1000, 1000, metadata),)
    chemistry = Composition((ConstituentInput("chloride", mass=Mass(5)), ConstituentInput("sulphate", mass=Mass(2))))
    system.configure_transport(
        ConservativeTransport((CHLORIDE, SULPHATE), boundaries={"supply": (chemistry,)}, process_accounts=accounts)
    )
    result = system.build().run(bytes(16)).physical
    outlet = result.sample("outlet", 0, view="incoming")
    assert outlet.water_count == 10
    assert outlet.mass_counts == {"chloride": 5000, "sulphate": 2000}
    assert outlet.quality["chloride"] is QualityState.UNSUPPORTED
    assert outlet.quality["sulphate"] is QualityState.SUPPORTED
    assert result.concentration("outlet", 0, "chloride", view="incoming") is None
    assert result.concentration("outlet", 0, "sulphate", view="incoming") == Fraction(1, 5)
    assert all(balance.residual == 0 for balance in result.basin_balances)
    supplied = result.metadata["inputs"]["process_accounts"]
    assert supplied[0]["incoming"] == 5000
    assert supplied[0]["outgoing"] == 4000
    assert result.process_balances[0].status is BalanceState.FAILED
    assert result.process_balances[0].residual == 1000
    if include_unknown_account:
        assert supplied[1]["start"] is None
        assert result.process_balances[1].status is BalanceState.INCOMPLETE
        assert result.process_balances[1].residual is None


def test_unknown_supplied_process_account_never_fabricates_supported_closure():
    system = _system()
    metadata = InputMetadata(source="specialist-model", version="3")
    unknown = ProcessAccount("component", "supply", 0, "chloride", None, 0, 5000, 5000, metadata)
    chemistry = Composition((ConstituentInput("chloride", mass=Mass(5)),))
    system.configure_transport(
        ConservativeTransport((CHLORIDE,), boundaries={"supply": (chemistry,)}, process_accounts=(unknown,))
    )
    result = system.build().run(bytes(16)).physical
    outlet = result.sample("outlet", 0, view="incoming")
    assert outlet.mass_counts["chloride"] == 5000
    assert outlet.quality["chloride"] is QualityState.UNSUPPORTED
    assert result.concentration("outlet", 0, "chloride", view="incoming") is None
    assert result.process_balances[0].status is BalanceState.INCOMPLETE
    assert result.process_balances[0].residual is None


@pytest.mark.parametrize(
    "declared_count,expected_quality",
    [
        (4000, QualityState.UNSUPPORTED),
        (5000, QualityState.SUPPORTED),
    ],
)
def test_closed_imported_account_must_match_realised_exchange(declared_count, expected_quality):
    system = _system()
    metadata = InputMetadata(source="specialist-salt-model", version="3")
    account = ProcessAccount("component", "supply", 0, "chloride", 0, 0, declared_count, declared_count, metadata)
    chemistry = Composition((ConstituentInput("chloride", mass=Mass(5)),))
    system.configure_transport(
        ConservativeTransport((CHLORIDE,), boundaries={"supply": (chemistry,)}, process_accounts=(account,))
    )
    result = system.build().run(bytes(16)).physical
    outlet = result.sample("outlet", 0, view="incoming")
    assert outlet.mass_counts["chloride"] == 5000
    assert outlet.quality["chloride"] is expected_quality
    assert result.process_balances[0].residual == 0
    assert result.process_balances[0].status is BalanceState.SUPPORTED
    assert all(balance.residual == 0 for balance in result.basin_balances)
    checks = result.metadata["process_account_exchange_checks"]
    assert checks
    # Preserve independent account closure while disclosing the exchange comparison.
    comparison = next(check for check in checks if check["owner"] == "component")
    assert comparison["term"] == "outgoing"
    assert comparison["supplied_count"] == declared_count
    assert comparison["realised_count"] == 5000
    assert comparison["difference_counts"] == declared_count - 5000
    assert comparison["support"] == ("unsupported" if declared_count == 4000 else "supported")


def test_delay_storage_recovers_known_mass_after_unknown_parcel_departs():
    system = make_water_system(3, "1 m3")
    system.source("supply", interval_volume([4.0, 4.0, 0.0]))
    system.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            boundaries={
                "supply": (
                    Composition((ConstituentInput("chloride"),)),
                    Composition((ConstituentInput("chloride", mass=Mass(2)),)),
                    Composition(),
                )
            },
        )
    )
    result = system.build().run(bytes(16)).physical
    first = result.sample("transit", 0, view="storage")
    second = result.sample("transit", 1, view="storage")
    final = result.sample("transit", 2, view="storage")
    assert first.mass_counts["chloride"] is None
    assert second.water_count == 4
    assert second.mass_counts["chloride"] == 2000
    assert second.quality["chloride"] is QualityState.SUPPORTED
    assert final.water_count == 0
    assert final.mass_counts["chloride"] == 0
    delivered = result.sample("outlet", 2, view="incoming")
    assert delivered.mass_counts["chloride"] == 2000
    assert delivered.quality["chloride"] is QualityState.SUPPORTED


def test_failed_water_process_account_invalidates_all_carried_constituents():
    system = _system()
    metadata = InputMetadata(source="water-component", version="2")
    account = ProcessAccount("component", "supply", 0, "water", 0, 0, 11, 10, metadata)
    chemistry = Composition((ConstituentInput("chloride", mass=Mass(5)), ConstituentInput("sulphate", mass=Mass(2))))
    system.configure_transport(
        ConservativeTransport((CHLORIDE, SULPHATE), boundaries={"supply": (chemistry,)}, process_accounts=(account,))
    )
    result = system.build().run(bytes(16)).physical
    outlet = result.sample("outlet", 0, view="incoming")
    assert outlet.water_count == 10
    assert outlet.mass_counts == {"chloride": 5000, "sulphate": 2000}
    assert outlet.quality == {"chloride": QualityState.UNSUPPORTED, "sulphate": QualityState.UNSUPPORTED}
    assert outlet.water_quality is QualityState.UNSUPPORTED
    assert result.process_balances[0].status is BalanceState.FAILED
    assert result.process_balances[0].residual == 1
    assert all(balance.residual == 0 for balance in result.basin_balances)


def test_travel_delay_refuses_initial_dry_mass_without_dated_cohort():
    system = make_water_system(2, "1 m3")
    system.source("supply", interval_volume([0.0, 0.0]))
    system.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    system.configure_transport(
        ConservativeTransport(
            (CHLORIDE,),
            initial={
                "transit": Composition((ConstituentInput("chloride", mass=Mass(1)),)),
            },
        )
    )
    with pytest.raises(ValueError, match="(?i)(delay|cohort|initial)"):
        system.build().run(bytes(16))


@pytest.mark.parametrize(
    "form,basis,unit,value",
    [
        ("Cl2", "as chloride", "kg/m3", Decimal(1)),
        ("Cl-", "as sodium", "kg/m3", Decimal(1)),
        ("Cl-", "as chloride", "degC", Decimal(1)),
        ("Cl-", "as chloride", "kg/m3", Decimal(-1)),
    ],
)
def test_imported_registered_chemistry_cannot_bypass_basis_unit_or_domain(form, basis, unit, value):
    state = ProcessState(
        "chloride",
        "river",
        0,
        value,
        unit,
        "water column concentration",
        InputMetadata(source="specialist", version="1"),
        chemical_form=form,
        reporting_basis=basis,
    )
    with pytest.raises(ValueError):
        ConservativeTransport((CHLORIDE,), process_states=(state,))
