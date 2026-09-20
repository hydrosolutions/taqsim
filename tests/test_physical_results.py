import json
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from taqsim.constituents import Constituent
from taqsim.persistence import load_run
from taqsim.physical_results import Balance, BalanceState, PhysicalSample, QualityState, TransportResult
from taqsim.water_system import TimeAxis
from tests import interval_volume, make_water_system


def sample(step=0, water=10, mass=5, state=QualityState.SUPPORTED):
    return PhysicalSample("reach", step, water, {"salt": mass}, {"salt": state}, {})


def result(samples=None, balances=(), basin_balances=(), metadata=None):
    samples = samples or (sample(),)
    return TransportResult(
        TimeAxis("2020-01-01", periods=len(samples)),
        Decimal(1),
        (Constituent("salt", "NaCl", "NaCl", Decimal(1)),),
        {"reach": samples},
        {"reach": samples},
        {"reach": samples},
        (),
        balances,
        basin_balances,
        metadata or {},
    )


def test_saved_run_preserves_arrivals(tmp_path):
    system = make_water_system(2, "1 m3")
    system.source("river", interval_volume([5.0, 0.0]))
    system.add_reach("reach", "river", "outlet")
    original = system.build().run(bytes(16))
    path = tmp_path / "run.json"
    original.save(path)
    restored = load_run(path)
    assert restored.arrivals("outlet") == original.arrivals("outlet")


def test_samples_and_nested_metadata_are_immutable():
    metadata = {
        "scenario": "baseline",
        "provenance": [{"source": "original", "version": "v1"}],
        "predecessor": "previous",
    }
    value = result(metadata=metadata)
    metadata["provenance"][0]["version"] = "changed"
    assert value.metadata["provenance"][0]["version"] == "v1"
    with pytest.raises(TypeError):
        value.outgoing["reach"][0].mass_counts["salt"] = 4
    with pytest.raises(TypeError):
        value.metadata["provenance"][0]["version"] = "changed"


def test_exact_huge_counts_and_decimal_projection_do_not_round():
    count = 10**50 + 3
    value = result((sample(water=count, mass=count - 1),))
    with localcontext() as context:
        context.prec = 6
        assert value.amount("reach", 0) == Decimal(count)
        assert value.amount("reach", 0, "salt") == Decimal(count - 1)
    assert value.concentration("reach", 0, "salt") == Fraction(count - 1, count)
    restored = TransportResult.from_dict(json.loads(json.dumps(value.to_dict())))
    assert restored == value
    assert restored.digest == value.digest


def test_balance_independent_residual_missing_and_failure():
    assert Balance("reach", 0, "salt", 4, 3, 2, 3).residual == 0
    failed = Balance("reach", 0, "salt", 4, 3, 2, 2)
    unknown = Balance("reach", 0, "salt", None, 3, 2, 3)
    assert failed.residual == 1
    assert failed.status is BalanceState.FAILED
    assert unknown.residual is None
    assert unknown.status is BalanceState.INCOMPLETE
    assert result(balances=(failed,)).concentration("reach", 0, "salt") is None
    assert result(balances=(failed,)).sample("reach", 0).quality["salt"] is QualityState.UNSUPPORTED
    assert result(balances=(unknown,)).sample("reach", 0).quality["salt"] is QualityState.SUPPORTED


def test_water_balance_failure_invalidates_quality():
    value = result(balances=(Balance("reach", 0, "water", 0, 0, 10, 9),))
    assert value.sample("reach", 0).quality["salt"] is QualityState.UNSUPPORTED


def test_aggregation_sums_amounts_before_ratio_and_retains_missing():
    value = result((sample(0, 10, 8), sample(1, 5, 1)))
    aggregate = value.aggregate("reach", 0, 2)
    assert aggregate.water_count == 15
    assert aggregate.mass_counts["salt"] == 9
    assert value.sample_concentration(aggregate, "salt") == Fraction(3, 5)
    missing = result((sample(0, 10, 8), sample(1, 5, None, QualityState.MISSING)))
    aggregate = missing.aggregate("reach", 0, 2)
    assert aggregate.mass_counts["salt"] is None
    assert aggregate.quality["salt"] is QualityState.MISSING
    assert missing.sample_concentration(aggregate, "salt") is None
    with pytest.raises(ValueError):
        value.aggregate("reach", 0, 0.5)
    with pytest.raises(ValueError):
        value.aggregate("reach", 0, 2, view="storage")


def test_dry_absent_unknown_and_outside_horizon_remain_distinct():
    value = result(
        (
            sample(0, 0, 4, QualityState.DRY),
            sample(1, 0, None, QualityState.ABSENT),
            sample(2, 1, None, QualityState.MISSING),
        )
    )
    assert value.amount("reach", 0, "salt") == Decimal(4)
    assert value.concentration("reach", 0, "salt") is None
    assert value.sample("reach", -1).water_count is None
    assert value.sample("reach", -1).quality["salt"] is QualityState.OUTSIDE_HORIZON
    assert value.sample("reach", 1).quality["salt"] is QualityState.ABSENT
    assert value.sample("reach", 2).quality["salt"] is QualityState.MISSING
    assert TransportResult.from_dict(value.to_dict()) == value


@pytest.mark.parametrize("count", [1.1, True, -1])
def test_noninteger_and_negative_counts_are_refused(count):
    with pytest.raises(ValueError):
        sample(water=count)


def test_known_balance_failure_survives_unknown_basin_balance():
    value = result(
        balances=(Balance("reach", 0, "salt", 4, 3, 2, 2),),
        basin_balances=(Balance("basin", 0, "salt", None, 3, 2, 3),),
    )
    assert value.sample("reach", 0).quality["salt"] is QualityState.UNSUPPORTED


def test_v4_water_cache_migrates_without_inventing_arrivals(tmp_path):
    from taqsim.persistence import _artifact_digest

    system = make_water_system(1, "1 m3")
    system.source("river", interval_volume([5.0]))
    system.add_reach("reach", "river", "outlet")
    original = system.build().run(bytes(16))
    path = tmp_path / "legacy.json"
    original.save(path)
    document = json.loads(path.read_text())
    document["format_version"] = 4
    document.pop("physical")
    document.pop("arrivals")
    document.pop("artifact_sha256")
    document["artifact_sha256"] = _artifact_digest(document)
    path.write_text(json.dumps(document))
    restored = load_run(path)
    assert restored.flow("reach") == original.flow("reach")
    with pytest.raises(ValueError, match="legacy"):
        restored.arrivals("outlet")
    restored.save(tmp_path / "migrated.json")
    migrated = load_run(tmp_path / "migrated.json")
    assert migrated.flow("reach") == original.flow("reach")
    with pytest.raises(ValueError, match="legacy"):
        migrated.arrivals("outlet")


def test_physical_document_retains_exchange_identity_and_all_metadata():
    from dataclasses import replace

    from taqsim.physical_results import TransferRecord

    metadata = {
        "scenario": "baseline",
        "reference": "natural",
        "input_versions": {"gauging": "v2"},
        "software": {"taqsim": "tested", "incidence": "pinned"},
        "provenance": ["declared source"],
        "precision": {"allocation": "integer remainder"},
        "uncertainty": {"description": "synthetic exact inputs", "source": "test"},
        "predecessor": {"step": -1, "water_count": 3},
        "locations": {"reach": "station-1"},
        "input_status": "corrected",
    }
    value = result(metadata=metadata)
    value = replace(value, transfers=(TransferRecord("upstream", "reach", "out", 0, "advection", sample()),))
    restored = TransportResult.from_dict(json.loads(json.dumps(value.to_dict())))
    assert restored == value
    assert restored.transfers[0].source == "upstream"
    assert restored.constituents[0].chemical_form == "NaCl"
    assert restored.constituents[0].reporting_basis == "NaCl"
    assert restored.digest == value.digest


def test_public_physical_run_saved_roundtrip(tmp_path):
    from taqsim.constituents import Composition, Concentration, ConservativeTransport, ConstituentInput

    system = make_water_system(2, "1 m3")
    system.source("river", interval_volume([5.0, 0.0]))
    system.add_reach("reach", "river", "outlet")
    system.configure_transport(
        ConservativeTransport(
            constituents=(Constituent("salt", "NaCl", "NaCl", Decimal("0.1")),),
            boundaries={"river": (Composition((ConstituentInput("salt", concentration=Concentration("0.5")),)),) * 2},
        )
    )
    original = system.build().run(bytes(16))
    path = tmp_path / "physical.json"
    original.save(path)
    restored = load_run(path)
    assert restored.physical == original.physical
    assert restored.physical.digest == original.physical.digest
    rewritten = tmp_path / "physical-rewritten.json"
    restored.save(rewritten)
    assert rewritten.read_bytes() == path.read_bytes()
    assert restored.arrivals("outlet") == original.arrivals("outlet")
    assert restored.physical.amount("outlet", 0, "salt", view="incoming") == Decimal("2.5")


def test_incomplete_basin_account_does_not_poison_supported_local_observation():
    value = result(basin_balances=(Balance("basin", 0, "salt", None, 3, 2, 3),))
    assert value.concentration("reach", 0, "salt") == Fraction(1, 2)
    assert value.basin_balances[0].status is BalanceState.INCOMPLETE


def test_incomplete_local_balance_preserves_known_incoming_and_zero_outgoing():
    from dataclasses import replace

    unknown_initial = Balance("reach", 0, "salt", None, None, 5, 0)
    value = result(balances=(unknown_initial,))
    assert value.sample("reach", 0, view="incoming").mass_counts["salt"] == 5
    assert value.concentration("reach", 0, "salt", view="incoming") == Fraction(1, 2)
    value = replace(value, outgoing={"reach": (sample(0, 0, 0, QualityState.DRY),)})
    assert value.sample("reach", 0).quality["salt"] is QualityState.DRY
    assert value.sample("reach", 0).mass_counts["salt"] == 0
    assert value.balances[0].status is BalanceState.INCOMPLETE


def test_unprovided_endpoint_mapping_is_saved_as_unavailable_not_empty(tmp_path):
    from taqsim.water_system import WaterSystemRun

    system = make_water_system(1, "1 m3")
    system.source("river", interval_volume([5.0]))
    system.add_reach("reach", "river", "outlet")
    live = system.build().run(bytes(16))
    direct = WaterSystemRun(live._completed, live.time, live.quantum, live.reaches, live._initial_counts)
    assert direct.endpoints is None
    saved = tmp_path / "unprovided.json"
    direct.save(saved)
    assert json.loads(saved.read_text())["arrivals"] is None
    loaded = load_run(saved)
    assert loaded.endpoints is None
    with pytest.raises(ValueError, match="unavailable from saved-run caches"):
        loaded.arrivals("outlet")
    loaded.save(saved)
    assert json.loads(saved.read_text())["arrivals"] is None


def test_failed_water_account_retains_numeric_water_with_explicit_invalid_support():
    value = result(balances=(Balance("reach", 0, "water", 0, 0, 10, 9),))
    observed = value.sample("reach", 0)
    assert observed.water_count == 10
    assert observed.water_quality is QualityState.UNSUPPORTED
    assert "failed water balance" in observed.water_reason
    assert value.sample("reach", -1).water_quality is QualityState.OUTSIDE_HORIZON
    aggregate = value.aggregate("reach", 0, 1)
    assert aggregate.water_count == 10
    assert aggregate.water_quality is QualityState.UNSUPPORTED
    restored = TransportResult.from_dict(value.to_dict())
    assert restored == value


def test_process_balance_is_separate_and_invalidates_affected_output():
    from dataclasses import replace

    failure = Balance("reach", 0, "water", 0, 0, 10, 9)
    value = replace(result(), process_balances=(failure,))
    assert value.balances == ()
    assert value.process_balances[0].status is BalanceState.FAILED
    assert value.sample("reach", 0).water_quality is QualityState.UNSUPPORTED
    assert TransportResult.from_dict(value.to_dict()) == value


def test_public_quality_state_distinguishes_unregistered_dry_outside_and_unsupported(tmp_path):
    from taqsim.constituents import Composition, Concentration, ConservativeTransport, ConstituentInput

    system = make_water_system(2, "1 m3")
    system.source("river", interval_volume([5.0, 0.0]))
    system.add_reach("reach", "river", "outlet")
    system.configure_transport(
        ConservativeTransport(
            constituents=(Constituent("salt", "NaCl", "NaCl", Decimal("0.1")),),
            boundaries={"river": (Composition((ConstituentInput("salt", concentration=Concentration("0.5")),)),) * 2},
        )
    )
    original = system.build().run(bytes(16))
    saved = tmp_path / "quality.json"
    original.save(saved)
    for run in (original, load_run(saved)):
        physical = run.physical
        assert physical.quality_state("outlet", 0, "chloride", view="incoming") is QualityState.ABSENT
        assert physical.concentration("outlet", 0, "chloride", view="incoming") is None
        assert physical.quality_state("outlet", 1, "salt", view="incoming") is QualityState.DRY
        assert physical.quality_state("outlet", -1, "salt", view="incoming") is QualityState.OUTSIDE_HORIZON
        with pytest.raises(KeyError):
            physical.quality_state("unknown-location", 0, "chloride", view="incoming")
    unsupported = result(balances=(Balance("reach", 0, "salt", 1, 0, 0, 0),))
    assert unsupported.quality_state("reach", 0, "salt") is QualityState.UNSUPPORTED
