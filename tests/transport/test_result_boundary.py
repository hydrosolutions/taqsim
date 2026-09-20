"""Public count/projection/replay results retain their declared temporal and support meanings."""

import json
from copy import deepcopy

import incidence as inc
import pytest

from taqsim import ConservationQuantum, Presence, TimeAxis, WaterSystemRun, load_run
from taqsim.water_system import WaterQuantumCount
from tests.transport.model_cases import (
    RUN_ID,
    SALT,
    WATER,
    Inventory,
    Quantity,
    assert_balances,
    available,
    counts,
    coupled,
    mixed_pool,
    network,
    rule,
    scheduled,
    stock,
)


def test_transfer_concentration_differs_from_final_storage_after_selective_loss():
    doc, stocks = mixed_pool((10, 50), ())
    rule(doc, "pool", "water", {"release": ("outlet", inc.literal(4)), "evaporate": ("evaporation", inc.literal(2))})
    coupled(doc, "pool", "salt", {"advected": ("release", "outlet")})
    run = inc.compile_model(doc).run(RUN_ID)
    transfer = counts(run, "outlet", "salt")[0] * SALT.quantum / counts(run, "outlet", "water")[0]
    retained = (
        stock(run, Inventory("pool", "salt", 50))[0] * SALT.quantum / stock(run, Inventory("pool", "water", 10))[0]
    )
    assert transfer == 0.5
    assert retained == 0.75
    assert transfer != retained
    assert_balances(run, (WATER, SALT), stocks, ("outlet", "evaporation"), frozenset({"pool"}), ())


def test_volume_weighted_aggregation_uses_mass_and_volume_not_mean_concentration():
    stocks = (
        Inventory("supply", "water", 15),
        Inventory("supply", "salt", 90),
        Inventory("pool", "water", 0),
        Inventory("pool", "salt", 0),
    )
    doc = network((WATER, SALT), stocks, 2)
    scheduled(doc, "supply", "water", {"entry": ("pool", (10.0, 5.0))})
    scheduled(doc, "supply", "salt", {"entry": ("pool", (8.0, 1.0))})
    rule(doc, "pool", "water", {"release": ("outlet", available(doc, "pool", "water", 0))})
    coupled(doc, "pool", "salt", {"carried": ("release", "outlet")})
    run = inc.compile_model(doc).run(RUN_ID)
    water = counts(run, "outlet", "water")
    mass = counts(run, "outlet", "salt")
    assert water == [10, 5] and mass == [80, 10]
    interval_concentration = [m * SALT.quantum / v for m, v in zip(mass, water, strict=True)]
    assert interval_concentration == [0.8, 0.2]
    aggregate = sum(mass) * SALT.quantum / sum(water)
    assert aggregate == 0.6 and aggregate != sum(interval_concentration) / 2
    # Public exchange needed by a future quality consumer, not Fishy integration:
    assert aggregate * 1000 == 600  # mg/l with this explicitly declared kg/m3 basis
    assert doc["calendar"]["timestep_seconds"] == 86400
    assert stock(run, Inventory("pool", "water", 0)) == [0, 0]
    assert_balances(run, (WATER, SALT), stocks, ("outlet",), frozenset({"pool"}), ("supply",))


def test_future_forcing_counts_are_excluded_at_the_physical_entry_boundary():
    stocks = (
        Inventory("supply", "water", 101),
        Inventory("supply", "salt", 1010),
        Inventory("pool", "water", 0),
        Inventory("pool", "salt", 0),
    )
    doc = network((WATER, SALT), stocks, 2)
    scheduled(doc, "supply", "water", {"entry": ("pool", (1.0, 100.0))})
    scheduled(doc, "supply", "salt", {"entry": ("pool", (1.0, 100.0))})
    water = available(doc, "pool", "water", 0)
    rule(
        doc,
        "pool",
        "water",
        {"left": ("left", inc.mul(water, inc.literal(0.5))), "right": ("right", inc.mul(water, inc.literal(0.5)))},
    )
    coupled(doc, "pool", "salt", {"left-mass": ("left", "left"), "right-mass": ("right", "right")})
    run = inc.compile_model(doc).run(RUN_ID)
    assert stock(run, Inventory("supply", "water", 101)) == [100, 0]
    assert stock(run, Inventory("supply", "salt", 1010)) == [1000, 0]
    assert stock(run, Inventory("pool", "water", 0)) == [1, 1]
    assert stock(run, Inventory("pool", "salt", 0)) == [10, 10]
    assert counts(run, "left", "salt") == [0, 500]
    assert counts(run, "right", "salt") == [0, 500]
    assert_balances(run, (WATER, SALT), stocks, ("left", "right"), frozenset({"pool"}), ("supply",))


def test_presence_and_missing_quality_keep_independently_known_components():
    sulphate = Quantity("sulphate", "kg", 0.1)
    stocks = (Inventory("pool", "water", 1), Inventory("pool", "sulphate", 10))
    doc = network((WATER, sulphate), stocks, 2)
    scheduled(doc, "pool", "water", {"release": ("outlet", (1.0, 0.0))})
    coupled(doc, "pool", "sulphate", {"carried": ("release", "outlet")})
    # Metadata belongs to the physical caller, not the engine. Missing chloride
    # is not registered with a fictitious zero initial inventory or zero load.
    supportedness = {"water": "supported", "sulphate": "supported", "chloride": "missing boundary chemistry"}
    original = deepcopy(supportedness)
    run = inc.compile_model(doc).run(RUN_ID)
    assert counts(run, "outlet", "water") == [1, 0]
    assert counts(run, "outlet", "sulphate") == [10, 0]
    chloride = run.transfer_count_series("outlet", "chloride", direction="incoming")
    assert list(chloride.values) == [None, None]
    assert list(chloride.presence) == ["not_modelled", "not_modelled"]
    water = run.transfer_count_series("outlet", "water", direction="incoming", first=0, last=2)
    assert list(water.values) == [1, 0, None]
    assert list(water.presence) == ["present", "present", "absent"]
    assert supportedness == original
    # A downstream chlorine result is incomplete; engine not_modelled by itself
    # does not determine the chemical reason, and present zero is not missing.
    assert "chloride" not in doc["substances"]
    projected = WaterSystemRun(
        run,
        TimeAxis("2020-01-01", 2),
        ConservationQuantum.CUBIC_METRE,
        frozenset({"pool"}),
        {"pool": WaterQuantumCount(1)},
    )
    assert projected.flow("pool", start="2019-12-31", end="2020-01-03").presence == (
        Presence.NOT_MODELLED,
        Presence.PRESENT,
        Presence.PRESENT,
        Presence.NOT_MODELLED,
    )
    with pytest.raises(KeyError):
        projected.flow("not-a-reach")


def test_existing_saved_water_projections_do_not_offer_mass_arrivals_or_restart(tmp_path):
    doc, stocks = mixed_pool((20, 40), ((10, 40),))
    rule(doc, "pool", "water", {"release": ("outlet", inc.literal(15))})
    coupled(doc, "pool", "salt", {"advected": ("release", "outlet")})
    model = inc.compile_model(doc)
    completed = model.run(RUN_ID)
    run = WaterSystemRun(
        completed,
        TimeAxis("2020-01-01", 1),
        ConservationQuantum.CUBIC_METRE,
        frozenset({"pool"}),
        {"pool": WaterQuantumCount(20)},
    )
    saved = tmp_path / "output.json"
    run.save(saved)
    loaded = load_run(saved)
    assert list(run.flow("pool").values) == [15.0]
    assert list(run.retained("pool").values) == [15.0]
    assert loaded.flow("pool") == run.flow("pool")
    assert loaded.retained("pool") == run.retained("pool")
    assert loaded.authoritative_log() == run.authoritative_log()
    assert loaded.model_digest == completed.model_digest
    assert counts(completed, "outlet", "salt") == [40]  # supported live mass readback
    completed.replay_against(model.run(RUN_ID))  # reconstruct by executing declared model
    assert completed.authoritative_log() == model.run(RUN_ID).authoritative_log()
    with pytest.raises(ValueError, match="unavailable from saved-run caches"):
        loaded.arrivals("outlet")
    artifact = json.loads(saved.read_text())
    assert artifact["format_version"] == 4
    assert set(artifact) == {
        "artifact_sha256",
        "authoritative_log",
        "flows",
        "format",
        "format_version",
        "incidence_version",
        "model_digest",
        "quantum",
        "reaches",
        "retained",
        "time",
    }
    # The opaque log is preserved, not decoded into unoffered physical results.
    assert "mass" not in artifact and "arrivals" not in artifact and "checkpoint" not in artifact
