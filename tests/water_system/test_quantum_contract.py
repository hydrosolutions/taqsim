"""quantum contract evidence : ModelDeclarations → Exact conserved run observations."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from taqsim import (
    CanalLosses,
    ConservationQuantum,
    EFlowSplit,
    MonthlyDistribution,
    Parameter,
    Presence,
    PriorityDistribution,
    ReservoirEvaporation,
    TimeAxis,
    VolumetricRate,
    WaterSystem,
    WaterSystemObjective,
    WaterVolume,
    ZoneRelease,
    load_run,
    optimize_water_system,
)
from tests import interval_volume, make_water_system


def _water_system(*, steps: int = 1, quantum: ConservationQuantum = ConservationQuantum.LITRE) -> WaterSystem:
    return make_water_system(steps, quantum)


def _count(value: float | None, quantum: ConservationQuantum) -> int:
    assert value is not None
    return round(value / quantum.quantum_m3)


def _count_value(count: int, quantum: ConservationQuantum) -> float:
    return float(count) * quantum.quantum_m3


def test_external_water_must_align_before_incidence_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = False

    def compile_model(document: object) -> None:
        nonlocal compiled
        compiled = True
        raise AssertionError("misaligned water reached incidence")

    monkeypatch.setattr("taqsim.water_system.incidence.compile_model", compile_model)
    source = _water_system(quantum=ConservationQuantum.LITRE)
    source.source("river", interval_volume([1.0001]))
    source.reach("canal", "river", "farm")
    with pytest.raises(ValueError, match=r"source 'river'.*timestep 0.*1\.0001.*quantum 0\.001"):
        source.build()
    assert not compiled

    initial = _water_system(quantum=ConservationQuantum.LITRE)
    initial.source("river", interval_volume([1.0]))
    initial.reach("canal", "river", "farm", initial_water=WaterVolume(0.0001, "m3"))
    with pytest.raises(ValueError, match=r"reach 'canal' initial stock.*position 0.*0\.0001.*quantum 0\.001"):
        initial.build()
    assert not compiled


def test_boundary_counts_are_aggregated_without_float_sum_authority() -> None:
    water_system = _water_system(steps=2, quantum=ConservationQuantum.LITRE)
    water_system.source("river", interval_volume([0.1, 0.2]))
    water_system.reach("canal", "river", "farm")
    model = water_system.build()
    stock = next(item for item in model.document["initial_stocks"] if item["compartment"] == "river")
    assert stock["amounts"][0]["amount"] == 0.3


def test_reported_canal_case_accounts_in_integer_counts_and_carries_residual() -> None:
    quantum = ConservationQuantum.LITRE
    water_system = _water_system(steps=2, quantum=quantum)
    water_system.source("river", interval_volume([10000.0, 8000.0]))
    water_system.reach("canal", "river", "farm", rule=CanalLosses(0.0014, 12.0))
    run = water_system.build().run(bytes(16))

    incoming = [_count(value, quantum) for value in run.arrivals("canal").values]
    delivered = [_count(value, quantum) for value in run.arrivals("farm").values]
    seepage = [_count(value, quantum) for value in run.arrivals("seepage").values]
    evaporation = [_count(value, quantum) for value in run.arrivals("evaporation").values]
    operational = [_count(value, quantum) for value in run.arrivals("operational-loss").values]
    retained = [_count(value, quantum) for value in run.retained("canal").values]
    previous = 0
    for step in range(2):
        assert (
            previous + incoming[step]
            == delivered[step] + seepage[step] + evaporation[step] + operational[step] + retained[step]
        )
        previous = retained[step]
    assert retained == [1, 1]


def test_computed_subquantum_zero_is_present_not_absent_or_not_modelled() -> None:
    water_system = _water_system(quantum=ConservationQuantum.CUBIC_METRE)
    water_system.source("river", interval_volume([1.0]))
    water_system.reach("canal", "river", "farm", rule=CanalLosses(1e-9, 1.0))
    run = water_system.build().run(bytes([2]) * 16)
    seepage = run.arrivals("seepage")
    assert list(seepage.values) == [0.0]
    assert seepage.presence == (Presence.PRESENT,)
    extended = run.flow("canal", start="2019-12-31", end="2020-01-02")
    assert extended.presence == (Presence.NOT_MODELLED, Presence.PRESENT, Presence.NOT_MODELLED)
    with pytest.raises(KeyError, match="unknown reach"):
        run.flow("never-modelled")


def test_quantum_is_bound_to_model_live_run_and_saved_run(tmp_path: Path) -> None:
    def build(quantum: ConservationQuantum):
        water_system = _water_system(quantum=quantum)
        water_system.source("river", interval_volume([1.0]))
        water_system.reach("canal", "river", "farm")
        return water_system.build()

    litre = build(ConservationQuantum.LITRE)
    millilitre = build(ConservationQuantum.MILLILITRE)
    assert litre.model_digest != millilitre.model_digest
    assert litre.quantum is ConservationQuantum.LITRE
    run = litre.run(bytes([3]) * 16)
    assert run.quantum is ConservationQuantum.LITRE
    saved = tmp_path / "run.json"
    run.save(saved)
    reopened = load_run(saved)
    assert reopened.quantum is ConservationQuantum.LITRE
    assert reopened.model_digest == run.model_digest == litre.model_digest
    document = json.loads(saved.read_text())
    assert document["quantum"] == ConservationQuantum.LITRE.value
    assert document["model_digest"] == litre.model_digest


def test_repeated_execution_has_identical_authoritative_digest() -> None:
    water_system = _water_system(steps=2, quantum=ConservationQuantum.LITRE)
    water_system.source("river", interval_volume([3.141, 2.718]))
    water_system.reach("canal", "river", "farm", rule=CanalLosses(0.0014, 12.0))
    model = water_system.build()
    run_id = bytes([4]) * 16
    assert model.run(run_id).authoritative_log_digest == model.run(run_id).authoritative_log_digest


def test_all_75_canal_grid_cases_complete() -> None:
    cases = [
        (flow, coefficient, length)
        for flow in (100.0, 500.0, 1000.0, 5000.0, 10000.0)
        for coefficient in (0.0001, 0.0005, 0.0014, 0.005, 0.01)
        for length in (1.0, 5.0, 12.0)
    ]
    for index, (flow, coefficient, length) in enumerate(cases, 1):
        water_system = _water_system(steps=2, quantum=ConservationQuantum.LITRE)
        water_system.source("river", interval_volume([flow, flow * 0.8]))
        water_system.reach("canal", "river", "farm", rule=CanalLosses(coefficient, length))
        assert water_system.build().run(index.to_bytes(16, "big")).authoritative_log_digest


def test_priority_optimizer_completes_first_generation() -> None:
    water_system = _water_system(quantum=ConservationQuantum.LITRE)
    water_system.source("river", interval_volume([_count_value(123_457, ConservationQuantum.LITRE)]))
    water_system.reach(
        "split",
        "river",
        "tail",
        rule=CanalLosses(Parameter("seepage", 0.0014, (0.0001, 0.01)), 7.0),
    )
    model = water_system.build()
    result = optimize_water_system(
        model,
        [WaterSystemObjective("priority", "maximize", lambda run: float(run.arrivals("tail").values[0] or 0.0))],
        pop_size=8,
        generations=1,
        seed=19,
    )
    assert result.solutions


def test_seeded_nonround_samples_cover_all_six_rule_compilers() -> None:
    rng = random.Random(11)
    factories = (
        lambda: PriorityDistribution("downstream", WaterVolume(rng.uniform(1.0, 900.0), "m3"), {"other": 1.0}),
        lambda: EFlowSplit({"downstream": 1.0}, {"other": 1.0}, rng.uniform(0.01, 0.6)),
        lambda: ReservoirEvaporation(
            (rng.uniform(0.5, 120.0),) * 12, ((WaterVolume(0.0, "m3"), 0.0), (WaterVolume(1000.0, "m3"), 10_000.0))
        ),
        lambda: (
            lambda fraction: MonthlyDistribution({"downstream": (fraction,) * 12, "other": (1.0 - fraction,) * 12})
        )(rng.uniform(0.05, 0.95)),
        lambda: ZoneRelease(
            WaterVolume(0.0, "m3"),
            WaterVolume(2_000.0, "m3"),
            WaterVolume(50_000.0, "m3"),
            VolumetricRate(rng.uniform(0.0001, 0.02), "m3/s"),
        ),
        lambda: CanalLosses(rng.uniform(1e-4, 1e-2), rng.uniform(0.5, 15.0)),
    )
    for shape_index, factory in enumerate(factories):
        for sample in range(60):
            water_system = _water_system(steps=2, quantum=ConservationQuantum.LITRE)
            water_system.source("source", interval_volume([1000.0, 800.0]))
            water_system.reach("structure", "source", "downstream", rule=factory())
            run_id = (1000 + shape_index * 60 + sample).to_bytes(16, "big")
            assert water_system.build().run(run_id).authoritative_log_digest


def test_sweep_and_replay_complete_at_ordinary_nonround_parameters() -> None:
    water_system = _water_system(quantum=ConservationQuantum.LITRE)
    water_system.source("river", interval_volume([_count_value(321_987, ConservationQuantum.LITRE)]))
    water_system.reach(
        "canal",
        "river",
        "farm",
        rule=CanalLosses(Parameter("seepage", 0.00143, (0.00017, 0.00983)), 7.321),
    )
    model = water_system.build()
    runs = model.sweep("canal.seepage", [0.000173, 0.002719, 0.008997])
    assert len(runs) == 3
    first = model.run(bytes([71]) * 16, {"canal.seepage": 0.002719})
    repeated = model.run(bytes([71]) * 16, {"canal.seepage": 0.002719})
    first.verify_replay(repeated)
    assert first.authoritative_log_digest == repeated.authoritative_log_digest


def test_multibranch_residual_is_retained_and_used_on_the_next_timestep() -> None:
    left = (0.45, 1.0) + (1.0,) * 10
    right = (0.45, 0.0) + (0.0,) * 10
    water_system = WaterSystem(
        time=TimeAxis("2020-01-01", periods=2, frequency="31d"),
        quantum=ConservationQuantum.LITRE,
    )
    water_system.source("river", interval_volume([0.010, 0.0], cadence="31d"))
    water_system.reach("split", "river", "unused", rule=MonthlyDistribution({"left": left, "right": right}))
    run = water_system.build().run(bytes([81]) * 16)
    assert list(run.retained("split").values) == [0.002, 0.0]
    assert list(run.arrivals("left").values) == [0.004, 0.002]
    incoming = np.array([_count(value, ConservationQuantum.LITRE) for value in run.arrivals("split").values])
    left_counts = np.array([_count(value, ConservationQuantum.LITRE) for value in run.arrivals("left").values])
    right_counts = np.array([_count(value, ConservationQuantum.LITRE) for value in run.arrivals("right").values])
    retained = np.array([_count(value, ConservationQuantum.LITRE) for value in run.retained("split").values])
    np.testing.assert_array_equal(incoming + np.array([0, 2]), left_counts + right_counts + retained)


def test_ambiguous_aggregate_initial_stock_is_refused_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 4_394_222_044_288_838
    value = _count_value(count, ConservationQuantum.MILLILITRE)
    water_system = _water_system(steps=2, quantum=ConservationQuantum.MILLILITRE)
    water_system.source("river", interval_volume([value, value]))
    water_system.reach("canal", "river", "farm")
    compiled = False

    def compile_model(document: object) -> None:
        nonlocal compiled
        compiled = True
        raise AssertionError("ambiguous aggregate reached incidence")

    monkeypatch.setattr("taqsim.water_system.incidence.compile_model", compile_model)
    with pytest.raises(
        ValueError,
        match=r"source 'river' aggregate initial stock at position 0.*8788444088577676.*quantum 1e-06",
    ):
        water_system.build()
    assert not compiled


def test_non_roundtripping_aggregate_initial_stock_is_refused_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = (680_958_065_804_247, 3_736_979_495_382_760)
    water_system = _water_system(steps=2, quantum=ConservationQuantum.LITRE)
    water_system.source("river", interval_volume([_count_value(count, ConservationQuantum.LITRE) for count in counts]))
    water_system.reach("canal", "river", "farm")
    compiled = False

    def compile_model(document: object) -> None:
        nonlocal compiled
        compiled = True
        raise AssertionError("non-roundtripping aggregate reached incidence")

    monkeypatch.setattr("taqsim.water_system.incidence.compile_model", compile_model)
    with pytest.raises(
        ValueError,
        match=r"source 'river' aggregate initial stock at position 0.*4417937561187007.*quantum 0.001",
    ):
        water_system.build()
    assert not compiled


def test_merged_transfer_counts_drive_retained_stock_and_saved_roundtrip(tmp_path: Path) -> None:
    quantum = ConservationQuantum.MILLILITRE
    individual_count = 4_394_222_044_288_838
    merged_count = 8_788_444_088_577_676
    value = _count_value(individual_count, quantum)
    water_system = _water_system(quantum=quantum)
    for source_name in ("source-a", "source-b"):
        water_system.source(source_name, interval_volume([value]))
        water_system.reach(f"{source_name}-feeder", source_name, "canal")
    water_system.reach("canal", "junction", "farm")

    run = water_system.build().run(bytes([0x62]) * 16)
    retained = run.retained("canal")
    assert list(retained.values) == [quantum.quantum_m3]
    assert retained.presence == (Presence.PRESENT,)

    saved = tmp_path / "merged-count-run.json"
    run.save(saved)
    reopened = load_run(saved)
    assert reopened.retained("canal") == retained

    completed = run._completed
    assert completed is not None
    incoming_counts = completed.transfer_count_series("canal", "water", direction="incoming")
    assert incoming_counts.values == [merged_count]
    assert incoming_counts.presence == [Presence.PRESENT.value]
