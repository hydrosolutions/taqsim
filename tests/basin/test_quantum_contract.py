"""quantum contract evidence : ModelDeclarations → Exact conserved run observations."""

from __future__ import annotations

import json
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from taqsim import (
    Basin,
    BasinObjective,
    CanalLosses,
    EFlowSplit,
    MonthlyDistribution,
    Parameter,
    Presence,
    PriorityDistribution,
    ReservoirEvaporation,
    Resolution,
    ZoneRelease,
    load_run,
    optimize_basin,
)


def _basin(*, steps: int = 1, resolution: Resolution = Resolution.LITRE) -> Basin:
    return Basin(start_date="2020-01-01", timesteps=steps, resolution=resolution)


def _count(value: float | None, resolution: Resolution) -> int:
    assert value is not None
    return round(value / resolution.quantum_m3)


def _count_value(count: int, resolution: Resolution) -> float:
    return float(count) * resolution.quantum_m3


def test_external_water_must_align_before_incidence_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = False

    def compile_model(document: object) -> None:
        nonlocal compiled
        compiled = True
        raise AssertionError("misaligned water reached incidence")

    monkeypatch.setattr("taqsim.basin.incidence.compile_model", compile_model)
    source = _basin(resolution=Resolution.LITRE)
    source.source("river", [1.0001])
    source.reach("canal", "river", "farm")
    with pytest.raises(ValueError, match=r"source 'river'.*timestep 0.*1\.0001.*quantum 0\.001"):
        source.build()
    assert not compiled

    initial = _basin(resolution=Resolution.LITRE)
    initial.source("river", [1.0])
    initial.reach("canal", "river", "farm", initial_water=0.0001)
    with pytest.raises(ValueError, match=r"reach 'canal' initial stock.*position 0.*0\.0001.*quantum 0\.001"):
        initial.build()
    assert not compiled


def test_boundary_counts_are_aggregated_without_float_sum_authority() -> None:
    basin = _basin(steps=2, resolution=Resolution.LITRE)
    basin.source("river", [0.1, 0.2])
    basin.reach("canal", "river", "farm")
    model = basin.build()
    stock = next(item for item in model.document["initial_stocks"] if item["compartment"] == "river")
    assert stock["amounts"][0]["amount"] == 0.3


def test_reported_canal_case_accounts_in_integer_counts_and_carries_residual() -> None:
    resolution = Resolution.LITRE
    basin = _basin(steps=2, resolution=resolution)
    basin.source("river", [10000.0, 8000.0])
    basin.reach("canal", "river", "farm", rule=CanalLosses(0.0014, 12.0))
    run = basin.build().run(bytes(16))

    incoming = [_count(value, resolution) for value in run.arrivals("canal").values]
    delivered = [_count(value, resolution) for value in run.arrivals("farm").values]
    seepage = [_count(value, resolution) for value in run.arrivals("seepage").values]
    evaporation = [_count(value, resolution) for value in run.arrivals("evaporation").values]
    operational = [_count(value, resolution) for value in run.arrivals("operational-loss").values]
    retained = [_count(value, resolution) for value in run.retained("canal").values]
    previous = 0
    for step in range(2):
        assert (
            previous + incoming[step]
            == delivered[step] + seepage[step] + evaporation[step] + operational[step] + retained[step]
        )
        previous = retained[step]
    assert retained == [1, 1]


def test_computed_subquantum_zero_is_present_not_absent_or_not_modelled() -> None:
    basin = _basin(resolution=Resolution.CUBIC_METRE)
    basin.source("river", [1.0])
    basin.reach("canal", "river", "farm", rule=CanalLosses(1e-9, 1.0))
    run = basin.build().run(bytes([2]) * 16)
    seepage = run.arrivals("seepage")
    assert list(seepage.values) == [0.0]
    assert seepage.presence == (Presence.PRESENT,)
    extended = run.flow("canal", start="2019-12-31", end="2020-01-02")
    assert extended.presence == (Presence.NOT_MODELLED, Presence.PRESENT, Presence.NOT_MODELLED)
    with pytest.raises(KeyError, match="unknown reach"):
        run.flow("never-modelled")


def test_resolution_is_bound_to_model_live_run_and_saved_run(tmp_path: Path) -> None:
    def build(resolution: Resolution):
        basin = _basin(resolution=resolution)
        basin.source("river", [1.0])
        basin.reach("canal", "river", "farm")
        return basin.build()

    litre = build(Resolution.LITRE)
    millilitre = build(Resolution.MILLILITRE)
    assert litre.model_digest != millilitre.model_digest
    assert litre.resolution is Resolution.LITRE
    run = litre.run(bytes([3]) * 16)
    assert run.resolution is Resolution.LITRE
    saved = tmp_path / "run.json"
    run.save(saved)
    reopened = load_run(saved)
    assert reopened.resolution is Resolution.LITRE
    assert reopened.model_digest == run.model_digest == litre.model_digest
    document = json.loads(saved.read_text())
    assert document["resolution"] == Resolution.LITRE.value
    assert document["model_digest"] == litre.model_digest


def test_repeated_execution_has_identical_authoritative_digest() -> None:
    basin = _basin(steps=2, resolution=Resolution.LITRE)
    basin.source("river", [3.141, 2.718])
    basin.reach("canal", "river", "farm", rule=CanalLosses(0.0014, 12.0))
    model = basin.build()
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
        basin = _basin(steps=2, resolution=Resolution.LITRE)
        basin.source("river", [flow, flow * 0.8])
        basin.reach("canal", "river", "farm", rule=CanalLosses(coefficient, length))
        assert basin.build().run(index.to_bytes(16, "big")).authoritative_log_digest


def test_priority_optimizer_completes_first_generation() -> None:
    basin = _basin(resolution=Resolution.LITRE)
    basin.source("river", [_count_value(123_457, Resolution.LITRE)])
    basin.reach(
        "split",
        "river",
        "tail",
        rule=PriorityDistribution("priority", Parameter("amount", 17.1234, (1.1111, 80.8888)), {"other": 1.0}),
    )
    model = basin.build()
    result = optimize_basin(
        model,
        [BasinObjective("priority", "maximize", lambda run: float(run.arrivals("priority").values[0] or 0.0))],
        pop_size=8,
        generations=1,
        seed=19,
    )
    assert result.solutions


def test_seeded_nonround_samples_cover_all_six_rule_compilers() -> None:
    rng = random.Random(11)
    factories = (
        lambda: PriorityDistribution("downstream", rng.uniform(1.0, 900.0), {"other": 1.0}),
        lambda: EFlowSplit({"downstream": 1.0}, {"other": 1.0}, rng.uniform(0.01, 0.6)),
        lambda: ReservoirEvaporation((rng.uniform(0.5, 120.0),) * 12, ((0.0, 0.0), (1000.0, 10_000.0))),
        lambda: (
            lambda fraction: MonthlyDistribution({"downstream": (fraction,) * 12, "other": (1.0 - fraction,) * 12})
        )(rng.uniform(0.05, 0.95)),
        lambda: ZoneRelease(0.0, 2_000.0, 50_000.0, rng.uniform(0.0001, 0.02)),
        lambda: CanalLosses(rng.uniform(1e-4, 1e-2), rng.uniform(0.5, 15.0)),
    )
    for shape_index, factory in enumerate(factories):
        for sample in range(60):
            basin = _basin(steps=2, resolution=Resolution.LITRE)
            basin.source("source", [1000.0, 800.0])
            basin.reach("structure", "source", "downstream", rule=factory())
            run_id = (1000 + shape_index * 60 + sample).to_bytes(16, "big")
            assert basin.build().run(run_id).authoritative_log_digest


def test_sweep_and_replay_complete_at_ordinary_nonround_parameters() -> None:
    basin = _basin(resolution=Resolution.LITRE)
    basin.source("river", [_count_value(321_987, Resolution.LITRE)])
    basin.reach(
        "canal",
        "river",
        "farm",
        rule=CanalLosses(Parameter("seepage", 0.00143, (0.00017, 0.00983)), 7.321),
    )
    model = basin.build()
    runs = model.sweep("canal.seepage", [0.000173, 0.002719, 0.008997])
    assert len(runs) == 3
    first = model.run(bytes([71]) * 16, {"canal.seepage": 0.002719})
    repeated = model.run(bytes([71]) * 16, {"canal.seepage": 0.002719})
    first.verify_replay(repeated)
    assert first.authoritative_log_digest == repeated.authoritative_log_digest


def test_multibranch_residual_is_retained_and_used_on_the_next_timestep() -> None:
    left = (0.45, 1.0) + (1.0,) * 10
    right = (0.45, 0.0) + (0.0,) * 10
    basin = Basin(
        start_date="2020-01-01",
        timesteps=2,
        timestep=timedelta(days=31),
        resolution=Resolution.LITRE,
    )
    basin.source("river", [0.010, 0.0])
    basin.reach("split", "river", "unused", rule=MonthlyDistribution({"left": left, "right": right}))
    run = basin.build().run(bytes([81]) * 16)
    assert list(run.retained("split").values) == [0.002, 0.0]
    assert list(run.arrivals("left").values) == [0.004, 0.002]
    incoming = np.array([_count(value, Resolution.LITRE) for value in run.arrivals("split").values])
    left_counts = np.array([_count(value, Resolution.LITRE) for value in run.arrivals("left").values])
    right_counts = np.array([_count(value, Resolution.LITRE) for value in run.arrivals("right").values])
    retained = np.array([_count(value, Resolution.LITRE) for value in run.retained("split").values])
    np.testing.assert_array_equal(incoming + np.array([0, 2]), left_counts + right_counts + retained)


def test_ambiguous_aggregate_initial_stock_is_refused_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 4_394_222_044_288_838
    value = _count_value(count, Resolution.MILLILITRE)
    basin = _basin(steps=2, resolution=Resolution.MILLILITRE)
    basin.source("river", [value, value])
    basin.reach("canal", "river", "farm")
    compiled = False

    def compile_model(document: object) -> None:
        nonlocal compiled
        compiled = True
        raise AssertionError("ambiguous aggregate reached incidence")

    monkeypatch.setattr("taqsim.basin.incidence.compile_model", compile_model)
    with pytest.raises(
        ValueError,
        match=r"source 'river' aggregate initial stock at position 0.*8788444088577676.*quantum 1e-06",
    ):
        basin.build()
    assert not compiled


def test_non_roundtripping_aggregate_initial_stock_is_refused_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = (680_958_065_804_247, 3_736_979_495_382_760)
    basin = _basin(steps=2, resolution=Resolution.LITRE)
    basin.source("river", [_count_value(count, Resolution.LITRE) for count in counts])
    basin.reach("canal", "river", "farm")
    compiled = False

    def compile_model(document: object) -> None:
        nonlocal compiled
        compiled = True
        raise AssertionError("non-roundtripping aggregate reached incidence")

    monkeypatch.setattr("taqsim.basin.incidence.compile_model", compile_model)
    with pytest.raises(
        ValueError,
        match=r"source 'river' aggregate initial stock at position 0.*4417937561187007.*quantum 0.001",
    ):
        basin.build()
    assert not compiled


def test_merged_transfer_counts_drive_retained_stock_and_saved_roundtrip(tmp_path: Path) -> None:
    resolution = Resolution.MILLILITRE
    individual_count = 4_394_222_044_288_838
    merged_count = 8_788_444_088_577_676
    value = _count_value(individual_count, resolution)
    basin = _basin(resolution=resolution)
    for source_name in ("source-a", "source-b"):
        basin.source(source_name, [value])
        basin.reach(f"{source_name}-feeder", source_name, "canal")
    basin.reach("canal", "junction", "farm")

    run = basin.build().run(bytes([0x62]) * 16)
    retained = run.retained("canal")
    assert list(retained.values) == [resolution.quantum_m3]
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
