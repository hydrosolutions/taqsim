"""acceptance : PublicWaterSystemDeclarations → ExactLifecycleWitnesses."""

from decimal import Decimal
from fractions import Fraction

import polars as pl
import pytest

from taqsim import ConservationQuantum, IntervalMeanRate, IntervalVolume, TimeAxis, WaterSystem, WaterVolume
from taqsim.constituents import (
    Composition,
    Concentration,
    ConservativeTransport,
    Constituent,
    ConstituentInput,
    Mass,
    MassCounts,
    Remobilisation,
)
from taqsim.physical_results import QualityState
from taqsim.vocabulary import EvaporateThenRelease, Hold, MonthlyDistribution, Release, TravelDelay
from taqsim.water_system import WaterSystemRun

SALT = Constituent("salt", "dissolved salt", "as salt", Decimal("0.000001"))
SULPHATE = Constituent("sulphate", "SO4", "as SO4", Decimal("0.000001"))


def composition(mass):
    return Composition((ConstituentInput("salt", mass=mass if isinstance(mass, MassCounts) else Mass(mass)),))


def volume(value):
    return WaterVolume(value, "m3")


def system(periods=1, frequency="1d"):
    return WaterSystem(time=TimeAxis("2020-02-29", periods, frequency), quantum=ConservationQuantum.LITRE)


def source(model, name, values):
    frame = pl.DataFrame(
        {
            "time": [model.time.datetime_at(i).replace(tzinfo=None) for i in range(len(values))],
            "value": [float(v) for v in values],
        }
    )
    model.source(name, IntervalVolume(frame, "m3", model.time.frequency, "0.001 m3"))


def execute(model, *, initial=None, boundaries=None, remobilisation=None, constituents=(SALT,)):
    model.configure_transport(
        ConservativeTransport(constituents, initial or {}, boundaries or {}, remobilisation or {})
    )
    return model.build().run(bytes(16))


def physical(run):
    assert run.physical is not None
    return run.physical


def amounts(result, location, step, water, mass, view="incoming"):
    assert result.amount(location, step, view=view) == Decimal(str(water))
    assert result.amount(location, step, "salt", view=view) == Decimal(str(mass))


def closes(result):
    assert result.balances and result.basin_balances
    for balance in (*result.balances, *result.basin_balances):
        assert balance.start + balance.incoming - balance.outgoing - balance.end == 0
        assert balance.residual == 0


def mixed(split=False, capacity=None):
    model = system()
    source(model, "a", [10])
    source(model, "b", [5])
    model.reach("a-reach", "a", "mix")
    model.reach("b-reach", "b", "mix")
    options = {}
    if split:
        options["rule"] = MonthlyDistribution({"left": (0.4,) * 12, "right": (0.6,) * 12})
    if capacity is not None:
        options.update(capacity=volume(capacity), overflow_destination="spill")
    model.reach("mix", "a-reach", "out", **options)
    return execute(model, boundaries={"a": (composition(8),), "b": (composition(1),)})


def test_a1_complete_mixing():
    result = physical(mixed())
    amounts(result, "out", 0, 15, 9)
    assert result.concentration("out", 0, "salt", view="incoming") == Fraction(3, 5)
    amounts(result, "mix", 0, 0, 0, "storage")
    closes(result)


def test_a2_branch_allocations():
    result = physical(mixed(split=True))
    amounts(result, "left", 0, 6, "3.6")
    amounts(result, "right", 0, 9, "5.4")
    closes(result)


@pytest.mark.parametrize("requested,water,mass,retained_water,retained_mass", [(40, 30, 8, 0, 0), (15, 15, 4, 15, 4)])
def test_a3_a4_realised_release(requested, water, mass, retained_water, retained_mass):
    model = system()
    source(model, "source", [10])
    model.reach("store", "source", "out", initial_water=volume(20), rule=Release(volume(requested)))
    result = physical(execute(model, initial={"store": composition(4)}, boundaries={"source": (composition(4),)}))
    amounts(result, "out", 0, water, mass)
    amounts(result, "store", 0, retained_water, retained_mass, "storage")
    assert result.concentration("out", 0, "salt", view="incoming") == Fraction(mass, water)
    if requested == 40:
        assert Decimal(result.metadata["water_deficit_m3"]["store"][0]) == 10
        assert result.concentration("store", 0, "salt", view="storage") is None
    closes(result)


@pytest.mark.parametrize(
    "evaporation,release,end_water,end_mass,out_mass", [(2, 0, 8, 5, 0), (0, 2, 8, 4, 1), (2, 4, 4, "2.5", "2.5")]
)
def test_a5_a6_and_post_evaporation_order(evaporation, release, end_water, end_mass, out_mass):
    model = system()
    source(model, "source", [0])
    model.reach(
        "store",
        "source",
        "seepage",
        initial_water=volume(10),
        rule=EvaporateThenRelease(volume(evaporation), volume(release)),
    )
    result = physical(execute(model, initial={"store": composition(5)}, boundaries={"source": (composition(0),)}))
    amounts(result, "evaporation", 0, evaporation, 0)
    amounts(result, "seepage", 0, release, out_mass)
    amounts(result, "store", 0, end_water, end_mass, "storage")
    closes(result)


def test_a7_capacity_routes_spill():
    result = physical(mixed(capacity=10))
    amounts(result, "out", 0, 10, 6)
    amounts(result, "spill", 0, 5, 3)
    closes(result)


def delayed():
    model = system(3)
    source(model, "source", [10, 5, 0])
    model.reach("transit", "source", "out", rule=TravelDelay(1))
    return execute(model, boundaries={"source": tuple(composition(m) for m in (8, 1, 0))})


def test_a8_variable_chemistry_parcels_and_aggregate():
    result = physical(delayed())
    for step, water, mass in [(0, 0, 0), (1, 10, 8), (2, 5, 1)]:
        amounts(result, "out", step, water, mass)
    amounts(result, "transit", 0, 10, 8, "storage")
    amounts(result, "transit", 1, 5, 1, "storage")
    aggregate = result.aggregate("out", 0, 3, view="incoming")
    assert result.sample_concentration(aggregate, "salt") == Fraction(3, 5)
    with pytest.raises((TypeError, ValueError)):
        result.aggregate("out", 0, 0.5, view="incoming")
    closes(result)


def use_run(diversion):
    model = system(2)
    source(model, "source", [diversion, 0])
    model.reach("use", "source", "transit", rule=EvaporateThenRelease((volume(4), volume(0)), volume(diversion)))
    model.reach("transit", "use", "return", rule=TravelDelay(1))
    return execute(model, boundaries={"source": (composition(5), composition(0))})


def test_a9_located_use_delayed_return_and_caller_comparison():
    baseline, alternative = (physical(use_run(value)) for value in (10, 6))
    amounts(baseline, "return", 1, 6, 5)
    amounts(baseline, "transit", 0, 6, 5, "storage")
    amounts(baseline, "use", 0, 0, 0, "storage")
    assert baseline.concentration("return", 1, "salt", view="incoming") == Fraction(5, 6)
    assert baseline.amount("use", 0, view="incoming") - alternative.amount("use", 0, view="incoming") == 4
    assert (
        baseline.amount("evaporation", 0, view="incoming") == alternative.amount("evaporation", 0, view="incoming") == 4
    )
    assert baseline.amount("return", 1, view="incoming") - alternative.amount("return", 1, view="incoming") == 4
    closes(baseline)
    closes(alternative)


@pytest.mark.parametrize("remobilisation", [Remobilisation.COMPLETE, Remobilisation.UNRESOLVED])
def test_a10_drying_rewetting_and_dependent_outflow(remobilisation):
    model = system(3)
    source(model, "source", [0, 4, 0])
    model.reach(
        "store",
        "source",
        "out",
        initial_water=volume(2),
        rule=EvaporateThenRelease(tuple(volume(v) for v in (2, 0, 0)), tuple(volume(v) for v in (0, 0, 2))),
    )
    result = physical(
        execute(
            model,
            initial={"store": composition(1)},
            boundaries={"source": tuple(composition(0) for _ in range(3))},
            remobilisation={"store": remobilisation},
        )
    )
    amounts(result, "store", 0, 0, 1, "storage")
    assert result.concentration("store", 0, "salt", view="storage") is None
    amounts(result, "store", 1, 4, 1, "storage")
    if remobilisation is Remobilisation.COMPLETE:
        assert result.concentration("store", 1, "salt", view="storage") == Fraction(1, 4)
        amounts(result, "out", 2, 2, "0.5")
    else:
        assert result.concentration("store", 1, "salt", view="storage") is None
        assert result.concentration("out", 2, "salt", view="incoming") is None
        assert result.sample("out", 2, view="incoming").quality["salt"] is QualityState.UNSUPPORTED
        amounts(result, "store", 2, 2, 1, "storage")
    closes(result)


def test_a11_missing_chloride_does_not_remove_sulphate_support():
    chloride = Constituent("chloride", "Cl", "as Cl", Decimal("0.000001"))
    model = system(3)
    source(model, "source", [10, 5, 0])
    model.reach("mix", "source", "transit", rule=MonthlyDistribution({"transit": (0.5,) * 12, "held": (0.5,) * 12}))
    model.reach("held", "mix", "unused", rule=Hold())
    model.reach("transit", "mix", "out", rule=TravelDelay(1))
    chemistry = tuple(
        Composition(
            (
                ConstituentInput("chloride") if i == 0 else ConstituentInput("chloride", mass=Mass(0)),
                ConstituentInput("sulphate", mass=Mass(m)),
            )
        )
        for i, m in enumerate((4, 2, 0))
    )
    result = physical(execute(model, constituents=(chloride, SULPHATE), boundaries={"source": chemistry}))
    assert result.amount("out", 1, view="incoming") == 5
    assert result.concentration("out", 1, "sulphate", view="incoming") == Fraction(2, 5)
    assert result.concentration("out", 1, "chloride", view="incoming") is None
    assert result.sample("held", 1, view="storage").quality["chloride"] is QualityState.MISSING
    assert result.concentration("held", 1, "sulphate", view="storage") == Fraction(2, 5)


@pytest.mark.parametrize("frequency,seconds", [("1d", 86400), ("1h", 3600), ("2d", 172800)])
def test_a12_rate_integration_once_on_leap_day(frequency, seconds):
    model = system(frequency=frequency)
    frame = pl.DataFrame({"time": [model.time.start.replace(tzinfo=None)], "value": [10.0]})
    rates = IntervalMeanRate(frame, "m3/s", frequency, "0.001 m3/s")
    model.source("source", rates.aggregate_to(model.time))
    model.reach("reach", "source", "out")
    chemistry = Composition((ConstituentInput("salt", concentration=Concentration(500, "mg/l")),))
    result = physical(execute(model, boundaries={"source": (chemistry,)}))
    amounts(result, "out", 0, 10 * seconds, 5 * seconds)
    assert result.metadata["topology"]["source_provenance"]["source"]["kind"] == "interval_mean_rate"
    assert result.metadata["topology"]["source_provenance"]["source"]["unit"] == "m3/s"
    closes(result)


def test_nonround_remainder_is_retained_and_explicitly_remobilised():
    salt = Constituent("salt", "dissolved salt", "as salt", Decimal(1))
    model = system(2)
    source(model, "source", [0, 3])
    model.reach(
        "split",
        "source",
        "out",
        initial_water=volume(3),
        rule=MonthlyDistribution({"one": (1 / 3,) * 12, "two": (2 / 3,) * 12}),
    )
    result = physical(
        execute(
            model,
            constituents=(salt,),
            initial={"split": composition(10)},
            boundaries={"source": (composition(0), composition(2))},
            remobilisation={"split": Remobilisation.COMPLETE},
        )
    )
    amounts(result, "one", 0, 1, 3)
    amounts(result, "two", 0, 2, 6)
    amounts(result, "split", 0, 0, 1, "storage")
    amounts(result, "one", 1, 1, 1)
    amounts(result, "two", 1, 2, 2)
    closes(result)


def test_exact_counts_above_float_precision_and_durable_projection(tmp_path):
    count = 2**53 + 1
    model = system()
    boundaries = {}
    for i in range(4):
        name = f"source-{i}"
        source(model, name, [1])
        model.reach(f"feed-{i}", name, "mix")
        boundaries[name] = (composition(MassCounts(count)),)
    model.reach("mix", "feed-0", "out", rule=MonthlyDistribution({"one": (0.25,) * 12, "three": (0.75,) * 12}))
    run = execute(model, boundaries=boundaries)
    result = physical(run)
    assert result.sample("one", 0, view="incoming").mass_counts["salt"] == count
    assert result.sample("three", 0, view="incoming").mass_counts["salt"] == 3 * count
    path = tmp_path / "run.taqsim"
    run.save(path)
    restored = WaterSystemRun.load(path)
    assert physical(restored).to_dict() == result.to_dict()
    assert physical(model.build().run(bytes(16))).digest == result.digest
    closes(result)


def test_delay_projection_roundtrip_and_presence(tmp_path):
    run = delayed()
    path = tmp_path / "delay.taqsim"
    run.save(path)
    restored = WaterSystemRun.load(path)
    result = physical(restored)
    assert result.to_dict() == physical(run).to_dict()
    assert result.amount("out", 0, view="incoming") == 0
    assert result.sample("out", 3, view="incoming").quality["salt"] is QualityState.OUTSIDE_HORIZON
    with pytest.raises(KeyError):
        result.sample("unknown", 0, view="incoming")


def test_transfer_and_terminal_concentration_are_distinct_after_evaporation():
    model = system()
    source(model, "source", [0])
    model.reach("release", "source", "store", initial_water=volume(10), rule=Release(volume(4)))
    model.reach("store", "release", "out", rule=EvaporateThenRelease(volume(2), volume(0)))
    result = physical(execute(model, initial={"release": composition(5)}, boundaries={"source": (composition(0),)}))
    assert result.concentration("release", 0, "salt") == Fraction(1, 2)
    assert result.concentration("store", 0, "salt", view="storage") == 1
    closes(result)


@pytest.mark.parametrize("remobilisation", [Remobilisation.COMPLETE, Remobilisation.UNRESOLVED])
def test_initial_dry_inventory_has_same_rewetting_contract(remobilisation):
    model = system()
    source(model, "source", [4])
    model.reach("store", "source", "out", rule=Hold())
    result = physical(
        execute(
            model,
            initial={"store": composition(1)},
            boundaries={"source": (composition(1),)},
            remobilisation={"store": remobilisation},
        )
    )
    amounts(result, "store", 0, 4, 2, "storage")
    expected = Fraction(1, 2) if remobilisation is Remobilisation.COMPLETE else None
    assert result.concentration("store", 0, "salt", view="storage") == expected
    closes(result)


def test_zero_realised_branch_does_not_advect_missing_chemistry():
    model = system()
    source(model, "unknown", [10])
    source(model, "known", [4])
    model.reach("blocked", "unknown", "clean", rule=Hold())
    model.reach("feed", "known", "clean")
    model.reach("clean", "feed", "out")
    result = physical(execute(model, boundaries={"unknown": (Composition(),), "known": (composition(2),)}))
    amounts(result, "out", 0, 4, 2)
    assert result.concentration("out", 0, "salt", view="incoming") == Fraction(1, 2)


def test_future_boundary_water_is_not_physical_initial_inventory():
    model = system(2)
    source(model, "source", [1, 1000000])
    model.reach("transit", "source", "out", rule=TravelDelay(1))
    result = physical(execute(model, boundaries={"source": (composition(1), composition(1000000))}))
    for balance in result.basin_balances:
        if balance.step == 0:
            assert balance.start == 0
    amounts(result, "transit", 0, 1, 1, "storage")
    closes(result)


def test_a8_horizon_ends_with_undelivered_transit_inventory():
    model = system()
    source(model, "source", [10])
    model.reach("transit", "source", "out", rule=TravelDelay(1))
    result = physical(execute(model, boundaries={"source": (composition(5),)}))
    amounts(result, "transit", 0, 10, 5, "storage")
    amounts(result, "out", 0, 0, 0)
    closes(result)


@pytest.mark.parametrize("chemistry", [Composition(), composition(1)])
def test_missing_and_unresolved_quality_survive_saved_projection(tmp_path, chemistry):
    model = system()
    source(model, "source", [4])
    model.reach("store", "source", "out", rule=Release(volume(2)))
    run = execute(
        model,
        initial={"store": chemistry},
        boundaries={"source": (composition(0) if chemistry.entries else Composition(),)},
    )
    result = physical(run)
    assert result.concentration("out", 0, "salt", view="incoming") is None
    path = tmp_path / "unsupported.taqsim"
    run.save(path)
    restored = physical(WaterSystemRun.load(path))
    assert restored.to_dict() == result.to_dict()
    assert restored.concentration("out", 0, "salt", view="incoming") is None
