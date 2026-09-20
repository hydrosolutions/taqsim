from __future__ import annotations

import pytest

from taqsim import (
    CanalLosses,
    CanalSeepageCoefficient,
    EFlowSplit,
    Length,
    MonthlyDistribution,
    Parameter,
    PriorityDistribution,
    ReservoirEvaporation,
    SurfaceArea,
    VolumetricRate,
    WaterDepth,
    WaterVolume,
    ZoneRelease,
)
from taqsim.constituents import ConservativeTransport, Constituent
from taqsim.vocabulary import EvaporateThenRelease, Hold, Release, RulePlan, TravelDelay
from tests import interval_volume, make_water_system


def _system(rule, periods=3):
    system = make_water_system(periods, "1 mL")
    system.source("supply", interval_volume([100.0] * periods))
    system.reach("structure", "supply", "outlet", rule=rule)
    return system


RULES = [
    None,
    Hold(),
    Release(WaterVolume(30, "m3")),
    TravelDelay(1),
    ZoneRelease(WaterVolume(0, "m3"), WaterVolume(30, "m3"), WaterVolume(80, "m3"), VolumetricRate(0.001, "m3/s")),
    EFlowSplit({"left": 1.0}, {"right": 1.0}),
    PriorityDistribution("left", WaterVolume(20, "m3"), {"right": 1.0}),
    MonthlyDistribution({"left": (0.6,) * 12, "right": (0.4,) * 12}),
    ReservoirEvaporation(
        (WaterDepth(10, "mm"),) * 12,
        ((WaterVolume(0, "m3"), SurfaceArea(0, "m2")), (WaterVolume(100, "m3"), SurfaceArea(1000, "m2"))),
    ),
    CanalLosses(
        CanalSeepageCoefficient(0.01, "sqrt(m3/s)/km"),
        Length(1, "km"),
        evaporation_depth=WaterDepth(1, "mm"),
        width=Length(1, "m"),
        operational_fraction=Parameter("operation", 0.01),
    ),
    EvaporateThenRelease(WaterVolume(2, "m3"), WaterVolume(40, "m3")),
]


@pytest.mark.parametrize("rule", RULES)
def test_observation_preserves_original_water_authority(rule):
    plain = _system(rule).build().run(bytes(16))
    system = _system(rule)
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    built = system.build()
    coupled = built.run(bytes(16))
    assert coupled.flow("structure") == plain.flow("structure")
    assert coupled.retained("structure") == plain.retained("structure")
    for endpoint in built.document["boundary_accounts"]:
        assert coupled.arrivals(endpoint) == plain.arrivals(endpoint)
    assert coupled._completed is not None
    assert built._topology is not None
    for branch in built._topology.branches["structure"]:
        incoming = coupled._completed.transfer_count_series(
            branch.observer, "water", direction="incoming", first=0, last=2
        )
        outgoing = coupled._completed.transfer_count_series(
            branch.observer, "water", direction="outgoing", first=0, last=2
        )
        assert incoming.values == outgoing.values


def test_travel_delay_real_retention_and_schedule():
    system = make_water_system(3, "1 m3")
    system.source("supply", interval_volume([10.0, 20.0, 0.0]))
    system.reach("transit", "supply", "outlet", rule=TravelDelay(1))
    run = system.build().run(bytes(16))
    assert tuple(run.arrivals("outlet").values) == (0.0, 10.0, 20.0)
    assert tuple(run.retained("transit").values) == (10.0, 20.0, 0.0)


def test_custom_rule_requires_declared_physical_semantics():
    class Undeclared:
        def compile(self, context, downstream):
            return RulePlan((("out", downstream, context.available),))

    system = _system(Undeclared())
    system.build().run(bytes(16))
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    with pytest.raises(ValueError, match="explicitly declare"):
        system.build()


def test_process_replacement_disables_both_old_water_directions():
    from taqsim.constituents import Composition, ConstituentInput, ExchangeReplacement, Mass

    system = make_water_system(1, "1 m3")
    system.source("old-input", interval_volume([10.0]))
    system.source("new-input", interval_volume([4.0]))
    system.reach("old-store", "old-input", "old-output", initial_water=WaterVolume(3, "m3"))
    system.reach("new-store", "new-input", "new-output")
    system.process_exchange("old-model", sources=("old-input",), branches=(("old-store", "out"),))
    system.process_exchange("new-model", sources=("new-input",), branches=(("new-store", "out"),))
    system.configure_transport(
        ConservativeTransport(
            (Constituent("salt", "salt", "salt"),),
            boundaries={
                "old-input": (Composition((ConstituentInput("salt", mass=Mass(5)),)),),
                "new-input": (Composition((ConstituentInput("salt", mass=Mass(2)),)),),
            },
            replacements=(ExchangeReplacement("old-model", "new-model"),),
        )
    )
    run = system.build().run(bytes(16))
    assert tuple(run.arrivals("old-store").values) == (0.0,)
    assert tuple(run.arrivals("old-output").values) == (0.0,)
    assert tuple(run.retained("old-store").values) == (3.0,)
    assert tuple(run.arrivals("new-output").values) == (4.0,)
    assert run.physical.amount("new-output", 0, "salt", view="incoming") == 2
    assert run.physical.amount("old-store", 0, "salt", view="incoming") == 0


def test_parameter_substitution_preserves_water_authority():
    rule = CanalLosses(
        CanalSeepageCoefficient(0.01, "sqrt(m3/s)/km"),
        Length(1, "km"),
        operational_fraction=Parameter("operation", 0.01),
    )
    plain = _system(rule).build().run(bytes(16), {"structure.operation": 0.2})
    system = _system(rule)
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    observed = system.build().run(bytes(16), {"structure.operation": 0.2})
    for endpoint in ("outlet", "seepage", "evaporation", "operational-loss"):
        assert observed.arrivals(endpoint) == plain.arrivals(endpoint)


def test_shared_destination_branches_stay_individually_observable():
    from taqsim import PriorityDistribution

    rule = PriorityDistribution("shared", WaterVolume(20, "m3"), {"shared": 1.0})
    plain = _system(rule).build().run(bytes(16))
    system = _system(rule)
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    built = system.build()
    observed = built.run(bytes(16))
    assert observed.arrivals("shared") == plain.arrivals("shared")
    branches = built._topology.branches["structure"]
    assert branches[0].observer != branches[1].observer
    counts = [
        observed._completed.transfer_count_series(
            branch.observer, "water", direction="incoming", first=0, last=2
        ).values
        for branch in branches
    ]
    assert tuple(counts[0]) == (20_000_000,) * 3
    assert tuple(counts[1]) == (80_000_000,) * 3


def test_observers_preserve_merged_exact_counts_when_float_projections_collide():
    system = make_water_system(2, "1 mm3")
    system.source("a", interval_volume([4503599627369506 * 1e-9, 0.0]))
    system.source("b", interval_volume([4503599627369516 * 1e-9, 0.0]))
    system.reach("ra", "a", "transit")
    system.reach("rb", "b", "transit")
    system.reach("transit", "ra", "outlet", rule=TravelDelay(1))
    plain = system.build().run(bytes(16))
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    coupled = system.build().run(bytes(16))
    # These public floats collide. Only the authoritative count comparison discriminates.
    assert plain.arrivals("outlet") == coupled.arrivals("outlet")
    assert plain._completed is not None
    assert coupled._completed is not None
    for location in ("ra", "rb", "transit", "outlet"):
        for direction in ("incoming", "outgoing"):
            expected = plain._completed.transfer_count_series(location, "water", direction=direction, first=0, last=1)
            actual = coupled._completed.transfer_count_series(location, "water", direction=direction, first=0, last=1)
            assert tuple(actual.values) == tuple(expected.values)


def test_capacity_overflow_zero_branch_and_downstream_storage_equivalence():
    system = make_water_system(3, "1 m3")
    system.source("supply", interval_volume([5.0, 15.0, 0.0]))
    system.reach("capacity", "supply", "reservoir", capacity=WaterVolume(10, "m3"), overflow_destination="spill")
    system.reach(
        "reservoir", "capacity", "outlet", initial_water=WaterVolume(4, "m3"), rule=Release(WaterVolume(3, "m3"))
    )
    plain = system.build().run(bytes(16))
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    observed = system.build().run(bytes(16))
    for name in ("capacity", "reservoir"):
        assert observed.flow(name) == plain.flow(name)
        assert observed.retained(name) == plain.retained(name)
    for name in ("spill", "outlet", "reservoir"):
        assert observed.arrivals(name) == plain.arrivals(name)
    assert tuple(observed.arrivals("spill").values) == (0.0, 5.0, 0.0)


@pytest.mark.parametrize("intervals", [0, -1, True, 1.5])
def test_invalid_travel_delay_is_refused(intervals):
    with pytest.raises(ValueError, match="positive integer"):
        TravelDelay(intervals)


def test_unknown_process_replacement_is_refused():
    from taqsim.constituents import ExchangeReplacement

    system = _system(None)
    system.configure_transport(
        ConservativeTransport(
            (Constituent("salt", "salt", "salt"),), replacements=(ExchangeReplacement("unknown", "replacement"),)
        )
    )
    with pytest.raises(ValueError, match="declared process owners"):
        system.build()


def test_transport_topology_is_immutable_and_excludes_observers():
    system = _system(None)
    system.configure_transport(ConservativeTransport((Constituent("salt", "salt", "salt"),)))
    built = system.build()
    assert built._topology.order == ("structure",)
    with pytest.raises(TypeError):
        built._topology.initial_counts["structure"] = 1
    assert all("observation" not in balance.location for balance in built.run(bytes(16)).physical.balances)
