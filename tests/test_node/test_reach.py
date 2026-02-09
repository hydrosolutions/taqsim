from dataclasses import dataclass
from typing import Any

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node.events import (
    WaterEnteredReach,
    WaterInTransit,
    WaterLost,
    WaterOutput,
    WaterReceived,
)
from taqsim.node.protocols import Receives
from taqsim.node.reach import Reach
from taqsim.node.strategies import NoReachLoss, NoRouting, ReachLossRule, RoutingModel
from taqsim.time import Frequency, Timestep

# --- Fakes ---


@dataclass(frozen=True)
class FakeReachLossRule:
    losses: dict[LossReason, float] = None

    def __post_init__(self):
        if self.losses is None:
            object.__setattr__(self, "losses", {})

    def calculate(self, reach: "Reach", flow: float, t: Timestep) -> dict[LossReason, float]:
        return self.losses


class PassThroughRouting:
    """Instantaneous pass-through: outflow = inflow, no storage."""

    def initial_state(self, reach: "Reach") -> None:
        return None

    def route(self, reach: "Reach", inflow: float, state: None, t: Timestep) -> tuple[float, None]:
        return (inflow, None)

    def storage(self, state: None) -> float:
        return 0.0


class BufferingRouting:
    """1-step delay: stores inflow, releases previous step's inflow."""

    def initial_state(self, reach: "Reach") -> float:
        return 0.0

    def route(self, reach: "Reach", inflow: float, state: float, t: Timestep) -> tuple[float, float]:
        outflow = state  # release what was buffered
        new_state = inflow  # buffer current inflow
        return (outflow, new_state)

    def storage(self, state: float) -> float:
        return state


class SpyRouting:
    """Records call order for verification."""

    def __init__(self):
        self.calls: list[str] = []
        self.inflows: list[float] = []

    def initial_state(self, reach: "Reach") -> None:
        return None

    def route(self, reach: "Reach", inflow: float, state: None, t: Timestep) -> tuple[float, None]:
        self.calls.append("route")
        self.inflows.append(inflow)
        return (inflow, None)

    def storage(self, state: None) -> float:
        return 0.0


T = Timestep(0, Frequency.MONTHLY)
T1 = Timestep(1, Frequency.MONTHLY)
T2 = Timestep(2, Frequency.MONTHLY)


def make_reach(
    id: str = "reach",
    routing_model: Any = None,
    loss_rule: Any = None,
) -> Reach:
    if routing_model is None:
        routing_model = PassThroughRouting()
    if loss_rule is None:
        loss_rule = FakeReachLossRule()
    return Reach(id=id, routing_model=routing_model, loss_rule=loss_rule)


# --- Tests ---


class TestReachInit:
    def test_requires_routing_model(self):
        with pytest.raises(ValueError, match="routing_model is required"):
            Reach(id="r", routing_model=None, loss_rule=FakeReachLossRule())

    def test_requires_loss_rule(self):
        with pytest.raises(ValueError, match="loss_rule is required"):
            Reach(id="r", routing_model=PassThroughRouting(), loss_rule=None)

    def test_creates_with_valid_params(self):
        reach = make_reach(id="canal")
        assert reach.id == "canal"

    def test_initializes_routing_state(self):
        reach = make_reach(routing_model=BufferingRouting())
        assert reach._routing_state == 0.0

    def test_water_in_transit_initially_zero_for_passthrough(self):
        reach = make_reach()
        assert reach.water_in_transit == 0.0

    def test_water_in_transit_initially_zero_for_buffering(self):
        reach = make_reach(routing_model=BufferingRouting())
        assert reach.water_in_transit == 0.0


class TestReachReceive:
    def test_receive_records_event(self):
        reach = make_reach()
        reach.receive(50.0, "upstream", T)

        events = reach.events_of_type(WaterReceived)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].source_id == "upstream"

    def test_receive_accumulates(self):
        reach = make_reach()
        reach.receive(30.0, "a", T)
        reach.receive(20.0, "b", T)

        events = reach.events_of_type(WaterReceived)
        assert len(events) == 2
        assert reach._received_this_step == 50.0

    def test_receive_returns_amount(self):
        reach = make_reach()
        result = reach.receive(75.0, "src", T)
        assert result == 75.0


class TestReachUpdate:
    def test_records_water_entered_reach(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)

        events = reach.events_of_type(WaterEnteredReach)
        assert len(events) == 1
        assert events[0].amount == 100.0

    def test_passthrough_outputs_full_amount(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 100.0

    def test_records_water_in_transit_snapshot(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)

        events = reach.events_of_type(WaterInTransit)
        assert len(events) == 1
        assert events[0].amount == 0.0  # passthrough has 0 storage

    def test_resets_received_after_update(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)
        assert reach._received_this_step == 0.0

    def test_losses_reduce_outflow(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 10.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 90.0

    def test_records_loss_events(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 10.0, SEEPAGE: 5.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        loss_events = reach.events_of_type(WaterLost)
        assert len(loss_events) == 2
        reasons = {e.reason: e.amount for e in loss_events}
        assert reasons[EVAPORATION] == 10.0
        assert reasons[SEEPAGE] == 5.0

    def test_scales_losses_when_exceeding_outflow(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 80.0, SEEPAGE: 20.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(10.0, "src", T)
        reach.update(T)

        # All water lost, nothing delivered
        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 0

        loss_events = reach.events_of_type(WaterLost)
        evap = next(e for e in loss_events if e.reason == EVAPORATION)
        seep = next(e for e in loss_events if e.reason == SEEPAGE)
        assert evap.amount == pytest.approx(8.0)
        assert seep.amount == pytest.approx(2.0)

    def test_no_output_event_when_zero_outflow(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 100.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 0


class TestReachMassBalance:
    def test_mass_balance_with_passthrough(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 10.0, SEEPAGE: 5.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        loss_total = sum(e.amount for e in reach.events_of_type(WaterLost))
        output_total = sum(e.amount for e in reach.events_of_type(WaterOutput))
        transit = reach.water_in_transit

        # inflow = losses + output + transit
        assert pytest.approx(loss_total + output_total + transit) == 100.0

    def test_mass_balance_with_buffering(self):
        reach = make_reach(routing_model=BufferingRouting())
        reach.receive(100.0, "src", T)
        reach.update(T)

        # First step: all buffered, outflow=0, transit=100
        output_total = sum(e.amount for e in reach.events_of_type(WaterOutput))
        assert output_total == 0.0
        assert reach.water_in_transit == 100.0


class TestReachReset:
    def test_reset_clears_events(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)
        assert len(reach.events) > 0

        reach.reset()
        assert len(reach.events) == 0

    def test_reset_reinitializes_routing_state(self):
        reach = make_reach(routing_model=BufferingRouting())
        reach.receive(100.0, "src", T)
        reach.update(T)
        assert reach.water_in_transit == 100.0

        reach.reset()
        assert reach.water_in_transit == 0.0

    def test_reset_clears_received_accumulator(self):
        reach = make_reach()
        reach.receive(50.0, "src", T)
        reach.reset()
        assert reach._received_this_step == 0.0


class TestReachFreshCopy:
    def test_creates_new_instance(self):
        reach = make_reach()
        copy = reach._fresh_copy()
        assert copy is not reach
        assert type(copy) is Reach

    def test_preserves_routing_model(self):
        model = PassThroughRouting()
        reach = make_reach(routing_model=model)
        copy = reach._fresh_copy()
        assert copy.routing_model is model

    def test_fresh_copy_has_clean_state(self):
        reach = make_reach(routing_model=BufferingRouting())
        reach.receive(100.0, "src", T)
        reach.update(T)

        copy = reach._fresh_copy()
        assert copy.water_in_transit == 0.0
        assert len(copy.events) == 0


class TestReachProtocolCompliance:
    def test_satisfies_receives_protocol(self):
        reach = make_reach()
        assert isinstance(reach, Receives)

    def test_routing_model_satisfies_protocol(self):
        assert isinstance(PassThroughRouting(), RoutingModel)
        assert isinstance(BufferingRouting(), RoutingModel)
        assert isinstance(NoRouting(), RoutingModel)

    def test_loss_rule_satisfies_protocol(self):
        assert isinstance(FakeReachLossRule(), ReachLossRule)
        assert isinstance(NoReachLoss(), ReachLossRule)


class TestReachEvents:
    def test_event_sequence_for_passthrough(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)

        event_types = [type(e).__name__ for e in reach.events]
        assert event_types == [
            "WaterReceived",
            "WaterEnteredReach",
            "WaterInTransit",
            "WaterOutput",
        ]

    def test_event_sequence_with_losses(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 10.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        event_types = [type(e).__name__ for e in reach.events]
        assert event_types == [
            "WaterReceived",
            "WaterEnteredReach",
            "WaterLost",
            "WaterInTransit",
            "WaterOutput",
        ]


class TestReachMultipleTimesteps:
    def test_buffering_releases_previous_inflow(self):
        reach = make_reach(routing_model=BufferingRouting())

        # Step 0: receive 100, output 0 (nothing buffered yet)
        reach.receive(100.0, "src", T)
        reach.update(T)

        # Step 1: receive 50, output 100 (buffered from step 0)
        reach.receive(50.0, "src", T1)
        reach.update(T1)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1  # only step 1 has output
        assert outputs[0].amount == 100.0
        assert outputs[0].t == 1

    def test_transit_water_tracks_buffered_amount(self):
        reach = make_reach(routing_model=BufferingRouting())

        reach.receive(100.0, "src", T)
        reach.update(T)
        assert reach.water_in_transit == 100.0

        reach.receive(50.0, "src", T1)
        reach.update(T1)
        assert reach.water_in_transit == 50.0

    def test_spy_records_all_inflows(self):
        spy = SpyRouting()
        reach = make_reach(routing_model=spy)

        reach.receive(100.0, "src", T)
        reach.update(T)
        reach.receive(50.0, "src", T1)
        reach.update(T1)

        assert spy.inflows == [100.0, 50.0]
        assert spy.calls == ["route", "route"]


class TestReachWithNoRouting:
    def test_no_routing_passes_through(self):
        reach = make_reach(routing_model=NoRouting(), loss_rule=NoReachLoss())
        reach.receive(100.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 100.0

    def test_no_routing_has_zero_transit(self):
        reach = make_reach(routing_model=NoRouting())
        reach.receive(100.0, "src", T)
        reach.update(T)
        assert reach.water_in_transit == 0.0
