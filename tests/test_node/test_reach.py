from dataclasses import dataclass
from typing import Any

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node.events import (
    WaterEnteredReach,
    WaterExitedReach,
    WaterInTransit,
    WaterLost,
    WaterOutput,
    WaterReceived,
    WaterSpilled,
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

    def test_exited_equals_losses_plus_output(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 10.0, SEEPAGE: 5.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        exited = sum(e.amount for e in reach.events_of_type(WaterExitedReach))
        losses = sum(e.amount for e in reach.events_of_type(WaterLost))
        output = sum(e.amount for e in reach.events_of_type(WaterOutput))

        assert exited == losses + output


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
            "WaterExitedReach",
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
            "WaterExitedReach",
            "WaterLost",
            "WaterInTransit",
            "WaterOutput",
        ]

    def test_water_exited_reach_records_routed_outflow(self):
        reach = make_reach()
        reach.receive(100.0, "src", T)
        reach.update(T)

        events = reach.events_of_type(WaterExitedReach)
        assert len(events) == 1
        assert events[0].amount == 100.0

    def test_water_exited_reach_amount_is_before_losses(self):
        loss_rule = FakeReachLossRule(losses={EVAPORATION: 10.0})
        reach = make_reach(loss_rule=loss_rule)
        reach.receive(100.0, "src", T)
        reach.update(T)

        exited = reach.events_of_type(WaterExitedReach)
        output = reach.events_of_type(WaterOutput)
        assert exited[0].amount == 100.0
        assert output[0].amount == 90.0

    def test_water_exited_reach_zero_when_buffering_delays_all(self):
        reach = make_reach(routing_model=BufferingRouting())
        reach.receive(100.0, "src", T)
        reach.update(T)

        exited = reach.events_of_type(WaterExitedReach)
        assert len(exited) == 1
        assert exited[0].amount == 0.0

    def test_water_exited_reach_emitted_when_buffered_water_released(self):
        reach = make_reach(routing_model=BufferingRouting())
        reach.receive(100.0, "src", T)
        reach.update(T)

        reach.receive(0.0, "src", T1)
        reach.update(T1)

        exited = reach.events_of_type(WaterExitedReach)
        assert len(exited) == 2
        assert exited[0].amount == 0.0  # step 0: nothing exits
        assert exited[1].amount == 100.0  # step 1: buffered water released


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


class TestReachCapacity:
    """Tests for the optional capacity field on Reach."""

    def _make(
        self,
        capacity: float | None = None,
        routing_model: Any = None,
        loss_rule: Any = None,
    ) -> Reach:
        if routing_model is None:
            routing_model = PassThroughRouting()
        if loss_rule is None:
            loss_rule = FakeReachLossRule()
        return Reach(
            id="r",
            routing_model=routing_model,
            loss_rule=loss_rule,
            capacity=capacity,
        )

    # --- Validation ---

    def test_capacity_none_is_valid(self):
        reach = make_reach()
        assert reach.capacity is None

    def test_capacity_positive_is_valid(self):
        reach = self._make(capacity=100.0)
        assert reach.capacity == 100.0

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            self._make(capacity=0)

    def test_capacity_negative_raises(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            self._make(capacity=-10)

    # --- No capacity (unlimited) ---

    def test_no_capacity_passes_all_flow(self):
        reach = self._make()
        reach.receive(200.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 200.0

    def test_no_capacity_emits_no_spill_events(self):
        reach = self._make()
        reach.receive(200.0, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 0

    # --- Below capacity ---

    def test_below_capacity_passes_all_flow(self):
        reach = self._make(capacity=100.0)
        reach.receive(50.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 50.0

    def test_below_capacity_emits_no_spill_event(self):
        reach = self._make(capacity=100.0)
        reach.receive(50.0, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 0

    def test_below_capacity_multiple_sources(self):
        reach = self._make(capacity=100.0)
        reach.receive(30.0, "A", T)
        reach.receive(20.0, "B", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 50.0

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 0

    # --- Equals capacity ---

    def test_at_capacity_boundary_no_spill(self):
        reach = self._make(capacity=100.0)
        reach.receive(100.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 100.0

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 0

    # --- Exceeds capacity ---

    def test_exceeds_capacity_limits_output(self):
        reach = self._make(capacity=100.0)
        reach.receive(150.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 100.0

    def test_exceeds_capacity_records_spill(self):
        reach = self._make(capacity=100.0)
        reach.receive(150.0, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].amount == 50.0

    def test_exceeds_capacity_correct_timestep(self):
        reach = self._make(capacity=100.0)
        reach.receive(150.0, "src", T1)
        reach.update(T1)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].t == T1.index

    def test_exceeds_capacity_multiple_sources(self):
        reach = self._make(capacity=100.0)
        reach.receive(80.0, "A", T)
        reach.receive(70.0, "B", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 100.0

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].amount == 50.0

    def test_exceeds_capacity_large_overflow(self):
        reach = self._make(capacity=10.0)
        reach.receive(1000.0, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].amount == 990.0

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 10.0

    # --- Capacity + routing ---

    def test_spy_sees_capped_inflow(self):
        spy = SpyRouting()
        reach = self._make(capacity=100.0, routing_model=spy)
        reach.receive(150.0, "src", T)
        reach.update(T)

        assert spy.inflows == [100.0]

    def test_buffering_buffers_capped_amount(self):
        reach = self._make(capacity=100.0, routing_model=BufferingRouting())

        # Step 0: receive 150, capped to 100, buffered (output=0)
        reach.receive(150.0, "src", T)
        reach.update(T)

        outputs_s0 = reach.events_of_type(WaterOutput)
        assert len(outputs_s0) == 0
        assert reach.water_in_transit == 100.0

        # Step 1: receive 0, release buffered 100
        reach.receive(0.0, "src", T1)
        reach.update(T1)

        outputs_s1 = reach.events_of_type(WaterOutput)
        assert len(outputs_s1) == 1
        assert outputs_s1[0].amount == 100.0

    def test_entered_reach_records_capped_amount(self):
        reach = self._make(capacity=100.0)
        reach.receive(150.0, "src", T)
        reach.update(T)

        entered = reach.events_of_type(WaterEnteredReach)
        assert len(entered) == 1
        assert entered[0].amount == 100.0

    # --- Capacity + losses ---

    def test_losses_applied_to_capped_flow(self):
        reach = self._make(
            capacity=100.0,
            loss_rule=FakeReachLossRule(losses={EVAPORATION: 10.0}),
        )
        reach.receive(150.0, "src", T)
        reach.update(T)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 90.0

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].amount == 50.0

    def test_buffering_plus_losses_on_capped(self):
        reach = self._make(
            capacity=100.0,
            routing_model=BufferingRouting(),
            loss_rule=FakeReachLossRule(losses={EVAPORATION: 5.0}),
        )

        # Step 0: receive 150 → spill=50, entered=100, buffered=100, output=0
        reach.receive(150.0, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].amount == 50.0

        entered = reach.events_of_type(WaterEnteredReach)
        assert entered[0].amount == 100.0

        outputs_s0 = reach.events_of_type(WaterOutput)
        assert len(outputs_s0) == 0

        # Step 1: receive 0 → outflow=100, loss=5, output=95
        reach.receive(0.0, "src", T1)
        reach.update(T1)

        outputs_s1 = reach.events_of_type(WaterOutput)
        assert len(outputs_s1) == 1
        assert outputs_s1[0].amount == 95.0

    def test_loss_scaling_after_cap(self):
        reach = self._make(
            capacity=50.0,
            loss_rule=FakeReachLossRule(losses={EVAPORATION: 80.0, SEEPAGE: 20.0}),
        )
        reach.receive(200.0, "src", T)
        reach.update(T)

        # Capacity caps inflow to 50; passthrough outflow=50
        # Losses (80+20=100) exceed outflow (50), so scaled: evap=40, seep=10
        loss_events = reach.events_of_type(WaterLost)
        evap = next(e for e in loss_events if e.reason == EVAPORATION)
        seep = next(e for e in loss_events if e.reason == SEEPAGE)
        assert evap.amount == pytest.approx(40.0)
        assert seep.amount == pytest.approx(10.0)

        # All capped flow lost, no output
        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 0

    # --- Mass balance ---

    def test_mass_balance_spilled_plus_entered_equals_received(self):
        reach = self._make(capacity=100.0)
        reach.receive(150.0, "src", T)
        reach.update(T)

        spill_total = sum(e.amount for e in reach.events_of_type(WaterSpilled))
        entered_total = sum(e.amount for e in reach.events_of_type(WaterEnteredReach))
        received_total = sum(e.amount for e in reach.events_of_type(WaterReceived))

        assert spill_total + entered_total == received_total

    def test_mass_balance_with_losses(self):
        reach = self._make(
            capacity=100.0,
            loss_rule=FakeReachLossRule(losses={EVAPORATION: 10.0}),
        )
        reach.receive(150.0, "src", T)
        reach.update(T)

        spill = sum(e.amount for e in reach.events_of_type(WaterSpilled))
        entered = sum(e.amount for e in reach.events_of_type(WaterEnteredReach))
        losses = sum(e.amount for e in reach.events_of_type(WaterLost))
        output = sum(e.amount for e in reach.events_of_type(WaterOutput))

        assert spill + entered == 150.0
        assert output + losses == pytest.approx(100.0)

    def test_mass_balance_multi_step_buffering(self):
        reach = self._make(capacity=80.0, routing_model=BufferingRouting())

        # Step 0: receive 100 → spill=20, entered=80, transit=80
        reach.receive(100.0, "src", T)
        reach.update(T)

        spill_s0 = sum(e.amount for e in reach.events_of_type(WaterSpilled))
        entered_s0 = sum(e.amount for e in reach.events_of_type(WaterEnteredReach))
        assert spill_s0 == 20.0
        assert entered_s0 == 80.0
        assert reach.water_in_transit == 80.0

        # Step 1: receive 0 → outflow=80
        reach.receive(0.0, "src", T1)
        reach.update(T1)

        outputs = reach.events_of_type(WaterOutput)
        assert len(outputs) == 1
        assert outputs[0].amount == 80.0

    # --- Multi-timestep ---

    def test_capacity_enforced_each_step(self):
        reach = self._make(capacity=100.0)

        # Step 0: receive 150 → spill=50, output=100
        reach.receive(150.0, "src", T)
        reach.update(T)

        spills_s0 = reach.events_of_type(WaterSpilled)
        outputs_s0 = reach.events_of_type(WaterOutput)
        assert len(spills_s0) == 1
        assert spills_s0[0].amount == 50.0
        assert outputs_s0[0].amount == 100.0

        # Step 1: receive 80 → no spill, output=80
        reach.receive(80.0, "src", T1)
        reach.update(T1)

        spills_s1 = reach.events_of_type(WaterSpilled)
        outputs_s1 = reach.events_of_type(WaterOutput)
        assert len(spills_s1) == 1  # still only the one from step 0
        assert len(outputs_s1) == 2
        assert outputs_s1[1].amount == 80.0

    def test_capacity_spill_correct_timesteps(self):
        reach = self._make(capacity=100.0)

        reach.receive(150.0, "src", T)
        reach.update(T)
        reach.receive(200.0, "src", T1)
        reach.update(T1)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 2
        assert spills[0].t == T.index
        assert spills[1].t == T1.index

    def test_capacity_no_accumulation_across_steps(self):
        reach = self._make(capacity=50.0)

        # Step 0: receive 30 → no spill
        reach.receive(30.0, "src", T)
        reach.update(T)

        spills_s0 = reach.events_of_type(WaterSpilled)
        assert len(spills_s0) == 0

        # Step 1: receive 60 → spill=10 (not 40)
        reach.receive(60.0, "src", T1)
        reach.update(T1)

        spills_s1 = reach.events_of_type(WaterSpilled)
        assert len(spills_s1) == 1
        assert spills_s1[0].amount == 10.0

    # --- Event sequence ---

    def test_event_sequence_with_spill_passthrough(self):
        reach = self._make(capacity=100.0)
        reach.receive(150.0, "src", T)
        reach.update(T)

        event_types = [type(e).__name__ for e in reach.events]
        assert event_types == [
            "WaterReceived",
            "WaterSpilled",
            "WaterEnteredReach",
            "WaterExitedReach",
            "WaterInTransit",
            "WaterOutput",
        ]

    def test_event_sequence_with_spill_and_losses(self):
        reach = self._make(
            capacity=100.0,
            loss_rule=FakeReachLossRule(losses={EVAPORATION: 10.0}),
        )
        reach.receive(150.0, "src", T)
        reach.update(T)

        event_types = [type(e).__name__ for e in reach.events]
        assert event_types == [
            "WaterReceived",
            "WaterSpilled",
            "WaterEnteredReach",
            "WaterExitedReach",
            "WaterLost",
            "WaterInTransit",
            "WaterOutput",
        ]

    # --- Edge cases ---

    def test_zero_flow_with_capacity(self):
        reach = self._make(capacity=100.0)
        reach.receive(0.0, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 0

    def test_very_small_overflow(self):
        reach = self._make(capacity=100.0)
        reach.receive(100.001, "src", T)
        reach.update(T)

        spills = reach.events_of_type(WaterSpilled)
        assert len(spills) == 1
        assert spills[0].amount == pytest.approx(0.001)

    def test_fresh_copy_preserves_capacity(self):
        reach = self._make(capacity=500.0)
        copy = reach._fresh_copy()
        assert copy.capacity == 500.0
