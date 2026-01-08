from dataclasses import FrozenInstanceError

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node.events import (
    DeficitRecorded,
    NodeEvent,
    WaterConsumed,
    WaterDistributed,
    WaterGenerated,
    WaterLost,
    WaterOutput,
    WaterPassedThrough,
    WaterReceived,
    WaterReleased,
    WaterSpilled,
    WaterStored,
)


class TestEventImmutability:
    def test_water_generated_is_frozen(self):
        event = WaterGenerated(amount=100.0, t=0)
        with pytest.raises(FrozenInstanceError):
            event.amount = 200.0

    def test_water_received_is_frozen(self):
        event = WaterReceived(amount=50.0, source_id="upstream", t=1)
        with pytest.raises(FrozenInstanceError):
            event.amount = 100.0

    def test_water_stored_is_frozen(self):
        event = WaterStored(amount=75.0, t=2)
        with pytest.raises(FrozenInstanceError):
            event.amount = 150.0

    def test_water_released_is_frozen(self):
        event = WaterReleased(amount=60.0, t=3)
        with pytest.raises(FrozenInstanceError):
            event.amount = 120.0

    def test_water_lost_is_frozen(self):
        event = WaterLost(amount=10.0, reason=EVAPORATION, t=4)
        with pytest.raises(FrozenInstanceError):
            event.amount = 20.0

    def test_water_spilled_is_frozen(self):
        event = WaterSpilled(amount=50.0, t=5)
        with pytest.raises(FrozenInstanceError):
            event.amount = 100.0

    def test_water_consumed_is_frozen(self):
        event = WaterConsumed(amount=30.0, t=5)
        with pytest.raises(FrozenInstanceError):
            event.amount = 60.0

    def test_water_distributed_is_frozen(self):
        event = WaterDistributed(amount=40.0, target_id="downstream", t=6)
        with pytest.raises(FrozenInstanceError):
            event.amount = 80.0

    def test_deficit_recorded_is_frozen(self):
        event = DeficitRecorded(required=100.0, actual=80.0, deficit=20.0, t=7)
        with pytest.raises(FrozenInstanceError):
            event.deficit = 30.0

    def test_water_output_is_frozen(self):
        event = WaterOutput(amount=100.0, t=8)
        with pytest.raises(FrozenInstanceError):
            event.amount = 200.0

    def test_water_passed_through_is_frozen(self):
        event = WaterPassedThrough(amount=50.0, t=9)
        with pytest.raises(FrozenInstanceError):
            event.amount = 100.0


class TestEventCreation:
    def test_water_generated_stores_amount_and_t(self):
        event = WaterGenerated(amount=100.0, t=5)
        assert event.amount == 100.0
        assert event.t == 5

    def test_water_received_stores_source_id(self):
        event = WaterReceived(amount=50.0, source_id="upstream", t=1)
        assert event.source_id == "upstream"
        assert event.amount == 50.0
        assert event.t == 1

    def test_water_stored_stores_amount_and_t(self):
        event = WaterStored(amount=200.0, t=3)
        assert event.amount == 200.0
        assert event.t == 3

    def test_water_released_stores_amount_and_t(self):
        event = WaterReleased(amount=150.0, t=4)
        assert event.amount == 150.0
        assert event.t == 4

    def test_water_lost_stores_reason(self):
        event = WaterLost(amount=10.0, reason=EVAPORATION, t=2)
        assert event.reason == EVAPORATION
        assert event.amount == 10.0
        assert event.t == 2

    def test_water_lost_stores_seepage_reason(self):
        event = WaterLost(amount=5.0, reason=SEEPAGE, t=3)
        assert event.reason == SEEPAGE

    def test_water_spilled_stores_amount_and_t(self):
        event = WaterSpilled(amount=50.0, t=4)
        assert event.amount == 50.0
        assert event.t == 4

    def test_water_consumed_stores_amount_and_t(self):
        event = WaterConsumed(amount=75.0, t=6)
        assert event.amount == 75.0
        assert event.t == 6

    def test_water_distributed_stores_target_id(self):
        event = WaterDistributed(amount=40.0, target_id="canal_1", t=7)
        assert event.target_id == "canal_1"
        assert event.amount == 40.0
        assert event.t == 7

    def test_deficit_recorded_stores_all_fields(self):
        event = DeficitRecorded(required=100.0, actual=80.0, deficit=20.0, t=3)
        assert event.required == 100.0
        assert event.actual == 80.0
        assert event.deficit == 20.0
        assert event.t == 3

    def test_water_output_stores_amount_and_t(self):
        event = WaterOutput(amount=120.0, t=8)
        assert event.amount == 120.0
        assert event.t == 8

    def test_water_passed_through_stores_amount_and_t(self):
        event = WaterPassedThrough(amount=85.0, t=9)
        assert event.amount == 85.0
        assert event.t == 9


class TestLossReason:
    def test_is_str_subclass(self):
        assert issubclass(LossReason, str)

    def test_evaporation_equals_string(self):
        assert EVAPORATION == "evaporation"

    def test_seepage_equals_string(self):
        assert SEEPAGE == "seepage"

    def test_evaporation_is_distinct_from_seepage(self):
        assert EVAPORATION != SEEPAGE

    def test_custom_loss_reason_works(self):
        custom = LossReason("custom_loss")
        assert custom == "custom_loss"
        assert isinstance(custom, LossReason)
        assert isinstance(custom, str)


class TestNodeEventProtocol:
    def test_water_generated_is_node_event(self):
        event = WaterGenerated(amount=100.0, t=0)
        assert isinstance(event, NodeEvent)

    def test_water_received_is_node_event(self):
        event = WaterReceived(amount=50.0, source_id="up", t=1)
        assert isinstance(event, NodeEvent)

    def test_water_stored_is_node_event(self):
        event = WaterStored(amount=75.0, t=2)
        assert isinstance(event, NodeEvent)

    def test_water_released_is_node_event(self):
        event = WaterReleased(amount=60.0, t=3)
        assert isinstance(event, NodeEvent)

    def test_water_lost_is_node_event(self):
        event = WaterLost(amount=10.0, reason=EVAPORATION, t=4)
        assert isinstance(event, NodeEvent)

    def test_water_spilled_is_node_event(self):
        event = WaterSpilled(amount=50.0, t=5)
        assert isinstance(event, NodeEvent)

    def test_water_consumed_is_node_event(self):
        event = WaterConsumed(amount=30.0, t=5)
        assert isinstance(event, NodeEvent)

    def test_water_distributed_is_node_event(self):
        event = WaterDistributed(amount=40.0, target_id="down", t=6)
        assert isinstance(event, NodeEvent)

    def test_deficit_recorded_is_node_event(self):
        event = DeficitRecorded(required=100.0, actual=80.0, deficit=20.0, t=7)
        assert isinstance(event, NodeEvent)

    def test_water_output_is_node_event(self):
        event = WaterOutput(amount=100.0, t=8)
        assert isinstance(event, NodeEvent)

    def test_water_passed_through_is_node_event(self):
        event = WaterPassedThrough(amount=50.0, t=9)
        assert isinstance(event, NodeEvent)
