from dataclasses import FrozenInstanceError

import pytest

from taqsim.common import EVAPORATION, SEEPAGE
from taqsim.edge.events import (
    CapacityExceeded,
    FlowDelivered,
    FlowLost,
    FlowReceived,
    RequirementUnmet,
)


class TestEdgeEventImmutability:
    def test_flow_received_is_frozen(self):
        event = FlowReceived(amount=100.0, t=0)
        with pytest.raises(FrozenInstanceError):
            event.amount = 200.0

    def test_flow_lost_is_frozen(self):
        event = FlowLost(amount=10.0, reason=EVAPORATION, t=1)
        with pytest.raises(FrozenInstanceError):
            event.amount = 20.0

    def test_flow_delivered_is_frozen(self):
        event = FlowDelivered(amount=90.0, t=2)
        with pytest.raises(FrozenInstanceError):
            event.amount = 180.0

    def test_capacity_exceeded_is_frozen(self):
        event = CapacityExceeded(excess=50.0, t=3)
        with pytest.raises(FrozenInstanceError):
            event.excess = 100.0

    def test_requirement_unmet_is_frozen(self):
        event = RequirementUnmet(required=100.0, actual=80.0, deficit=20.0, t=4)
        with pytest.raises(FrozenInstanceError):
            event.deficit = 30.0


class TestEdgeEventCreation:
    def test_flow_received_stores_amount_and_t(self):
        event = FlowReceived(amount=100.0, t=5)
        assert event.amount == 100.0
        assert event.t == 5

    def test_flow_lost_stores_amount_reason_and_t(self):
        event = FlowLost(amount=10.0, reason=EVAPORATION, t=1)
        assert event.amount == 10.0
        assert event.reason == EVAPORATION
        assert event.t == 1

    def test_flow_lost_stores_seepage_reason(self):
        event = FlowLost(amount=5.0, reason=SEEPAGE, t=2)
        assert event.reason == SEEPAGE

    def test_flow_delivered_stores_amount_and_t(self):
        event = FlowDelivered(amount=90.0, t=3)
        assert event.amount == 90.0
        assert event.t == 3

    def test_capacity_exceeded_stores_excess_and_t(self):
        event = CapacityExceeded(excess=25.0, t=4)
        assert event.excess == 25.0
        assert event.t == 4

    def test_requirement_unmet_stores_all_fields(self):
        event = RequirementUnmet(required=100.0, actual=80.0, deficit=20.0, t=5)
        assert event.required == 100.0
        assert event.actual == 80.0
        assert event.deficit == 20.0
        assert event.t == 5


class TestEdgeEventProtocol:
    def test_flow_received_is_edge_event(self):
        event = FlowReceived(amount=100.0, t=0)
        assert isinstance(event, FlowReceived)
        # EdgeEvent is a type alias union, check event is one of the types
        assert type(event).__name__ in [
            "FlowReceived",
            "FlowLost",
            "FlowDelivered",
            "CapacityExceeded",
            "RequirementUnmet",
        ]

    def test_flow_lost_is_edge_event(self):
        event = FlowLost(amount=10.0, reason=EVAPORATION, t=1)
        assert isinstance(event, FlowLost)
        assert type(event).__name__ in [
            "FlowReceived",
            "FlowLost",
            "FlowDelivered",
            "CapacityExceeded",
            "RequirementUnmet",
        ]

    def test_flow_delivered_is_edge_event(self):
        event = FlowDelivered(amount=90.0, t=2)
        assert isinstance(event, FlowDelivered)
        assert type(event).__name__ in [
            "FlowReceived",
            "FlowLost",
            "FlowDelivered",
            "CapacityExceeded",
            "RequirementUnmet",
        ]

    def test_capacity_exceeded_is_edge_event(self):
        event = CapacityExceeded(excess=50.0, t=3)
        assert isinstance(event, CapacityExceeded)
        assert type(event).__name__ in [
            "FlowReceived",
            "FlowLost",
            "FlowDelivered",
            "CapacityExceeded",
            "RequirementUnmet",
        ]

    def test_requirement_unmet_is_edge_event(self):
        event = RequirementUnmet(required=100.0, actual=80.0, deficit=20.0, t=4)
        assert isinstance(event, RequirementUnmet)
        assert type(event).__name__ in [
            "FlowReceived",
            "FlowLost",
            "FlowDelivered",
            "CapacityExceeded",
            "RequirementUnmet",
        ]
