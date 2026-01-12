from dataclasses import FrozenInstanceError

import pytest

from taqsim.common import CAPACITY_EXCEEDED, EVAPORATION, SEEPAGE
from taqsim.edge.events import (
    WaterDelivered,
    WaterLost,
    WaterReceived,
)


class TestEdgeEventImmutability:
    def test_water_received_is_frozen(self):
        event = WaterReceived(amount=100.0, t=0)
        with pytest.raises(FrozenInstanceError):
            event.amount = 200.0

    def test_water_lost_is_frozen(self):
        event = WaterLost(amount=10.0, reason=EVAPORATION, t=1)
        with pytest.raises(FrozenInstanceError):
            event.amount = 20.0

    def test_water_delivered_is_frozen(self):
        event = WaterDelivered(amount=90.0, t=2)
        with pytest.raises(FrozenInstanceError):
            event.amount = 180.0


class TestEdgeEventCreation:
    def test_water_received_stores_amount_and_t(self):
        event = WaterReceived(amount=100.0, t=5)
        assert event.amount == 100.0
        assert event.t == 5

    def test_water_lost_stores_amount_reason_and_t(self):
        event = WaterLost(amount=10.0, reason=EVAPORATION, t=1)
        assert event.amount == 10.0
        assert event.reason == EVAPORATION
        assert event.t == 1

    def test_water_lost_stores_seepage_reason(self):
        event = WaterLost(amount=5.0, reason=SEEPAGE, t=2)
        assert event.reason == SEEPAGE

    def test_water_lost_stores_capacity_exceeded_reason(self):
        event = WaterLost(amount=50.0, reason=CAPACITY_EXCEEDED, t=0)
        assert event.reason == CAPACITY_EXCEEDED

    def test_water_delivered_stores_amount_and_t(self):
        event = WaterDelivered(amount=90.0, t=3)
        assert event.amount == 90.0
        assert event.t == 3


class TestEdgeEventProtocol:
    def test_water_received_is_edge_event(self):
        event = WaterReceived(amount=100.0, t=0)
        assert isinstance(event, WaterReceived)
        # EdgeEvent is a type alias union, check event is one of the types
        assert type(event).__name__ in [
            "WaterReceived",
            "WaterLost",
            "WaterDelivered",
        ]

    def test_water_lost_is_edge_event(self):
        event = WaterLost(amount=10.0, reason=EVAPORATION, t=1)
        assert isinstance(event, WaterLost)
        assert type(event).__name__ in [
            "WaterReceived",
            "WaterLost",
            "WaterDelivered",
        ]

    def test_water_delivered_is_edge_event(self):
        event = WaterDelivered(amount=90.0, t=2)
        assert isinstance(event, WaterDelivered)
        assert type(event).__name__ in [
            "WaterReceived",
            "WaterLost",
            "WaterDelivered",
        ]
