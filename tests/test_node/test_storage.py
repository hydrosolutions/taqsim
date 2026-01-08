import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node.events import (
    WaterLost,
    WaterOutput,
    WaterReceived,
    WaterReleased,
    WaterSpilled,
    WaterStored,
)
from taqsim.node.protocols import Loses, Receives, Stores
from taqsim.node.storage import Storage


class FakeReleaseRule:
    def __init__(self, release_amount: float = 0.0):
        self._release_amount = release_amount

    def release(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
        return self._release_amount


class FakeLossRule:
    def __init__(self, losses: dict[LossReason, float] | None = None):
        self._losses = losses if losses is not None else {}

    def calculate(self, storage: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
        return self._losses


def make_storage(
    capacity: float = 1000.0,
    initial_storage: float = 0.0,
    release_amount: float = 0.0,
    losses: dict[LossReason, float] | None = None,
) -> Storage:
    return Storage(
        id="test_storage",
        capacity=capacity,
        initial_storage=initial_storage,
        release_rule=FakeReleaseRule(release_amount),
        loss_rule=FakeLossRule(losses),
    )


class TestStorageInit:
    def test_creates_with_valid_params(self):
        storage = make_storage(capacity=1000.0, initial_storage=500.0)
        assert storage.capacity == 1000.0
        assert storage.storage == 500.0

    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            make_storage(capacity=0.0)

    def test_capacity_cannot_be_negative(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            make_storage(capacity=-100.0)

    def test_initial_storage_cannot_be_negative(self):
        with pytest.raises(ValueError, match="initial_storage cannot be negative"):
            make_storage(capacity=1000.0, initial_storage=-1.0)

    def test_initial_storage_cannot_exceed_capacity(self):
        with pytest.raises(ValueError, match="initial_storage cannot exceed capacity"):
            make_storage(capacity=1000.0, initial_storage=1001.0)

    def test_initial_storage_equals_capacity_is_valid(self):
        storage = make_storage(capacity=1000.0, initial_storage=1000.0)
        assert storage.storage == 1000.0

    def test_release_rule_required(self):
        with pytest.raises(ValueError, match="release_rule is required"):
            Storage(
                id="test",
                capacity=1000.0,
                release_rule=None,
                loss_rule=FakeLossRule(),
            )

    def test_loss_rule_required(self):
        with pytest.raises(ValueError, match="loss_rule is required"):
            Storage(
                id="test",
                capacity=1000.0,
                release_rule=FakeReleaseRule(),
                loss_rule=None,
            )


class TestStorageReceive:
    def test_receive_records_water_received_event(self):
        storage = make_storage()
        storage.receive(100.0, "source_1", t=0)

        events = storage.events_of_type(WaterReceived)
        assert len(events) == 1
        assert events[0].amount == 100.0
        assert events[0].source_id == "source_1"
        assert events[0].t == 0

    def test_receive_returns_amount_received(self):
        storage = make_storage()
        result = storage.receive(50.0, "source_1", t=0)
        assert result == 50.0

    def test_receive_accumulates_for_step(self):
        storage = make_storage()
        storage.receive(100.0, "source_1", t=0)
        storage.receive(50.0, "source_2", t=0)

        events = storage.events_of_type(WaterReceived)
        assert len(events) == 2
        assert sum(e.amount for e in events) == 150.0

    def test_receive_from_multiple_sources(self):
        storage = make_storage()
        storage.receive(100.0, "river", t=0)
        storage.receive(50.0, "tributary", t=0)

        events = storage.events_of_type(WaterReceived)
        sources = {e.source_id for e in events}
        assert sources == {"river", "tributary"}


class TestStorageStore:
    def test_store_within_capacity(self):
        storage = make_storage(capacity=1000.0, initial_storage=0.0)
        stored, spilled = storage.store(500.0, t=0, dt=1.0)

        assert stored == 500.0
        assert spilled == 0.0
        assert storage.storage == 500.0

    def test_store_records_water_stored_event(self):
        storage = make_storage(capacity=1000.0)
        storage.store(300.0, t=0, dt=1.0)

        events = storage.events_of_type(WaterStored)
        assert len(events) == 1
        assert events[0].amount == 300.0
        assert events[0].t == 0

    def test_store_over_capacity_spills(self):
        storage = make_storage(capacity=100.0, initial_storage=80.0)
        stored, spilled = storage.store(50.0, t=0, dt=1.0)

        assert stored == 20.0
        assert spilled == 30.0
        assert storage.storage == 100.0

    def test_store_over_capacity_records_spilled_event(self):
        storage = make_storage(capacity=100.0, initial_storage=80.0)
        storage.store(50.0, t=0, dt=1.0)

        events = storage.events_of_type(WaterSpilled)
        assert len(events) == 1
        assert events[0].amount == 30.0
        assert events[0].t == 0

    def test_store_at_full_capacity_spills_all(self):
        storage = make_storage(capacity=100.0, initial_storage=100.0)
        stored, spilled = storage.store(50.0, t=0, dt=1.0)

        assert stored == 0.0
        assert spilled == 50.0
        assert storage.storage == 100.0

    def test_store_zero_amount(self):
        storage = make_storage(capacity=100.0, initial_storage=50.0)
        stored, spilled = storage.store(0.0, t=0, dt=1.0)

        assert stored == 0.0
        assert spilled == 0.0
        assert storage.storage == 50.0


class TestStorageLose:
    def test_lose_reduces_storage(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            losses={EVAPORATION: 10.0},
        )
        total_loss = storage.lose(t=0, dt=1.0)

        assert total_loss == 10.0
        assert storage.storage == 490.0

    def test_lose_records_water_lost_event(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            losses={EVAPORATION: 10.0},
        )
        storage.lose(t=0, dt=1.0)

        events = storage.events_of_type(WaterLost)
        assert len(events) == 1
        assert events[0].amount == 10.0
        assert events[0].reason == EVAPORATION
        assert events[0].t == 0

    def test_lose_multiple_reasons(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            losses={
                EVAPORATION: 10.0,
                SEEPAGE: 5.0,
            },
        )
        total_loss = storage.lose(t=0, dt=1.0)

        assert total_loss == 15.0
        assert storage.storage == 485.0

        events = storage.events_of_type(WaterLost)
        assert len(events) == 2
        reasons = {e.reason for e in events}
        assert reasons == {EVAPORATION, SEEPAGE}

    def test_lose_clamps_to_available_storage(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=10.0,
            losses={EVAPORATION: 100.0},
        )
        total_loss = storage.lose(t=0, dt=1.0)

        assert total_loss == 10.0
        assert storage.storage == 0.0

    def test_lose_scales_multiple_losses_proportionally(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=10.0,
            losses={
                EVAPORATION: 80.0,
                SEEPAGE: 20.0,
            },
        )
        storage.lose(t=0, dt=1.0)

        events = storage.events_of_type(WaterLost)
        evap = next(e for e in events if e.reason == EVAPORATION)
        seep = next(e for e in events if e.reason == SEEPAGE)

        assert evap.amount == pytest.approx(8.0)
        assert seep.amount == pytest.approx(2.0)
        assert storage.storage == pytest.approx(0.0)

    def test_lose_no_losses(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            losses={},
        )
        total_loss = storage.lose(t=0, dt=1.0)

        assert total_loss == 0.0
        assert storage.storage == 500.0
        assert len(storage.events_of_type(WaterLost)) == 0

    def test_lose_zero_loss_amount_no_event(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            losses={EVAPORATION: 0.0},
        )
        storage.lose(t=0, dt=1.0)

        events = storage.events_of_type(WaterLost)
        assert len(events) == 0


class TestStorageRelease:
    def test_release_reduces_storage(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            release_amount=100.0,
        )
        released = storage.release(inflow=0.0, t=0, dt=1.0)

        assert released == 100.0
        assert storage.storage == 400.0

    def test_release_records_water_released_event(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            release_amount=100.0,
        )
        storage.release(inflow=0.0, t=0, dt=1.0)

        events = storage.events_of_type(WaterReleased)
        assert len(events) == 1
        assert events[0].amount == 100.0
        assert events[0].t == 0

    def test_release_clamps_to_available_storage(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=50.0,
            release_amount=100.0,
        )
        released = storage.release(inflow=0.0, t=0, dt=1.0)

        assert released == 50.0
        assert storage.storage == 0.0

    def test_release_zero_no_event(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            release_amount=0.0,
        )
        released = storage.release(inflow=0.0, t=0, dt=1.0)

        assert released == 0.0
        assert len(storage.events_of_type(WaterReleased)) == 0

    def test_release_negative_rule_output_clamped_to_zero(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=500.0,
            release_amount=-50.0,
        )
        released = storage.release(inflow=0.0, t=0, dt=1.0)

        assert released == 0.0
        assert storage.storage == 500.0


class TestStorageUpdate:
    def test_update_stores_received_water(self):
        storage = make_storage(capacity=1000.0, initial_storage=0.0)
        storage.receive(100.0, "source", t=0)
        storage.update(t=0, dt=1.0)

        assert storage.storage == 100.0
        events = storage.events_of_type(WaterStored)
        assert len(events) == 1
        assert events[0].amount == 100.0

    def test_update_applies_losses(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=100.0,
            losses={EVAPORATION: 10.0},
        )
        storage.update(t=0, dt=1.0)

        assert storage.storage == 90.0

    def test_update_releases_water(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=100.0,
            release_amount=20.0,
        )
        storage.update(t=0, dt=1.0)

        assert storage.storage == 80.0

    def test_update_records_water_output_for_released_water(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=100.0,
            release_amount=50.0,
        )
        storage.update(t=0, dt=1.0)

        events = storage.events_of_type(WaterOutput)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].t == 0

    def test_update_records_water_output_for_spilled_plus_released(self):
        storage = make_storage(
            capacity=100.0,
            initial_storage=90.0,
            release_amount=10.0,
        )
        storage.receive(30.0, "source", t=0)
        storage.update(t=0, dt=1.0)

        events = storage.events_of_type(WaterOutput)
        assert len(events) == 1
        assert events[0].amount == 30.0
        assert events[0].t == 0

    def test_update_no_water_output_when_zero_outflow(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=100.0,
            release_amount=0.0,
        )
        storage.update(t=0, dt=1.0)

        events = storage.events_of_type(WaterOutput)
        assert len(events) == 0

    def test_update_resets_received_for_next_step(self):
        storage = make_storage(capacity=1000.0, initial_storage=0.0)
        storage.receive(100.0, "source", t=0)
        storage.update(t=0, dt=1.0)

        storage.update(t=1, dt=1.0)

        stored_events = storage.events_of_type(WaterStored)
        t1_stored = [e for e in stored_events if e.t == 1]
        assert len(t1_stored) == 1
        assert t1_stored[0].amount == 0.0

    def test_update_order_store_then_lose_then_release(self):
        storage = make_storage(
            capacity=1000.0,
            initial_storage=100.0,
            losses={EVAPORATION: 10.0},
            release_amount=50.0,
        )
        storage.receive(50.0, "source", t=0)
        storage.update(t=0, dt=1.0)

        assert storage.storage == 90.0

    def test_update_mass_balance(self):
        initial = 100.0
        received = 50.0
        evap_loss = 10.0
        seep_loss = 5.0
        release_amt = 20.0
        capacity = 1000.0

        storage = make_storage(
            capacity=capacity,
            initial_storage=initial,
            losses={
                EVAPORATION: evap_loss,
                SEEPAGE: seep_loss,
            },
            release_amount=release_amt,
        )
        storage.receive(received, "source", t=0)
        storage.update(t=0, dt=1.0)

        expected_storage = initial + received - evap_loss - seep_loss - release_amt
        assert storage.storage == pytest.approx(expected_storage)

        output_events = storage.events_of_type(WaterOutput)
        assert output_events[0].amount == release_amt


class TestStorageProtocolCompliance:
    def test_satisfies_receives_protocol(self):
        storage = make_storage()
        assert isinstance(storage, Receives)

    def test_satisfies_stores_protocol(self):
        storage = make_storage()
        assert isinstance(storage, Stores)

    def test_satisfies_loses_protocol(self):
        storage = make_storage()
        assert isinstance(storage, Loses)

    def test_storage_property_returns_float(self):
        storage = make_storage(initial_storage=250.0)
        assert isinstance(storage.storage, float)
        assert storage.storage == 250.0

    def test_capacity_property_returns_float(self):
        storage = make_storage(capacity=500.0)
        assert isinstance(storage.capacity, float)
        assert storage.capacity == 500.0
