from taqsim.node.demand import Demand
from taqsim.node.events import (
    DeficitRecorded,
    WaterConsumed,
    WaterOutput,
    WaterReceived,
)
from taqsim.node.protocols import Consumes, Receives


class FakeTimeSeries:
    def __init__(self, values: dict[int, float] | None = None, default: float = 0.0):
        self._values = values or {}
        self._default = default

    def __getitem__(self, t: int) -> float:
        return self._values.get(t, self._default)


class TestDemandInit:
    def test_creates_with_id_and_requirement(self):
        ts = FakeTimeSeries(default=10.0)
        node = Demand(id="demand_1", requirement=ts)
        assert node.id == "demand_1"

    def test_starts_with_zero_received(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        assert node._received_this_step == 0.0

    def test_starts_with_empty_events(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        assert node.events == []


class TestDemandReceive:
    def test_accepts_water_and_returns_amount(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        accepted = node.receive(amount=100.0, source_id="source_a", t=0)
        assert accepted == 100.0

    def test_records_water_received_event(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        node.receive(amount=50.0, source_id="upstream", t=3)

        events = node.events_of_type(WaterReceived)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].source_id == "upstream"
        assert events[0].t == 3

    def test_accumulates_received_water(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        node.receive(amount=30.0, source_id="a", t=0)
        node.receive(amount=20.0, source_id="b", t=0)
        assert node._received_this_step == 50.0

    def test_records_multiple_receive_events(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        node.receive(amount=10.0, source_id="a", t=0)
        node.receive(amount=20.0, source_id="b", t=0)

        events = node.events_of_type(WaterReceived)
        assert len(events) == 2

    def test_receive_with_zero_amount(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        accepted = node.receive(amount=0.0, source_id="source", t=0)
        assert accepted == 0.0


class TestDemandConsume:
    def test_consumes_up_to_requirement(self):
        ts = FakeTimeSeries(values={0: 50.0})
        node = Demand(id="d1", requirement=ts)
        consumed, remaining = node.consume(available=100.0, t=0, dt=1.0)
        assert consumed == 50.0
        assert remaining == 50.0

    def test_consumes_all_when_less_than_requirement(self):
        ts = FakeTimeSeries(values={0: 100.0})
        node = Demand(id="d1", requirement=ts)
        consumed, remaining = node.consume(available=30.0, t=0, dt=1.0)
        assert consumed == 30.0
        assert remaining == 0.0

    def test_records_water_consumed_event(self):
        ts = FakeTimeSeries(values={0: 50.0})
        node = Demand(id="d1", requirement=ts)
        node.consume(available=100.0, t=0, dt=1.0)

        events = node.events_of_type(WaterConsumed)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].t == 0

    def test_records_deficit_when_underfulfilled(self):
        ts = FakeTimeSeries(values={0: 100.0})
        node = Demand(id="d1", requirement=ts)
        node.consume(available=60.0, t=0, dt=1.0)

        deficits = node.events_of_type(DeficitRecorded)
        assert len(deficits) == 1
        assert deficits[0].required == 100.0
        assert deficits[0].actual == 60.0
        assert deficits[0].deficit == 40.0
        assert deficits[0].t == 0

    def test_no_deficit_when_fully_satisfied(self):
        ts = FakeTimeSeries(values={0: 50.0})
        node = Demand(id="d1", requirement=ts)
        node.consume(available=100.0, t=0, dt=1.0)

        deficits = node.events_of_type(DeficitRecorded)
        assert len(deficits) == 0

    def test_dt_scales_requirement(self):
        ts = FakeTimeSeries(values={0: 10.0})
        node = Demand(id="d1", requirement=ts)
        consumed, remaining = node.consume(available=100.0, t=0, dt=3.0)
        assert consumed == 30.0  # 10 * 3
        assert remaining == 70.0

    def test_consume_with_zero_available(self):
        ts = FakeTimeSeries(values={0: 50.0})
        node = Demand(id="d1", requirement=ts)
        consumed, remaining = node.consume(available=0.0, t=0, dt=1.0)
        assert consumed == 0.0
        assert remaining == 0.0

    def test_consume_with_zero_requirement(self):
        ts = FakeTimeSeries(values={0: 0.0})
        node = Demand(id="d1", requirement=ts)
        consumed, remaining = node.consume(available=100.0, t=0, dt=1.0)
        assert consumed == 0.0
        assert remaining == 100.0


class TestDemandUpdate:
    def test_consume_then_output_sequence(self):
        ts = FakeTimeSeries(values={0: 30.0})
        node = Demand(id="d1", requirement=ts)
        node._received_this_step = 100.0
        node.update(t=0, dt=1.0)

        consumed_events = node.events_of_type(WaterConsumed)
        assert len(consumed_events) == 1
        assert consumed_events[0].amount == 30.0

        output_events = node.events_of_type(WaterOutput)
        assert len(output_events) == 1
        assert output_events[0].amount == 70.0
        assert output_events[0].t == 0

    def test_resets_received_after_update(self):
        ts = FakeTimeSeries(values={0: 10.0})
        node = Demand(id="d1", requirement=ts)
        node._received_this_step = 50.0
        node.update(t=0, dt=1.0)
        assert node._received_this_step == 0.0

    def test_full_receive_update_cycle(self):
        ts = FakeTimeSeries(values={0: 25.0})
        node = Demand(id="d1", requirement=ts)
        node.receive(amount=100.0, source_id="upstream", t=0)
        node.update(t=0, dt=1.0)

        received = node.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 100.0

        consumed = node.events_of_type(WaterConsumed)
        assert len(consumed) == 1
        assert consumed[0].amount == 25.0

        output = node.events_of_type(WaterOutput)
        assert len(output) == 1
        assert output[0].amount == 75.0
        assert output[0].t == 0

    def test_update_with_zero_received(self):
        ts = FakeTimeSeries(values={0: 50.0})
        node = Demand(id="d1", requirement=ts)
        node.update(t=0, dt=1.0)

        consumed = node.events_of_type(WaterConsumed)
        assert len(consumed) == 1
        assert consumed[0].amount == 0.0

        deficits = node.events_of_type(DeficitRecorded)
        assert len(deficits) == 1
        assert deficits[0].deficit == 50.0

    def test_no_output_when_all_consumed(self):
        ts = FakeTimeSeries(values={0: 100.0})
        node = Demand(id="d1", requirement=ts)
        node._received_this_step = 50.0
        node.update(t=0, dt=1.0)

        output = node.events_of_type(WaterOutput)
        assert len(output) == 0

    def test_no_output_when_zero_remaining(self):
        ts = FakeTimeSeries(values={0: 100.0})
        node = Demand(id="d1", requirement=ts)
        node._received_this_step = 100.0
        node.update(t=0, dt=1.0)

        output = node.events_of_type(WaterOutput)
        assert len(output) == 0


class TestDemandProtocolCompliance:
    def test_satisfies_receives_protocol(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        assert isinstance(node, Receives)

    def test_satisfies_consumes_protocol(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        assert isinstance(node, Consumes)

    def test_receive_signature_matches_protocol(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        result = node.receive(amount=10.0, source_id="src", t=0)
        assert isinstance(result, float)

    def test_consume_signature_matches_protocol(self):
        ts = FakeTimeSeries()
        node = Demand(id="d1", requirement=ts)
        result = node.consume(available=10.0, t=0, dt=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
