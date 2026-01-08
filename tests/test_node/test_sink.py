from taqsim.node.events import WaterReceived
from taqsim.node.protocols import Receives
from taqsim.node.sink import Sink


class TestSinkInit:
    def test_creates_with_id(self):
        sink = Sink(id="sink_1")
        assert sink.id == "sink_1"

    def test_starts_with_empty_events(self):
        sink = Sink(id="sink_1")
        assert sink.events == []


class TestSinkReceive:
    def test_receive_returns_amount(self):
        sink = Sink(id="sink_1")
        result = sink.receive(amount=100.0, source_id="source_a", t=0)
        assert result == 100.0

    def test_receive_records_event(self):
        sink = Sink(id="sink_1")
        sink.receive(amount=50.0, source_id="upstream", t=5)
        assert len(sink.events) == 1
        event = sink.events[0]
        assert isinstance(event, WaterReceived)
        assert event.amount == 50.0
        assert event.source_id == "upstream"
        assert event.t == 5

    def test_receive_multiple_times_records_all(self):
        sink = Sink(id="sink_1")
        sink.receive(amount=10.0, source_id="a", t=0)
        sink.receive(amount=20.0, source_id="b", t=1)
        sink.receive(amount=30.0, source_id="c", t=2)
        assert len(sink.events) == 3

    def test_receive_zero_amount(self):
        sink = Sink(id="sink_1")
        result = sink.receive(amount=0.0, source_id="source", t=0)
        assert result == 0.0
        assert len(sink.events) == 1


class TestSinkUpdate:
    def test_update_does_nothing(self):
        sink = Sink(id="sink_1")
        sink.receive(amount=100.0, source_id="source", t=0)
        initial_events = len(sink.events)
        sink.update(t=0, dt=1.0)
        assert len(sink.events) == initial_events

    def test_update_can_be_called_multiple_times(self):
        sink = Sink(id="sink_1")
        sink.update(t=0, dt=1.0)
        sink.update(t=1, dt=1.0)
        sink.update(t=2, dt=1.0)
        assert sink.events == []


class TestSinkProtocol:
    def test_implements_receives_protocol(self):
        sink = Sink(id="sink_1")
        assert isinstance(sink, Receives)


class TestSinkIsTerminal:
    def test_has_no_distribute_method(self):
        sink = Sink(id="sink_1")
        assert not hasattr(sink, "distribute")

    def test_is_terminal_node(self):
        sink = Sink(id="sink_1")
        sink.receive(amount=100.0, source_id="source", t=0)
        sink.update(t=0, dt=1.0)
        received_events = sink.events_of_type(WaterReceived)
        assert len(received_events) == 1
        assert received_events[0].amount == 100.0
