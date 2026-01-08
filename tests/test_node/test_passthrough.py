from taqsim.node.events import WaterOutput, WaterPassedThrough, WaterReceived
from taqsim.node.passthrough import PassThrough
from taqsim.node.protocols import Receives


class TestPassThroughInit:
    def test_creates_with_id(self):
        pt = PassThrough(id="turbine_1")
        assert pt.id == "turbine_1"

    def test_starts_with_empty_events(self):
        pt = PassThrough(id="turbine_1")
        assert pt.events == []


class TestPassThroughReceive:
    def test_receive_returns_amount(self):
        pt = PassThrough(id="turbine_1")
        result = pt.receive(amount=100.0, source_id="upstream", t=0)
        assert result == 100.0

    def test_receive_records_water_received_event(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=50.0, source_id="reservoir", t=5)
        assert len(pt.events) == 1
        event = pt.events[0]
        assert isinstance(event, WaterReceived)
        assert event.amount == 50.0
        assert event.source_id == "reservoir"
        assert event.t == 5

    def test_receive_accumulates_from_multiple_sources(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=30.0, source_id="a", t=0)
        pt.receive(amount=20.0, source_id="b", t=0)
        pt.receive(amount=50.0, source_id="c", t=0)
        assert len(pt.events) == 3
        pt.update(t=0, dt=1.0)
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1
        assert output_events[0].amount == 100.0

    def test_receive_zero_amount(self):
        pt = PassThrough(id="turbine_1")
        result = pt.receive(amount=0.0, source_id="source", t=0)
        assert result == 0.0
        assert len(pt.events) == 1


class TestPassThroughUpdate:
    def test_update_records_water_passed_through(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=75.0, source_id="upstream", t=0)
        pt.update(t=0, dt=1.0)
        passed_events = pt.events_of_type(WaterPassedThrough)
        assert len(passed_events) == 1
        assert passed_events[0].amount == 75.0
        assert passed_events[0].t == 0

    def test_update_records_water_output(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=75.0, source_id="upstream", t=0)
        pt.update(t=0, dt=1.0)
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1
        assert output_events[0].amount == 75.0
        assert output_events[0].t == 0

    def test_update_resets_counter(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=100.0, source_id="upstream", t=0)
        pt.update(t=0, dt=1.0)
        pt.update(t=1, dt=1.0)
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1

    def test_update_with_zero_received_does_not_record_events(self):
        pt = PassThrough(id="turbine_1")
        pt.update(t=0, dt=1.0)
        assert len(pt.events) == 0

    def test_update_passes_100_percent_through(self):
        pt = PassThrough(id="turbine_1")
        received = 123.456
        pt.receive(amount=received, source_id="upstream", t=0)
        pt.update(t=0, dt=1.0)
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == received

    def test_multiple_timesteps(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=50.0, source_id="up", t=0)
        pt.update(t=0, dt=1.0)
        pt.receive(amount=75.0, source_id="up", t=1)
        pt.update(t=1, dt=1.0)
        pt.receive(amount=25.0, source_id="up", t=2)
        pt.update(t=2, dt=1.0)
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 3
        assert [e.amount for e in output_events] == [50.0, 75.0, 25.0]


class TestPassThroughProtocols:
    def test_implements_receives_protocol(self):
        pt = PassThrough(id="turbine_1")
        assert isinstance(pt, Receives)
