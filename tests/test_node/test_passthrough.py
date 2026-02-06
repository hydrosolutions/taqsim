import pytest

from taqsim.node.events import WaterOutput, WaterPassedThrough, WaterReceived, WaterSpilled
from taqsim.node.passthrough import PassThrough
from taqsim.node.protocols import Receives
from taqsim.time import Frequency, Timestep


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
        result = pt.receive(amount=100.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        assert result == 100.0

    def test_receive_records_water_received_event(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=50.0, source_id="reservoir", t=Timestep(5, Frequency.MONTHLY))
        assert len(pt.events) == 1
        event = pt.events[0]
        assert isinstance(event, WaterReceived)
        assert event.amount == 50.0
        assert event.source_id == "reservoir"
        assert event.t == 5

    def test_receive_accumulates_from_multiple_sources(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=30.0, source_id="a", t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=20.0, source_id="b", t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=50.0, source_id="c", t=Timestep(0, Frequency.MONTHLY))
        assert len(pt.events) == 3
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1
        assert output_events[0].amount == 100.0

    def test_receive_zero_amount(self):
        pt = PassThrough(id="turbine_1")
        result = pt.receive(amount=0.0, source_id="source", t=Timestep(0, Frequency.MONTHLY))
        assert result == 0.0
        assert len(pt.events) == 1


class TestPassThroughUpdate:
    def test_update_records_water_passed_through(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=75.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        passed_events = pt.events_of_type(WaterPassedThrough)
        assert len(passed_events) == 1
        assert passed_events[0].amount == 75.0
        assert passed_events[0].t == 0

    def test_update_records_water_output(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=75.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1
        assert output_events[0].amount == 75.0
        assert output_events[0].t == 0

    def test_update_resets_counter(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=100.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1

    def test_update_with_zero_received_does_not_record_events(self):
        pt = PassThrough(id="turbine_1")
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        assert len(pt.events) == 0

    def test_update_passes_100_percent_through(self):
        pt = PassThrough(id="turbine_1")
        received = 123.456
        pt.receive(amount=received, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == received

    def test_multiple_timesteps(self):
        pt = PassThrough(id="turbine_1")
        pt.receive(amount=50.0, source_id="up", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=75.0, source_id="up", t=Timestep(1, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))
        pt.receive(amount=25.0, source_id="up", t=Timestep(2, Frequency.MONTHLY))
        pt.update(t=Timestep(2, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 3
        assert [e.amount for e in output_events] == [50.0, 75.0, 25.0]


class TestPassThroughProtocols:
    def test_implements_receives_protocol(self):
        pt = PassThrough(id="turbine_1")
        assert isinstance(pt, Receives)


class TestPassThroughCapacity:
    # --- Validation ---
    def test_capacity_none_is_valid(self):
        pt = PassThrough(id="pt")
        assert pt.capacity is None

    def test_capacity_positive_is_valid(self):
        pt = PassThrough(id="pt", capacity=100.0)
        assert pt.capacity == 100.0

    def test_capacity_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            PassThrough(id="pt", capacity=0.0)

    def test_capacity_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            PassThrough(id="pt", capacity=-50.0)

    # --- No Capacity (Unlimited) ---
    def test_no_capacity_passes_all_through(self):
        pt = PassThrough(id="pt")
        pt.receive(amount=1000.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert len(output_events) == 1
        assert output_events[0].amount == 1000.0

    def test_no_capacity_no_spill_events(self):
        pt = PassThrough(id="pt")
        pt.receive(amount=999999.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        assert len(pt.events_of_type(WaterSpilled)) == 0

    # --- Flow Below Capacity ---
    def test_flow_below_capacity_passes_all(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=75.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == 75.0
        passed_events = pt.events_of_type(WaterPassedThrough)
        assert passed_events[0].amount == 75.0

    def test_flow_below_capacity_no_spill(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=50.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        assert len(pt.events_of_type(WaterSpilled)) == 0

    def test_flow_below_capacity_from_multiple_sources(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=30.0, source_id="a", t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=40.0, source_id="b", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == 70.0
        assert len(pt.events_of_type(WaterSpilled)) == 0

    # --- Flow Equals Capacity (Edge) ---
    def test_flow_equals_capacity_passes_all(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=100.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == 100.0
        passed_events = pt.events_of_type(WaterPassedThrough)
        assert passed_events[0].amount == 100.0

    def test_flow_equals_capacity_no_spill(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=100.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        assert len(pt.events_of_type(WaterSpilled)) == 0

    # --- Flow Exceeds Capacity ---
    def test_flow_exceeds_capacity_limits_output(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == 100.0

    def test_flow_exceeds_capacity_limits_passed_through(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        passed_events = pt.events_of_type(WaterPassedThrough)
        assert passed_events[0].amount == 100.0

    def test_flow_exceeds_capacity_records_spill(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        spill_events = pt.events_of_type(WaterSpilled)
        assert len(spill_events) == 1
        assert spill_events[0].amount == 50.0

    def test_flow_exceeds_capacity_spill_has_correct_timestep(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="upstream", t=Timestep(5, Frequency.MONTHLY))
        pt.update(t=Timestep(5, Frequency.MONTHLY))
        spill_events = pt.events_of_type(WaterSpilled)
        assert spill_events[0].t == 5

    def test_flow_exceeds_capacity_from_multiple_sources(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=80.0, source_id="a", t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=60.0, source_id="b", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == 100.0
        spill_events = pt.events_of_type(WaterSpilled)
        assert spill_events[0].amount == 40.0

    def test_large_overflow_spills_correct_amount(self):
        pt = PassThrough(id="pt", capacity=50.0)
        pt.receive(amount=500.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        output_events = pt.events_of_type(WaterOutput)
        assert output_events[0].amount == 50.0
        spill_events = pt.events_of_type(WaterSpilled)
        assert spill_events[0].amount == 450.0

    # --- Multi-Timestep ---
    def test_capacity_applies_each_timestep(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="up", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=80.0, source_id="up", t=Timestep(1, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))
        pt.receive(amount=120.0, source_id="up", t=Timestep(2, Frequency.MONTHLY))
        pt.update(t=Timestep(2, Frequency.MONTHLY))

        output_events = pt.events_of_type(WaterOutput)
        assert [e.amount for e in output_events] == [100.0, 80.0, 100.0]

        spill_events = pt.events_of_type(WaterSpilled)
        assert len(spill_events) == 2
        assert [e.amount for e in spill_events] == [50.0, 20.0]

    def test_spill_events_have_correct_timesteps(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="up", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=80.0, source_id="up", t=Timestep(1, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))
        pt.receive(amount=120.0, source_id="up", t=Timestep(2, Frequency.MONTHLY))
        pt.update(t=Timestep(2, Frequency.MONTHLY))

        spill_events = pt.events_of_type(WaterSpilled)
        assert [e.t for e in spill_events] == [0, 2]

    # --- Trace Extraction ---
    def test_spill_trace_extraction(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="up", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=120.0, source_id="up", t=Timestep(1, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))

        trace = pt.trace(WaterSpilled)
        assert trace.values() == [50.0, 20.0]

    def test_spill_trace_sum(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=150.0, source_id="up", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=130.0, source_id="up", t=Timestep(1, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))

        trace = pt.trace(WaterSpilled)
        assert trace.sum() == 80.0

    def test_spill_trace_empty_when_no_spill(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=50.0, source_id="up", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        pt.receive(amount=75.0, source_id="up", t=Timestep(1, Frequency.MONTHLY))
        pt.update(t=Timestep(1, Frequency.MONTHLY))

        trace = pt.trace(WaterSpilled)
        assert len(trace) == 0

    # --- Edge Cases ---
    def test_zero_flow_with_capacity(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        assert len(pt.events_of_type(WaterOutput)) == 0
        assert len(pt.events_of_type(WaterSpilled)) == 0
        assert len(pt.events_of_type(WaterPassedThrough)) == 0

    def test_very_small_overflow(self):
        pt = PassThrough(id="pt", capacity=100.0)
        pt.receive(amount=100.001, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        pt.update(t=Timestep(0, Frequency.MONTHLY))
        spill_events = pt.events_of_type(WaterSpilled)
        assert len(spill_events) == 1
        assert spill_events[0].amount == pytest.approx(0.001)

    def test_receive_returns_full_amount_regardless_of_capacity(self):
        pt = PassThrough(id="pt", capacity=100.0)
        result = pt.receive(amount=500.0, source_id="upstream", t=Timestep(0, Frequency.MONTHLY))
        assert result == 500.0
