from taqsim.node.events import WaterGenerated, WaterOutput
from taqsim.node.protocols import Generates
from taqsim.node.source import Source
from taqsim.time import Frequency, Timestep


class FakeTimeSeries:
    def __init__(self, values: dict[int, float]):
        self._values = values

    def __getitem__(self, t: int) -> float:
        return self._values.get(t, 0.0)


class TestSourceInit:
    def test_creates_with_required_fields(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)
        assert source.id == "src1"
        assert source.inflow is ts

    def test_starts_with_empty_events(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)
        assert source.events == []


class TestSourceGenerate:
    def test_returns_inflow_value(self):
        ts = FakeTimeSeries({0: 10.0, 1: 20.0, 2: 15.0})
        source = Source(id="src1", inflow=ts)

        result = source.generate(t=Timestep(0, Frequency.MONTHLY))
        assert result == 10.0

    def test_returns_inflow_for_different_frequency(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)

        result = source.generate(t=Timestep(0, Frequency.DAILY))
        assert result == 10.0

    def test_records_water_generated_event(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)

        source.generate(t=Timestep(0, Frequency.MONTHLY))

        events = source.events_of_type(WaterGenerated)
        assert len(events) == 1
        assert events[0].amount == 10.0
        assert events[0].t == 0

    def test_handles_different_timesteps(self):
        ts = FakeTimeSeries({5: 100.0})
        source = Source(id="src1", inflow=ts)

        result = source.generate(t=Timestep(5, Frequency.MONTHLY))
        assert result == 100.0

    def test_returns_zero_for_missing_timestep(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)

        result = source.generate(t=Timestep(999, Frequency.MONTHLY))
        assert result == 0.0


class TestSourceUpdate:
    def test_generates_and_records_output(self):
        ts = FakeTimeSeries({0: 50.0})
        source = Source(id="src1", inflow=ts)

        source.update(t=Timestep(0, Frequency.MONTHLY))

        generated = source.events_of_type(WaterGenerated)
        outputs = source.events_of_type(WaterOutput)

        assert len(generated) == 1
        assert generated[0].amount == 50.0

        assert len(outputs) == 1
        assert outputs[0].amount == 50.0
        assert outputs[0].t == 0

    def test_events_in_correct_sequence(self):
        ts = FakeTimeSeries({0: 100.0})
        source = Source(id="src1", inflow=ts)

        source.update(t=Timestep(0, Frequency.MONTHLY))

        assert len(source.events) == 2
        assert isinstance(source.events[0], WaterGenerated)
        assert isinstance(source.events[1], WaterOutput)

    def test_no_output_event_when_generated_is_zero(self):
        ts = FakeTimeSeries({0: 0.0})
        source = Source(id="src1", inflow=ts)

        source.update(t=Timestep(0, Frequency.MONTHLY))

        assert len(source.events_of_type(WaterGenerated)) == 1
        assert len(source.events_of_type(WaterOutput)) == 0

    def test_multiple_updates_accumulate_events(self):
        ts = FakeTimeSeries({0: 10.0, 1: 20.0, 2: 30.0})
        source = Source(id="src1", inflow=ts)

        source.update(t=Timestep(0, Frequency.MONTHLY))
        source.update(t=Timestep(1, Frequency.MONTHLY))
        source.update(t=Timestep(2, Frequency.MONTHLY))

        generated = source.events_of_type(WaterGenerated)
        outputs = source.events_of_type(WaterOutput)
        assert len(generated) == 3
        assert [e.amount for e in generated] == [10.0, 20.0, 30.0]
        assert len(outputs) == 3
        assert [e.amount for e in outputs] == [10.0, 20.0, 30.0]


class TestSourceProtocolCompliance:
    def test_satisfies_generates_protocol(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)

        assert isinstance(source, Generates)

    def test_generates_method_signature_matches_protocol(self):
        ts = FakeTimeSeries({0: 10.0})
        source = Source(id="src1", inflow=ts)

        result = source.generate(t=Timestep(0, Frequency.MONTHLY))
        assert isinstance(result, float)
