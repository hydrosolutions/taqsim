import pytest

from taqsim.node.events import WaterDistributed, WaterGenerated
from taqsim.node.protocols import Generates, Gives
from taqsim.node.source import Source


class FakeTimeSeries:
    def __init__(self, values: dict[int, float]):
        self._values = values

    def __getitem__(self, t: int) -> float:
        return self._values.get(t, 0.0)


class FakeSplitStrategy:
    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


class RecordingSplitStrategy:
    def __init__(self):
        self.calls: list[tuple[float, list[str], int]] = []

    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
        self.calls.append((amount, targets.copy(), t))
        share = amount / len(targets) if targets else 0.0
        return dict.fromkeys(targets, share)


class TestSourceInit:
    def test_creates_with_required_fields(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)
        assert source.id == "src1"
        assert source.inflow is ts
        assert source.split_strategy is strategy

    def test_default_targets_is_empty_list(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)
        assert source.targets == []

    def test_creates_with_custom_targets(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)
        assert source.targets == ["a", "b"]

    def test_requires_split_strategy(self):
        ts = FakeTimeSeries({0: 10.0})
        with pytest.raises(ValueError, match="split_strategy is required"):
            Source(id="src1", inflow=ts, split_strategy=None)

    def test_starts_with_empty_events(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)
        assert source.events == []


class TestSourceGenerate:
    def test_returns_inflow_times_dt(self):
        ts = FakeTimeSeries({0: 10.0, 1: 20.0, 2: 15.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        result = source.generate(t=0, dt=1.0)
        assert result == 10.0

    def test_scales_by_dt(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        result = source.generate(t=0, dt=0.5)
        assert result == 5.0

    def test_records_water_generated_event(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        source.generate(t=0, dt=1.0)

        events = source.events_of_type(WaterGenerated)
        assert len(events) == 1
        assert events[0].amount == 10.0
        assert events[0].t == 0

    def test_handles_different_timesteps(self):
        ts = FakeTimeSeries({5: 100.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        result = source.generate(t=5, dt=2.0)
        assert result == 200.0

    def test_returns_zero_for_missing_timestep(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        result = source.generate(t=999, dt=1.0)
        assert result == 0.0


class TestSourceDistribute:
    def test_uses_split_strategy(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = RecordingSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        source.distribute(amount=100.0, t=5)

        assert len(strategy.calls) == 1
        assert strategy.calls[0] == (100.0, ["a", "b"], 5)

    def test_returns_allocation_from_strategy(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        allocation = source.distribute(amount=100.0, t=0)

        assert allocation == {"a": 50.0, "b": 50.0}

    def test_records_water_distributed_per_target(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        source.distribute(amount=100.0, t=0)

        events = source.events_of_type(WaterDistributed)
        assert len(events) == 2
        targets_and_amounts = {(e.target_id, e.amount) for e in events}
        assert targets_and_amounts == {("a", 50.0), ("b", 50.0)}
        assert all(e.t == 0 for e in events)

    def test_returns_empty_dict_when_no_targets(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=[], split_strategy=strategy)

        allocation = source.distribute(amount=100.0, t=0)

        assert allocation == {}

    def test_returns_empty_dict_when_amount_is_zero(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        allocation = source.distribute(amount=0.0, t=0)

        assert allocation == {}

    def test_returns_empty_dict_when_amount_is_negative(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        allocation = source.distribute(amount=-10.0, t=0)

        assert allocation == {}

    def test_no_events_when_no_targets(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=[], split_strategy=strategy)

        source.distribute(amount=100.0, t=0)

        assert source.events_of_type(WaterDistributed) == []

    def test_no_events_when_amount_is_zero(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        source.distribute(amount=0.0, t=0)

        assert source.events_of_type(WaterDistributed) == []


class TestSourceUpdate:
    def test_generates_and_distributes(self):
        ts = FakeTimeSeries({0: 50.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a", "b"], split_strategy=strategy)

        source.update(t=0, dt=1.0)

        generated = source.events_of_type(WaterGenerated)
        distributed = source.events_of_type(WaterDistributed)

        assert len(generated) == 1
        assert generated[0].amount == 50.0

        assert len(distributed) == 2
        total_distributed = sum(e.amount for e in distributed)
        assert total_distributed == 50.0

    def test_events_in_correct_sequence(self):
        ts = FakeTimeSeries({0: 100.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a"], split_strategy=strategy)

        source.update(t=0, dt=1.0)

        assert len(source.events) == 2
        assert isinstance(source.events[0], WaterGenerated)
        assert isinstance(source.events[1], WaterDistributed)

    def test_passes_generated_amount_to_distribute(self):
        ts = FakeTimeSeries({0: 30.0})
        strategy = RecordingSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["x"], split_strategy=strategy)

        source.update(t=0, dt=2.0)

        assert strategy.calls[0][0] == 60.0

    def test_no_distribution_when_no_targets(self):
        ts = FakeTimeSeries({0: 100.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=[], split_strategy=strategy)

        source.update(t=0, dt=1.0)

        assert len(source.events_of_type(WaterGenerated)) == 1
        assert len(source.events_of_type(WaterDistributed)) == 0

    def test_multiple_updates_accumulate_events(self):
        ts = FakeTimeSeries({0: 10.0, 1: 20.0, 2: 30.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a"], split_strategy=strategy)

        source.update(t=0, dt=1.0)
        source.update(t=1, dt=1.0)
        source.update(t=2, dt=1.0)

        generated = source.events_of_type(WaterGenerated)
        assert len(generated) == 3
        assert [e.amount for e in generated] == [10.0, 20.0, 30.0]


class TestSourceProtocolCompliance:
    def test_satisfies_generates_protocol(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        assert isinstance(source, Generates)

    def test_satisfies_gives_protocol(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        assert isinstance(source, Gives)

    def test_generates_method_signature_matches_protocol(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, split_strategy=strategy)

        result = source.generate(t=0, dt=1.0)
        assert isinstance(result, float)

    def test_distribute_method_signature_matches_protocol(self):
        ts = FakeTimeSeries({0: 10.0})
        strategy = FakeSplitStrategy()
        source = Source(id="src1", inflow=ts, targets=["a"], split_strategy=strategy)

        result = source.distribute(amount=10.0, t=0)
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result)
        assert all(isinstance(v, float) for v in result.values())
