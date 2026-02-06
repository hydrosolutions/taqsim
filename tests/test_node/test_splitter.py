from typing import TYPE_CHECKING

import pytest

from taqsim.node.events import WaterDistributed, WaterReceived
from taqsim.node.protocols import Receives
from taqsim.node.splitter import Splitter
from taqsim.time import Frequency, Timestep

if TYPE_CHECKING:
    pass


class FakeEqualSplitRule:
    def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


class FakeFixedSplitRule:
    def __init__(self, ratios: dict[str, float]):
        self._ratios = ratios

    def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
        return {target: amount * self._ratios.get(target, 0) for target in node.targets}


class TestSplitterInit:
    def test_creates_with_id_and_strategy(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        assert splitter.id == "splitter_1"

    def test_starts_with_empty_targets(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        assert splitter.targets == []

    def test_starts_with_empty_events(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        assert splitter.events == []

    def test_raises_without_strategy(self):
        with pytest.raises(ValueError, match="split_rule is required"):
            Splitter(id="splitter_1")

    def test_sets_targets_via_set_targets(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1", "t2"])
        assert splitter.targets == ["t1", "t2"]


class TestSplitterReceive:
    def test_receive_returns_amount(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        result = splitter.receive(amount=100.0, source_id="source_a", t=Timestep(0, Frequency.MONTHLY))
        assert result == 100.0

    def test_receive_records_event(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter.receive(amount=50.0, source_id="upstream", t=Timestep(5, Frequency.MONTHLY))
        assert len(splitter.events) == 1
        event = splitter.events[0]
        assert isinstance(event, WaterReceived)
        assert event.amount == 50.0
        assert event.source_id == "upstream"
        assert event.t == 5

    def test_receive_accumulates_for_distribution(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1"])
        splitter.receive(amount=30.0, source_id="a", t=Timestep(0, Frequency.MONTHLY))
        splitter.receive(amount=20.0, source_id="b", t=Timestep(0, Frequency.MONTHLY))
        splitter.update(t=Timestep(0, Frequency.MONTHLY))
        distributed = splitter.events_of_type(WaterDistributed)
        assert len(distributed) == 1
        assert distributed[0].amount == 50.0


class TestSplitterDistribute:
    def test_distribute_uses_strategy(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1", "t2"])
        allocation = splitter.distribute(amount=100.0, t=Timestep(0, Frequency.MONTHLY))
        assert allocation == {"t1": 50.0, "t2": 50.0}

    def test_distribute_records_events(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1", "t2"])
        splitter.distribute(amount=100.0, t=Timestep(0, Frequency.MONTHLY))
        distributed = splitter.events_of_type(WaterDistributed)
        assert len(distributed) == 2

    def test_distribute_records_correct_target_ids(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["target_a", "target_b"])
        splitter.distribute(amount=80.0, t=Timestep(3, Frequency.MONTHLY))
        distributed = splitter.events_of_type(WaterDistributed)
        target_ids = {e.target_id for e in distributed}
        assert target_ids == {"target_a", "target_b"}

    def test_distribute_empty_targets_returns_empty(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        allocation = splitter.distribute(amount=100.0, t=Timestep(0, Frequency.MONTHLY))
        assert allocation == {}

    def test_distribute_zero_amount_returns_empty(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1"])
        allocation = splitter.distribute(amount=0.0, t=Timestep(0, Frequency.MONTHLY))
        assert allocation == {}

    def test_distribute_negative_amount_returns_empty(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1"])
        allocation = splitter.distribute(amount=-10.0, t=Timestep(0, Frequency.MONTHLY))
        assert allocation == {}

    def test_distribute_with_fixed_ratios(self):
        strategy = FakeFixedSplitRule(ratios={"t1": 0.7, "t2": 0.3})
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1", "t2"])
        allocation = splitter.distribute(amount=100.0, t=Timestep(0, Frequency.MONTHLY))
        assert allocation["t1"] == pytest.approx(70.0)
        assert allocation["t2"] == pytest.approx(30.0)


class TestSplitterUpdate:
    def test_update_distributes_received_water(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1"])
        splitter.receive(amount=100.0, source_id="source", t=Timestep(0, Frequency.MONTHLY))
        splitter.update(t=Timestep(0, Frequency.MONTHLY))
        distributed = splitter.events_of_type(WaterDistributed)
        assert len(distributed) == 1
        assert distributed[0].amount == 100.0

    def test_update_resets_received_for_next_step(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1"])
        splitter.receive(amount=100.0, source_id="source", t=Timestep(0, Frequency.MONTHLY))
        splitter.update(t=Timestep(0, Frequency.MONTHLY))
        splitter.update(t=Timestep(1, Frequency.MONTHLY))
        distributed = splitter.events_of_type(WaterDistributed)
        assert len(distributed) == 1

    def test_update_across_multiple_timesteps(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        splitter._set_targets(["t1"])
        splitter.receive(amount=50.0, source_id="s", t=Timestep(0, Frequency.MONTHLY))
        splitter.update(t=Timestep(0, Frequency.MONTHLY))

        splitter.receive(amount=30.0, source_id="s", t=Timestep(1, Frequency.MONTHLY))
        splitter.update(t=Timestep(1, Frequency.MONTHLY))

        distributed = splitter.events_of_type(WaterDistributed)
        assert len(distributed) == 2
        assert distributed[0].amount == 50.0
        assert distributed[1].amount == 30.0


class TestSplitterProtocols:
    def test_implements_receives_protocol(self):
        strategy = FakeEqualSplitRule()
        splitter = Splitter(id="splitter_1", split_rule=strategy)
        assert isinstance(splitter, Receives)
