from taqsim.node import Demand, PassThrough, Source, Splitter, Storage, TimeSeries
from taqsim.time import Frequency, Timestep

# Use fixtures from conftest.py for fake strategies


class TestBaseNodeReset:
    """Tests for BaseNode.reset() - inherited by all nodes."""

    def test_reset_clears_events(self):
        source = Source(id="s", inflow=TimeSeries(values=[100.0]))
        source.update(t=Timestep(0, Frequency.MONTHLY))

        assert len(source.events) > 0
        source.reset()
        assert len(source.events) == 0


class TestStorageReset:
    """Tests for Storage.reset()."""

    def test_reset_restores_initial_storage(self, fake_release_rule, fake_loss_rule):
        storage = Storage(
            id="dam",
            capacity=1000.0,
            initial_storage=500.0,
            release_rule=fake_release_rule,
            loss_rule=fake_loss_rule,
        )

        # Simulate some activity
        storage.receive(200.0, "upstream", t=Timestep(0, Frequency.MONTHLY))
        storage.update(t=Timestep(0, Frequency.MONTHLY))

        # Storage level changed
        assert storage.storage != 500.0

        storage.reset()

        assert storage.storage == 500.0
        assert storage._received_this_step == 0.0
        assert len(storage.events) == 0


class TestDemandReset:
    """Tests for Demand.reset()."""

    def test_reset_clears_received_accumulator(self):
        demand = Demand(
            id="city",
            requirement=TimeSeries(values=[50.0]),
        )

        demand.receive(30.0, "upstream", t=Timestep(0, Frequency.MONTHLY))
        assert demand._received_this_step == 30.0

        demand.reset()

        assert demand._received_this_step == 0.0
        assert len(demand.events) == 0


class TestSplitterReset:
    """Tests for Splitter.reset()."""

    def test_reset_clears_received_accumulator(self, fake_split_rule):
        splitter = Splitter(
            id="junction",
            split_rule=fake_split_rule,
        )
        splitter._set_targets(["a", "b"])

        splitter.receive(100.0, "upstream", t=Timestep(0, Frequency.MONTHLY))
        assert splitter._received_this_step == 100.0

        splitter.reset()

        assert splitter._received_this_step == 0.0
        assert len(splitter.events) == 0


class TestPassThroughReset:
    """Tests for PassThrough.reset()."""

    def test_reset_clears_received_accumulator(self):
        pt = PassThrough(id="turbine")

        pt.receive(100.0, "upstream", t=Timestep(0, Frequency.MONTHLY))
        assert pt._received_this_step == 100.0

        pt.reset()

        assert pt._received_this_step == 0.0
        assert len(pt.events) == 0
