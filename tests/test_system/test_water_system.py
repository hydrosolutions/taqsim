import pytest

from taqsim.node import WaterReceived
from taqsim.node.events import WaterGenerated
from taqsim.system import ValidationError, WaterSystem
from taqsim.time import Frequency, Timestep

from .conftest import (
    FakeEdgeLossRule,
    make_edge,
    make_sink,
    make_source,
    make_splitter,
    make_storage,
)


class TestWaterSystemInit:
    def test_creates_with_frequency(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.frequency == Frequency.MONTHLY

    def test_creates_with_custom_frequency(self):
        system = WaterSystem(frequency=Frequency.DAILY)
        assert system.frequency == Frequency.DAILY

    def test_starts_with_empty_nodes(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.nodes == {}

    def test_starts_with_empty_edges(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.edges == {}


class TestAddNode:
    def test_adds_source_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source(id="src1")
        system.add_node(source)

        assert "src1" in system.nodes
        assert system.nodes["src1"] is source

    def test_adds_sink_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        sink = make_sink(id="snk1")
        system.add_node(sink)

        assert "snk1" in system.nodes
        assert system.nodes["snk1"] is sink

    def test_raises_on_duplicate_node_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source(id="node1")
        system.add_node(source)

        with pytest.raises(ValueError, match="Node 'node1' already exists"):
            system.add_node(make_sink(id="node1"))

    def test_invalidates_after_add(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.validate()

        assert system._validated is True

        # Adding node should invalidate
        system.add_node(make_sink(id="sink2"))
        assert system._validated is False


class TestAddEdge:
    def test_adds_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        edge = make_edge(id="e1", source="a", target="b")
        system.add_edge(edge)

        assert "e1" in system.edges
        assert system.edges["e1"] is edge

    def test_raises_on_duplicate_edge_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        edge1 = make_edge(id="edge1", source="a", target="b")
        edge2 = make_edge(id="edge1", source="c", target="d")

        system.add_edge(edge1)

        with pytest.raises(ValueError, match="Edge 'edge1' already exists"):
            system.add_edge(edge2)

    def test_invalidates_after_add(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.validate()

        assert system._validated is True

        # Adding edge should invalidate
        system.add_edge(make_edge(id="edge2", source="source", target="sink"))
        assert system._validated is False


class TestValidation:
    def test_validates_simple_network(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        system.add_edge(make_edge(source="source", target="sink"))

        system.validate()
        assert system._validated is True

    def test_fails_on_missing_source_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_sink())
        system.add_edge(make_edge(source="nonexistent", target="sink"))

        with pytest.raises(ValidationError, match="source node 'nonexistent' does not exist"):
            system.validate()

    def test_fails_on_missing_target_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_edge(make_edge(source="source", target="nonexistent"))

        with pytest.raises(ValidationError, match="target node 'nonexistent' does not exist"):
            system.validate()

    def test_fails_on_cycle(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        splitter1 = make_splitter(id="s1")
        splitter2 = make_splitter(id="s2")
        sink = make_sink()

        system.add_node(splitter1)
        system.add_node(splitter2)
        system.add_node(sink)

        # Create cycle: s1 -> s2 -> s1 (and s2 -> sink)
        system.add_edge(make_edge(id="e1", source="s1", target="s2"))
        system.add_edge(make_edge(id="e2", source="s2", target="s1"))
        system.add_edge(make_edge(id="e3", source="s2", target="sink"))

        with pytest.raises(ValidationError, match="contains cycles"):
            system.validate()

    def test_fails_on_disconnected_nodes(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source(id="src1"))
        system.add_node(make_sink(id="snk1"))
        system.add_node(make_source(id="src2"))  # disconnected
        system.add_node(make_sink(id="snk2"))  # disconnected

        system.add_edge(make_edge(id="e1", source="src1", target="snk1"))
        system.add_edge(make_edge(id="e2", source="src2", target="snk2"))

        with pytest.raises(ValidationError, match="not connected"):
            system.validate()

    def test_fails_on_source_with_incoming_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink = make_sink()

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink)

        # Edge going into source (invalid)
        system.add_edge(make_edge(id="e1", source="splitter", target="source"))
        system.add_edge(make_edge(id="e2", source="source", target="sink"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink"))

        with pytest.raises(ValidationError, match="Source 'source' must have in_degree=0"):
            system.validate()

    def test_fails_on_sink_with_outgoing_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        other_sink = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(sink)
        system.add_node(other_sink)

        system.add_edge(make_edge(id="e1", source="source", target="sink"))
        system.add_edge(make_edge(id="e2", source="sink", target="sink2"))

        with pytest.raises(ValidationError, match="Sink 'sink' must have out_degree=0"):
            system.validate()

    def test_fails_on_non_splitter_with_multiple_outputs(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(sink1)
        system.add_node(sink2)

        # Source (non-splitter) with two outputs
        system.add_edge(make_edge(id="e1", source="source", target="sink1"))
        system.add_edge(make_edge(id="e2", source="source", target="sink2"))

        with pytest.raises(ValidationError, match="must have exactly 1 outgoing edge"):
            system.validate()

    def test_allows_splitter_with_multiple_outputs(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.validate()
        assert system._validated is True

    def test_fails_on_node_without_path_to_sink(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        storage = make_storage(id="storage")
        sink = make_sink()

        system.add_node(source)
        system.add_node(storage)
        system.add_node(sink)

        # Source -> sink, but storage is orphaned (no edges from/to it)
        # This will fail at "not connected" first
        system.add_edge(make_edge(id="e1", source="source", target="sink"))

        with pytest.raises(ValidationError, match="not connected"):
            system.validate()

    def test_populates_targets_on_validate(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(id="edge1", source="source", target="sink"))

        system.validate()

        # Source should have sink (target node ID) in its targets
        assert "sink" in source.targets


class TestSimulation:
    def test_runs_for_given_timesteps(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        system.simulate(timesteps=3)

        # Check source generated water at each timestep
        gen_events = source.events_of_type(WaterGenerated)
        assert len(gen_events) == 3
        assert [e.t for e in gen_events] == [0, 1, 2]

    def test_validates_if_not_validated(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        assert system._validated is False
        system.simulate(timesteps=1)
        assert system._validated is True

    def test_routes_water_through_network(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        system.simulate(timesteps=1)

        # Sink should have received water
        received_events = sink.events_of_type(WaterReceived)
        assert len(received_events) == 1
        assert received_events[0].t == 0

    def test_processes_in_topological_order(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        storage = make_storage()
        sink = make_sink()

        system.add_node(source)
        system.add_node(storage)
        system.add_node(sink)

        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))

        system.simulate(timesteps=1)

        # All nodes should have been processed
        source_gen = source.events_of_type(WaterGenerated)
        storage_recv = storage.events_of_type(WaterReceived)
        sink_recv = sink.events_of_type(WaterReceived)

        assert len(source_gen) == 1
        assert len(storage_recv) == 1
        assert len(sink_recv) == 1

    def test_splitter_distributes_to_multiple_targets(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.simulate(timesteps=1)

        # Both sinks should have received water
        sink1_recv = sink1.events_of_type(WaterReceived)
        sink2_recv = sink2.events_of_type(WaterReceived)

        assert len(sink1_recv) == 1
        assert len(sink2_recv) == 1

    def test_edge_losses_reduce_delivered_amount(self):
        from taqsim.common import SEEPAGE

        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()

        # Edge with 50% loss
        loss_rule = FakeEdgeLossRule(losses={SEEPAGE: 50.0})
        edge = make_edge(source="source", target="sink", loss_rule=loss_rule)

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        system.simulate(timesteps=1)

        # Sink should receive 50 (100 - 50 loss)
        received_events = sink.events_of_type(WaterReceived)
        assert len(received_events) == 1
        assert received_events[0].amount == 50.0


class TestSplitterRouting:
    """Regression tests for splitter routing with node IDs."""

    def test_splitter_routes_water_to_downstream_nodes(self):
        """SplitPolicy returning node IDs routes water correctly."""

        class ProportionalSplit:
            def split(self, node, amount: float, t: Timestep) -> dict[str, float]:
                return {"sink1": amount * 0.6, "sink2": amount * 0.4}

        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter(split_policy=ProportionalSplit())
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.simulate(timesteps=1)

        # sink1 should receive 60% (60.0)
        sink1_recv = sink1.events_of_type(WaterReceived)
        assert len(sink1_recv) == 1
        assert sink1_recv[0].amount == 60.0

        # sink2 should receive 40% (40.0)
        sink2_recv = sink2.events_of_type(WaterReceived)
        assert len(sink2_recv) == 1
        assert sink2_recv[0].amount == 40.0

    def test_splitter_node_targets_are_node_ids(self):
        """After validation, node.targets contains downstream node IDs."""
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.validate()

        # Splitter targets should be node IDs, not edge IDs
        assert set(splitter.targets) == {"sink1", "sink2"}

    def test_splitter_raises_on_invalid_target_node(self):
        """SplitPolicy returning invalid node ID raises ValueError."""

        class BadSplit:
            def split(self, node, amount: float, t: Timestep) -> dict[str, float]:
                return {"nonexistent_node": amount}

        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter(split_policy=BadSplit())
        sink = make_sink()

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink"))

        with pytest.raises(ValueError, match="nonexistent_node"):
            system.simulate(timesteps=1)

    def test_single_output_node_routes_via_node_id(self):
        """Single-output nodes route correctly with node ID targets."""
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="sink"))

        system.simulate(timesteps=1)

        # Verify water flows correctly from source to sink
        received_events = sink.events_of_type(WaterReceived)
        assert len(received_events) == 1
        assert received_events[0].amount == 100.0  # Default source inflow
