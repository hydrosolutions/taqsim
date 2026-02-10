import pytest

from taqsim.node import WaterReceived
from taqsim.system import WaterSystem
from taqsim.testing import ProportionalReachLoss, make_passthrough, make_reach
from taqsim.time import Frequency

from .conftest import make_edge, make_sink, make_source, make_splitter


class TestConnectPlain:
    def test_creates_edge_with_correct_source_and_target(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink")

        edge = system.edges["source_to_sink"]
        assert edge.source == "source"
        assert edge.target == "sink"

    def test_auto_generates_edge_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink")

        assert "source_to_sink" in system.edges

    def test_returns_self_for_chaining(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        result = system.connect("source", "sink")

        assert result is system

    def test_passes_tags_to_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink", tags=frozenset({"main"}))

        edge = system.edges["source_to_sink"]
        assert edge.tags == frozenset({"main"})

    def test_passes_metadata_to_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink", metadata={"length": 10})

        edge = system.edges["source_to_sink"]
        assert edge.metadata == {"length": 10}

    def test_defaults_tags_to_empty_frozenset(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink")

        edge = system.edges["source_to_sink"]
        assert edge.tags == frozenset()

    def test_defaults_metadata_to_empty_dict(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink")

        edge = system.edges["source_to_sink"]
        assert edge.metadata == {}


class TestConnectViaReach:
    def test_adds_reach_node_to_system(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        system.connect("source", "sink", via=reach)

        assert "reach" in system.nodes

    def test_creates_two_edges_with_correct_ids(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        system.connect("source", "sink", via=reach)

        assert "source_to_reach" in system.edges
        assert "reach_to_sink" in system.edges

    def test_first_edge_source_to_reach(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        system.connect("source", "sink", via=reach)

        edge = system.edges["source_to_reach"]
        assert edge.source == "source"
        assert edge.target == "reach"

    def test_second_edge_reach_to_target(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        system.connect("source", "sink", via=reach)

        edge = system.edges["reach_to_sink"]
        assert edge.source == "reach"
        assert edge.target == "sink"

    def test_passes_tags_to_both_edges(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        system.connect("source", "sink", via=reach, tags=frozenset({"main"}))

        assert system.edges["source_to_reach"].tags == frozenset({"main"})
        assert system.edges["reach_to_sink"].tags == frozenset({"main"})

    def test_skips_add_node_if_reach_already_present(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        system.add_node(reach)
        system.connect("source", "sink", via=reach)

        assert system.nodes["reach"] is reach
        assert "source_to_reach" in system.edges
        assert "reach_to_sink" in system.edges


class TestConnectErrors:
    def test_raises_on_duplicate_plain_edge_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        system.connect("source", "sink")

        with pytest.raises(ValueError, match="Edge 'source_to_sink' already exists"):
            system.connect("source", "sink")

    def test_raises_on_duplicate_reach_node_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source(id="src1"))
        system.add_node(make_source(id="src2"))
        system.add_node(make_sink(id="sink1"))
        system.add_node(make_sink(id="sink2"))
        reach1 = make_reach(id="reach")
        reach2 = make_reach(id="reach")

        system.connect("src1", "sink1", via=reach1)

        with pytest.raises(ValueError, match="Node 'reach' already exists"):
            system.connect("src2", "sink2", via=reach2)

    def test_raises_on_self_loop(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())

        with pytest.raises(ValueError, match="source and target cannot be the same"):
            system.connect("source", "source")

    def test_raises_on_empty_source(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_sink())

        with pytest.raises(ValueError, match="source cannot be empty"):
            system.connect("", "sink")

    def test_raises_on_empty_target(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())

        with pytest.raises(ValueError, match="target cannot be empty"):
            system.connect("source", "")

    def test_raises_on_invalid_via_type(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())

        with pytest.raises(TypeError, match="via must be a Reach node"):
            system.connect("source", "sink", via="not_a_reach")

    def test_atomicity_no_partial_state_on_edge_failure(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        reach = make_reach(id="reach")

        # Pre-add the second edge ID so it collides
        system.add_edge(make_edge(id="reach_to_sink", source="reach", target="sink"))

        nodes_before = dict(system.nodes)
        edges_before = dict(system.edges)

        with pytest.raises(ValueError, match="Edge 'reach_to_sink' already exists"):
            system.connect("source", "sink", via=reach)

        # No reach node added, no source_to_reach edge added
        assert "reach" not in system.nodes
        assert "source_to_reach" not in system.edges
        # Only the pre-existing edge remains
        assert system.nodes == nodes_before
        assert system.edges == edges_before


class TestConnectInvalidation:
    def test_plain_connect_sets_validated_false(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="sink"))
        system.validate()

        assert system._validated is True

        # Remove old edge and connect fresh (need a different edge id)
        # Instead, just build a new valid system and connect an additional edge
        system2 = WaterSystem(frequency=Frequency.MONTHLY)
        system2.add_node(make_source())
        system2.add_node(make_sink())
        system2.add_edge(make_edge(id="e1", source="source", target="sink"))
        system2.validate()

        assert system2._validated is True

        system2.add_node(make_sink(id="sink2"))
        system2.connect("source", "sink2")

        assert system2._validated is False

    def test_via_reach_connect_sets_validated_false(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="sink"))
        system.validate()

        assert system._validated is True

        system.add_node(make_sink(id="sink2"))
        reach = make_reach(id="reach")
        system.connect("source", "sink2", via=reach)

        assert system._validated is False


class TestConnectSimulation:
    def test_plain_connection_delivers_water(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        system.add_node(source)
        system.add_node(sink)

        system.connect("source", "sink")
        system.simulate(timesteps=1)

        received = sink.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 100.0

    def test_reach_passthrough_delivers_full_amount(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        reach = make_reach(id="reach")
        system.add_node(source)
        system.add_node(sink)

        system.connect("source", "sink", via=reach)
        system.simulate(timesteps=1)

        received = sink.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 100.0

    def test_reach_losses_reduce_delivery(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        reach = make_reach(id="reach", loss_rule=ProportionalReachLoss(loss_fraction=0.1))
        system.add_node(source)
        system.add_node(sink)

        system.connect("source", "sink", via=reach)
        system.simulate(timesteps=1)

        received = sink.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 90.0


class TestConnectChaining:
    def test_multiple_plain_connections(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        pt = make_passthrough(id="passthrough")
        sink = make_sink()
        system.add_node(source)
        system.add_node(pt)
        system.add_node(sink)

        system.connect("source", "passthrough").connect("passthrough", "sink")
        system.simulate(timesteps=1)

        received = sink.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 100.0

    def test_splitter_with_multiple_connects(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")
        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.connect("source", "splitter").connect("splitter", "sink1").connect("splitter", "sink2")
        system.simulate(timesteps=1)

        received1 = sink1.events_of_type(WaterReceived)
        received2 = sink2.events_of_type(WaterReceived)
        assert len(received1) == 1
        assert len(received2) == 1
        assert received1[0].amount == 50.0
        assert received2[0].amount == 50.0
