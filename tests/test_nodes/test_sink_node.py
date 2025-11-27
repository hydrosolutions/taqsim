import pytest

from taqsim.nodes import SinkNode


class MockEdge:
    def __init__(self, source_id: str, flow_after_losses: list[float] | None = None):
        self.source = type("obj", (object,), {"id": source_id})
        self.flow_after_losses = flow_after_losses if flow_after_losses else []

    def set_flow_at_timestep(self, timestep: int, flow: float):
        while len(self.flow_after_losses) <= timestep:
            self.flow_after_losses.append(0.0)
        self.flow_after_losses[timestep] = flow


class TestSinkNodeConstructor:
    def setup_method(self):
        SinkNode.all_ids.clear()

    def test_valid_initialization_with_constant_min_flow(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=5.0, num_time_steps=12)

        assert node.id == "sink1"
        assert node.easting == 1000.0
        assert node.northing == 2000.0
        assert node.min_flows == [5.0] * 12
        assert node.flow_history == []
        assert node.flow_deficits == []
        assert node.inflow_edges == {}
        assert "sink1" in SinkNode.all_ids

    def test_fallback_to_zeros_when_no_min_flow_specified(self):
        node = SinkNode(id="sink2", easting=1000.0, northing=2000.0, num_time_steps=10)

        assert node.min_flows == [0] * 10

    def test_registry_tracks_multiple_nodes(self):
        SinkNode(id="sink1", easting=1000.0, northing=2000.0, num_time_steps=5)
        SinkNode(id="sink2", easting=1100.0, northing=2100.0, num_time_steps=5)

        assert len(SinkNode.all_ids) == 2
        assert "sink1" in SinkNode.all_ids
        assert "sink2" in SinkNode.all_ids

    def test_invalid_node_id_raises_error(self):
        with pytest.raises(ValueError, match="SinkNodeID cannot be empty"):
            SinkNode(id="", easting=1000.0, northing=2000.0, num_time_steps=5)

    def test_invalid_coordinates_raise_error(self):
        with pytest.raises(ValueError):
            SinkNode(id="sink1", easting=None, northing=2000.0, num_time_steps=5)


class TestSinkNodeAddInflowEdge:
    def setup_method(self):
        SinkNode.all_ids.clear()

    def test_stores_edge_keyed_by_source_id(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, num_time_steps=5)
        edge = MockEdge(source_id="source1")

        node.add_inflow_edge(edge)

        assert "source1" in node.inflow_edges
        assert node.inflow_edges["source1"] is edge

    def test_handles_multiple_inflows(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, num_time_steps=5)
        edge1 = MockEdge(source_id="source1")
        edge2 = MockEdge(source_id="source2")
        edge3 = MockEdge(source_id="source3")

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)
        node.add_inflow_edge(edge3)

        assert len(node.inflow_edges) == 3
        assert "source1" in node.inflow_edges
        assert "source2" in node.inflow_edges
        assert "source3" in node.inflow_edges


class TestSinkNodeUpdate:
    def setup_method(self):
        SinkNode.all_ids.clear()

    def test_single_inflow_no_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=5.0, num_time_steps=5)
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [10.0]
        assert node.flow_deficits == [0.0]

    def test_single_inflow_with_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=10.0, num_time_steps=5)
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 3.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [3.0]
        assert node.flow_deficits == [7.0]

    def test_multiple_inflows_sum_correctly(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=15.0, num_time_steps=5)
        edge1 = MockEdge(source_id="source1")
        edge1.set_flow_at_timestep(0, 5.0)
        edge2 = MockEdge(source_id="source2")
        edge2.set_flow_at_timestep(0, 7.0)
        edge3 = MockEdge(source_id="source3")
        edge3.set_flow_at_timestep(0, 3.0)

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)
        node.add_inflow_edge(edge3)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [15.0]
        assert node.flow_deficits == [0.0]

    def test_multiple_inflows_with_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=20.0, num_time_steps=5)
        edge1 = MockEdge(source_id="source1")
        edge1.set_flow_at_timestep(0, 5.0)
        edge2 = MockEdge(source_id="source2")
        edge2.set_flow_at_timestep(0, 7.0)

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [12.0]
        assert node.flow_deficits == [8.0]

    def test_exact_min_flow_met_no_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=10.0, num_time_steps=5)
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [10.0]
        assert node.flow_deficits == [0.0]

    def test_zero_min_flow_never_has_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=0.0, num_time_steps=5)
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 5.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [5.0]
        assert node.flow_deficits == [0.0]

    def test_multiple_timesteps_accumulate_history(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=10.0, num_time_steps=3)
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 12.0)
        edge.set_flow_at_timestep(1, 8.0)
        edge.set_flow_at_timestep(2, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)
        node.update(time_step=1, dt=3600.0)
        node.update(time_step=2, dt=3600.0)

        assert node.flow_history == [12.0, 8.0, 10.0]
        assert node.flow_deficits == [0.0, 2.0, 0.0]

    def test_no_inflow_edges_results_in_full_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=10.0, num_time_steps=5)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [0.0]
        assert node.flow_deficits == [10.0]

    def test_zero_flows_in_edges_results_in_full_deficit(self):
        node = SinkNode(id="sink1", easting=1000.0, northing=2000.0, constant_min_flow=15.0, num_time_steps=5)
        edge1 = MockEdge(source_id="source1")
        edge1.set_flow_at_timestep(0, 0.0)
        edge2 = MockEdge(source_id="source2")
        edge2.set_flow_at_timestep(0, 0.0)

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)

        node.update(time_step=0, dt=3600.0)

        assert node.flow_history == [0.0]
        assert node.flow_deficits == [15.0]
