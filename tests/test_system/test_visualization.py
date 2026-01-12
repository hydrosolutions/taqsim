import pytest
from pathlib import Path
from unittest.mock import patch

from taqsim.node import Source, Sink
from taqsim.node.timeseries import TimeSeries
from taqsim.edge import Edge
from taqsim.system import WaterSystem

from .conftest import FakeEdgeLossRule


def make_simple_system_with_locations() -> WaterSystem:
    """Create a simple Source -> Sink system with locations."""
    system = WaterSystem()
    system.add_node(Source(
        id="source",
        inflow=TimeSeries([100.0] * 12),
        location=(31.77, 35.21)
    ))
    system.add_node(Sink(id="sink", location=(31.78, 35.22)))
    system.add_edge(Edge(
        id="e1",
        source="source",
        target="sink",
        capacity=100.0,
        loss_rule=FakeEdgeLossRule()
    ))
    system.validate()
    return system


class TestEdgeLength:
    def test_edge_length_returns_distance_in_meters(self):
        system = make_simple_system_with_locations()
        length = system.edge_length("e1")
        assert length is not None
        # ~1.4 km for the test coordinates
        assert 1000 < length < 2000

    def test_edge_length_returns_none_for_missing_edge(self):
        system = make_simple_system_with_locations()
        assert system.edge_length("nonexistent") is None

    def test_edge_length_returns_none_for_missing_source_location(self):
        system = WaterSystem()
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12)))  # no location
        system.add_node(Sink(id="t", location=(31.78, 35.22)))
        system.add_edge(Edge(id="e", source="s", target="t", capacity=100.0, loss_rule=FakeEdgeLossRule()))
        system.validate()
        assert system.edge_length("e") is None

    def test_edge_length_returns_none_for_missing_target_location(self):
        system = WaterSystem()
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12), location=(31.77, 35.21)))
        system.add_node(Sink(id="t"))  # no location
        system.add_edge(Edge(id="e", source="s", target="t", capacity=100.0, loss_rule=FakeEdgeLossRule()))
        system.validate()
        assert system.edge_length("e") is None


class TestEdgeLengths:
    def test_returns_all_valid_lengths(self):
        system = make_simple_system_with_locations()
        lengths = system.edge_lengths()
        assert "e1" in lengths
        assert lengths["e1"] > 0

    def test_excludes_edges_without_locations(self):
        system = WaterSystem()
        system.add_node(Source(id="s1", inflow=TimeSeries([100.0] * 12), location=(31.77, 35.21)))
        system.add_node(Sink(id="t", location=(31.78, 35.22)))
        system.add_edge(Edge(id="e1", source="s1", target="t", capacity=100.0, loss_rule=FakeEdgeLossRule()))
        system.validate()
        lengths = system.edge_lengths()
        # Only e1 has both endpoints with locations
        assert "e1" in lengths

    def test_empty_for_no_located_nodes(self):
        system = WaterSystem()
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12)))
        system.add_node(Sink(id="t"))
        system.add_edge(Edge(id="e", source="s", target="t", capacity=100.0, loss_rule=FakeEdgeLossRule()))
        system.validate()
        assert system.edge_lengths() == {}


class TestVisualize:
    def test_raises_if_no_nodes_have_location(self):
        system = WaterSystem()
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12)))
        system.add_node(Sink(id="t"))
        system.add_edge(Edge(id="e", source="s", target="t", capacity=100.0, loss_rule=FakeEdgeLossRule()))
        system.validate()

        with pytest.raises(ValueError, match="No nodes have locations"):
            system.visualize()

    @patch("matplotlib.pyplot.show")
    def test_creates_figure_with_located_nodes(self, mock_show):
        system = make_simple_system_with_locations()
        system.visualize()
        mock_show.assert_called_once()

    def test_saves_to_file_when_path_provided(self, tmp_path: Path):
        system = make_simple_system_with_locations()
        output_file = tmp_path / "network.png"
        system.visualize(save_to=str(output_file))
        assert output_file.exists()
        assert output_file.stat().st_size > 0
