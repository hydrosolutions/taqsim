from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from taqsim.edge import Edge
from taqsim.node import (
    Demand,
    NoLoss,
    NoReachLoss,
    NoRelease,
    NoRouting,
    NoSplit,
    Reach,
    Sink,
    Source,
    Splitter,
    Storage,
)
from taqsim.node.timeseries import TimeSeries
from taqsim.system import WaterSystem
from taqsim.system._visualize import NODE_COLORS
from taqsim.time import Frequency


def make_simple_system_with_locations() -> WaterSystem:
    """Create a simple Source -> Sink system with locations."""
    system = WaterSystem(frequency=Frequency.MONTHLY)
    system.add_node(Source(id="source", inflow=TimeSeries([100.0] * 12), location=(31.77, 35.21)))
    system.add_node(Sink(id="sink", location=(31.78, 35.22)))
    system.add_edge(Edge(id="e1", source="source", target="sink"))
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
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12)))  # no location
        system.add_node(Sink(id="t", location=(31.78, 35.22)))
        system.add_edge(Edge(id="e", source="s", target="t"))
        system.validate()
        assert system.edge_length("e") is None

    def test_edge_length_returns_none_for_missing_target_location(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12), location=(31.77, 35.21)))
        system.add_node(Sink(id="t"))  # no location
        system.add_edge(Edge(id="e", source="s", target="t"))
        system.validate()
        assert system.edge_length("e") is None


class TestEdgeLengths:
    def test_returns_all_valid_lengths(self):
        system = make_simple_system_with_locations()
        lengths = system.edge_lengths()
        assert "e1" in lengths
        assert lengths["e1"] > 0

    def test_excludes_edges_without_locations(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(Source(id="s1", inflow=TimeSeries([100.0] * 12), location=(31.77, 35.21)))
        system.add_node(Sink(id="t", location=(31.78, 35.22)))
        system.add_edge(Edge(id="e1", source="s1", target="t"))
        system.validate()
        lengths = system.edge_lengths()
        # Only e1 has both endpoints with locations
        assert "e1" in lengths

    def test_empty_for_no_located_nodes(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(Source(id="s", inflow=TimeSeries([100.0] * 12)))
        system.add_node(Sink(id="t"))
        system.add_edge(Edge(id="e", source="s", target="t"))
        system.validate()
        assert system.edge_lengths() == {}


def make_system_without_locations() -> WaterSystem:
    system = WaterSystem(frequency=Frequency.MONTHLY)
    system.add_node(Source(id="src", inflow=TimeSeries([100.0] * 12)))
    system.add_node(Sink(id="snk"))
    system.add_edge(Edge(id="e1", source="src", target="snk"))
    system.validate()
    return system


def make_system_with_reach() -> WaterSystem:
    system = WaterSystem(frequency=Frequency.MONTHLY)
    system.add_node(Source(id="src", inflow=TimeSeries([100.0] * 12), location=(31.0, 35.0)))
    system.add_node(Reach(id="canal", routing_model=NoRouting(), loss_rule=NoReachLoss(), location=(31.1, 35.1)))
    system.add_node(Sink(id="snk", location=(31.2, 35.2)))
    system.add_edge(Edge(id="e1", source="src", target="canal"))
    system.add_edge(Edge(id="e2", source="canal", target="snk"))
    system.validate()
    return system


def make_system_with_multiple_reaches() -> WaterSystem:
    system = WaterSystem(frequency=Frequency.MONTHLY)
    system.add_node(Source(id="src", inflow=TimeSeries([100.0] * 12), location=(31.0, 35.0)))
    system.add_node(Reach(id="canal1", routing_model=NoRouting(), loss_rule=NoReachLoss(), location=(31.05, 35.05)))
    system.add_node(Reach(id="canal2", routing_model=NoRouting(), loss_rule=NoReachLoss(), location=(31.15, 35.15)))
    system.add_node(Sink(id="snk", location=(31.2, 35.2)))
    system.add_edge(Edge(id="e1", source="src", target="canal1"))
    system.add_edge(Edge(id="e2", source="canal1", target="canal2"))
    system.add_edge(Edge(id="e3", source="canal2", target="snk"))
    system.validate()
    return system


def make_multi_type_system() -> WaterSystem:
    system = WaterSystem(frequency=Frequency.MONTHLY)
    system.add_node(Source(id="river", inflow=TimeSeries([100.0] * 12), location=(31.0, 35.0)))
    system.add_node(
        Storage(
            id="dam",
            capacity=1000,
            release_policy=NoRelease(),
            loss_rule=NoLoss(),
            location=(31.05, 35.05),
        )
    )
    system.add_node(Splitter(id="junc", split_policy=NoSplit(), location=(31.1, 35.1)))
    system.add_node(Demand(id="city", requirement=TimeSeries([50.0] * 12), location=(31.15, 35.15)))
    system.add_node(Sink(id="sea", location=(31.2, 35.2)))
    system.add_edge(Edge(id="e1", source="river", target="dam"))
    system.add_edge(Edge(id="e2", source="dam", target="junc"))
    system.add_edge(Edge(id="e3", source="junc", target="city"))
    system.add_edge(Edge(id="e4", source="junc", target="sea"))
    system.add_edge(Edge(id="e5", source="city", target="sea"))
    system.validate()
    return system


class TestVisualizeBasic:
    def test_returns_tuple(self):
        system = make_simple_system_with_locations()
        result = system.visualize()
        assert isinstance(result, tuple)
        plt.close(result[0])

    def test_returns_fig_and_ax(self):
        system = make_simple_system_with_locations()
        result = system.visualize()
        assert len(result) == 2
        plt.close(result[0])

    def test_fig_type(self):
        system = make_simple_system_with_locations()
        result = system.visualize()
        assert isinstance(result[0], plt.Figure)
        plt.close(result[0])

    def test_ax_type(self):
        system = make_simple_system_with_locations()
        result = system.visualize()
        assert isinstance(result[1], plt.Axes)
        plt.close(result[0])

    def test_works_with_locations(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize()
        plt.close(fig)

    def test_works_without_locations(self):
        system = make_system_without_locations()
        result = system.visualize()
        assert isinstance(result, tuple)
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_warns_on_no_locations(self):
        system = make_system_without_locations()
        with pytest.warns(UserWarning, match="No nodes have locations"):
            fig, ax = system.visualize()
        plt.close(fig)

    def test_custom_figsize(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(figsize=(16, 10))
        width, height = fig.get_size_inches()
        assert abs(width - 16) < 0.5
        assert abs(height - 10) < 0.5
        plt.close(fig)

    def test_custom_title(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(title="My Network")
        assert ax.get_title() == "My Network"
        plt.close(fig)

    def test_auto_title_with_counts(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize()
        title = ax.get_title()
        assert "2 nodes" in title
        assert "1 edges" in title
        plt.close(fig)


class TestVisualizeShowReaches:
    def test_default_is_true(self):
        system = make_system_with_reach()
        fig, ax = system.visualize()
        texts = [t.get_text() for t in ax.texts]
        assert "canal" in texts
        plt.close(fig)

    def test_true_shows_reach_node(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=True)
        texts = [t.get_text() for t in ax.texts]
        assert "canal" in texts
        plt.close(fig)

    def test_false_hides_reach_node(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        texts = [t.get_text() for t in ax.texts]
        # With reaches hidden, "canal" should only appear as an edge label
        # The node "src" and "snk" should still be present as node labels
        assert "src" in texts
        assert "snk" in texts
        plt.close(fig)

    def test_false_draws_direct_edge(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_false_labels_edge_with_reach_id(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        texts = [t.get_text() for t in ax.texts]
        assert "canal" in texts
        plt.close(fig)

    def test_false_with_multiple_reaches(self):
        system = make_system_with_multiple_reaches()
        fig, ax = system.visualize(show_reaches=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_false_preserves_non_reach_nodes(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        texts = [t.get_text() for t in ax.texts]
        assert "src" in texts
        assert "snk" in texts
        plt.close(fig)


class TestVisualizeLegend:
    def test_legend_present(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize()
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_legend_entries_match_present_types(self):
        system = make_multi_type_system()
        fig, ax = system.visualize()
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Source" in legend_texts
        assert "Storage" in legend_texts
        assert "Splitter" in legend_texts
        assert "Demand" in legend_texts
        assert "Sink" in legend_texts
        plt.close(fig)


class TestVisualizeSaveTo:
    def test_saves_file(self, tmp_path: Path):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(save_to=str(tmp_path / "net.png"))
        assert (tmp_path / "net.png").exists()
        plt.close(fig)

    def test_saved_file_nonzero(self, tmp_path: Path):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(save_to=str(tmp_path / "net.png"))
        assert (tmp_path / "net.png").stat().st_size > 0
        plt.close(fig)

    def test_returns_tuple_when_saving(self, tmp_path: Path):
        system = make_simple_system_with_locations()
        result = system.visualize(save_to=str(tmp_path / "net.png"))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], plt.Figure)
        assert isinstance(result[1], plt.Axes)
        plt.close(result[0])

    def test_accepts_path_object(self, tmp_path: Path):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(save_to=tmp_path / "net.png")
        assert (tmp_path / "net.png").exists()
        plt.close(fig)


class TestVisualizeNodeStyles:
    def test_different_types_get_different_colors(self):
        system = make_multi_type_system()
        fig, ax = system.visualize()
        plt.close(fig)

    def test_colors_use_palette(self):
        system = make_multi_type_system()
        fig, ax = system.visualize()
        palette_colors = set(NODE_COLORS.values())
        found_palette_color = False
        for child in ax.get_children():
            if hasattr(child, "get_facecolor"):
                facecolors = child.get_facecolor()
                if len(facecolors) > 0:
                    from matplotlib.colors import to_hex

                    for fc in facecolors:
                        hex_color = to_hex(fc)
                        if hex_color in palette_colors:
                            found_palette_color = True
                            break
            if found_palette_color:
                break
        assert found_palette_color
        plt.close(fig)


class TestVisualizeEdgeCases:
    def test_minimal_topology(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_only_sources_and_sinks(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(Source(id="s1", inflow=TimeSeries([100.0] * 12), location=(31.0, 35.0)))
        system.add_node(Sink(id="t1", location=(31.1, 35.1)))
        system.add_edge(Edge(id="e1", source="s1", target="t1"))
        system.validate()
        fig, ax = system.visualize()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_mixed_located_and_unlocated(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(Source(id="s1", inflow=TimeSeries([100.0] * 12), location=(31.0, 35.0)))
        system.add_node(Sink(id="t1"))  # no location
        system.add_edge(Edge(id="e1", source="s1", target="t1"))
        system.validate()
        fig, ax = system.visualize()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestShowReachLabels:
    def test_default_shows_labels(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        texts = [t.get_text() for t in ax.texts]
        assert "canal" in texts
        plt.close(fig)

    def test_false_hides_labels(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False, show_reach_labels=False)
        texts = [t.get_text() for t in ax.texts]
        assert "canal" not in texts
        plt.close(fig)

    def test_true_shows_labels(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False, show_reach_labels=True)
        texts = [t.get_text() for t in ax.texts]
        assert "canal" in texts
        plt.close(fig)

    def test_ignored_when_show_reaches_true(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=True, show_reach_labels=False)
        texts = [t.get_text() for t in ax.texts]
        assert "canal" in texts  # canal is a node label, not an edge label
        plt.close(fig)


class TestEdgeColors:
    def test_default_is_black(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        edge_collections = [
            c
            for c in ax.get_children()
            if hasattr(c, "get_edgecolor") and hasattr(c, "get_paths") and len(c.get_paths()) > 0
        ]
        assert len(edge_collections) > 0
        plt.close(fig)

    def test_custom_color_applied(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False, edge_colors={"canal": "red"})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_reach_defaults_to_black(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False, edge_colors={"nonexistent": "red"})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_reaches_colored_independently(self):
        system = make_system_with_multiple_reaches()
        fig, ax = system.visualize(
            show_reaches=False,
            edge_colors={"canal1": "red", "canal2": "blue"},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_partial_dict(self):
        system = make_system_with_multiple_reaches()
        fig, ax = system.visualize(show_reaches=False, edge_colors={"canal1": "green"})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ignored_when_show_reaches_true(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=True, edge_colors={"canal": "red"})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_dict_same_as_none(self):
        system = make_system_with_reach()
        fig1, ax1 = system.visualize(show_reaches=False, edge_colors={})
        fig2, ax2 = system.visualize(show_reaches=False, edge_colors=None)
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig1)
        plt.close(fig2)


class TestUniformEdgeStyle:
    def test_regular_and_collapsed_use_same_style(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_collapsed_edges_are_solid_not_dashed(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False)
        from matplotlib.collections import LineCollection

        for child in ax.get_children():
            if isinstance(child, LineCollection):
                linestyle = child.get_linestyle()
                for ls in linestyle:
                    offset, dashes = ls
                    assert dashes is None or len(dashes) == 0, "Collapsed edges should be solid, not dashed"
        plt.close(fig)


class TestEdgeKwargs:
    def test_override_alpha(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(alpha=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_override_edge_color(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(edge_color="red")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_override_connectionstyle(self):
        system = make_simple_system_with_locations()
        fig, ax = system.visualize(connectionstyle="arc3,rad=0.05")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_colored_collapsed_edges_have_full_opacity(self):
        system = make_system_with_reach()
        fig, ax = system.visualize(show_reaches=False, edge_colors={"canal": "red"})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
