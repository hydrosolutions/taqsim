from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from taqsim.edge import Edge
from taqsim.node import (
    BaseNode,
    Demand,
    PassThrough,
    Reach,
    Sink,
    Source,
    Splitter,
    Storage,
)

NODE_COLORS: dict[type, str] = {
    Source: "#3498db",
    Storage: "#27ae60",
    Splitter: "#e67e22",
    Demand: "#e74c3c",
    PassThrough: "#9b59b6",
    Reach: "#8B4513",
    Sink: "#7f8c8d",
}

NODE_SIZES: dict[type, int] = {
    Source: 600,
    Storage: 700,
    Splitter: 500,
    Demand: 500,
    PassThrough: 400,
    Reach: 350,
    Sink: 500,
}

NODE_LABELS: dict[type, str] = {
    Source: "Source",
    Storage: "Storage",
    Splitter: "Splitter",
    Demand: "Demand",
    PassThrough: "PassThrough",
    Reach: "Reach",
    Sink: "Sink",
}


@dataclass(frozen=True)
class CollapsedEdge:
    source: str
    target: str
    reach_id: str


def visualize_system(
    nodes: dict[str, BaseNode],
    edges: dict[str, Edge],
    *,
    show_reaches: bool = True,
    save_to: str | Path | None = None,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if show_reaches:
        active_nodes = nodes
        active_edges = edges
        collapsed_edges: list[CollapsedEdge] = []
    else:
        active_nodes, active_edges, collapsed_edges = _collapse_reaches(nodes, edges)

    graph = _build_graph(active_nodes, active_edges)
    pos = _compute_positions(graph, active_nodes)

    fig, ax = plt.subplots(figsize=figsize)

    handles = _draw_nodes(graph, pos, active_nodes, ax)
    _draw_edges(graph, pos, ax, collapsed_edges)
    _draw_labels(graph, pos, ax)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Water System ({len(active_nodes)} nodes, {len(active_edges) + len(collapsed_edges)} edges)")

    ax.legend(handles=handles, loc="best")
    ax.set_axis_off()
    plt.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    return fig, ax


def _build_graph(
    nodes: dict[str, BaseNode],
    edges: dict[str, Edge],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, node in nodes.items():
        graph.add_node(node_id, node_type=type(node))
    for edge in edges.values():
        graph.add_edge(edge.source, edge.target)
    return graph


def _collapse_reaches(
    nodes: dict[str, BaseNode],
    edges: dict[str, Edge],
) -> tuple[dict[str, BaseNode], dict[str, Edge], list[CollapsedEdge]]:
    reach_ids: set[str] = {nid for nid, n in nodes.items() if isinstance(n, Reach)}

    adjacency: dict[str, list[str]] = {}
    for edge in edges.values():
        adjacency.setdefault(edge.source, []).append(edge.target)

    reverse: dict[str, list[str]] = {}
    for edge in edges.values():
        reverse.setdefault(edge.target, []).append(edge.source)

    def _resolve_source(node_id: str) -> list[str]:
        if node_id not in reach_ids:
            return [node_id]
        return [s for pred in reverse.get(node_id, []) for s in _resolve_source(pred)]

    def _resolve_target(node_id: str) -> list[str]:
        if node_id not in reach_ids:
            return [node_id]
        return [t for succ in adjacency.get(node_id, []) for t in _resolve_target(succ)]

    edges_to_remove: set[str] = set()
    collapsed: list[CollapsedEdge] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for reach_id in reach_ids:
        for edge in edges.values():
            if edge.target == reach_id or edge.source == reach_id:
                edges_to_remove.add(edge.id)

        sources = _resolve_source(reach_id)
        targets = _resolve_target(reach_id)
        for src in sources:
            for tgt in targets:
                key = (src, tgt, reach_id)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    collapsed.append(CollapsedEdge(source=src, target=tgt, reach_id=reach_id))

    filtered_nodes = {nid: n for nid, n in nodes.items() if nid not in reach_ids}
    remaining_edges = {eid: e for eid, e in edges.items() if eid not in edges_to_remove}

    return filtered_nodes, remaining_edges, collapsed


def _compute_positions(
    graph: nx.DiGraph,
    nodes: dict[str, BaseNode],
) -> dict[str, tuple[float, float]]:
    located: dict[str, tuple[float, float]] = {
        nid: (node.location[1], node.location[0]) for nid, node in nodes.items() if node.location is not None
    }

    if len(located) == len(nodes) and len(nodes) > 0:
        return located

    if not located:
        warnings.warn(
            "No nodes have locations; using automatic layout.",
            stacklevel=2,
        )
        return nx.kamada_kawai_layout(graph)

    return nx.spring_layout(graph, pos=located, fixed=list(located.keys()))


def _draw_nodes(
    graph: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    nodes: dict[str, BaseNode],
    ax: plt.Axes,
) -> list[Any]:
    groups: dict[type, list[str]] = {}
    for node_id, node in nodes.items():
        node_type = type(node)
        groups.setdefault(node_type, []).append(node_id)

    handles: list[Any] = []
    for node_type, ids in groups.items():
        collection = nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=ids,
            node_color=NODE_COLORS[node_type],
            node_size=NODE_SIZES[node_type],
            label=NODE_LABELS[node_type],
            ax=ax,
        )
        handles.append(collection)

    return handles


def _draw_edges(
    graph: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    ax: plt.Axes,
    collapsed_edges: list[CollapsedEdge],
) -> None:
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#cccccc",
        arrows=True,
        arrowsize=15,
        width=1.5,
    )

    if collapsed_edges:
        collapsed_graph = nx.DiGraph()
        for ce in collapsed_edges:
            collapsed_graph.add_edge(ce.source, ce.target)

        collapsed_pos = {n: pos[n] for n in collapsed_graph.nodes() if n in pos}

        nx.draw_networkx_edges(
            collapsed_graph,
            collapsed_pos,
            ax=ax,
            edge_color="#8B4513",
            style="dashed",
            arrows=True,
            arrowsize=15,
            width=2.0,
        )

        labels = {(ce.source, ce.target): ce.reach_id for ce in collapsed_edges}
        nx.draw_networkx_edge_labels(
            collapsed_graph,
            collapsed_pos,
            edge_labels=labels,
            ax=ax,
            font_color="#8B4513",
            font_size=8,
        )


def _draw_labels(
    graph: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    ax: plt.Axes,
) -> None:
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_weight="bold")
