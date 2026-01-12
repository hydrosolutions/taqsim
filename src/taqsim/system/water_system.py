from dataclasses import dataclass, field

import networkx as nx

from taqsim.edge import Edge
from taqsim.node import BaseNode, Receives, Sink, Source, Splitter
from taqsim.node.events import WaterDistributed, WaterOutput

from .validation import ValidationError


@dataclass
class WaterSystem:
    dt: float = 2629800.0

    _nodes: dict[str, BaseNode] = field(default_factory=dict, init=False, repr=False)
    _edges: dict[str, Edge] = field(default_factory=dict, init=False, repr=False)
    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)

    def add_node(self, node: BaseNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists")
        self._nodes[node.id] = node
        self._graph.add_node(node.id)
        self._validated = False

    def add_edge(self, edge: Edge) -> None:
        if edge.id in self._edges:
            raise ValueError(f"Edge '{edge.id}' already exists")
        self._edges[edge.id] = edge
        self._validated = False

    def validate(self) -> None:
        errors: list[str] = []

        # 1. Check node existence for edge endpoints
        for edge_id, edge in self._edges.items():
            if edge.source not in self._nodes:
                errors.append(f"Edge '{edge_id}': source node '{edge.source}' does not exist")
            if edge.target not in self._nodes:
                errors.append(f"Edge '{edge_id}': target node '{edge.target}' does not exist")

        if errors:
            raise ValidationError("\n".join(errors))

        # 2. Build graph edges from edge definitions
        for edge in self._edges.values():
            self._graph.add_edge(edge.source, edge.target, edge_id=edge.id)

        # 3. Check acyclic (DAG)
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValidationError("Network contains cycles")

        # 4. Check connectivity (weakly connected)
        if len(self._nodes) > 0 and not nx.is_weakly_connected(self._graph):
            raise ValidationError("Network is not connected")

        # 5. Check terminal structure
        for node_id, node in self._nodes.items():
            in_degree = self._graph.in_degree(node_id)
            out_degree = self._graph.out_degree(node_id)

            if isinstance(node, Source) and in_degree != 0:
                errors.append(f"Source '{node_id}' must have in_degree=0, got {in_degree}")
            elif isinstance(node, Sink) and out_degree != 0:
                errors.append(f"Sink '{node_id}' must have out_degree=0, got {out_degree}")

        if errors:
            raise ValidationError("\n".join(errors))

        # 6. Check single output for non-Splitter nodes (excluding Sink which has out_degree=0)
        for node_id, node in self._nodes.items():
            if isinstance(node, (Splitter, Sink)):
                continue
            out_degree = self._graph.out_degree(node_id)
            if out_degree != 1:
                errors.append(f"Non-splitter node '{node_id}' must have exactly 1 outgoing edge, got {out_degree}")

        if errors:
            raise ValidationError("\n".join(errors))

        # 7. Check path to sink for all nodes
        sink_ids = {nid for nid, n in self._nodes.items() if isinstance(n, Sink)}
        for node_id in self._nodes:
            if node_id in sink_ids:
                continue
            has_path = any(nx.has_path(self._graph, node_id, sink_id) for sink_id in sink_ids)
            if not has_path:
                errors.append(f"Node '{node_id}' has no path to any Sink")

        if errors:
            raise ValidationError("\n".join(errors))

        # 8. Populate targets via _set_targets()
        self._set_targets()

        self._validated = True

    def _set_targets(self) -> None:
        for node_id, node in self._nodes.items():
            if isinstance(node, Sink):
                continue

            # Find outgoing edges for this node
            outgoing_edges = [edge for edge in self._edges.values() if edge.source == node_id]

            target_edge_ids = [e.id for e in outgoing_edges]
            node._set_targets(target_edge_ids)

    def simulate(self, timesteps: int) -> None:
        if not self._validated:
            self.validate()

        for t in range(timesteps):
            for node_id in nx.topological_sort(self._graph):
                node = self._nodes[node_id]
                node.update(t, self.dt)
                self._route_output(node_id, t)

    def _route_output(self, node_id: str, t: int) -> None:
        node = self._nodes[node_id]
        events = node.events_at(t)

        # Handle WaterOutput events (single-output nodes: Source, Storage, Demand, PassThrough)
        output_events = [e for e in events if isinstance(e, WaterOutput)]
        for event in output_events:
            # Single-output nodes have exactly one target edge
            if not node.targets:
                continue
            target_edge_id = node.targets[0]
            self._deliver_to_edge(target_edge_id, event.amount, t)

        # Handle WaterDistributed events (Splitter with multiple targets)
        distributed_events = [e for e in events if isinstance(e, WaterDistributed)]
        for event in distributed_events:
            self._deliver_to_edge(event.target_id, event.amount, t)

    def _deliver_to_edge(self, edge_id: str, amount: float, t: int) -> None:
        if edge_id not in self._edges:
            return

        edge = self._edges[edge_id]
        edge.receive(amount, t)
        delivered = edge.update(t, self.dt)

        # Route to target node
        target_node = self._nodes[edge.target]
        if isinstance(target_node, Receives):
            target_node.receive(delivered, edge.id, t)

    @property
    def nodes(self) -> dict[str, BaseNode]:
        return self._nodes

    @property
    def edges(self) -> dict[str, Edge]:
        return self._edges
