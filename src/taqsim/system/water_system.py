from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

from taqsim.common import ParamSpec
from taqsim.edge import Edge
from taqsim.geo import haversine
from taqsim.node import BaseNode, Demand, PassThrough, Receives, Sink, Source, Splitter, Storage
from taqsim.node.events import WaterDistributed, WaterOutput
from taqsim.time import Frequency, Timestep

from .validation import ValidationError

if TYPE_CHECKING:
    from taqsim.constraints import ConstraintSpec


@dataclass
class WaterSystem:
    frequency: Frequency

    _nodes: dict[str, BaseNode] = field(default_factory=dict, init=False, repr=False)
    _edges: dict[str, Edge] = field(default_factory=dict, init=False, repr=False)
    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)
    _source_target_to_edge: dict[tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)

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
        """Set targets (downstream node IDs) for each node and build edge lookup."""
        self._source_target_to_edge.clear()

        for node_id, node in self._nodes.items():
            if isinstance(node, Sink):
                continue

            outgoing_edges = [edge for edge in self._edges.values() if edge.source == node_id]

            # Set targets as downstream NODE IDs (not edge IDs)
            target_node_ids = [e.target for e in outgoing_edges]
            node._set_targets(target_node_ids)

            # Build edge lookup: (source_node, target_node) -> edge_id
            for edge in outgoing_edges:
                self._source_target_to_edge[(node_id, edge.target)] = edge.id

    def _get_edge_to(self, source_node_id: str, target_node_id: str) -> str:
        """Get the edge ID connecting source to target node.

        Raises:
            ValueError: If no edge exists between the nodes.
        """
        key = (source_node_id, target_node_id)
        if key not in self._source_target_to_edge:
            raise ValueError(
                f"No edge from '{source_node_id}' to '{target_node_id}'. "
                f"Check that SplitRule returns valid downstream node IDs."
            )
        return self._source_target_to_edge[key]

    def simulate(self, timesteps: int) -> None:
        if not self._validated:
            self.validate()
        self._validate_time_varying_lengths(timesteps)

        for i in range(timesteps):
            t = Timestep(index=i, frequency=self.frequency)
            for node_id in nx.topological_sort(self._graph):
                node = self._nodes[node_id]
                node.update(t)
                self._route_output(node_id, t)

    def _route_output(self, node_id: str, t: Timestep) -> None:
        node = self._nodes[node_id]
        events = node.events_at(t.index)

        # Handle WaterOutput events (single-output nodes: Source, Storage, Demand, PassThrough)
        output_events = [e for e in events if isinstance(e, WaterOutput)]
        for event in output_events:
            if not node.targets:
                continue
            target_node_id = node.targets[0]  # Now a node ID
            edge_id = self._get_edge_to(node_id, target_node_id)
            self._deliver_to_edge(edge_id, event.amount, t)

        # Handle WaterDistributed events (Splitter with multiple targets)
        distributed_events = [e for e in events if isinstance(e, WaterDistributed)]
        for event in distributed_events:
            edge_id = self._get_edge_to(node_id, event.target_id)  # target_id is now a node ID
            self._deliver_to_edge(edge_id, event.amount, t)

    def _deliver_to_edge(self, edge_id: str, amount: float, t: Timestep) -> None:
        if edge_id not in self._edges:
            raise ValueError(f"Edge '{edge_id}' not found in system")

        edge = self._edges[edge_id]
        edge.receive(amount, t)
        delivered = edge.update(t)

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

    def param_schema(self) -> list[ParamSpec]:
        """Discover all tunable parameters from node strategies.

        Returns a sorted list of ParamSpec objects describing each
        tunable parameter. Only operational strategies (inheriting from
        Strategy) are included - physical models are excluded.
        """
        specs: list[ParamSpec] = []

        for node_id, node in sorted(self._nodes.items()):
            for strategy_name, strategy in node.strategies().items():
                for param_name, value in strategy.params().items():
                    path = f"{node_id}.{strategy_name}.{param_name}"
                    specs.extend(self._flatten_param(path, value))

        return sorted(specs, key=lambda s: s.path)

    def to_vector(self) -> list[float]:
        """Flatten all tunable parameters to a vector for GA optimization."""
        return [spec.value for spec in self.param_schema()]

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Collect bounds for all tunable parameters.

        Raises:
            ValueError: If any strategy parameter lacks bounds.
        """
        bounds: dict[str, tuple[float, float]] = {}
        missing: list[str] = []

        for node_id, node in sorted(self._nodes.items()):
            for strategy_name, strategy in node.strategies().items():
                strategy_bounds = strategy.bounds(node)
                for param_name, value in strategy.params().items():
                    base_path = f"{node_id}.{strategy_name}.{param_name}"
                    if param_name not in strategy_bounds:
                        missing.append(base_path)
                        continue
                    param_bound = strategy_bounds[param_name]

                    if isinstance(value, tuple):
                        for i in range(len(value)):
                            bounds[f"{base_path}[{i}]"] = param_bound
                    else:
                        bounds[base_path] = param_bound

        if missing:
            raise ValueError(f"Missing bounds for parameters: {missing}")
        return bounds

    def bounds_vector(self) -> list[tuple[float, float]]:
        """Return bounds matching to_vector() order."""
        all_bounds = self.param_bounds()
        return [all_bounds.get(spec.path, (float("-inf"), float("inf"))) for spec in self.param_schema()]

    def constraint_specs(self) -> list["ConstraintSpec"]:
        """Collect constraints with resolved paths and bounds.

        Returns fully resolved ConstraintSpec objects for use with make_repair.
        """
        from taqsim.constraints import ConstraintSpec

        result: list[ConstraintSpec] = []
        all_bounds = self.param_bounds()

        for node_id, node in sorted(self._nodes.items()):
            for strategy_name, strategy in node.strategies().items():
                prefix = f"{node_id}.{strategy_name}"
                param_values = strategy.params()

                for constraint in strategy.constraints(node):
                    # Build param_paths: local name -> full path
                    param_paths = {p: f"{prefix}.{p}" for p in constraint.params}

                    # Build param_bounds: local name -> bounds
                    # For time-varying params, bounds are indexed (e.g., path[0]), so look up first index
                    param_bounds = {}
                    for p, full_path in param_paths.items():
                        if full_path in all_bounds:
                            param_bounds[p] = all_bounds[full_path]
                        elif f"{full_path}[0]" in all_bounds:
                            # Time-varying param: use bounds from first index (all share same bounds)
                            param_bounds[p] = all_bounds[f"{full_path}[0]"]
                        else:
                            raise KeyError(f"No bounds found for {full_path}")

                    # Determine which constraint params are time-varying
                    time_varying_params = frozenset(
                        p for p in constraint.params if isinstance(param_values.get(p), tuple)
                    )

                    spec = ConstraintSpec(
                        constraint=constraint,
                        prefix=prefix,
                        param_paths=param_paths,
                        param_bounds=param_bounds,
                        time_varying_params=time_varying_params,
                    )
                    result.append(spec)

        return result

    def with_vector(self, vector: list[float]) -> "WaterSystem":
        """Create a new WaterSystem with parameters from vector.

        The original system is unchanged (immutable pattern).
        """
        schema = self.param_schema()
        if len(vector) != len(schema):
            raise ValueError(f"Vector length {len(vector)} does not match schema length {len(schema)}")

        # Group values by node.strategy path
        updates: dict[str, dict[str, float | tuple[float, ...]]] = {}
        # Track indexed params separately: key -> {param_name: {idx: value}}
        indexed_params: dict[str, dict[str, dict[int, float]]] = {}

        for spec, value in zip(schema, vector, strict=True):
            parts = spec.path.split(".")
            node_id = parts[0]
            strategy_name = parts[1]
            param_part = parts[2]

            key = f"{node_id}.{strategy_name}"

            # Parse param_part for index: "rate[5]" -> ("rate", 5)
            if "[" in param_part:
                bracket_idx = param_part.index("[")
                param_name = param_part[:bracket_idx]
                idx = int(param_part[bracket_idx + 1 : -1])

                if key not in indexed_params:
                    indexed_params[key] = {}
                if param_name not in indexed_params[key]:
                    indexed_params[key][param_name] = {}
                indexed_params[key][param_name][idx] = value
            else:
                # Scalar param, store directly
                if key not in updates:
                    updates[key] = {}
                updates[key][param_part] = value

        # Convert indexed_params to tuples and merge into updates
        for key, params in indexed_params.items():
            if key not in updates:
                updates[key] = {}
            for param_name, idx_values in params.items():
                # Build tuple from collected values in index order
                max_idx = max(idx_values.keys())
                updates[key][param_name] = tuple(idx_values[i] for i in range(max_idx + 1))

        # Build new system with updated strategies
        return self._clone_with_updates(updates)

    def reset(self) -> None:
        """Reset all nodes and edges for a fresh simulation run.

        Preserves topology and strategies, clears all events and
        resets accumulators to initial state.
        """
        for node in self._nodes.values():
            node.reset()
        for edge in self._edges.values():
            edge.reset()

    def _flatten_param(self, path: str, value: float | tuple[float, ...]) -> list[ParamSpec]:
        """Create ParamSpec(s) for a scalar or time-varying parameter."""
        if isinstance(value, tuple):
            return [ParamSpec(path=f"{path}[{i}]", value=v) for i, v in enumerate(value)]
        return [ParamSpec(path=path, value=value)]

    def _validate_time_varying_lengths(self, timesteps: int) -> None:
        """Validate all time-varying parameters have sufficient length."""
        from taqsim.system.validation import InsufficientLengthError

        for node_id, node in sorted(self._nodes.items()):
            for strategy_name, strategy in node.strategies().items():
                cyclical_params = frozenset(strategy.cyclical())
                for param_name, value in strategy.params().items():
                    if isinstance(value, tuple) and len(value) < timesteps:
                        if param_name in cyclical_params:
                            continue  # Skip validation for cyclical params
                        path = f"{node_id}.{strategy_name}.{param_name}"
                        raise InsufficientLengthError(path, len(value), timesteps)

    def edge_length(self, edge_id: str) -> float | None:
        """Compute geodesic length of an edge in meters.

        Returns None if edge doesn't exist or either endpoint lacks location.
        """
        if edge_id not in self._edges:
            return None

        edge = self._edges[edge_id]
        source_node = self._nodes.get(edge.source)
        target_node = self._nodes.get(edge.target)

        if source_node is None or target_node is None:
            return None
        if source_node.location is None or target_node.location is None:
            return None

        lat1, lon1 = source_node.location
        lat2, lon2 = target_node.location

        return haversine(lat1, lon1, lat2, lon2)

    def edge_lengths(self) -> dict[str, float]:
        """Compute geodesic lengths for all edges where both endpoints have locations."""
        return {edge_id: length for edge_id in self._edges if (length := self.edge_length(edge_id)) is not None}

    def visualize(self, save_to: str | None = None, figsize: tuple[int, int] = (12, 8)) -> None:
        """Visualize the water network on a geographic plot.

        Plots nodes at their (lon, lat) positions with colors/markers by type.
        Draws edges as arrows between connected nodes.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_context("paper", font_scale=1.3)

        # Filter nodes with locations
        located_nodes = {nid: node for nid, node in self._nodes.items() if node.location is not None}

        if not located_nodes:
            raise ValueError("No nodes have locations set. Cannot visualize.")

        # Node type styling
        node_styles = {
            Source: {"color": "blue", "marker": "^", "label": "Source"},
            Storage: {"color": "green", "marker": "s", "label": "Storage"},
            Demand: {"color": "orange", "marker": "o", "label": "Demand"},
            Sink: {"color": "gray", "marker": "v", "label": "Sink"},
            Splitter: {"color": "purple", "marker": "D", "label": "Splitter"},
            PassThrough: {"color": "cyan", "marker": "h", "label": "PassThrough"},
        }

        fig, ax = plt.subplots(figsize=figsize)

        # Collect positions: (lon, lat) -> (x, y)
        positions = {}
        for nid, node in located_nodes.items():
            lat, lon = node.location
            positions[nid] = (lon, lat)  # x=lon, y=lat

        # Plot nodes by type
        plotted_types: set[type] = set()
        for nid, node in located_nodes.items():
            x, y = positions[nid]
            for node_type, style in node_styles.items():
                if isinstance(node, node_type):
                    label = style["label"] if node_type not in plotted_types else None
                    ax.scatter(x, y, c=style["color"], marker=style["marker"], s=150, zorder=3, label=label)
                    plotted_types.add(node_type)
                    break

            # Add node ID label
            ax.annotate(nid, (x, y), textcoords="offset points", xytext=(5, 5), zorder=4)

        # Draw edges as arrows
        for edge in self._edges.values():
            if edge.source in positions and edge.target in positions:
                x1, y1 = positions[edge.source]
                x2, y2 = positions[edge.target]
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
                    zorder=2,
                )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Water System Network")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(useOffset=False)

        plt.tight_layout()
        sns.despine(fig=fig)

        if save_to:
            plt.savefig(save_to, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def _clone_with_updates(self, updates: dict[str, dict[str, float]]) -> "WaterSystem":
        """Create a new system with updated strategy parameters."""
        import copy

        new_system = WaterSystem(frequency=self.frequency)

        # Clone nodes with updated strategies
        for node_id, node in self._nodes.items():
            new_node = copy.deepcopy(node)
            new_node.reset()  # Clear events and state

            # Apply strategy updates
            for strategy_name, strategy in node.strategies().items():
                key = f"{node_id}.{strategy_name}"
                if key in updates:
                    new_strategy = strategy.with_params(**updates[key])
                    setattr(new_node, strategy_name, new_strategy)

            new_system._nodes[node_id] = new_node
            new_system._graph.add_node(node_id)

        # Clone edges
        for edge_id, edge in self._edges.items():
            new_edge = copy.deepcopy(edge)
            new_edge.reset()
            new_system._edges[edge_id] = new_edge

        # Rebuild graph edges
        for edge in new_system._edges.values():
            new_system._graph.add_edge(edge.source, edge.target, edge_id=edge.id)

        new_system._validated = False
        return new_system
