import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from taqsim.common import ParamSpec
from taqsim.edge import Edge
from taqsim.geo import haversine
from taqsim.node import (
    BaseNode,
    Demand,
    NoLoss,
    NoReachLoss,
    NoRelease,
    NoRouting,
    NoSplit,
    PassThrough,
    Reach,
    Receives,
    Sink,
    Source,
    Splitter,
    Storage,
)
from taqsim.node.events import WaterDistributed, WaterOutput
from taqsim.time import Frequency, Timestep, time_index

from .validation import ValidationError

if TYPE_CHECKING:
    from taqsim.constraints import ConstraintSpec

_FREQUENCY_MAP: dict[str, Frequency] = {
    "daily": Frequency.DAILY,
    "weekly": Frequency.WEEKLY,
    "monthly": Frequency.MONTHLY,
    "yearly": Frequency.YEARLY,
}


def _parse_common_fields(data: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {"id": data["id"]}
    if "location" in data:
        loc = data["location"]
        fields["location"] = (loc[0], loc[1])
    if "tags" in data:
        fields["tags"] = frozenset(data["tags"])
    if "metadata" in data:
        fields["metadata"] = dict(data["metadata"])
    if "auxiliary_data" in data:
        fields["auxiliary_data"] = dict(data["auxiliary_data"])
    return fields


def _parse_source(data: dict[str, Any]) -> Source:
    fields = _parse_common_fields(data)
    return Source(inflow=None, **fields)


def _parse_storage(data: dict[str, Any]) -> Storage:
    fields = _parse_common_fields(data)
    if "capacity" not in data:
        raise ValueError(f"Storage '{data['id']}': 'capacity' is required")
    fields["capacity"] = data["capacity"]
    if "initial_storage" in data:
        fields["initial_storage"] = data["initial_storage"]
    if "dead_storage" in data:
        fields["dead_storage"] = data["dead_storage"]
    fields["release_policy"] = NoRelease()
    fields["loss_rule"] = NoLoss()
    return Storage(**fields)


def _parse_demand(data: dict[str, Any]) -> Demand:
    fields = _parse_common_fields(data)
    if "consumption_fraction" in data:
        fields["consumption_fraction"] = data["consumption_fraction"]
    if "efficiency" in data:
        fields["efficiency"] = data["efficiency"]
    return Demand(requirement=None, **fields)


def _parse_splitter(data: dict[str, Any]) -> Splitter:
    fields = _parse_common_fields(data)
    fields["split_policy"] = NoSplit()
    return Splitter(**fields)


def _parse_reach(data: dict[str, Any]) -> Reach:
    fields = _parse_common_fields(data)
    fields["routing_model"] = NoRouting()
    fields["loss_rule"] = NoReachLoss()
    if "capacity" in data:
        fields["capacity"] = data["capacity"]
    return Reach(**fields)


def _parse_passthrough(data: dict[str, Any]) -> PassThrough:
    fields = _parse_common_fields(data)
    if "capacity" in data:
        fields["capacity"] = data["capacity"]
    return PassThrough(**fields)


def _parse_sink(data: dict[str, Any]) -> Sink:
    fields = _parse_common_fields(data)
    return Sink(**fields)


def _parse_edge(data: dict[str, Any]) -> Edge:
    edge_id = data.get("id")
    if not edge_id:
        raise ValueError("Edge is missing required field 'id'")
    source = data.get("source")
    if not source:
        raise ValueError(f"Edge '{edge_id}': 'source' is required")
    target = data.get("target")
    if not target:
        raise ValueError(f"Edge '{edge_id}': 'target' is required")
    tags = frozenset(data.get("tags", []))
    metadata = dict(data.get("metadata", {}))
    return Edge(id=edge_id, source=source, target=target, tags=tags, metadata=metadata)


_NODE_PARSERS: dict[str, Any] = {
    "source": _parse_source,
    "storage": _parse_storage,
    "demand": _parse_demand,
    "splitter": _parse_splitter,
    "reach": _parse_reach,
    "passthrough": _parse_passthrough,
    "sink": _parse_sink,
}

_FREQUENCY_REVERSE: dict[Frequency, str] = {v: k for k, v in _FREQUENCY_MAP.items()}


def _serialize_common_fields(node: BaseNode) -> dict[str, Any]:
    fields: dict[str, Any] = {"id": node.id}
    if node.location is not None:
        fields["location"] = list(node.location)
    if node.tags:
        fields["tags"] = sorted(node.tags)
    if node.metadata:
        fields["metadata"] = dict(node.metadata)
    if node.auxiliary_data:
        fields["auxiliary_data"] = dict(node.auxiliary_data)
    return fields


def _serialize_source(node: Source) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "source"
    return fields


def _serialize_storage(node: Storage) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "storage"
    fields["capacity"] = node.capacity
    fields["initial_storage"] = node.initial_storage
    fields["dead_storage"] = node.dead_storage
    return fields


def _serialize_demand(node: Demand) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "demand"
    fields["consumption_fraction"] = node.consumption_fraction
    fields["efficiency"] = node.efficiency
    return fields


def _serialize_splitter(node: Splitter) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "splitter"
    return fields


def _serialize_reach(node: Reach) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "reach"
    if node.capacity is not None:
        fields["capacity"] = node.capacity
    return fields


def _serialize_passthrough(node: PassThrough) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "passthrough"
    if node.capacity is not None:
        fields["capacity"] = node.capacity
    return fields


def _serialize_sink(node: Sink) -> dict[str, Any]:
    fields = _serialize_common_fields(node)
    fields["type"] = "sink"
    return fields


def _serialize_edge(edge: Edge) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
    }
    if edge.tags:
        fields["tags"] = sorted(edge.tags)
    if edge.metadata:
        fields["metadata"] = dict(edge.metadata)
    return fields


_NODE_SERIALIZERS: dict[type, Any] = {
    Source: _serialize_source,
    Storage: _serialize_storage,
    Demand: _serialize_demand,
    Splitter: _serialize_splitter,
    Reach: _serialize_reach,
    PassThrough: _serialize_passthrough,
    Sink: _serialize_sink,
}


@dataclass
class WaterSystem:
    frequency: Frequency
    start_date: date | None = None

    _nodes: dict[str, BaseNode] = field(default_factory=dict, init=False, repr=False)
    _edges: dict[str, Edge] = field(default_factory=dict, init=False, repr=False)
    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)
    _source_target_to_edge: dict[tuple[str, str], str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_json(cls, source: str | Path) -> "WaterSystem":
        if isinstance(source, Path):
            raw = source.read_text(encoding="utf-8")
        elif isinstance(source, str) and source.strip().startswith("{"):
            raw = source
        else:
            raw = Path(source).read_text(encoding="utf-8")
        data = json.loads(raw)

        freq_str = data.get("frequency")
        if not freq_str:
            raise ValueError("JSON must include a 'frequency' field")
        freq_key = freq_str.strip().lower()
        if freq_key not in _FREQUENCY_MAP:
            raise ValueError(f"Invalid frequency '{freq_str}'. Valid options: {', '.join(sorted(_FREQUENCY_MAP))}")
        frequency = _FREQUENCY_MAP[freq_key]

        start_date: date | None = None
        if "start_date" in data:
            try:
                start_date = date.fromisoformat(data["start_date"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid start_date '{data['start_date']}': {exc}") from exc

        system = cls(frequency=frequency, start_date=start_date)

        for node_data in data.get("nodes", []):
            node_type = node_data.get("type")
            if not node_type:
                raise ValueError(f"Node is missing required field 'type': {node_data}")
            if "id" not in node_data:
                raise ValueError(f"Node is missing required field 'id': {node_data}")
            type_key = node_type.strip().lower()
            parser = _NODE_PARSERS.get(type_key)
            if parser is None:
                raise ValueError(f"Unknown node type '{node_type}'. Valid types: {', '.join(sorted(_NODE_PARSERS))}")
            system.add_node(parser(node_data))

        for edge_data in data.get("edges", []):
            system.add_edge(_parse_edge(edge_data))

        return system

    def to_json(self, save_to: str | Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"frequency": _FREQUENCY_REVERSE[self.frequency]}
        if self.start_date is not None:
            result["start_date"] = self.start_date.isoformat()
        result["nodes"] = [_NODE_SERIALIZERS[type(node)](node) for node in self._nodes.values()]
        result["edges"] = [_serialize_edge(edge) for edge in self._edges.values()]
        if save_to is not None:
            Path(save_to).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

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

    def connect(
        self,
        source: str,
        target: str,
        *,
        via: Reach | None = None,
        tags: frozenset[str] = frozenset(),
        metadata: dict[str, Any] | None = None,
    ) -> "WaterSystem":
        if not source:
            raise ValueError("source cannot be empty")
        if not target:
            raise ValueError("target cannot be empty")
        if source == target:
            raise ValueError("source and target cannot be the same")
        if via is not None and not isinstance(via, Reach):
            raise TypeError("via must be a Reach node")

        # Pre-check edge ID collisions
        if via is None:
            edge_id = f"{source}_to_{target}"
            if edge_id in self._edges:
                raise ValueError(f"Edge '{edge_id}' already exists")
        else:
            edge_id_1 = f"{source}_to_{via.id}"
            edge_id_2 = f"{via.id}_to_{target}"
            if edge_id_1 in self._edges:
                raise ValueError(f"Edge '{edge_id_1}' already exists")
            if edge_id_2 in self._edges:
                raise ValueError(f"Edge '{edge_id_2}' already exists")

        # Check via node identity if already registered
        if via is not None and via.id in self._nodes and self._nodes[via.id] is not via:
            raise ValueError(f"Node '{via.id}' already exists")

        meta = metadata or {}

        if via is None:
            edge = Edge(id=f"{source}_to_{target}", source=source, target=target, tags=tags, metadata=meta)
            self.add_edge(edge)
        else:
            # Add reach node (skip if already present with identity match)
            if via.id not in self._nodes:
                self.add_node(via)
            edge1 = Edge(id=f"{source}_to_{via.id}", source=source, target=via.id, tags=tags, metadata=meta)
            edge2 = Edge(id=f"{via.id}_to_{target}", source=via.id, target=target, tags=tags, metadata=meta)
            self.add_edge(edge1)
            self.add_edge(edge2)

        return self

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

        # 8. Validate auxiliary data requirements
        self._validate_auxiliary_data()

        # 8b. Validate required TimeSeries fields
        ts_errors = self._validate_timeseries()
        if ts_errors:
            raise ValidationError("\n".join(ts_errors))

        # 9. Populate targets via _set_targets()
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
                f"Check that SplitPolicy returns valid downstream node IDs."
            )
        return self._source_target_to_edge[key]

    def simulate(self, timesteps: int) -> None:
        if not self._validated:
            self.validate()
        self._validate_time_varying_lengths(timesteps)

        topo_order = tuple(nx.topological_sort(self._graph))
        for i in range(timesteps):
            t = Timestep(index=i, frequency=self.frequency)
            for node_id in topo_order:
                node = self._nodes[node_id]
                node.update(t)
                self._route_output(node_id, t)

    def _route_output(self, node_id: str, t: Timestep) -> None:
        node = self._nodes[node_id]
        for event in node.take_step_outputs():
            if isinstance(event, WaterOutput):
                if not node.targets:
                    continue
                target_node_id = node.targets[0]
                edge_id = self._get_edge_to(node_id, target_node_id)
                self._deliver_to_edge(edge_id, event.amount, t)
            elif isinstance(event, WaterDistributed):
                edge_id = self._get_edge_to(node_id, event.target_id)
                self._deliver_to_edge(edge_id, event.amount, t)

    def _deliver_to_edge(self, edge_id: str, amount: float, t: Timestep) -> None:
        if edge_id not in self._edges:
            raise ValueError(f"Edge '{edge_id}' not found in system")

        edge = self._edges[edge_id]

        # Edge is pure topology — pass amount directly to target node
        target_node = self._nodes[edge.target]
        if isinstance(target_node, Receives):
            target_node.receive(amount, edge.id, t)

    @property
    def nodes(self) -> dict[str, BaseNode]:
        return self._nodes

    @property
    def edges(self) -> dict[str, Edge]:
        return self._edges

    def time_index(self, n: int) -> tuple[date, ...]:
        if self.start_date is None:
            raise ValueError("time_index() requires start_date to be set")
        return time_index(self.start_date, self.frequency, n)

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

    def _validate_auxiliary_data(self) -> None:
        """Validate that nodes provide auxiliary_data required by their physical models."""
        from dataclasses import fields as dc_fields

        from taqsim.system.validation import MissingAuxiliaryDataError

        for node in self._nodes.values():
            for f in dc_fields(node):
                value = getattr(node, f.name)
                required: frozenset[str] | None = getattr(value, "required_auxiliary", None)
                if required is None:
                    continue
                missing = required - node.auxiliary_data.keys()
                if missing:
                    raise MissingAuxiliaryDataError(
                        node_id=node.id,
                        field_name=f.name,
                        model_type=type(value).__name__,
                        missing_keys=frozenset(missing),
                    )

    def _validate_timeseries(self) -> list[str]:
        errors: list[str] = []
        for node_id, node in self._nodes.items():
            if isinstance(node, Source) and node.inflow is None:
                errors.append(f"Source '{node_id}': 'inflow' is required but not set")
            if isinstance(node, Demand) and node.requirement is None:
                errors.append(f"Demand '{node_id}': 'requirement' is required but not set")
        return errors

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

    def visualize(
        self,
        *,
        show_reaches: bool = True,
        show_reach_labels: bool = True,
        edge_colors: dict[str, str] | None = None,
        save_to: str | Path | None = None,
        figsize: tuple[float, float] = (12, 8),
        title: str | None = None,
    ) -> tuple:
        from taqsim.system._visualize import visualize_system

        return visualize_system(
            self._nodes,
            self._edges,
            show_reaches=show_reaches,
            show_reach_labels=show_reach_labels,
            edge_colors=edge_colors,
            save_to=save_to,
            figsize=figsize,
            title=title,
        )

    def _clone_with_updates(self, updates: dict[str, dict[str, float]]) -> "WaterSystem":
        """Create a new system with updated strategy parameters."""
        new_system = WaterSystem(frequency=self.frequency, start_date=self.start_date)

        # Clone nodes with updated strategies
        for node_id, node in self._nodes.items():
            overrides: dict[str, object] = {}
            for strategy_name, strategy in node.strategies().items():
                key = f"{node_id}.{strategy_name}"
                if key in updates:
                    overrides[strategy_name] = strategy.with_params(**updates[key])

            new_node = node._fresh_copy(**overrides)
            new_system._nodes[node_id] = new_node
            new_system._graph.add_node(node_id)

        # Clone edges
        for edge_id, edge in self._edges.items():
            new_system._edges[edge_id] = edge._fresh_copy()

        # Rebuild graph edges
        for edge in new_system._edges.values():
            new_system._graph.add_edge(edge.source, edge.target, edge_id=edge.id)

        new_system._set_targets()
        new_system._validated = True
        return new_system
