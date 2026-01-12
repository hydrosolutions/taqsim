# WaterSystem Architecture

## Overview

`WaterSystem` is the orchestrator that:

1. Manages nodes and edges
2. Derives topology from edge definitions
3. Validates network structure
4. Runs simulation in topological order
5. Routes water between nodes via edges

## Design Principles

### Separation of Concerns

| Component | Responsibility |
|-----------|----------------|
| **Nodes** | Process water, record events (know nothing about topology) |
| **Edges** | Transport water, apply losses (know source/target IDs) |
| **WaterSystem** | Derive topology, validate, orchestrate simulation |

### Nodes Don't Know Targets

Nodes never specify their downstream connections. Instead:

1. You define edges with `source` and `target` node IDs
2. `WaterSystem.validate()` builds the graph and populates node targets
3. Nodes output via events (`WaterOutput`, `WaterDistributed`)
4. `WaterSystem` reads events and routes water through edges

### Events as Communication API

Nodes communicate output through events:

- **Single-output nodes** (Source, Storage, Demand, PassThrough): Record `WaterOutput`
- **Multi-output nodes** (Splitter): Record `WaterDistributed` for each target

`WaterSystem._route_output()` reads these events and delivers water to downstream nodes.

## Class Definition

```python
from dataclasses import dataclass, field
import networkx as nx

@dataclass
class WaterSystem:
    dt: float = 2629800.0  # ~1 month in seconds

    _nodes: dict[str, BaseNode]
    _edges: dict[str, Edge]
    _graph: nx.DiGraph
    _validated: bool
```

## Key Methods

### add_node(node)

Registers a node in the system.

```python
system.add_node(Source(id="river", inflow=inflow_data))
```

### add_edge(edge)

Registers an edge connecting two nodes.

```python
system.add_edge(Edge(id="e1", source="river", target="dam", capacity=500, loss_rule=no_loss))
```

### validate()

Validates network structure and derives topology:

1. Check all edge endpoints exist
2. Build networkx graph
3. Check acyclic (DAG)
4. Check connectivity
5. Validate terminal structure (Source in_degree=0, Sink out_degree=0)
6. Check single output for non-Splitter nodes
7. Check path to sink for all nodes
8. Populate targets on all nodes via `_set_targets()`

### simulate(timesteps)

Runs the simulation:

1. Auto-validates if not validated
2. For each timestep:
   - Process nodes in topological order
   - Route output from each node to downstream via edges

## Usage Example

```python
from taqsim.node import Source, Storage, PassThrough, Splitter, Demand, Sink, TimeSeries
from taqsim.edge import Edge
from taqsim.system import WaterSystem

# Create nodes (no targets specified)
source = Source(id="river", inflow=TimeSeries([100.0] * 12))
dam = Storage(id="dam", capacity=1000, release_rule=rule, loss_rule=losses)
turbine = PassThrough(id="turbine")
junction = Splitter(id="junction", split_strategy=equal_split)
farm = Demand(id="farm", requirement=TimeSeries([30.0] * 12))
city = Sink(id="city")

# Create system
system = WaterSystem(dt=86400)  # 1 day

# Add nodes
system.add_node(source)
system.add_node(dam)
system.add_node(turbine)
system.add_node(junction)
system.add_node(farm)
system.add_node(city)

# Add edges (topology defined here)
system.add_edge(Edge(id="e1", source="river", target="dam", capacity=500, loss_rule=no_loss))
system.add_edge(Edge(id="e2", source="dam", target="turbine", capacity=500, loss_rule=no_loss))
system.add_edge(Edge(id="e3", source="turbine", target="junction", capacity=500, loss_rule=no_loss))
system.add_edge(Edge(id="e4", source="junction", target="farm", capacity=200, loss_rule=seepage))
system.add_edge(Edge(id="e5", source="junction", target="city", capacity=100, loss_rule=no_loss))

# Validate (derives targets from edges)
system.validate()

# Simulate 12 timesteps
system.simulate(timesteps=12)

# Analyze results via node events
total_received = sum(e.amount for e in city.events_of_type(WaterReceived))
```

## Graph Structure

The internal graph is a `networkx.DiGraph`:

- Nodes are node IDs (strings)
- Edges store `edge_id` attribute linking to the Edge object

```python
# After validation, inspect topology
system._graph.nodes()  # ['river', 'dam', 'turbine', 'junction', 'farm', 'city']
system._graph.edges()  # [('river', 'dam'), ('dam', 'turbine'), ...]
```

## Simulation Flow

```
For each timestep t:
    For each node_id in topological_sort(graph):
        node.update(t, dt)           # Node processes water, records events
        _route_output(node_id, t)    # System reads events, routes via edges
```

### Routing Logic

1. Read `WaterOutput` events → route to single outgoing edge
2. Read `WaterDistributed` events → route to specified target edge
3. For each routed amount:
   - `edge.receive(amount, t)`
   - `delivered = edge.update(t, dt)` (applies losses)
   - `target_node.receive(delivered, edge_id, t)`

## Parameter Exposure

WaterSystem provides methods for parameter discovery and manipulation, enabling optimization workflows.

### Methods

| Method | Purpose |
|--------|---------|
| `param_schema()` | Returns list of all tunable parameters as `ParamSpec` objects |
| `to_vector()` | Flattens parameters to `list[float]` for optimization |
| `with_vector(vector)` | Creates new system with parameters from vector (immutable) |
| `reset()` | Clears all events and resets node state for fresh simulation |

### Example

```python
# Discover parameters
schema = system.param_schema()
for spec in schema:
    print(f"{spec.path}: {spec.value}")

# Vectorize and modify
vector = system.to_vector()
new_system = system.with_vector(modified_vector)
new_system.simulate(12)
```

See [Parameter Exposure](03_parameter_exposure.md) for complete documentation.