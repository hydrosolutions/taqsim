# JSON Serialization

## Overview

`WaterSystem.from_json()` loads network **topology** (nodes, edges, scalar config) from JSON. Strategies and TimeSeries are injected programmatically after loading.

## JSON Schema

```json
{
  "frequency": "monthly",
  "start_date": "2024-01-15",
  "nodes": [
    {"type": "source", "id": "river"},
    {"type": "storage", "id": "dam", "capacity": 500.0, "initial_storage": 200.0},
    {"type": "demand", "id": "city", "consumption_fraction": 0.8},
    {"type": "splitter", "id": "junction"},
    {"type": "reach", "id": "canal", "capacity": 2000.0},
    {"type": "passthrough", "id": "gauge", "capacity": 1000.0},
    {"type": "sink", "id": "ocean"}
  ],
  "edges": [
    {"id": "e1", "source": "river", "target": "dam"},
    {"id": "e2", "source": "dam", "target": "ocean"}
  ]
}
```

### Top-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `frequency` | Yes | `"daily"`, `"weekly"`, `"monthly"`, or `"yearly"` (case-insensitive) |
| `start_date` | No | ISO date string, e.g. `"2024-01-15"` |
| `nodes` | No | Array of node objects |
| `edges` | No | Array of edge objects |

### Common Node Fields

All nodes support these optional fields alongside the required `type` and `id`:

| Field | Type | Default |
|-------|------|---------|
| `location` | `[lat, lon]` | `null` |
| `tags` | `string[]` | `[]` |
| `metadata` | `object` | `{}` |
| `auxiliary_data` | `object` | `{}` |

### Node Types

| Type | Required Fields | Optional Fields | Placeholder Strategies |
|------|----------------|-----------------|----------------------|
| `source` | `id` | — | `inflow=None` |
| `storage` | `id`, `capacity` | `initial_storage`, `dead_storage` | `release_policy=NoRelease()`, `loss_rule=NoLoss()` |
| `demand` | `id` | `consumption_fraction`, `efficiency` | `requirement=None` |
| `splitter` | `id` | — | `split_policy=NoSplit()` |
| `reach` | `id` | `capacity` | `routing_model=NoRouting()`, `loss_rule=NoReachLoss()` |
| `passthrough` | `id` | `capacity` | — |
| `sink` | `id` | — | — |

### Edge Fields

| Field | Required | Default |
|-------|----------|---------|
| `id` | Yes | — |
| `source` | Yes | — |
| `target` | Yes | — |
| `tags` | No | `[]` |
| `metadata` | No | `{}` |

## Usage

```python
from taqsim import WaterSystem, TimeSeries

# Load topology
system = WaterSystem.from_json("network.json")

# Inject strategies and time series
system.nodes["river"].inflow = TimeSeries([100.0] * 12)
system.nodes["dam"].release_policy = MyRelease(rate=50)
system.nodes["dam"].loss_rule = EvaporationLoss()
system.nodes["city"].requirement = TimeSeries([80.0] * 12)
system.nodes["junction"].split_policy = ProportionalSplit(ratios={"canal": 0.6, "city": 0.4})

# Simulate
system.simulate(timesteps=12)
```

### Input Sources

`from_json()` accepts three input types:

```python
# File path as string
system = WaterSystem.from_json("path/to/network.json")

# Path object
from pathlib import Path
system = WaterSystem.from_json(Path("network.json"))

# JSON string (auto-detected by leading '{')
system = WaterSystem.from_json('{"frequency": "monthly", ...}')
```

## Placeholder Behavior

Nodes loaded from JSON use **placeholder strategies** that raise `RuntimeError` if called without replacement:

- `NoRelease` on Storage nodes
- `NoSplit` on Splitter nodes
- `NoRouting` on Reach nodes (pass-through, not a real placeholder — just passes inflow straight through)
- `NoReachLoss` on Reach nodes (no loss — safe default)
- `NoLoss` on Storage nodes (no loss — safe default)

Source `inflow` and Demand `requirement` are set to `None`. Validation catches these before simulation.

## Validation

`from_json()` does **not** call `validate()`. Validation runs automatically when `simulate()` is called, or can be triggered manually:

```python
system = WaterSystem.from_json("network.json")
# ... inject strategies ...
system.validate()  # Optional — simulate() calls this automatically
system.simulate(12)
```

### TimeSeries Validation

If `Source.inflow` or `Demand.requirement` is still `None` at validation time:

```
ValidationError: Source 'river': 'inflow' is required but not set
```

## Error Reference

| Error | Cause |
|-------|-------|
| `ValueError: JSON must include a 'frequency' field` | Missing `frequency` in JSON |
| `ValueError: Invalid frequency '...'` | Unrecognized frequency value |
| `ValueError: Node is missing required field 'type'` | Node without `type` |
| `ValueError: Node is missing required field 'id'` | Node without `id` |
| `ValueError: Unknown node type '...'` | Unrecognized node type |
| `ValueError: Storage '...': 'capacity' is required` | Storage without `capacity` |
| `ValueError: Node '...' already exists` | Duplicate node IDs |
| `ValueError: Edge '...' already exists` | Duplicate edge IDs |
| `ValueError: Invalid start_date '...'` | Unparseable date string |
| `FileNotFoundError` | File path doesn't exist |
| `json.JSONDecodeError` | Invalid JSON syntax |

---

## Exporting to JSON -- `to_json()`

### Signature

```python
def to_json(self, save_to: str | Path | None = None) -> dict[str, Any]:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_to` | `str \| Path \| None` | `None` | File path to write JSON. If `None`, only returns the dict. |

### What's Serialized

| Data | Serialized | Notes |
|------|-----------|-------|
| Frequency | Yes | Lowercase string (e.g., `"monthly"`) |
| Start date | Yes | ISO format, omitted when `None` |
| Node IDs and types | Yes | |
| Node location | Yes | Tuple -> list conversion |
| Node tags | Yes | Frozenset -> sorted list |
| Node metadata | Yes | |
| Node auxiliary_data | Yes | |
| Storage: capacity, initial_storage, dead_storage | Yes | Always included |
| Demand: consumption_fraction, efficiency | Yes | Always included |
| Reach/PassThrough: capacity | Yes | Only when not `None` |
| Edge ID, source, target | Yes | |
| Edge tags, metadata | Yes | Only when non-empty |
| Strategies (release_policy, split_policy, etc.) | **No** | Must be re-injected after loading |
| TimeSeries (inflow, requirement) | **No** | Must be re-injected after loading |
| Runtime state (events, accumulators) | **No** | |

### Output Example

```python
system.to_json()
# {
#     "frequency": "monthly",
#     "start_date": "2024-01-15",
#     "nodes": [
#         {"type": "source", "id": "river", "location": [31.5, 35.2]},
#         {"type": "storage", "id": "dam", "capacity": 500.0, "initial_storage": 200.0, "dead_storage": 0.0},
#         {"type": "sink", "id": "ocean"}
#     ],
#     "edges": [
#         {"id": "e1", "source": "river", "target": "dam"},
#         {"id": "e2", "source": "dam", "target": "ocean"}
#     ]
# }
```

### Saving to File

```python
# Save to file (also returns the dict)
result = system.to_json(save_to="network.json")

# Accepts Path objects
from pathlib import Path
result = system.to_json(save_to=Path("output") / "network.json")
```

---

## Round-Trip Guarantee

`to_json()` and `from_json()` form a round-trip for topology data:

```python
# Original system
system = WaterSystem.from_json("network.json")

# Round-trip
exported = system.to_json()
restored = WaterSystem.from_json(json.dumps(exported))

# Topology is preserved
assert set(restored.nodes.keys()) == set(system.nodes.keys())
assert set(restored.edges.keys()) == set(system.edges.keys())
```

### What's Preserved

- Frequency and start_date
- All node IDs, types, and topology fields (location, tags, metadata, auxiliary_data)
- Type-specific fields (capacity, initial_storage, etc.)
- All edge IDs, source/target, tags, metadata

### What Changes After Round-Trip

Strategies reset to placeholders:

| Node Type | Reset Strategy |
|-----------|---------------|
| Source | `inflow=None` |
| Storage | `release_policy=NoRelease()`, `loss_rule=NoLoss()` |
| Demand | `requirement=None` |
| Splitter | `split_policy=NoSplit()` |
| Reach | `routing_model=NoRouting()`, `loss_rule=NoReachLoss()` |

These must be re-injected before simulation, just as with `from_json()`.
