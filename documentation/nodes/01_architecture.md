# Node Architecture

## Overview

The `taqsim.node` module provides the building blocks for water system simulation using event sourcing. Nodes record events instead of mutating state directly.

## Design Principles

### Event Sourcing

Nodes record events instead of mutating state. State is derived from the event history.

Benefits:

- Audit trail: See exactly what happened
- Replay: Reconstruct any historical state
- Debugging: Trace issues through event sequence
- Testability: Assert on events, not internal state

```python
# Instead of:
node.storage = node.storage + inflow - outflow  # mutation

# We record:
node.record(WaterStored(amount=inflow, t=0))
node.record(WaterReleased(amount=outflow, t=0))

# State is derived:
current_storage = sum(stored) - sum(released) - sum(lost)
```

### Capability Composition

Nodes are defined by their capabilities:

| Capability | Method | Description |
|------------|--------|-------------|
| Generates | `generate(t, dt)` | Produce water |
| Receives | `receive(amount, source_id, t)` | Accept water from upstream |
| Stores | `store(amount, t, dt)` | Buffer water temporally |
| Loses | `lose(t, dt)` | Physical losses |
| Consumes | `consume(amount, t, dt)` | Remove water from system |
| Gives | `distribute(amount, t)` | Send water downstream |

Node types are combinations:

```
Source    = Generates + Gives
Splitter  = Receives + Gives
Reservoir = Receives + Stores + Loses + Gives
Demand    = Receives + Consumes + Gives
Sink      = Receives
```

## Node Lifecycle

```
create(id) → [update(t, dt) → record events]* → derive state from events
```

## BaseNode

All nodes extend `BaseNode` which provides:

```python
@dataclass
class BaseNode:
    id: str
    events: list[NodeEvent]  # append-only event log

    def record(event) → None      # append event
    def events_at(t) → list       # filter by timestep
    def events_of_type(T) → list  # filter by event class
    def clear_events() → None     # reset (for testing)
    def update(t, dt) → None      # MUST be implemented by subclasses
```

## Universal update() Pattern

All nodes implement `update(t, dt)` which:

1. Generate water (if source)
2. Receive from upstream edges
3. Store/release (if has storage)
4. Consume (if demand)
5. Calculate losses
6. Distribute downstream

Each step records appropriate events.

## Data Flow

```
Source ──[edge]──> Reservoir ──[edge]──> Demand ──[edge]──> Sink
   │                   │                    │                 │
   ▼                   ▼                    ▼                 ▼
Generated          Received             Received          Received
Distributed        Stored               Consumed          (terminal)
                   Lost
                   Released
                   Distributed
```
