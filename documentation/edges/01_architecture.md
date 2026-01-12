# Edge Architecture

## Overview

The `taqsim.edge` module represents connections between nodes, managing water transport with capacity constraints and losses. Like nodes, edges use event sourcing.

## Design Principles

### Event Sourcing

Edges record events instead of mutating state. State is derived from the event history.

Benefits:

- Audit trail: See exactly what flowed through
- Replay: Reconstruct any historical state
- Debugging: Trace flow issues through event sequence
- Testability: Assert on events, not internal state

```python
# Instead of:
edge.delivered = edge.received - edge.losses  # mutation

# We record:
edge.record(WaterReceived(amount=100.0, t=0))
edge.record(WaterLost(amount=5.0, reason=LossReason.SEEPAGE, t=0))
edge.record(WaterDelivered(amount=95.0, t=0))

# State is derived:
total_delivered = sum(e.amount for e in edge.events_of_type(WaterDelivered))
```

### Strategy Pattern

Configurable behaviors via protocols:

- `EdgeLossRule` - customizable loss calculation for transport losses

## Edge vs Node

| Aspect | Node | Edge |
|--------|------|------|
| Purpose | Process water | Transport water |
| Events | Water* prefix | Water* prefix |
| Losses | Storage-based | Flow-based |
| State | Complex (storage, generation) | Simple (received, delivered) |

## Edge Lifecycle

```
create(id, source, target) -> [receive(amount, t) -> update(t, dt)]* -> derive state from events
```

## Data Flow

```
receive(amount) -> capacity check -> loss calculation -> deliver
      |                  |                 |               |
      v                  v                 v               v
WaterReceived    WaterLost            WaterLost      WaterDelivered
                 (CAPACITY_EXCEEDED)  (SEEPAGE, etc)
```

## Edge Interface

Key methods:

- `receive(amount, t)` - Accept flow from upstream node
- `update(t, dt)` - Process flow and record events
- `reset()` - Clears events and step accumulator

## Update Cycle

1. **Receive**: Edge accumulates water via `receive()` calls
2. **Capacity Check**: Excess flow is recorded as `WaterLost(reason=CAPACITY_EXCEEDED)` if over capacity
3. **Loss Calculation**: Apply `EdgeLossRule` to calculate transport losses
4. **Delivery**: Record final delivered amount
5. **Reset**: Clear accumulated flow for next timestep
