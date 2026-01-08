# Edge Class

## Overview

The `Edge` class represents a connection between two nodes, handling water transport with capacity constraints, losses, and minimum flow requirements.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | `str` | Yes | - | Unique identifier |
| `source` | `str` | Yes | - | Source node ID |
| `target` | `str` | Yes | - | Target node ID |
| `capacity` | `float` | Yes | - | Maximum flow capacity |
| `loss_rule` | `EdgeLossRule` | Yes | - | Loss calculation strategy |
| `requirement` | `TimeSeries \| None` | No | `None` | Minimum flow requirement per timestep |
| `targets` | `list[str]` | No | `[]` | Additional target nodes |

## Validation Rules

The Edge validates parameters on creation:

- `id` cannot be empty
- `source` cannot be empty
- `target` cannot be empty
- `capacity` must be positive
- `loss_rule` is required

```python
# These raise ValueError:
Edge(id="", source="a", target="b", capacity=100, loss_rule=rule)  # empty id
Edge(id="e", source="", target="b", capacity=100, loss_rule=rule)  # empty source
Edge(id="e", source="a", target="b", capacity=0, loss_rule=rule)   # zero capacity
Edge(id="e", source="a", target="b", capacity=100, loss_rule=None) # no loss rule
```

## Methods

### receive(amount, t) -> float

Accept water from the source node.

```python
received = edge.receive(100.0, t=0)
# Records FlowReceived event
# Returns amount received
```

### update(t, dt) -> float

Process accumulated flow and deliver to target.

```python
delivered = edge.update(t=0, dt=1.0)
# Returns amount delivered after losses
```

### record(event) -> None

Append an event to the event log.

```python
edge.record(FlowReceived(amount=50.0, t=0))
```

### events_of_type(event_type) -> list[T]

Filter events by type.

```python
deliveries = edge.events_of_type(FlowDelivered)
losses = edge.events_of_type(FlowLost)
```

### clear_events() -> None

Reset the event log. Primarily for testing.

```python
edge.clear_events()
assert len(edge.events) == 0
```

## Update Cycle

The `update()` method performs these steps:

1. **Capacity Check**: If received > capacity, record `CapacityExceeded` and cap flow
2. **Loss Calculation**: Apply `loss_rule.calculate()` to determine losses
3. **Loss Scaling**: If total losses > available flow, scale proportionally
4. **Record Losses**: Create `FlowLost` event for each loss type
5. **Calculate Delivered**: delivered = received - total_losses
6. **Requirement Check**: If requirement set and delivered < required, record `RequirementUnmet`
7. **Record Delivery**: Create `FlowDelivered` event
8. **Reset**: Clear accumulated flow for next timestep
9. **Return**: Return delivered amount

## Usage Examples

### Basic Edge

```python
from taqsim.edge import Edge, EdgeLossRule
from taqsim.common import LossReason
from dataclasses import dataclass

@dataclass
class ZeroLoss:
    def calculate(self, flow: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
        return {}

edge = Edge(
    id="canal_1",
    source="reservoir",
    target="farm",
    capacity=100.0,
    loss_rule=ZeroLoss()
)

# Receive water
edge.receive(80.0, t=0)

# Process and deliver
delivered = edge.update(t=0, dt=1.0)
assert delivered == 80.0
```

### Edge with Losses

```python
@dataclass
class SeepageLoss:
    fraction: float

    def calculate(self, flow: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
        return {LossReason.SEEPAGE: flow * self.fraction}

edge = Edge(
    id="leaky_canal",
    source="reservoir",
    target="farm",
    capacity=100.0,
    loss_rule=SeepageLoss(fraction=0.1)  # 10% loss
)

edge.receive(100.0, t=0)
delivered = edge.update(t=0, dt=1.0)
assert delivered == 90.0  # 100 - 10% loss
```

### Edge with Minimum Requirement

```python
from taqsim.node import TimeSeries

edge = Edge(
    id="environmental_flow",
    source="dam",
    target="river",
    capacity=1000.0,
    loss_rule=ZeroLoss(),
    requirement=TimeSeries(values=[50.0, 50.0, 100.0])  # minimum flow per timestep
)

edge.receive(30.0, t=0)  # below requirement
delivered = edge.update(t=0, dt=1.0)

# Check for unmet requirement
unmet = edge.events_of_type(RequirementUnmet)
assert len(unmet) == 1
assert unmet[0].deficit == 20.0  # 50 required - 30 delivered
```

### Capacity Exceeded

```python
edge = Edge(
    id="small_pipe",
    source="tank",
    target="outlet",
    capacity=50.0,
    loss_rule=ZeroLoss()
)

edge.receive(100.0, t=0)  # exceeds capacity
delivered = edge.update(t=0, dt=1.0)

assert delivered == 50.0  # capped at capacity

exceeded = edge.events_of_type(CapacityExceeded)
assert len(exceeded) == 1
assert exceeded[0].excess == 50.0  # 100 - 50 capacity
```

## Properties

### events

The append-only event log.

```python
edge.events  # list[EdgeEvent]
```

## Internal State

The edge tracks `_received_this_step` internally to accumulate flow between `receive()` calls before `update()`. This is reset to 0 after each `update()`.
