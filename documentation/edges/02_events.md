# Event System

## Overview

The `taqsim.edge` module uses event sourcing. All events are frozen dataclasses with `slots=True` for immutability and memory efficiency.

## Common Fields

All events share:
- `t: int` - timestep when event occurred
- `amount: float` - water volume in m3

## Event Types

### WaterReceived

Edge receives water from source node.

```python
WaterReceived(amount=100.0, t=0)
```

### WaterLost

Transport losses (seepage, evaporation, capacity exceeded).

```python
from taqsim.common import LossReason

WaterLost(amount=5.0, reason=LossReason.SEEPAGE, t=0)
WaterLost(amount=2.0, reason=LossReason.EVAPORATION, t=0)
WaterLost(amount=50.0, reason=LossReason.CAPACITY_EXCEEDED, t=0)
```

### WaterDelivered

Water successfully transported to target node.

```python
WaterDelivered(amount=93.0, t=0)
```

## Deriving State from Events

Total delivered:

```python
total = sum(e.amount for e in edge.events_of_type(WaterDelivered))
```

Total losses:

```python
losses = sum(e.amount for e in edge.events_of_type(WaterLost))
```

Losses by reason:

```python
from taqsim.common import LossReason

seepage = sum(
    e.amount for e in edge.events_of_type(WaterLost)
    if e.reason == LossReason.SEEPAGE
)
```

Capacity exceeded losses:

```python
capacity_losses = sum(
    e.amount for e in edge.events_of_type(WaterLost)
    if e.reason == LossReason.CAPACITY_EXCEEDED
)
```

## Event Ordering

Within a timestep, events are recorded in `update()` execution order:

1. WaterReceived (during receive calls)
2. WaterLost (capacity exceeded, transport losses)
3. WaterDelivered

## Querying Events

```python
# All delivery events
deliveries = edge.events_of_type(WaterDelivered)

# Total losses
losses = sum(e.amount for e in edge.events_of_type(WaterLost))

# Check for capacity exceeded
capacity_exceeded = [
    e for e in edge.events_of_type(WaterLost)
    if e.reason == LossReason.CAPACITY_EXCEEDED
]
if capacity_exceeded:
    total_excess = sum(e.amount for e in capacity_exceeded)
    print(f"Capacity exceeded {len(capacity_exceeded)} times, total: {total_excess}")

# Filter losses by reason
seepage_losses = [
    e for e in edge.events_of_type(WaterLost)
    if e.reason == LossReason.SEEPAGE
]
```

## EdgeEvent Union Type

All event types are part of the `EdgeEvent` union:

```python
EdgeEvent = WaterReceived | WaterLost | WaterDelivered
```
