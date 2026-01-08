# Event System

## Overview

The `taqsim.node` module uses event sourcing. All events are frozen dataclasses with `slots=True` for immutability and memory efficiency.

## Common Fields

All events share:
- `t: int` - timestep when event occurred
- `amount: float` - water volume in m³

## Event Types

### WaterGenerated

Source nodes produce water.

```python
WaterGenerated(amount=100.0, t=0)
```

### WaterReceived

Node receives water from upstream.

```python
WaterReceived(amount=50.0, source_id="upstream_node", t=0)
```

### WaterStored

Water added to storage.

```python
WaterStored(amount=30.0, t=0)
```

### WaterReleased

Water released from storage.

```python
WaterReleased(amount=20.0, t=0)
```

### WaterLost

Physical losses (evaporation, seepage, overflow, or custom reasons).

```python
from taqsim.common import LossReason, EVAPORATION, SEEPAGE, OVERFLOW

# Using standard constants
WaterLost(amount=5.0, reason=EVAPORATION, t=0)
WaterLost(amount=2.0, reason=SEEPAGE, t=0)
WaterLost(amount=100.0, reason=OVERFLOW, t=0)

# Custom loss reasons
WaterLost(amount=10.0, reason=LossReason("infiltration"), t=0)
```

### WaterConsumed

Water removed from system by demand.

```python
WaterConsumed(amount=40.0, t=0)
```

### WaterDistributed

Water sent to downstream node.

```python
WaterDistributed(amount=60.0, target_id="downstream_node", t=0)
```

### DeficitRecorded

Unmet demand or minimum flow requirement.

```python
DeficitRecorded(required=100.0, actual=80.0, deficit=20.0, t=0)
```

## Deriving State from Events

Current storage:

```python
def current_storage(node: BaseNode, initial: float = 0.0) -> float:
    balance = initial
    for event in node.events:
        match event:
            case WaterStored(amount=a):
                balance += a
            case WaterReleased(amount=a) | WaterLost(amount=a):
                balance -= a
    return balance
```

Total generated:

```python
total = sum(e.amount for e in node.events_of_type(WaterGenerated))
```

## Event Ordering

Within a timestep, events are recorded in `update()` execution order:

1. WaterGenerated (sources)
2. WaterReceived
3. WaterStored / WaterReleased
4. WaterLost
5. WaterConsumed
6. WaterDistributed

## Querying Events

```python
# All events at timestep 5
events_t5 = node.events_at(5)

# All storage events
stored = node.events_of_type(WaterStored)

# Total losses
losses = sum(e.amount for e in node.events_of_type(WaterLost))

# Losses by reason (using constants from taqsim.common)
evap = sum(
    e.amount for e in node.events_of_type(WaterLost)
    if e.reason == EVAPORATION
)
```

## NodeEvent Union Type

All event types are part of the `NodeEvent` union:

```python
NodeEvent = (
    WaterGenerated | WaterReceived | WaterStored | WaterReleased |
    WaterLost | WaterConsumed | WaterDistributed | DeficitRecorded
)
```
