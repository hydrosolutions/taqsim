# Event System

## Overview

The `taqsim.edge` module uses event sourcing. All events are frozen dataclasses with `slots=True` for immutability and memory efficiency.

## Common Fields

All events share:
- `t: int` - timestep when event occurred
- `amount: float` - water volume in m3 (except CapacityExceeded)

## Event Types

### FlowReceived

Edge receives water from source node.

```python
FlowReceived(amount=100.0, t=0)
```

### FlowLost

Transport losses (seepage, evaporation).

```python
from taqsim.common import LossReason

FlowLost(amount=5.0, reason=LossReason.SEEPAGE, t=0)
FlowLost(amount=2.0, reason=LossReason.EVAPORATION, t=0)
```

### FlowDelivered

Water successfully transported to target node.

```python
FlowDelivered(amount=93.0, t=0)
```

### CapacityExceeded

Flow exceeds edge capacity.

```python
CapacityExceeded(excess=20.0, t=0)
```

Fields:
- `excess: float` - amount over capacity (not `amount`)
- `t: int` - timestep

### RequirementUnmet

Flow below minimum requirement.

```python
RequirementUnmet(required=100.0, actual=80.0, deficit=20.0, t=0)
```

Fields:
- `required: float` - minimum flow requirement
- `actual: float` - actual delivered flow
- `deficit: float` - shortfall (required - actual)
- `t: int` - timestep

## Deriving State from Events

Total delivered:

```python
total = sum(e.amount for e in edge.events_of_type(FlowDelivered))
```

Total losses:

```python
losses = sum(e.amount for e in edge.events_of_type(FlowLost))
```

Losses by reason:

```python
from taqsim.common import LossReason

seepage = sum(
    e.amount for e in edge.events_of_type(FlowLost)
    if e.reason == LossReason.SEEPAGE
)
```

Total excess (over capacity):

```python
excess = sum(e.excess for e in edge.events_of_type(CapacityExceeded))
```

## Event Ordering

Within a timestep, events are recorded in `update()` execution order:

1. FlowReceived (during receive calls)
2. CapacityExceeded (if over capacity)
3. FlowLost (transport losses)
4. RequirementUnmet (if below minimum)
5. FlowDelivered

## Querying Events

```python
# All delivery events
deliveries = edge.events_of_type(FlowDelivered)

# Total losses
losses = sum(e.amount for e in edge.events_of_type(FlowLost))

# Check for capacity issues
exceeded = edge.events_of_type(CapacityExceeded)
if exceeded:
    print(f"Capacity exceeded {len(exceeded)} times")

# Check for requirement violations
unmet = edge.events_of_type(RequirementUnmet)
total_deficit = sum(e.deficit for e in unmet)
```

## EdgeEvent Union Type

All event types are part of the `EdgeEvent` union:

```python
EdgeEvent = FlowReceived | FlowLost | FlowDelivered | CapacityExceeded | RequirementUnmet
```
