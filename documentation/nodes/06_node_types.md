# Node Types

## Overview

Five node types compose the water system. Each combines specific capabilities.

| Node | Generates | Receives | Stores | Loses | Consumes | Gives |
|------|-----------|----------|--------|-------|----------|-------|
| Source | X | | | | | X |
| Sink | | X | | | | |
| Splitter | | X | | | | X |
| Demand | | X | | | X | X |
| Storage | | X | X | X | | X |

---

## Source

Water entry point. Generates inflow and distributes downstream.

### Capabilities

- **Generates**: Produces water from `inflow` TimeSeries
- **Gives**: Distributes to targets via `split_strategy`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `inflow` | `TimeSeries` | Yes | Inflow rates per timestep |
| `targets` | `list[str]` | No | Downstream node IDs |
| `split_strategy` | `SplitStrategy` | Yes | Distribution strategy |

### Events Recorded

- `WaterGenerated(amount, t)` - when water is generated
- `WaterDistributed(amount, target_id, t)` - for each target

### Update Cycle

1. Generate water: `amount = inflow[t] * dt`
2. Distribute to targets

### Example

```python
from taqsim.node import Source, TimeSeries

source = Source(
    id="river_intake",
    inflow=TimeSeries(values=[100.0, 150.0, 120.0]),
    targets=["reservoir"],
    split_strategy=equal_split
)
```

---

## Sink

Terminal node. Receives water and exits the system.

### Capabilities

- **Receives**: Accepts water from upstream

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |

### Events Recorded

- `WaterReceived(amount, source_id, t)` - when water arrives

### Update Cycle

No-op. Sink is passive; it only records received water.

### Example

```python
from taqsim.node import Sink

sink = Sink(id="ocean_outflow")
```

---

## Splitter

Distribution node. Receives water and splits among targets.

### Capabilities

- **Receives**: Accepts water from upstream
- **Gives**: Distributes to targets via `split_strategy`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `targets` | `list[str]` | No | Downstream node IDs |
| `split_strategy` | `SplitStrategy` | Yes | Distribution strategy |

### Events Recorded

- `WaterReceived(amount, source_id, t)` - when water arrives
- `WaterDistributed(amount, target_id, t)` - for each target

### Update Cycle

1. Distribute all received water to targets
2. Reset received counter

### Example

```python
from taqsim.node import Splitter

splitter = Splitter(
    id="canal_junction",
    targets=["irrigation_north", "irrigation_south"],
    split_strategy=proportional_split
)
```

---

## Demand

Consumption node. Receives water, consumes requirement, passes remainder.

### Capabilities

- **Receives**: Accepts water from upstream
- **Consumes**: Removes water to meet requirement
- **Gives**: Distributes remaining to targets

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `requirement` | `TimeSeries` | Yes | Demand rates per timestep |
| `targets` | `list[str]` | No | Downstream node IDs |
| `split_strategy` | `SplitStrategy` | Yes | Distribution strategy |

### Events Recorded

- `WaterReceived(amount, source_id, t)` - when water arrives
- `WaterConsumed(amount, t)` - amount consumed
- `DeficitRecorded(required, actual, deficit, t)` - if demand not met
- `WaterDistributed(amount, target_id, t)` - for remaining water

### Deficit Tracking

When received water is less than required, a `DeficitRecorded` event captures:
- `required`: demand for the timestep
- `actual`: amount consumed
- `deficit`: unmet demand

### Update Cycle

1. Consume up to requirement: `consumed = min(received, requirement[t] * dt)`
2. Record deficit if `consumed < required`
3. Distribute remaining water to targets
4. Reset received counter

### Example

```python
from taqsim.node import Demand, TimeSeries

demand = Demand(
    id="city_water",
    requirement=TimeSeries(values=[50.0, 60.0, 55.0]),
    targets=["wastewater_treatment"],
    split_strategy=equal_split
)
```

---

## Storage

Reservoir node. Receives, stores, loses, releases, and distributes water.

### Capabilities

- **Receives**: Accepts water from upstream
- **Stores**: Buffers water up to capacity
- **Loses**: Physical losses (evaporation, seepage)
- **Gives**: Distributes released + spilled water

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `capacity` | `float` | Yes | Maximum storage volume |
| `initial_storage` | `float` | No | Starting volume (default: 0) |
| `release_rule` | `ReleaseRule` | Yes | Release calculation |
| `loss_rule` | `LossRule` | Yes | Loss calculation |
| `split_strategy` | `SplitStrategy` | Yes | Distribution strategy |
| `targets` | `list[str]` | No | Downstream node IDs |

### Validation

- `capacity` must be positive
- `initial_storage` must be >= 0 and <= capacity

### Events Recorded

- `WaterReceived(amount, source_id, t)` - when water arrives
- `WaterStored(amount, t)` - amount added to storage
- `WaterSpilled(amount, t)` - overflow when capacity exceeded
- `WaterLost(amount, reason, t)` - per loss type (evaporation, seepage)
- `WaterReleased(amount, t)` - controlled release
- `WaterDistributed(amount, target_id, t)` - for each target

### Spillway Handling

When inflow exceeds available capacity:
1. Store up to capacity
2. Record `WaterSpilled` for overflow
3. Spilled water joins released water for downstream distribution

### Loss Handling

Losses are scaled if they exceed current storage:
```python
if total_loss > current_storage:
    scale = current_storage / total_loss
    losses = {reason: amount * scale for reason, amount in losses.items()}
```

### Storage Property

Access current storage level:
```python
reservoir.storage  # returns current volume
```

### Update Cycle

1. Store inflow (record spill if overflow)
2. Calculate and apply losses
3. Calculate and apply release
4. Distribute (released + spilled) to targets
5. Reset received counter

### Example

```python
from taqsim.node import Storage

reservoir = Storage(
    id="main_dam",
    capacity=10000.0,
    initial_storage=5000.0,
    release_rule=fixed_release,
    loss_rule=evaporation_loss,
    split_strategy=equal_split,
    targets=["downstream_river"]
)

# Check current storage
print(reservoir.storage)  # 5000.0
```
