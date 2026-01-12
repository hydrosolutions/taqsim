# Node Types

## Overview

Six node types compose the water system. Each combines specific capabilities. Nodes process water and record events but **do not know their targets** — topology is derived from edges by `WaterSystem`.

| Node | Generates | Receives | Stores | Loses | Consumes | Output Event |
|------|-----------|----------|--------|-------|----------|--------------|
| Source | ✓ | | | | | WaterOutput |
| PassThrough | | ✓ | | | | WaterOutput |
| Splitter | | ✓ | | | | WaterDistributed |
| Demand | | ✓ | | | ✓ | WaterOutput |
| Storage | | ✓ | ✓ | ✓ | | WaterOutput |
| Sink | | ✓ | | | | (terminal) |

---

## Source

Water entry point. Generates inflow from a TimeSeries.

### Capabilities

- **Generates**: Produces water from `inflow` TimeSeries

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `inflow` | `TimeSeries` | Yes | Inflow rates per timestep |

### Events Recorded

- `WaterGenerated(amount, t)` — when water is generated
- `WaterOutput(amount, t)` — water available for downstream

### Update Cycle

1. Generate water: `amount = inflow[t] * dt`
2. Record `WaterOutput` for routing

### Example

```python
from taqsim.node import Source, TimeSeries

source = Source(
    id="river_intake",
    inflow=TimeSeries(values=[100.0, 150.0, 120.0])
)
```

---

## PassThrough

Transparent node for turbines, measurement points, or junctions. Passes 100% of received water downstream.

### Capabilities

- **Receives**: Accepts water from upstream

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |

### Events Recorded

- `WaterReceived(amount, source_id, t)` — when water arrives
- `WaterPassedThrough(amount, t)` — for analysis (e.g., turbine power)
- `WaterOutput(amount, t)` — water available for downstream

### Update Cycle

1. Record `WaterPassedThrough` for the accumulated amount
2. Record `WaterOutput` for routing
3. Reset counter

### Example

```python
from taqsim.node import PassThrough

turbine = PassThrough(id="hydropower_turbine")
```

---

## Splitter

Distribution node. Receives water and splits among multiple targets using a split strategy.

### Capabilities

- **Receives**: Accepts water from upstream

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `split_strategy` | `SplitStrategy` | Yes | Distribution strategy |

> **Note**: Targets are derived from edges by `WaterSystem` and populated via `_set_targets()` during validation.

### Events Recorded

- `WaterReceived(amount, source_id, t)` — when water arrives
- `WaterDistributed(amount, target_id, t)` — for each target edge

### Update Cycle

1. Apply split strategy to distribute water
2. Record `WaterDistributed` for each target
3. Reset counter

### Example

```python
from taqsim.node import Splitter

splitter = Splitter(
    id="canal_junction",
    split_strategy=proportional_split
)
```

---

## Demand

Consumption node. Receives water, consumes requirement, passes remainder downstream.

### Capabilities

- **Receives**: Accepts water from upstream
- **Consumes**: Removes water to meet requirement

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `requirement` | `TimeSeries` | Yes | Demand rates per timestep |

### Events Recorded

- `WaterReceived(amount, source_id, t)` — when water arrives
- `WaterConsumed(amount, t)` — amount consumed
- `DeficitRecorded(required, actual, deficit, t)` — if demand not met
- `WaterOutput(amount, t)` — remaining water for downstream

### Deficit Tracking

When received water is less than required, a `DeficitRecorded` event captures:
- `required`: demand for the timestep
- `actual`: amount consumed
- `deficit`: unmet demand

### Update Cycle

1. Consume up to requirement: `consumed = min(received, requirement[t] * dt)`
2. Record deficit if `consumed < required`
3. Record `WaterOutput` for remaining water
4. Reset counter

### Example

```python
from taqsim.node import Demand, TimeSeries

city = Demand(
    id="city_water",
    requirement=TimeSeries(values=[50.0, 60.0, 55.0])
)
```

---

## Storage

Reservoir node. Receives, stores, loses, releases, and outputs water.

### Capabilities

- **Receives**: Accepts water from upstream
- **Stores**: Buffers water up to capacity
- **Loses**: Physical losses (evaporation, seepage)

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |
| `capacity` | `float` | Yes | Maximum storage volume [m³] |
| `initial_storage` | `float` | No | Starting volume (default: 0) [m³] |
| `dead_storage` | `float` | No | Volume of water that cannot be released (default: 0.0) [m³] |
| `release_rule` | `ReleaseRule` | Yes | Release calculation |
| `loss_rule` | `LossRule` | Yes | Loss calculation |

### Validation

- `capacity` must be positive
- `initial_storage` must be >= 0 and <= capacity
- `dead_storage` cannot be negative
- `dead_storage` cannot exceed capacity

### Note on Dead Storage

Water below the dead storage level cannot be released (it is below the lowest outlet), but losses still apply to the full storage volume. Water at dead level can still evaporate or seep.

### Events Recorded

- `WaterReceived(amount, source_id, t)` — when water arrives
- `WaterStored(amount, t)` — amount added to storage
- `WaterSpilled(amount, t)` — overflow when capacity exceeded
- `WaterLost(amount, reason, t)` — per loss type (evaporation, seepage)
- `WaterReleased(amount, t)` — controlled release
- `WaterOutput(amount, t)` — released + spilled for downstream

### Spillway Handling

When inflow exceeds available capacity:
1. Store up to capacity
2. Record `WaterSpilled` for overflow
3. Spilled water joins released water in `WaterOutput`

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
4. Record `WaterOutput` for (released + spilled)
5. Reset counter

### Example

```python
from taqsim.node import Storage

reservoir = Storage(
    id="main_dam",
    capacity=10000.0,
    initial_storage=5000.0,
    release_rule=fixed_release,
    loss_rule=evaporation_loss
)

# Check current storage
print(reservoir.storage)  # 5000.0
```

---

## Sink

Terminal node. Receives water and exits the system. Represents system boundaries (ocean, aquifer, etc.).

### Capabilities

- **Receives**: Accepts water from upstream

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | Yes | Unique identifier |

### Events Recorded

- `WaterReceived(amount, source_id, t)` — when water arrives

### Update Cycle

No-op. Sink is passive; it only records received water.

### Example

```python
from taqsim.node import Sink

sink = Sink(id="ocean_outflow")
```
