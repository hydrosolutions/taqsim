# Parameter Introspection

## Overview

Nodes expose their configurable strategies for parameter discovery through introspection methods. This enables optimization algorithms, sensitivity analysis tools, and debugging interfaces to discover and modify node parameters at runtime.

---

## The strategies() Method

Every node inherits `strategies()` from `BaseNode`. This method auto-discovers all fields that are `Strategy` instances.

### Signature

```python
def strategies(self) -> dict[str, Strategy]:
    """Return all strategy fields on this node."""
```

### Behavior

- Scans node fields for instances inheriting from `Strategy`
- Returns a mapping of field name to strategy instance
- **Physical models (`LossRule`) are excluded** — they don't inherit from `Strategy`

### Example

```python
from taqsim.node import Storage
from taqsim.node.strategies import FixedRelease, ZeroLoss

storage = Storage(
    id="reservoir",
    capacity=1000.0,
    release_rule=FixedRelease(rate=50.0),
    loss_rule=ZeroLoss()
)

storage.strategies()
# -> {"release_rule": FixedRelease(rate=50.0)}
# Note: loss_rule is NOT included (LossRule is not a Strategy)
```

---

## Auto-Discovery Mechanism

The `strategies()` method uses structural typing to find strategies:

```python
# Simplified internal logic
{
    name: field
    for name, field in self.__dict__.items()
    if isinstance(field, Strategy)
}
```

### Benefits

- **No manual registration** — strategies are discovered automatically
- **Extensible** — works for any new `Strategy` subclass
- **Consistent** — all nodes use the same discovery mechanism

---

## The reset() Method

Clears node state between simulation runs. Essential for optimization workflows that run multiple simulations with different parameters.

### Signature

```python
def reset(self) -> None:
    """Reset node state for a fresh simulation run."""
```

### Node-Specific Behavior

| Node Type | Reset Behavior |
|-----------|----------------|
| `BaseNode` | Clears events via `clear_events()` |
| `Storage` | Clears events + resets `_current_storage` to `initial_storage` |
| `PassThrough` | Clears events + resets `_received_this_step` to 0 |
| `Splitter` | Clears events + resets `_received_this_step` to 0 |
| `Demand` | Clears events + resets `_received_this_step` to 0 |
| `Sink` | Clears events only |

### Example

```python
from taqsim import WaterSystem

# After a simulation run
system.simulate(timesteps=12)

# Reset all nodes and edges for a fresh run
system.reset()

# Run again with same or modified parameters
system.simulate(timesteps=12)
```

---

## Usage: Inspecting Node Parameters

Iterate over all nodes to discover their tunable strategies:

```python
for node_id, node in system.nodes.items():
    strategies = node.strategies()
    for name, strategy in strategies.items():
        print(f"{node_id}.{name}: {strategy.params()}")
```

### Example Output

```
river_intake.split_strategy: {"weights": {"canal_a": 0.6, "canal_b": 0.4}}
reservoir.release_rule: {"rate": 50.0}
junction.split_strategy: {"weights": {"farm_1": 0.5, "farm_2": 0.5}}
```

---

## Optimization Workflow

Combining introspection with reset enables parameter optimization:

```python
def evaluate(system: WaterSystem, params: dict) -> float:
    # Apply parameters
    system.nodes["reservoir"].release_rule.rate = params["release_rate"]

    # Reset state
    system.reset()

    # Run simulation
    system.simulate(timesteps=365)

    # Compute objective (e.g., total deficit)
    return compute_deficit(system)

# Optimize
best_params = optimizer.minimize(
    lambda p: evaluate(system, p),
    bounds={"release_rate": (10.0, 100.0)}
)
```
