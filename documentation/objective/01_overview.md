# Objective System Overview

## Purpose

The `taqsim.objective` module provides declarative optimization objectives for genetic algorithm (GA) optimization. Instead of manually computing fitness scores, you declare what to optimize and the system handles evaluation.

## Core Concepts

### Objective

An `Objective` encapsulates what to optimize:

```python
@dataclass(frozen=True, slots=True)
class Objective:
    name: str                                    # Human-readable identifier
    direction: Literal["minimize", "maximize"]   # Optimization direction
    evaluate: Callable[[WaterSystem], float]     # Computes score from simulation
    priority: int = 1                            # For multi-objective weighting
```

### Trace

A `Trace` is a time-indexed series of values. It bridges simulation events and objective computation, supporting arithmetic operations and aggregations.

### lift

The `@lift` decorator transforms scalar functions to work on Traces, enabling custom physics transformations.

## Quick Example

```python
from taqsim.objective import minimize, maximize, Objective

# Use built-in objectives via registries
objectives = [
    minimize.spill("reservoir_1"),      # Minimize water spilled
    minimize.deficit("city_demand"),    # Minimize unmet demand
    maximize.delivery("irrigation"),    # Maximize water delivered
]

# Objectives evaluate against a simulated system
system.simulate(timesteps=12)
scores = [obj.evaluate(system) for obj in objectives]
```

## Registries

Two registries provide fluent access to built-in objectives:

- `minimize` - objectives where lower is better (spill, deficit, loss)
- `maximize` - objectives where higher is better (delivery)

```python
from taqsim.objective import minimize, maximize

# Fluent API
obj1 = minimize.spill("reservoir")
obj2 = maximize.delivery("downstream_node")

# With priority for multi-objective optimization
obj3 = minimize.deficit("demand", priority=2)
```

## Custom Objectives

Create custom objectives by defining an evaluate function:

```python
from taqsim.objective import Objective
from taqsim.node.events import WaterReleased

def hydropower(turbine_id: str, efficiency: float = 0.85) -> Objective:
    def evaluate(system):
        node = system.nodes[turbine_id]
        releases = node.trace(WaterReleased)
        power = releases.map(lambda q: q * 9.81 * 50 * efficiency)  # P = Q * g * h * eta
        return power.sum()

    return Objective(
        name=f"{turbine_id}.hydropower",
        direction="maximize",
        evaluate=evaluate,
    )
```
