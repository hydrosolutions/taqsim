# Parameter Exposure

## Overview

Parameter exposure enables genetic algorithm (GA) optimization by providing a standardized interface to discover, read, and modify tunable parameters across a `WaterSystem`. The system treats operational strategies as optimizable while keeping physical models fixed.

## Semantic Distinction

Not all configurable objects are optimizable. The system distinguishes between:

| Type | Category | Optimizable | Examples |
|------|----------|-------------|----------|
| **Operational Policy** | Strategy | Yes | `ReleaseRule`, `SplitStrategy` |
| **Physical Model** | Rule | No | `LossRule`, `EdgeLossRule` |

### Why the Distinction?

- **Operational policies** represent decisions: how much water to release, how to split flow. These are tunable parameters a GA can optimize.
- **Physical models** represent reality: evaporation rates, seepage coefficients. These are measured properties of the physical system, not optimization variables.

Only classes that inherit from `Strategy` expose parameters for optimization.

## Strategy Base Class

Located in `src/taqsim/common.py`:

```python
class Strategy:
    __params__: ClassVar[tuple[str, ...]] = ()

    def params(self) -> dict[str, ParamValue]:
        """Return current parameter values."""
        return {name: getattr(self, name) for name in self.__params__}

    def with_params(self, **kwargs: ParamValue) -> Self:
        """Create new instance with updated parameters (immutable)."""
        invalid = set(kwargs) - set(self.__params__)
        if invalid:
            raise ValueError(f"Unknown parameters: {invalid}")
        return replace(self, **kwargs)
```

### Implementing a Strategy

Concrete strategies must:

1. Inherit from `Strategy`
2. Be frozen dataclasses
3. Declare `__params__` listing optimizable field names

```python
from dataclasses import dataclass
from typing import ClassVar
from taqsim.common import Strategy

@dataclass(frozen=True)
class FixedRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate",)
    rate: float = 50.0

    def release(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
        return min(self.rate * dt, storage)
```

### Non-Optimizable Fields

Fields not listed in `__params__` are excluded from optimization:

```python
@dataclass(frozen=True)
class ThresholdRelease(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("rate", "threshold")
    rate: float = 50.0
    threshold: float = 0.8
    name: str = "threshold_release"  # NOT in __params__ - excluded
```

## ParamSpec Dataclass

Each tunable parameter is described by a `ParamSpec`:

```python
@dataclass(frozen=True, slots=True)
class ParamSpec:
    path: str              # e.g., "dam.release_rule.rate"
    value: float           # flattened scalar value
    index: int | None = None  # position in tuple, None for scalar
```

| Field | Description | Example |
|-------|-------------|---------|
| `path` | Dot-separated path to parameter | `"dam.release_rule.rate"` |
| `value` | Current scalar value | `50.0` |
| `index` | Position in tuple (None for scalar) | `0` for first element |

## WaterSystem Vectorization API

### param_schema()

Discover all tunable parameters:

```python
system = build_system()
schema = system.param_schema()

for spec in schema:
    print(f"{spec.path} = {spec.value}")
```

Output:
```
dam.release_rule.rate = 50.0
junction.split_strategy.ratios = 0.6  (index=0)
junction.split_strategy.ratios = 0.4  (index=1)
```

### to_vector()

Flatten parameters to a list of floats:

```python
vector = system.to_vector()
# [50.0, 0.6, 0.4]
```

### with_vector(vector)

Create a new system with modified parameters (immutable):

```python
new_vector = [100.0, 0.7, 0.3]
new_system = system.with_vector(new_vector)

# Original unchanged
assert system.to_vector() == [50.0, 0.6, 0.4]
# New system has updated values
assert new_system.to_vector() == [100.0, 0.7, 0.3]
```

Raises `ValueError` if vector length doesn't match schema.

### reset()

Clear state for fresh simulation:

```python
system.simulate(12)
# Events accumulated, storage changed

system.reset()
# Events cleared, storage back to initial value
# Topology and strategies preserved
```

## Tuple Parameter Handling

Tuple parameters (like split ratios) are flattened to individual `ParamSpec` entries:

```python
@dataclass(frozen=True)
class ProportionalSplit(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("ratios",)
    ratios: tuple[float, ...] = (0.6, 0.4)
```

Schema output:

| path | value | index |
|------|-------|-------|
| `junction.split_strategy.ratios` | `0.6` | `0` |
| `junction.split_strategy.ratios` | `0.4` | `1` |

Vector: `[0.6, 0.4]`

When reconstructing via `with_vector()`, values are reassembled into tuples based on index.

## GA Optimization Loop Pattern

The vectorization API enables a standard optimization loop:

```python
from taqsim import WaterSystem

def build_system() -> WaterSystem:
    # ... create and configure system
    pass

def evaluate(system: WaterSystem) -> float:
    # ... calculate fitness score
    pass

def evolve(population: list[list[float]], scores: list[float]) -> list[list[float]]:
    # ... GA selection, crossover, mutation
    pass

# Initialize
system = build_system()
base_vector = system.to_vector()
population = [mutate(base_vector) for _ in range(100)]

# Optimization loop
for generation in range(100):
    scores = []

    for vec in population:
        # Create candidate (original unchanged)
        candidate = system.with_vector(vec)

        # Run simulation
        candidate.simulate(12)

        # Evaluate fitness
        scores.append(evaluate(candidate))

    # Evolve population
    population = evolve(population, scores)

# Best solution
best_idx = scores.index(max(scores))
best_system = system.with_vector(population[best_idx])
```

### Key Properties

1. **Immutable**: `with_vector()` never modifies the original system
2. **Isolated**: Each candidate runs independently
3. **Resettable**: Use `reset()` if reusing a system instance
4. **Consistent**: Schema order is deterministic (sorted by path, then index)
