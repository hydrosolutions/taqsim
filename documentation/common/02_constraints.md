# Constraints

Constraints define relationships between strategy parameters that must hold after genetic operators modify values during optimization.

## Built-in Constraint Types

### SumToOne

Parameters must sum to 1.0. Common for split ratios.

```python
from taqsim.constraints import SumToOne

@dataclass(frozen=True)
class ProportionalSplit(Strategy):
    __params__ = ("ratio_a", "ratio_b")
    __bounds__ = {"ratio_a": (0, 1), "ratio_b": (0, 1)}
    __constraints__ = (SumToOne(("ratio_a", "ratio_b")),)

    ratio_a: float = 0.5
    ratio_b: float = 0.5
```

**Repair behavior:**
- Normalizes values to sum to 1.0
- Clamps negative values to 0 before normalizing
- If all zeros, distributes equally (1/n each)

### Ordered

Ensures `low <= high`. Common for threshold parameters.

```python
from taqsim.constraints import Ordered

@dataclass(frozen=True)
class BandedRelease(Strategy):
    __params__ = ("min_rate", "max_rate")
    __bounds__ = {"min_rate": (0, 100), "max_rate": (0, 200)}
    __constraints__ = (Ordered(low="min_rate", high="max_rate"),)

    min_rate: float = 10.0
    max_rate: float = 50.0
```

**Repair behavior:**
- Swaps values if low > high

## Custom Constraints

Implement the `Constraint` protocol:

```python
from typing import Protocol

class Constraint(Protocol):
    params: tuple[str, ...]

    def repair(self, values: dict[str, float]) -> dict[str, float]:
        """Return corrected values. Must not mutate input."""
        ...
```

## Validation

Constraints are validated at class definition time. Referencing unknown parameters raises `TypeError`:

```python
# Raises TypeError: BadStrategy: constraint references unknown params: {'missing'}
@dataclass(frozen=True)
class BadStrategy(Strategy):
    __params__ = ("rate",)
    __constraints__ = (SumToOne(("missing",)),)
```
