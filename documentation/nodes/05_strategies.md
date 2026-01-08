# Strategy Protocols

## Overview

Strategies define configurable behaviors for nodes. Each is a `Protocol` that allows custom implementations to be injected at node creation.

## Protocol Definitions

### ReleaseRule

Controls how storage nodes release water.

```python
@runtime_checkable
class ReleaseRule(Protocol):
    def release(
        self,
        storage: float,    # current storage volume
        capacity: float,   # maximum capacity
        inflow: float,     # inflow this timestep
        t: int,            # timestep
        dt: float          # timestep duration
    ) -> float: ...        # amount to release
```

### SplitStrategy

Determines how water is distributed among multiple targets.

```python
@runtime_checkable
class SplitStrategy(Protocol):
    def split(
        self,
        amount: float,       # total amount to distribute
        targets: list[str],  # target node IDs
        t: int               # timestep
    ) -> dict[str, float]: ...  # {target_id: amount}
```

### LossRule

Calculates physical losses from storage.

```python
@runtime_checkable
class LossRule(Protocol):
    def calculate(
        self,
        storage: float,   # current storage volume
        capacity: float,  # maximum capacity
        t: int,           # timestep
        dt: float         # timestep duration
    ) -> dict[LossReason, float]: ...  # {reason: amount}
```

`LossReason` is an enum with values:

- `LossReason.EVAPORATION`
- `LossReason.SEEPAGE`

## Custom Implementations

### Equal Split Strategy

```python
from dataclasses import dataclass

@dataclass
class EqualSplit:
    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
        if not targets:
            return {}
        share = amount / len(targets)
        return {target: share for target in targets}
```

### Proportional Split Strategy

```python
@dataclass
class ProportionalSplit:
    weights: dict[str, float]

    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
        total_weight = sum(self.weights.get(t, 0) for t in targets)
        if total_weight == 0:
            return {t: 0 for t in targets}
        return {
            target: amount * (self.weights.get(target, 0) / total_weight)
            for target in targets
        }
```

### Fixed Release Rule

```python
@dataclass
class FixedRelease:
    rate: float  # constant release rate

    def release(
        self, storage: float, capacity: float, inflow: float, t: int, dt: float
    ) -> float:
        return min(self.rate * dt, storage)
```

### Percentage Release Rule

```python
@dataclass
class PercentageRelease:
    fraction: float  # 0.0 to 1.0

    def release(
        self, storage: float, capacity: float, inflow: float, t: int, dt: float
    ) -> float:
        return storage * self.fraction
```

### Zero Loss Rule

```python
from taqsim.node import LossReason

@dataclass
class ZeroLoss:
    def calculate(
        self, storage: float, capacity: float, t: int, dt: float
    ) -> dict[LossReason, float]:
        return {}
```

### Evaporation Loss Rule

```python
@dataclass
class EvaporationLoss:
    rate: float  # evaporation rate per timestep

    def calculate(
        self, storage: float, capacity: float, t: int, dt: float
    ) -> dict[LossReason, float]:
        loss = min(self.rate * dt, storage)
        return {LossReason.EVAPORATION: loss}
```

### Combined Loss Rule

```python
@dataclass
class CombinedLoss:
    evap_rate: float
    seepage_rate: float

    def calculate(
        self, storage: float, capacity: float, t: int, dt: float
    ) -> dict[LossReason, float]:
        return {
            LossReason.EVAPORATION: self.evap_rate * dt,
            LossReason.SEEPAGE: self.seepage_rate * dt
        }
```

## Usage

```python
from taqsim.node import Source, Storage, TimeSeries

# Create strategies
split = EqualSplit()
release = FixedRelease(rate=5.0)
loss = ZeroLoss()

# Use with nodes
source = Source(
    id="river",
    inflow=TimeSeries(values=[10.0, 15.0]),
    targets=["dam"],
    split_strategy=split
)

storage = Storage(
    id="dam",
    capacity=1000.0,
    release_rule=release,
    loss_rule=loss,
    split_strategy=split,
    targets=["downstream"]
)
```

## Protocol Checking

Strategies satisfy protocols via structural typing:

```python
from taqsim.node import SplitStrategy, ReleaseRule, LossRule

split = EqualSplit()
assert isinstance(split, SplitStrategy)

release = FixedRelease(rate=5.0)
assert isinstance(release, ReleaseRule)

loss = ZeroLoss()
assert isinstance(loss, LossRule)
```
