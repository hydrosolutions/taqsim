# Built-in Objectives

## Overview

Taqsim provides four built-in objective factories covering common water system optimization goals. Access them via the `minimize` and `maximize` registries.

## minimize.spill

Minimize water spilled from a node (typically a reservoir).

```python
from taqsim.objective import minimize

obj = minimize.spill("reservoir_1")
obj = minimize.spill("reservoir_1", priority=2)
```

**What it measures:** Sum of `WaterSpilled` event amounts at the specified node.

**Use case:** Reduce wasteful overflow from storage nodes. Spills indicate the reservoir exceeded capacity.

## minimize.deficit

Minimize unmet demand at a node.

```python
from taqsim.objective import minimize

obj = minimize.deficit("city_demand")
```

**What it measures:** Sum of `DeficitRecorded` event deficits at the specified node.

**Use case:** Ensure demand nodes receive their required water. Deficits indicate supply fell short of requirements.

## minimize.loss

Minimize water lost from a node or edge.

```python
from taqsim.objective import minimize

obj = minimize.loss("canal_1")        # Edge losses
obj = minimize.loss("reservoir_1")    # Node losses (evaporation, seepage)
```

**What it measures:** Sum of `WaterLost` event amounts at the specified target.

**Use case:** Reduce physical losses in the system. Works for both edge transmission losses and node storage losses.

## maximize.delivery

Maximize water delivered to a target.

```python
from taqsim.objective import maximize

obj = maximize.delivery("irrigation_edge")  # Edge delivery
obj = maximize.delivery("demand_node")      # Node received
```

**What it measures:**
- For edges: Sum of `WaterDelivered` events
- For nodes: Sum of `WaterReceived` events

**Use case:** Maximize water reaching a destination. Useful for irrigation delivery, environmental flows, or downstream transfers.

## Priority Parameter

All built-in objectives accept a `priority` parameter for multi-objective weighting:

```python
objectives = [
    minimize.deficit("municipal", priority=3),   # Highest priority
    minimize.deficit("irrigation", priority=1),  # Lower priority
    minimize.spill("dam", priority=2),
]
```

Higher priority objectives carry more weight in optimization.

## Error Handling

Objectives raise `ValueError` if the target node or edge doesn't exist:

```python
obj = minimize.spill("nonexistent")
obj.evaluate(system)  # ValueError: Node 'nonexistent' not found
```
