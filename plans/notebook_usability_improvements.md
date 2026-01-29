# Notebook Usability Improvements Plan

## Goal

Make the `_4obj` and `_5obj` Jupyter notebooks more approachable for students and teachers by improving parameter organization, reactivity, and overall learning experience.

------------------------------------------------------------------------

## Status Legend

-   ⬜ Not started
-   🟡 In discussion / planning
-   🔵 In progress
-   ✅ Completed

------------------------------------------------------------------------

## Improvement 1: Reactive Parameter Configuration (Lego Brick Pattern)

**Status**: 🟡 In discussion

### Problem Statement

The infrastructure elements (reservoirs, turbines, cities, etc.) are assembled like "lego bricks", but their parameters are currently scattered throughout the notebook: - Node capacities defined at instantiation - Strategy parameters defined separately - Visualization axes hardcoded to specific ranges - Changing one parameter (e.g., reservoir capacity 150 → 1000) requires manual updates in multiple places

### Desired Behavior

1.  **Single source of truth**: All configurable parameters defined in ONE location
2.  **Logical grouping**: Parameters organized by physical location/component
3.  **Automatic propagation**: When a parameter changes:
    -   Dependent calculations update (e.g., h-V curve)
    -   Visualization axes adjust (e.g., storage plot y-axis)
    -   Strategy bounds adapt (e.g., SLOP thresholds)
4.  **Intuitive exploration**: Students can easily perform sensitivity analyses

### Current Architecture

```         
Parameters scattered:
├─ Cell N:   RESERVOIR_CAPACITY = 150
├─ Cell N+3: Storage(capacity=150, ...)
├─ Cell N+7: h-V curve uses hardcoded 150
├─ Cell N+12: Plot y-axis [0, 160]
└─ Cell N+15: SLOP bounds reference 100 (head max)
```

### Proposed Approaches

#### Approach A: Centralized Configuration Dataclass

``` python
@dataclass
class ReservoirConfig:
    capacity: float = 150.0
    dead_storage: float = 10.0
    initial_storage: float = 75.0

    @property
    def active_storage(self) -> float:
        return self.capacity - self.dead_storage

    def head_from_volume(self, volume: float) -> float:
        """Linear h-V relationship."""
        return 20.0 + (volume - self.dead_storage) * 80.0 / self.active_storage
```

**Pros**: Pure Python, no external dependencies, clear structure **Cons**: Still requires re-running cells manually after changes

#### Approach B: Interactive Widgets (ipywidgets)

``` python
capacity_slider = widgets.FloatSlider(min=50, max=2000, value=150, description='Capacity')

@interact(capacity=capacity_slider)
def update_system(capacity):
    config.capacity = capacity
    rerun_visualization()
```

**Pros**: Interactive, immediate feedback **Cons**: Can become cluttered, callbacks complex, not all students comfortable with widgets

#### Approach C: marimo Reactive Notebooks

``` python
# marimo cell
capacity = mo.ui.slider(50, 2000, value=150, label="Reservoir Capacity")

# Downstream cell automatically re-runs when capacity.value changes
reservoir = Storage(id="reservoir", capacity=capacity.value, ...)
```

**Pros**: Built-in reactivity, cleaner than ipywidgets, modern UI **Cons**: Requires migration from Jupyter, learning curve

#### Approach D: Parameter Registry with Dependency Graph

``` python
class ParamRegistry:
    def __init__(self):
        self._params = {}
        self._dependencies = defaultdict(set)

    def register(self, name, value, depends_on=None): ...
    def update(self, name, value): ...  # triggers dependent updates
```

**Pros**: Explicit dependency tracking, powerful **Cons**: Over-engineered for educational notebooks

### Discussion Points

1.  Which approach best balances simplicity vs. power?
2.  Should we support multiple approaches (e.g., config dataclass + optional widgets)?
3.  What's the minimum viable version for initial release?

### Dependencies Requiring Update When Reservoir Capacity Changes

-   [ ] Storage node instantiation
-   [ ] h-V curve calculation
-   [ ] Head bounds in SLOPRelease strategy
-   [ ] Reservoir storage plot y-axis limits
-   [ ] Initial storage (should it be proportional?)
-   [ ] Dead storage (should it scale with capacity?)

------------------------------------------------------------------------

## Improvement 2: System Schematic Visualization

**Status**: ⬜ Not started

### Problem Statement

Students need a visual mental model of the water system topology.

### Desired Behavior

-   Auto-generated network diagram showing nodes and edges
-   Node icons indicating type (reservoir = lake, turbine = generator, etc.)
-   Edge labels showing capacities
-   Updates when system structure changes

------------------------------------------------------------------------

## Improvement 3: Parameter Sensitivity Dashboard

**Status**: ⬜ Not started

### Problem Statement

Difficult to understand how individual parameters affect system behavior.

### Desired Behavior

-   One-at-a-time parameter sweeps
-   Tornado diagrams showing sensitivity
-   Clear visual feedback on trade-offs

------------------------------------------------------------------------

## Improvement 4: Progressive Complexity Cells

**Status**: ⬜ Not started

### Problem Statement

Notebooks show full complexity upfront, which can overwhelm beginners.

### Desired Behavior

-   "Basic" cells with defaults hidden
-   "Advanced" expandable sections
-   Clear learning progression markers

------------------------------------------------------------------------

## Improvement 5: Inline Documentation & Tooltips

**Status**: ⬜ Not started

### Problem Statement

Parameter meanings not immediately obvious.

### Desired Behavior

-   Hover tooltips explaining parameters
-   Inline LaTeX equations where relevant
-   Links to theory/documentation

------------------------------------------------------------------------

## Backlog (Future Ideas)

-   [ ] Export configurations to YAML/JSON
-   [ ] Load pre-configured scenarios
-   [ ] Comparison mode (side-by-side configurations)
-   [ ] Animated time series playback
-   [ ] Student exercise templates

------------------------------------------------------------------------

## Implementation Log

| Date       | Improvement | Action                    | Notes              |
|------------|-------------|---------------------------|--------------------|
| 2026-01-29 | #1          | Created planning document | Initial discussion |

------------------------------------------------------------------------

## References

-   Notebooks: `notebooks/reservoir_optimization_4obj.ipynb`, `notebooks/reservoir_optimization_5obj_smoothness.ipynb`
-   Strategy definitions: `src/taqsim/strategies/`
-   Node types: `src/taqsim/nodes/`