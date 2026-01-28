# Plan: Extend Reservoir Optimization Notebook

## Goal
Update `notebooks/reservoir_optimization_showcase_tobi.ipynb` to:
1. ~~Add irrigation and thermal plant deficit objectives~~ DONE
2. ~~Visualize irrigation vs thermal trade-off with Pareto front~~ DONE
3. ~~Add water availability visualization over time~~ DONE
4. ~~Add parallel coordinates plot~~ DONE
5. ~~Add interactive Plotly version with filtering~~ DONE
6. ~~Track and visualize optimization progress over generations~~ DONE
7. ~~Add hypervolume tracking to convergence visualization~~ DONE

---

## New Implementation Steps: Hypervolume Tracking

### 1. Add pymoo Dependency
```bash
uv add pymoo
```

### 2. Update History Callback (cell-12)

Add hypervolume calculation to the callback:

```python
from pymoo.indicators.hv import HV

def make_history_callback(objectives: list[Objective]) -> tuple[callable, dict]:
    """Create callback that collects per-generation statistics including hypervolume."""

    # Reference point for hypervolume (worst possible values, slightly beyond observed)
    # For minimize objectives: use large positive value
    # For maximize objectives: use large negative value (since stored negated)
    ref_point = np.array([0.0, 2500.0, 2000.0, 2500.0])  # [hp(neg), flood, irr, therm]
    hv_indicator = HV(ref=ref_point)

    history = {
        "generation": [],
        "n_pareto": [],
        "hypervolume": [],  # NEW
        "obj_min": {...},
        "obj_max": {...},
        "obj_mean": {...},
    }

    def callback(pop: Population, gen: int) -> bool:
        pareto_mask = pop.rank == 0
        pareto_obj = pop.objectives[pareto_mask]

        # Calculate hypervolume (objectives are already in minimize form internally)
        hv = hv_indicator(pareto_obj)
        history["hypervolume"].append(float(hv))

        # ... rest of existing callback code ...

        return False

    return callback, history
```

### 3. Update Convergence Plot (cell-16)

Replace the empty 6th subplot with hypervolume:

```python
# Change from 2x3 grid to use all 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# ... existing plots in positions [0,0], [0,1], [0,2], [1,0], [1,1] ...

# Plot 6: Hypervolume (replaces empty subplot)
ax = axes[1, 2]
ax.plot(gens, history["hypervolume"], "k-", linewidth=2)
ax.set_xlabel("Generation")
ax.set_ylabel("Hypervolume")
ax.set_title("Hypervolume (↑ better)")
ax.grid(True, alpha=0.3)
```

### 4. Reference Point Selection

The reference point must dominate all Pareto solutions. Strategy:
- **Option A (static)**: Set conservatively large values based on problem knowledge
- **Option B (dynamic)**: Compute from worst values in first generation + margin

We'll use **Option A** with values slightly worse than expected worst case:
- Hydropower: 0.0 GWh (stored as 0.0 since negated internally)
- City Flood: 2500 (well above observed ~1900 max)
- Irrigation Deficit: 2000 (above observed ~1400 max)
- Thermal Deficit: 2500 (above observed ~1800 max)

---

## Files to Modify
- `notebooks/reservoir_optimization_showcase_tobi.ipynb`
  - cell-12: Update `make_history_callback` to compute hypervolume
  - cell-16: Add hypervolume subplot

## Dependencies to Add
- `pymoo` (via `uv add pymoo`)

## Verification
1. Run `uv add pymoo` to install dependency
2. Restart kernel and run notebook from cell-12
3. Verify hypervolume plot shows monotonically increasing trend
4. Confirm hypervolume stabilizes as other metrics stabilize
