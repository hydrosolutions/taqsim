# Toy System Analysis: 5-Objective Optimization with Operational Smoothness

## Purpose

Document the 5-objective extension that adds operational smoothness to the original 4-objective toy system.

**Related files:**
- Notebook: `notebooks/reservoir_optimization_5obj_smoothness.ipynb`
- Base analysis: `plans/toy_system_analysis_4obj.md`
- Base notebook: `notebooks/reservoir_optimization_showcase_4obj.ipynb`

------------------------------------------------------------------------

## Motivation

The original 4-objective optimization produces solutions with highly erratic month-to-month variability. While mathematically optimal, these solutions are **operationally unrealistic**:

- Real operators cannot implement wild release swings
- Downstream users need predictable water availability
- Infrastructure stress from rapid flow changes
- Regulatory constraints often mandate gradual changes

By adding smoothness as an explicit 5th objective, we:
1. **Reveal the trade-off** between performance and operability
2. Let decision-makers **choose their smoothness level**
3. Generate solutions that are actually **implementable**

------------------------------------------------------------------------

## System Overview

### Topology (unchanged from 4-objective)

```
River → Reservoir → Turbine → City → Splitter → Irrigation → Sink
                                          ↓
                                    Thermal Plant → Sink
```

### Configuration (unchanged)

| Component  | Parameter       | Value                       |
|------------|-----------------|-----------------------------|
| Simulation | Timesteps       | 120 (10 years × 12 months)  |
| Reservoir  | Capacity        | 150 units                   |
| Reservoir  | Dead storage    | 10 units                    |
| Reservoir  | Initial storage | 75 units                    |
| Turbine    | Capacity        | 60 units                    |
| City       | Capacity        | 41 units (flood bottleneck) |
| Irrigation | Consumption     | 100% (fully consumptive)    |
| Thermal    | Consumption     | 0% (non-consumptive)        |

### Optimization Parameters (72 total, unchanged)

| Strategy             | Parameters                      | Count |
|----------------------|---------------------------------|-------|
| SLOP Release Rule    | h1, h2, w, m1, m2 × 12 months   | 60    |
| Seasonal Split Ratio | irrigation_fraction × 12 months | 12    |

### Objectives (5 total)

| Objective              | Direction | Description                                    |
|------------------------|-----------|------------------------------------------------|
| Hydropower             | Maximize  | Total electricity generated (GWh)              |
| City Flood             | Minimize  | Total spillage at city node                    |
| Irrigation Deficit     | Minimize  | Unmet irrigation demand                        |
| Thermal Deficit        | Minimize  | Unmet cooling water demand                     |
| **Op. Variability**    | **Minimize** | **Combined release + allocation smoothness** |

------------------------------------------------------------------------

## The 5th Objective: Operational Variability

### Formula

```
Operational Variability = Release Variability + Allocation Variability

where:
    Release Variability    = Σ|R(t) - R(t-1)|           (turbine flow changes)
    Allocation Variability = Σ|frac(t) - frac(t-1)|×100 (irrigation split changes)
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Combine vs separate** | Single combined metric | Keeps 5 objectives (not 6); simpler for students |
| **Allocation scaling** | ×100 | Makes 10% fraction change ≈ 10 units (comparable to flow) |
| **Weight parameter** | `allocation_weight=1.0` | Equal importance; can be tuned if needed |
| **Edge case (zero flow)** | Use 0.5 fraction | Neutral value when no water reaches splitter |

### Why Combined Metric?

Initial implementation with release-only smoothness revealed that:
- Turbine flow could be smooth, but...
- Irrigation/thermal allocation still swung wildly month-to-month

The combined metric ensures the optimizer penalizes **both**:
1. Erratic reservoir releases
2. Erratic allocation between irrigation and thermal plant

------------------------------------------------------------------------

## Implementation Details

### Objective Function

```python
def operational_variability_objective(
    turbine_id: str,
    irrigation_id: str,
    thermal_id: str,
    allocation_weight: float = 1.0,
) -> Objective:
    """Minimize total operational variability: release + allocation changes."""

    def evaluate(system: WaterSystem) -> float:
        # Release variability: Σ|R(t) - R(t-1)|
        flows = np.array(turbine.trace(WaterPassedThrough).values())
        release_var = np.abs(np.diff(flows)).sum()

        # Allocation variability: Σ|frac(t) - frac(t-1)| × 100
        irr_received = np.array(irrigation.trace(WaterReceived).values())
        therm_received = np.array(thermal.trace(WaterReceived).values())
        total = irr_received + therm_received
        irr_fraction = np.where(total > 0, irr_received / total, 0.5)
        allocation_var = np.abs(np.diff(irr_fraction)).sum() * 100

        return release_var + allocation_weight * allocation_var
```

### Hypervolume Reference Point

Extended to 5 dimensions:
```python
ref_point = np.array([0.0, 2500.0, 2000.0, 2500.0, 5000.0])
#                     hp   flood   irr_def therm_def op_var
```

------------------------------------------------------------------------

## Expected Trade-offs

### Direct Trade-offs

| Pair | Mechanism | Expected Pattern |
|------|-----------|------------------|
| Hydropower ↔ City Flood | Same water release causes both | Clear front (unchanged) |
| Irrigation ↔ Thermal | Splitter allocates between them | Clear front (unchanged) |
| **Hydropower ↔ Op. Variability** | High power may need variable releases | **New trade-off** |
| **Deficits ↔ Op. Variability** | Meeting demands may need allocation swings | **New trade-off** |

### Indirect Relationships

| Pair | Mechanism |
|------|-----------|
| Flood ↔ Op. Variability | Complex — smooth can mean conservative or aggressive |
| Irrigation ↔ Op. Variability | Allocation smoothness directly affects irrigation stability |

------------------------------------------------------------------------

## Visualization Updates

### Parallel Coordinates

- Now shows 5 axes
- Colored by Operational Variability (Viridis_r: lower = brighter)
- Interactive filtering to find smooth solutions

### Pairwise Projections

- 2×3 grid showing key pairs
- New plots: Hydropower vs Op. Variability, Flood vs Op. Variability, etc.
- All colored by smoothness or other objectives

### Time Series

- Interactive Plotly figure (click legend to show/hide)
- 4 representative solutions: Smoothest, Low Flood, Balanced, High Power
- **Variability breakdown table** shows both components:
  - Release variability
  - Allocation variability
  - Total

------------------------------------------------------------------------

## Comparison with 4-Objective Version

| Aspect | 4 Objectives | 5 Objectives |
|--------|--------------|--------------|
| Objectives | 4 | 5 (+operational variability) |
| Erratic solutions | Common | Explicitly penalized |
| Decision support | Performance only | Performance + Operability |
| Visualization | 4D parallel coords | 5D parallel coords |
| Solution selection | By performance | Can filter by smoothness first |

------------------------------------------------------------------------

## Usage Recommendations

### For Teaching

1. Start with 4-objective notebook to show raw optimization results
2. Discuss why erratic solutions are problematic
3. Introduce 5-objective notebook to show smoothness trade-off
4. Use variability breakdown to explain both components

### For Decision-Makers

1. Filter Pareto front by acceptable Op. Variability threshold
2. Among smooth solutions, choose based on other priorities
3. Breakdown table helps identify if release or allocation dominates

### Suggested Smoothness Thresholds

| Level | Op. Variability | Interpretation |
|-------|-----------------|----------------|
| Very smooth | < 1500 | Highly constrained operations |
| Smooth | 1500-2500 | Operationally realistic |
| Moderate | 2500-3500 | Some variability acceptable |
| Variable | > 3500 | Erratic, may be hard to implement |

*Note: These thresholds should be calibrated after running the optimization.*

------------------------------------------------------------------------

## Open Questions

1. **Allocation weight**: Is 1.0 the right balance, or should allocation smoothness be weighted differently?
2. **Scaling factor**: Is ×100 appropriate for making fraction changes comparable to flow changes?
3. **Alternative metrics**: Would variance-based metrics behave differently than sum-of-absolute-differences?
4. **Comparison study**: How do the Pareto fronts differ between 4-obj and 5-obj versions?
