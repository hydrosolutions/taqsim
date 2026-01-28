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

### Release Rule Behavior

The SLOP release rule is **reactive** — it determines release based solely on current storage volume (converted to hydraulic head), not on inflow forecasts:

```
Release = f(head, month)    # head = volume_to_head(storage)
```

This means:
- The policy cannot anticipate incoming floods
- Smoothness must emerge from the policy parameters themselves
- More sophisticated approaches (MPC, forecast-based) could enable both smooth *and* responsive operations

### Water Balance

The reservoir water balance is computed implicitly in `Storage.update()`:

```
S(t+1) = S(t) + inflow - spilled - losses - released
```

Each timestep:
1. **Receive** — water arrives from upstream
2. **Store** — add to storage up to capacity; excess spills
3. **Lose** — subtract evaporation/seepage (zero in this toy system)
4. **Release** — subtract controlled release through turbine

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
        turbine = system.nodes[turbine_id]
        irrigation = system.nodes[irrigation_id]
        thermal = system.nodes[thermal_id]

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

## Observed Results

### Pareto Front Statistics

From a run with `pop_size=100`, `generations=50`, `seed=96530`:

| Objective              |    Min |     Max |   Range |
|------------------------|-------:|--------:|--------:|
| Hydropower (GWh)       |   1.24 |    4.68 |    3.44 |
| City Flood             |  790.9 |  2206.7 |  1415.8 |
| Irrigation Deficit     |  103.8 |  1312.1 |  1208.2 |
| Thermal Deficit        |    8.5 |  2043.6 |  2035.1 |
| Operational Variability| 1799.1 |  9456.2 |  7657.0 |

### Representative Solutions

| Strategy       | Hydropower | City Flood | Irr. Deficit | Therm. Deficit | Op. Var |
|----------------|------------|------------|--------------|----------------|---------|
| Smoothest      |       3.46 |     2044.1 |        140.7 |         1264.5 |  1799.1 |
| Low Flood Risk |       1.24 |      790.9 |        998.7 |         1067.5 |  7865.1 |
| Balanced       |       3.11 |     1347.4 |        117.1 |         1762.7 |  4205.1 |
| High Power     |       4.68 |     1763.0 |        857.3 |          579.7 |  4080.9 |

### Variability Breakdown

| Strategy       | Release Var | Alloc. Var | Total   |
|----------------|-------------|------------|---------|
| Smoothest      |       449.3 |     1349.9 |  1799.1 |
| Low Flood Risk |      2807.3 |     5057.8 |  7865.1 |
| Balanced       |      1243.4 |     2961.7 |  4205.1 |
| High Power     |       747.0 |     3333.9 |  4080.9 |

**Key insight**: Allocation variability dominates release variability in most solutions. The "Smoothest" solution achieves low variability in *both* components.

### Convergence Behavior

- Hypervolume improved +110% from generation 0 to 50
- Pareto front reached full population size (100) by generation 2
- **Non-monotonic hypervolume**: Observed ~10-15% oscillations in later generations
  - This is expected with NSGA-II, which optimizes Pareto dominance + crowding distance, not hypervolume directly
  - For smoother convergence, consider SMS-EMOA (hypervolume-based) or larger populations

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

Based on observed results (range 1799–9456):

| Level | Op. Variability | Interpretation | Example |
|-------|-----------------|----------------|---------|
| Very smooth | < 2000 | Highly constrained operations | "Smoothest" solution (1799) |
| Smooth | 2000-3500 | Operationally realistic | — |
| Moderate | 3500-5000 | Some variability acceptable | "High Power" (4081), "Balanced" (4205) |
| Variable | > 5000 | Erratic, may be hard to implement | "Low Flood Risk" (7865) |

**Note**: The "Low Flood Risk" solution is highly variable because conservative release requires rapid adjustments when inflows spike.

------------------------------------------------------------------------

## Further Smoothing Approaches

The current approach penalizes *output* variability (turbine flow and allocation fractions). Several strategies could achieve even smoother policies:

### 1. Parameter-Level Smoothness

Penalize month-to-month jumps in the 72 SLOP parameters themselves:

```python
def parameter_smoothness_objective(strategy_params: dict) -> float:
    """Penalize month-to-month jumps in the 72 parameters."""
    total = 0.0
    for param in ["h1", "h2", "w", "m1", "m2", "irrigation_fraction"]:
        values = strategy_params[param]  # 12 monthly values
        total += sum(abs(values[i] - values[(i+1) % 12]) for i in range(12))
    return total
```

This prevents erratic *policies* even if outputs happen to be smooth.

### 2. Fourier Basis Parameterization (Recommended)

**Core idea**: Instead of optimizing 12 independent monthly values per parameter, represent seasonal patterns using Fourier harmonics. This makes smoothness *structural* rather than penalized.

#### Mathematical Foundation

Any periodic function can be approximated by a Fourier series:

```
f(m) = a₀ + Σₖ [aₖ·cos(2πkm/12) + bₖ·sin(2πkm/12)]
```

where:
- `m` = month (0-11)
- `a₀` = annual mean
- `a₁, b₁` = first harmonic (annual cycle)
- `a₂, b₂` = second harmonic (semi-annual cycle)
- Higher harmonics capture finer variations

#### Implementation

```python
from math import cos, sin, pi

def fourier_seasonal(a0: float, a1: float, b1: float, a2: float, b2: float) -> tuple[float, ...]:
    """Convert 5 Fourier coefficients → 12 monthly values.

    Args:
        a0: Mean level (constant term)
        a1: Cosine amplitude for annual cycle (peak in winter)
        b1: Sine amplitude for annual cycle (peak in spring)
        a2: Cosine amplitude for semi-annual cycle
        b2: Sine amplitude for semi-annual cycle

    Returns:
        Tuple of 12 monthly values with smooth seasonal pattern
    """
    return tuple(
        a0 + a1 * cos(2 * pi * m / 12) + b1 * sin(2 * pi * m / 12)
           + a2 * cos(4 * pi * m / 12) + b2 * sin(4 * pi * m / 12)
        for m in range(12)
    )
```

#### Parameter Reduction

| Strategy | Original | Fourier (2 harmonics) | Reduction |
|----------|----------|----------------------|-----------|
| SLOP (h1, h2, w, m1, m2) | 5 × 12 = 60 | 5 × 5 = 25 | 58% |
| Seasonal Ratio | 1 × 12 = 12 | 1 × 5 = 5 | 58% |
| **Total** | **72** | **30** | **58%** |

#### Why Fourier Works

1. **Forced smoothness**: Fourier series are inherently smooth — no month-to-month jumps possible
2. **Physical interpretability**:
   - `a₀` = baseline release rate
   - `a₁, b₁` = annual seasonality (matches hydrological year)
   - `a₂, b₂` = semi-annual patterns (e.g., bimodal rainfall)
3. **Smaller search space**: 30 parameters instead of 72 → faster convergence
4. **No additional objectives needed**: Smoothness is guaranteed by construction

#### Example Patterns

```
# High winter release, low summer (snow melt management)
fourier_seasonal(a0=40, a1=15, b1=0, a2=0, b2=0)
# → [55, 52, 45, 35, 28, 25, 25, 28, 35, 45, 52, 55]

# Peak release in spring (March-April)
fourier_seasonal(a0=40, a1=0, b1=15, a2=0, b2=0)
# → [40, 47, 52, 55, 52, 47, 40, 33, 28, 25, 28, 33]

# Bimodal: peaks in spring and fall
fourier_seasonal(a0=40, a1=0, b1=0, a2=10, b2=0)
# → [50, 45, 35, 30, 35, 45, 50, 45, 35, 30, 35, 45]
```

#### Integration with taqsim

Would require modifying `SLOPRelease` to accept Fourier coefficients:

```python
@dataclass(frozen=True)
class FourierSLOPRelease(Strategy):
    """SLOP with Fourier-parameterized seasonal patterns."""

    __params__: ClassVar[tuple[str, ...]] = (
        "w_a0", "w_a1", "w_b1", "w_a2", "w_b2",  # base release Fourier
        "h1_a0", "h1_a1", ...                      # thresholds Fourier
    )

    # Optimizer tunes 30 Fourier coefficients instead of 72 monthly values
```

#### Critical Limitation: Extreme Event Response

**Problem**: Fourier-smoothed parameters may not respond adequately to rare, extreme events (e.g., 1-in-10,000-year floods).

The SLOP flood control zone releases water when storage is high:

```python
if head > h2_t:
    release = w_t + m2_t * (head - h2_t)  # Flood control slope
```

With Fourier parameterization, `m2` (flood control aggressiveness) and `h2` (threshold) vary smoothly across months. If an extreme flood hits during a month where:
- `m2` is low (conservative policy for that season)
- `h2` is high (expecting low inflows)

...the reservoir cannot empty quickly enough, risking dam overtopping.

**Why this matters**: Real dam safety is governed by the **Probable Maximum Flood (PMF)** or **Spillway Design Flood (SDF)** — events so rare they fall outside normal optimization. Smoothness objectives should never compromise safety.

#### Solutions for Safe Fourier Implementation

**1. Emergency Override (Recommended)**

Hard-coded safety rule that bypasses the optimized policy:

```python
def release_with_emergency(self, node: Storage, inflow: float, t: int, dt: float) -> float:
    # Normal SLOP release from Fourier-parameterized policy
    policy_release = self._fourier_slop_release(node, t, dt)

    # Emergency override — NOT optimized, fixed safety threshold
    head = volume_to_head(node.storage)
    if head > EMERGENCY_THRESHOLD:  # e.g., 95% of max head
        emergency_release = EMERGENCY_RATE * (head - EMERGENCY_THRESHOLD)
        return max(policy_release, emergency_release)

    return policy_release
```

This mirrors real reservoir operations where **spillway gates** operate independently of normal release rules.

**2. Constrain Fourier Coefficients**

Ensure flood response parameters never drop below safe minimums:

```python
# Bounds for m2 Fourier coefficients
"m2_a0": (1.5, 3.0),   # Mean slope always aggressive
"m2_a1": (-0.5, 0.5),  # Limited seasonal variation
"m2_b1": (-0.5, 0.5),  # Amplitude constrained
```

This guarantees `m2(month) >= m2_a0 - |m2_a1| - |m2_b1| >= 0.5` for all months.

**3. Hybrid Parameterization**

Use Fourier only for parameters governing normal operations:

| Parameter | Parameterization | Rationale |
|-----------|------------------|-----------|
| `w` (base release) | Fourier | Smooth seasonal operations |
| `h1` (conservation threshold) | Fourier | Seasonal water conservation |
| `h2` (flood threshold) | **Constant** | Safety-critical, no seasonal variation |
| `m1` (conservation slope) | Fourier | Smooth drawdown behavior |
| `m2` (flood slope) | **Constant or min-bounded** | Must respond to any extreme |

This reduces parameters from 30 to ~20 while maintaining safety.

**4. Worst-Case Objective**

Add an objective that evaluates extreme event response:

```python
def extreme_flood_response_objective(reservoir_id: str, flood_scenarios: list[float]) -> Objective:
    """Minimize time to evacuate storage under extreme inflows."""

    def evaluate(system: WaterSystem) -> float:
        worst_case_time = 0
        for flood_inflow in flood_scenarios:
            time_to_safe_level = simulate_flood_response(system, flood_inflow)
            worst_case_time = max(worst_case_time, time_to_safe_level)
        return worst_case_time

    return Objective(name="flood_response", direction="minimize", evaluate=evaluate)
```

#### Recommendation

For any real-world application of Fourier-smoothed policies:

1. **Always implement emergency overrides** — safety rules outside the optimization
2. **Use hybrid parameterization** — Fourier for normal ops, constants for safety-critical
3. **Validate against PMF/SDF** — ensure optimized policies survive design floods
4. **Document the separation** — make clear which rules are optimized vs. fixed

The Fourier approach is excellent for operational smoothness under normal conditions, but **dam safety must never be compromised for smoothness**.

### 3. Squared Differences (Penalize Large Swings More)

Replace absolute differences with squared:

```python
release_var = float(np.sum(np.diff(flows)**2))
```

This penalizes a single 20-unit swing more than two 10-unit swings (400 vs 200), discouraging occasional large changes.

### 4. Acceleration Penalty (Second Derivative)

Penalize changes in the *rate of change*:

```python
acceleration = np.diff(flows, n=2)  # Second difference
accel_penalty = np.sum(np.abs(acceleration))
```

This allows gradual trends but penalizes sudden reversals (e.g., release going up-down-up).

### 5. Hard Constraints on Rate of Change

Add maximum allowable change per timestep:

```python
max_change = 0.2 * np.mean(flows)  # 20% of mean
violations = np.maximum(0, np.abs(np.diff(flows)) - max_change)
constraint_penalty = violations.sum() * 1000  # Large penalty
```

This creates "forbidden zones" rather than soft penalties.

### Comparison of Approaches

| Approach | Pros | Cons |
|----------|------|------|
| Output penalty (current) | Simple, flexible | Still allows erratic policies |
| Parameter penalty | Targets root cause | Adds objective |
| **Fourier basis** | **Structural smoothness, smaller search** | **Requires strategy refactor** |
| Squared differences | Penalizes extremes | May over-penalize |
| Acceleration | Allows trends | More complex |
| Hard constraints | Guaranteed limits | May be infeasible |

### Recommendation

For a future iteration, **Fourier basis parameterization** offers the best combination of:
- Guaranteed smooth seasonal patterns
- Reduced optimization dimensionality (72 → 30)
- Physical interpretability of parameters
- No need for smoothness objectives

The trade-off is implementation complexity — requires refactoring the strategy classes.

------------------------------------------------------------------------

## Open Questions

1. **Allocation weight**: Is 1.0 the right balance, or should allocation smoothness be weighted differently?
2. **Scaling factor**: Is ×100 appropriate for making fraction changes comparable to flow changes?
3. **Alternative metrics**: Would variance-based metrics behave differently than sum-of-absolute-differences?
4. **Comparison study**: How do the Pareto fronts differ between 4-obj and 5-obj versions?
5. **Fourier harmonics**: Would 1 harmonic (3 params) suffice, or are 2 harmonics (5 params) needed for realistic patterns?
6. **Hybrid approach**: Could Fourier be used for some parameters (e.g., `w`) while keeping monthly resolution for others (e.g., thresholds)?
7. **Emergency threshold calibration**: What storage level should trigger emergency overrides? How does this interact with optimized policies?
8. **Robustness testing**: How should optimized policies be validated against PMF/SDF scenarios that weren't in the training data?
