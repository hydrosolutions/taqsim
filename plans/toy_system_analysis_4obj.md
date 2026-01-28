# Toy System Analysis: Reservoir Optimization Showcase

## Purpose

Document the current toy example system and identify logical, structured extensions for future development.

------------------------------------------------------------------------

## Current System Overview

### Topology

```         
River → Reservoir → Turbine → City → Splitter → Irrigation → Sink
                                          ↓
                                    Thermal Plant → Sink
```

### Configuration

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

### Optimization Parameters (72 total)

| Strategy             | Parameters                      | Count |
|----------------------|---------------------------------|-------|
| SLOP Release Rule    | h1, h2, w, m1, m2 × 12 months   | 60    |
| Seasonal Split Ratio | irrigation_fraction × 12 months | 12    |

### Objectives (4 total)

| Objective          | Direction | Description                       |
|--------------------|-----------|-----------------------------------|
| Hydropower         | Maximize  | Total electricity generated (GWh) |
| City Flood         | Minimize  | Total spillage at city node       |
| Irrigation Deficit | Minimize  | Unmet irrigation demand           |
| Thermal Deficit    | Minimize  | Unmet cooling water demand        |

------------------------------------------------------------------------

## Trade-off Analysis

### Direct Trade-offs (Clear Pareto Fronts)

| Pair | Mechanism | Visualization |
|-----------------|-----------------------|--------------------------------|
| Hydropower ↔ City Flood | Same water release causes both power and flooding | Clear front |
| Irrigation ↔ Thermal | Splitter allocates between them | Clear front |

### Indirect Relationships (Scattered Clouds)

| Pair | Mechanism | Visualization |
|-----------------|-----------------------|--------------------------------|
| Hydropower ↔ Irrigation | Both benefit from release, but compete via splitter | Cloud |
| Hydropower ↔ Thermal | Connected through system dynamics | Cloud |

### Key Insight

The parallel coordinates plot shows a characteristic "bow-tie" pattern at the irrigation/thermal crossing, clearly demonstrating the allocation competition mediated by the splitter.

------------------------------------------------------------------------

## Strengths of Current System

1.  **Pedagogically effective**: Clear direct trade-offs alongside indirect relationships
2.  **Physically intuitive**: Students can understand why more power = more flooding
3.  **Appropriate complexity**: 72 parameters and 4 objectives hit a sweet spot
4.  **Good visualization**: All trade-offs can be shown in 2D projections and parallel coordinates
5.  **Realistic structure**: Mirrors real-world reservoir operation decisions

------------------------------------------------------------------------

## Proposed Extensions

### Tier 1: Minimal Changes (Keep 4 objectives, same topology)

| Extension | Implementation | Effect on Trade-offs |
|------------------|------------------------|-------------------------------|
| **Tighten city capacity** | Change from 41 to 35 units | Sharper hydropower ↔ flood front |
| **Add reservoir evaporation** | Implement storage loss rule | Storage decisions more consequential |
| **Asymmetric demand scaling** | Adjust irrigation/thermal peak ratios | Change shape of irrigation ↔ thermal front |

### Tier 2: Add Constraints (Same objectives, new constraints)

| Extension | Implementation | Effect on Trade-offs |
|------------------|------------------------|-------------------------------|
| **Minimum environmental flow** | Add constraint: city flow ≥ 15 units | Creates floor on release; limits both conservation and deficit strategies |
| **Maximum drawdown rate** | Limit month-to-month storage change | Smoother release patterns; may reduce optimal hydropower |
| **Seasonal storage targets** | Require storage ≥ X in month Y | Forces anticipatory operations |

### Tier 3: Add Objectives (Expand to 5-6 objectives)

| Extension | New Objective | Effect |
|-----------------------|-------------------------------|------------------|
| **Environmental flow deficit** | Minimize months below min flow | Explicit ecology objective |
| **Revenue maximization** | Maximize: power_price × generation | Economic framing for decision-makers |
| **Reliability** | Maximize: % months meeting demands | Risk-based objective |
| **Storage stability** | Minimize storage variance | Operational consistency |

### Tier 4: Topology Extensions

| Extension | Change | Complexity Impact |
|--------------------|------------------|-----------------------------------|
| **Return flow from thermal** | Add edge: thermal → downstream node | Shows non-consumptive nature explicitly |
| **Groundwater recharge** | Add sink with recharge from irrigation | Irrigation becomes partially non-consumptive |
| **Downstream reservoir** | Add second storage node | Multi-reservoir coordination; doubles parameters |
| **Parallel demand branches** | Split city into residential + industrial | More complex allocation |

------------------------------------------------------------------------

## Recommended Next Steps

### For Teaching/Demonstration

1.  Keep current system as-is (it works well)
2.  Add minimum environmental flow constraint (Tier 2)
3.  Frame results in economic terms (convert to \$/year)

### For Research/Publication

1.  Implement Tier 2 constraints
2.  Add 1-2 Tier 3 objectives (reliability, revenue)
3.  Compare with real-world case study data

### For Library Showcase

1.  Implement all Tier 1-3 extensions as optional configurations
2.  Create comparison notebook showing effect of each extension
3.  Add stochastic inflow scenarios for robustness analysis

------------------------------------------------------------------------

## Implementation Priority Matrix

| Extension                  | Effort  | Impact | Priority |
|----------------------------|---------|--------|----------|
| **Rate-of-change constraint** | Low     | High   | **P0**   |
| Minimum environmental flow | Low     | High   | **P1**   |
| Tighten city capacity      | Trivial | Medium | P2       |
| Revenue objective          | Low     | High   | **P1**   |
| Reservoir evaporation      | Low     | Medium | P2       |
| Return flow from thermal   | Medium  | Low    | P3       |
| Downstream reservoir       | High    | High   | P3       |

**Note**: Rate-of-change constraint elevated to P0 because current solutions exhibit operationally unrealistic variability that undermines the pedagogical value of the results.

------------------------------------------------------------------------

## Design Decisions

### Thermal Plant Constraint: Soft vs. Hard

**Question**: Should the thermal power plant cooling water requirement be a hard constraint (violation = irreversible plant damage) rather than a soft objective?

**Decision**: Keep as soft objective.

**Rationale**:

| Consideration | Analysis |
|---------------|----------|
| **Trade-off preservation** | The irrigation ↔ thermal allocation via the splitter creates the "bow-tie" pattern in parallel coordinates. A hard constraint would collapse this trade-off (thermal gets minimum first, irrigation gets remainder). |
| **Complexity budget** | Hard constraints require explaining feasibility rules, penalty methods, or ε-constraint handling in multi-objective optimization. Adds conceptual overhead without proportional pedagogical payoff. |
| **Physical realism** | Real thermal plants have safety shutdowns before catastrophic failure. Binary "destroyed" state is less realistic than gradual deficit costs. |
| **Visualization** | 4D Pareto front with soft objectives is easier to interpret than 3D front + feasibility boundary. |

**Alternative**: If a hard constraint is needed for teaching purposes, use **minimum environmental flow** instead. It's physically intuitive, doesn't eliminate existing trade-offs, and adds new tension rather than removing one.

------------------------------------------------------------------------

### Operational Variability in Pareto Solutions

**Observation**: Time series plots of Pareto-optimal solutions reveal highly erratic month-to-month variability in releases, storage, and downstream allocations. Solutions swing between extremes (e.g., turbine flow oscillating between 10 and 60 units in consecutive months).

**Question**: Is this behavior realistic, and should it be addressed?

#### Root Cause Analysis

The high variability stems from the optimizer exploiting unconstrained degrees of freedom:

| Factor | Mechanism | Contribution |
|--------|-----------|--------------|
| **60 monthly SLOP parameters** | Each month has independent h1, h2, w, m1, m2 values | Allows completely different release rules per month |
| **12 monthly split ratios** | Irrigation fraction optimized independently per month | Allocation can swing wildly between irrigation and thermal |
| **No smoothing penalty** | Objectives measure totals/sums, not variability | Optimizer indifferent between steady and erratic paths to same total |
| **No rate-of-change constraints** | Storage and release can change arbitrarily between timesteps | Physically possible but operationally unrealistic |

The optimizer finds that aggressive, variable strategies can exploit system dynamics better than smooth ones. For example, releasing heavily when inflow is high and conserving when low may yield better total hydropower than a steady release - even if the pattern looks chaotic.

#### Why This Matters

**Operationally unrealistic**: Real reservoir operators avoid erratic releases because:

- Downstream users (cities, farms) cannot adapt to wild swings
- Infrastructure stress from rapid flow changes
- Alternating flood/drought conditions harm riparian ecosystems
- Communication and coordination overhead with stakeholders
- Regulatory constraints often mandate gradual changes

**Pedagogically confusing**: Students may question whether "optimal" solutions that look chaotic are actually useful. This is actually a valuable teaching moment about the gap between mathematical optimality and operational feasibility.

**Masks true trade-offs**: The erratic behavior may obscure the fundamental trade-offs we want to demonstrate. Smoother solutions might reveal cleaner Pareto fronts.

#### Potential Solutions

Three approaches can address operational variability, each with different complexity and trade-off implications:

##### Option A: Hard Constraint on Rate of Change

**Implementation**: Add constraint limiting month-to-month storage or release change.

```
|S(t) - S(t-1)| ≤ ΔS_max    (storage drawdown limit)
|R(t) - R(t-1)| ≤ ΔR_max    (release change limit)
```

| Aspect | Analysis |
|--------|----------|
| **Pros** | Simple to implement; directly enforces smoothness; mirrors real operational rules |
| **Cons** | Reduces feasible region; may cut off some Pareto-optimal solutions; requires choosing threshold values |
| **Threshold selection** | Could be based on physical limits (e.g., max valve adjustment rate) or policy (e.g., ±20% change allowed) |
| **Effect on trade-offs** | May shift entire Pareto front toward less extreme solutions; could reveal cleaner trade-off structure |

##### Option B: Add Smoothness as a 5th Objective

**Implementation**: Add objective to minimize operational variability.

```
Minimize: Σ|R(t) - R(t-1)|    (total release variation)
   or
Minimize: Var(R)              (release variance)
   or
Minimize: Σ|S(t) - S(t-1)|    (total storage swing)
```

| Aspect | Analysis |
|--------|----------|
| **Pros** | Preserves all feasible solutions; lets decision-maker choose smoothness level; explicit trade-off between performance and operability |
| **Cons** | Increases dimensionality (5 objectives harder to visualize); adds complexity for students |
| **Visualization** | Parallel coordinates can handle 5 objectives; would show smoothness trading against other goals |
| **Pedagogical value** | Demonstrates that "operability" is itself an objective in real systems |

##### Option C: Reduce Parameter Freedom

**Implementation**: Use fewer, shared parameters across months instead of fully independent monthly values.

| Current | Alternative | Parameter reduction |
|---------|-------------|---------------------|
| 12 monthly SLOP rules (60 params) | 4 seasonal SLOP rules (20 params) | 67% reduction |
| 12 monthly SLOP rules (60 params) | 1 annual SLOP rule (5 params) | 92% reduction |
| 12 monthly split ratios | 4 seasonal split ratios | 67% reduction |
| 12 monthly split ratios | 1 annual split ratio | 92% reduction |

| Aspect | Analysis |
|--------|----------|
| **Pros** | Inherently smoother solutions; faster optimization (fewer dimensions); simpler to explain |
| **Cons** | May sacrifice significant performance; reduces optimizer's ability to adapt to seasonal patterns |
| **Middle ground** | Seasonal parameters (4 seasons × 5 SLOP = 20 params) balance flexibility and smoothness |
| **Pedagogical value** | Shows impact of model structure on solution character |

##### Option D: Penalty Function Approach

**Implementation**: Add smoothness penalty to existing objectives rather than as separate objective.

```
Hydropower_adjusted = Hydropower - λ × Σ|R(t) - R(t-1)|
```

| Aspect | Analysis |
|--------|----------|
| **Pros** | Keeps 4 objectives; implicitly trades smoothness against performance |
| **Cons** | Requires tuning penalty weight λ; obscures the trade-off (not visible in Pareto front); conflates distinct concerns |
| **When appropriate** | When smoothness preference is fixed and known, not subject to decision-maker choice |

#### Recommendation

For the toy system's goal of **minimal complexity with interesting dynamics**:

**Primary recommendation**: **Option A (hard constraint)** with a moderate threshold.

- Simplest to implement and explain
- Directly mirrors real operational rules
- Keeps 4 objectives and existing trade-off structure
- Removes pathological solutions without hiding trade-offs

**Secondary recommendation**: If variability trade-offs are pedagogically important, use **Option B (5th objective)** to explicitly show that smooth operation has a cost.

**Avoid**: Option D (penalty function) obscures rather than reveals trade-offs, which contradicts the pedagogical goals.

#### Suggested Constraint Values

Based on the current system parameters:

| Constraint | Conservative | Moderate | Permissive |
|------------|--------------|----------|------------|
| Storage change `\|ΔS\|` | ≤ 15 units/month | ≤ 25 units/month | ≤ 40 units/month |
| Release change `\|ΔR\|` | ≤ 10 units/month | ≤ 20 units/month | ≤ 30 units/month |

These should be calibrated against the natural inflow variability and demand patterns to ensure feasibility while enforcing meaningful smoothness.

------------------------------------------------------------------------

## Open Questions

1.  ~~Should environmental flow be a hard constraint or a soft objective?~~ → Soft objective preferred for thermal; env. flow is better hard constraint candidate if needed
2.  What's the right balance between realism and pedagogical clarity?
3.  Should we add stochastic scenarios or keep deterministic for simplicity?
4.  How many objectives can parallel coordinates effectively display? (Current: 4, Max useful: \~6-7)