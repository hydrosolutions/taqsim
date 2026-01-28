# Toy System Analysis: Reservoir Optimization Showcase

## Purpose

Document the current toy example system and identify logical, structured extensions for future development.

---

## Current System Overview

### Topology

```
River → Reservoir → Turbine → City → Splitter → Irrigation → Sink
                                          ↓
                                    Thermal Plant → Sink
```

### Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| Simulation | Timesteps | 120 (10 years × 12 months) |
| Reservoir | Capacity | 150 units |
| Reservoir | Dead storage | 10 units |
| Reservoir | Initial storage | 75 units |
| Turbine | Capacity | 60 units |
| City | Capacity | 41 units (flood bottleneck) |
| Irrigation | Consumption | 100% (fully consumptive) |
| Thermal | Consumption | 0% (non-consumptive) |

### Optimization Parameters (72 total)

| Strategy | Parameters | Count |
|----------|------------|-------|
| SLOP Release Rule | h1, h2, w, m1, m2 × 12 months | 60 |
| Seasonal Split Ratio | irrigation_fraction × 12 months | 12 |

### Objectives (4 total)

| Objective | Direction | Description |
|-----------|-----------|-------------|
| Hydropower | Maximize | Total electricity generated (GWh) |
| City Flood | Minimize | Total spillage at city node |
| Irrigation Deficit | Minimize | Unmet irrigation demand |
| Thermal Deficit | Minimize | Unmet cooling water demand |

---

## Trade-off Analysis

### Direct Trade-offs (Clear Pareto Fronts)

| Pair | Mechanism | Visualization |
|------|-----------|---------------|
| Hydropower ↔ City Flood | Same water release causes both power and flooding | Clear front |
| Irrigation ↔ Thermal | Splitter allocates between them | Clear front |

### Indirect Relationships (Scattered Clouds)

| Pair | Mechanism | Visualization |
|------|-----------|---------------|
| Hydropower ↔ Irrigation | Both benefit from release, but compete via splitter | Cloud |
| Hydropower ↔ Thermal | Connected through system dynamics | Cloud |

### Key Insight

The parallel coordinates plot shows a characteristic "bow-tie" pattern at the irrigation/thermal crossing, clearly demonstrating the allocation competition mediated by the splitter.

---

## Strengths of Current System

1. **Pedagogically effective**: Clear direct trade-offs alongside indirect relationships
2. **Physically intuitive**: Students can understand why more power = more flooding
3. **Appropriate complexity**: 72 parameters and 4 objectives hit a sweet spot
4. **Good visualization**: All trade-offs can be shown in 2D projections and parallel coordinates
5. **Realistic structure**: Mirrors real-world reservoir operation decisions

---

## Proposed Extensions

### Tier 1: Minimal Changes (Keep 4 objectives, same topology)

| Extension | Implementation | Effect on Trade-offs |
|-----------|----------------|---------------------|
| **Tighten city capacity** | Change from 41 to 35 units | Sharper hydropower ↔ flood front |
| **Add reservoir evaporation** | Implement storage loss rule | Storage decisions more consequential |
| **Asymmetric demand scaling** | Adjust irrigation/thermal peak ratios | Change shape of irrigation ↔ thermal front |

### Tier 2: Add Constraints (Same objectives, new constraints)

| Extension | Implementation | Effect on Trade-offs |
|-----------|----------------|---------------------|
| **Minimum environmental flow** | Add constraint: city flow ≥ 15 units | Creates floor on release; limits both conservation and deficit strategies |
| **Maximum drawdown rate** | Limit month-to-month storage change | Smoother release patterns; may reduce optimal hydropower |
| **Seasonal storage targets** | Require storage ≥ X in month Y | Forces anticipatory operations |

### Tier 3: Add Objectives (Expand to 5-6 objectives)

| Extension | New Objective | Effect |
|-----------|---------------|--------|
| **Environmental flow deficit** | Minimize months below min flow | Explicit ecology objective |
| **Revenue maximization** | Maximize: power_price × generation | Economic framing for decision-makers |
| **Reliability** | Maximize: % months meeting demands | Risk-based objective |
| **Storage stability** | Minimize storage variance | Operational consistency |

### Tier 4: Topology Extensions

| Extension | Change | Complexity Impact |
|-----------|--------|-------------------|
| **Return flow from thermal** | Add edge: thermal → downstream node | Shows non-consumptive nature explicitly |
| **Groundwater recharge** | Add sink with recharge from irrigation | Irrigation becomes partially non-consumptive |
| **Downstream reservoir** | Add second storage node | Multi-reservoir coordination; doubles parameters |
| **Parallel demand branches** | Split city into residential + industrial | More complex allocation |

---

## Recommended Next Steps

### For Teaching/Demonstration

1. Keep current system as-is (it works well)
2. Add minimum environmental flow constraint (Tier 2)
3. Frame results in economic terms (convert to $/year)

### For Research/Publication

1. Implement Tier 2 constraints
2. Add 1-2 Tier 3 objectives (reliability, revenue)
3. Compare with real-world case study data

### For Library Showcase

1. Implement all Tier 1-3 extensions as optional configurations
2. Create comparison notebook showing effect of each extension
3. Add stochastic inflow scenarios for robustness analysis

---

## Implementation Priority Matrix

| Extension | Effort | Impact | Priority |
|-----------|--------|--------|----------|
| Minimum environmental flow | Low | High | **P1** |
| Tighten city capacity | Trivial | Medium | P2 |
| Revenue objective | Low | High | **P1** |
| Reservoir evaporation | Low | Medium | P2 |
| Return flow from thermal | Medium | Low | P3 |
| Downstream reservoir | High | High | P3 |

---

## Open Questions

1. Should environmental flow be a hard constraint or a soft objective?
2. What's the right balance between realism and pedagogical clarity?
3. Should we add stochastic scenarios or keep deterministic for simplicity?
4. How many objectives can parallel coordinates effectively display? (Current: 4, Max useful: ~6-7)
