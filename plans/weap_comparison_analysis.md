# TaqSim vs WEAP: Feature Comparison and Gap Analysis

27 January 2026, Tobias and Claude.

## Executive Summary

This document analyzes TaqSim's current capabilities against WEAP (Water Evaluation And Planning System). The analysis distinguishes between two modeling domains:

1.  **Water Systems Modeling** - Infrastructure and allocation (TaqSim's focus)
2.  **Hydrological Modeling** - Natural processes and runoff generation (outside TaqSim's scope)

TaqSim is a **water systems model** that simulates water allocation, storage operations, and infrastructure management. It assumes inflows are provided externally (e.g., from a separate hydrological model or observed data). WEAP combines both domains in a single tool.

**Key Findings:**

-   TaqSim excels in multi-objective optimization with NSGA-II and strategy-based parameter tuning
-   For water systems modeling, critical gaps are: demand priorities, water quality, groundwater supply
-   Hydrological modeling (catchments, snow, glaciers) is outside TaqSim's current scope
-   TaqSim's Protocol-based architecture enables clean extension for water systems features

------------------------------------------------------------------------

## 1. Current TaqSim Architecture

TaqSim is a water systems simulation and optimization framework with the following components:

### 1.1 Node Types

| Node Type | Purpose | Protocols Implemented |
|-----------------|--------------------------------|-----------------------|
| **Source** | Generates water from TimeSeries inflow | `Generates` |
| **Sink** | Terminal node, accepts all water | `Receives` |
| **PassThrough** | Routes water with optional capacity | `Receives` |
| **Storage** | Reservoir with release/loss rules | `Receives`, `Stores`, `Loses` |
| **Demand** | Consumes water with efficiency/return flow | `Receives`, `Consumes` |
| **Splitter** | Distributes water to multiple targets | `Receives` |

### 1.2 Edge System

-   Single `Edge` class with pluggable `EdgeLossRule` protocol
-   Capacity constraints with spillage handling
-   Loss mechanisms: evaporation, seepage, inefficiency, capacity_exceeded

### 1.3 Strategy System

-   `Strategy` mixin for optimizable operational policies
-   `ReleaseRule`, `SplitRule`, `LossRule` protocols
-   Time-varying and cyclical parameter support
-   Built-in constraints: `SumToOne`, `Ordered`

### 1.4 Optimization Capabilities

-   NSGA-II multi-objective optimization via `ctrl_freak`
-   Parameter vectorization with hierarchical paths
-   Constraint repair during GA operations
-   `Trace` class for objective extraction with union semantics
-   Immutable system cloning for parallel evaluation

------------------------------------------------------------------------

## 2. WEAP Water Systems Features

This section covers WEAP features relevant to water systems modeling (infrastructure, allocation, operations).

### 2.1 Demand Analysis

-   **Demand Sites** with activity levels and water use rates
-   **Demand disaggregation** (sectors, sub-sectors)
-   **Demand priorities** (1-99 scale) for allocation
-   **Losses and reuse** modeling
-   **Demand-side management** scenarios

### 2.2 Supply Infrastructure

-   **Rivers and reaches** with routing
-   **Reservoirs** with operating rules (Top of Conservation, Buffer, Inactive zones)
-   **Groundwater** resources and aquifers (as supply sources)
-   **Transmission links** with supply preferences
-   **Return flows** with routing to specific nodes
-   **Flow requirements** (environmental flows)

### 2.3 Reservoirs and Hydropower

-   Multi-zone reservoir operation (conservation, buffer, inactive)
-   Volume-elevation curves
-   Hydropower generation with turbine efficiency curves
-   Run-of-river power plants

### 2.4 Water Quality

-   Constituent tracking (BOD, DO, TSS, temperature, nutrients)
-   First-order decay modeling
-   Pollution loading from demand sites
-   Wastewater treatment plants (WWTP)
-   QUAL2K integration for detailed river quality

### 2.5 Groundwater (as Supply Source)

-   Aquifer storage and yield
-   Pumping and recharge
-   Surface water-groundwater interaction

### 2.6 Financial Analysis

-   Capital costs and O&M costs
-   Benefits modeling
-   Net Present Value (NPV) calculations
-   Cost-benefit analysis for scenarios

### 2.7 Scenarios

-   Reference scenario as baseline
-   Multiple alternative scenarios
-   Water Year Method for climate variability
-   Scenario comparison and evaluation

------------------------------------------------------------------------

## 3. Water Systems Feature Comparison

| Feature Category          | WEAP    | TaqSim               | Gap Level     |
|---------------------------|---------|----------------------|---------------|
| **Demand Modeling**       |         |                      |               |
| Basic demand sites        | Yes     | Yes                  | None          |
| Demand disaggregation     | Yes     | No                   | Medium        |
| Demand priorities (1-99)  | Yes     | No                   | **Critical**  |
| Efficiency/losses         | Yes     | Yes                  | None          |
| Return flows              | Yes     | Partial              | Low           |
| **Supply Infrastructure** |         |                      |               |
| River reaches             | Yes     | Yes (Source)         | None          |
| Reservoirs                | Yes     | Yes (Storage)        | None          |
| Multi-zone operation      | Yes     | Partial              | Medium        |
| Groundwater supply        | Yes     | No                   | **Critical**  |
| Flow requirements         | Yes     | No                   | Medium        |
| **Routing**               |         |                      |               |
| Network topology          | DAG     | DAG                  | None          |
| Transmission links        | Yes     | Yes (Edge)           | None          |
| Link preferences          | Yes     | No                   | Medium        |
| **Reservoirs**            |         |                      |               |
| Storage capacity          | Yes     | Yes                  | None          |
| Operating rules           | Yes     | Yes (ReleaseRule)    | None          |
| Dead storage              | Yes     | Yes                  | None          |
| Zone-based operation      | Yes     | Partial              | Medium        |
| Evaporation/seepage       | Yes     | Yes (LossRule)       | None          |
| **Hydropower**            |         |                      |               |
| Power calculation         | Yes     | Custom               | None          |
| Volume-head curve         | Yes     | Custom               | None          |
| Turbine efficiency        | Yes     | Custom               | None          |
| Run-of-river              | Yes     | Via PassThrough      | None          |
| **Water Quality**         |         |                      |               |
| Constituent tracking      | Yes     | No                   | **Critical**  |
| Decay modeling            | Yes     | No                   | **Critical**  |
| Pollution loading         | Yes     | No                   | **Critical**  |
| WWTP modeling             | Yes     | No                   | **Critical**  |
| Temperature               | Yes     | No                   | **Critical**  |
| **Groundwater Supply**    |         |                      |               |
| Aquifer storage           | Yes     | No                   | **Critical**  |
| GW-SW interaction         | Yes     | No                   | **Critical**  |
| Pumping                   | Yes     | No                   | **Critical**  |
| Recharge                  | Yes     | No                   | Medium        |
| **Financial**             |         |                      |               |
| Cost modeling             | Yes     | No                   | Medium        |
| Benefit modeling          | Yes     | No                   | Medium        |
| NPV calculation           | Yes     | No                   | Medium        |
| **Optimization**          |         |                      |               |
| Multi-objective           | Limited | **Yes (NSGA-II)**    | TaqSim Better |
| Parameter tuning          | Manual  | **Automated**        | TaqSim Better |
| Pareto fronts             | No      | **Yes**              | TaqSim Better |
| Constraints               | Manual  | **Automated repair** | TaqSim Better |
| **Time Series**           |         |                      |               |
| External data             | Yes     | Yes                  | None          |
| Time-varying params       | Yes     | Yes                  | None          |
| Cyclical patterns         | Yes     | Yes                  | None          |

**Legend:** None = Equivalent, Low = Minor enhancement, Medium = Significant work, **Critical** = Major gap, TaqSim Better = TaqSim exceeds WEAP

------------------------------------------------------------------------

## 4. Gap Analysis: Water Systems Features

### 4.1 Demand Priority System

**WEAP Capability:** - Priorities 1 (highest) to 99 (lowest) - Allocation follows priority during scarcity - Multiple demand sites competing for limited supply

**TaqSim Gap:** - No priority-based allocation - All demands receive water equally (first-come basis via topology) - No mechanism for preferential allocation

**Impact:** Cannot model real-world water rights and allocation policies

### 4.2 Water Quality Modeling

**WEAP Capability:** - Track multiple constituents through network - First-order decay in river reaches - Pollution generation at demand sites - WWTP removal efficiency - Temperature modeling - QUAL2K integration for detailed biochemistry

**TaqSim Gap:** - No constituent tracking - No water quality state variables - No decay/reaction modeling - No pollution loading

**Impact:** Cannot assess water quality impacts of management decisions

### 4.3 Groundwater as Supply Source

**WEAP Capability:** - Aquifer storage and yield - Pumping and recharge - Surface water-groundwater interaction

**TaqSim Gap:** - No groundwater representation - No aquifer storage - No SW-GW exchange

**Impact:** Cannot model conjunctive use or aquifer management

------------------------------------------------------------------------

## 5. Implementation Plans: Water Systems Features

### 5.1 Demand Priority System

**Objective:** Enable priority-based water allocation during scarcity

**Design:**

``` python
# New protocol
@runtime_checkable
class HasPriority(Protocol):
    @property
    def priority(self) -> int: ...  # 1-99, lower = higher priority

# Extended Demand node
@dataclass
class Demand(BaseNode):
    requirement: TimeSeries
    priority: int = 50  # Default middle priority
    consumption_fraction: float = 1.0
    efficiency: float = 1.0
```

**Implementation Steps:**

1.  **Add priority attribute to Demand** (1 day)
    -   File: `src/taqsim/node/demand.py`
    -   Add `priority: int = 50` field
    -   Validate range 1-99 in `__post_init__`
2.  **Create AllocationRule protocol** (2 days)
    -   File: `src/taqsim/allocation/rules.py` (new)
    -   Protocol: `AllocationRule.allocate(demands: list[Demand], available: float, t: int) -> dict[str, float]`
    -   Built-in: `PriorityAllocation`, `ProportionalAllocation`, `EqualAllocation`
3.  **Extend WaterSystem for multi-demand allocation** (3 days)
    -   File: `src/taqsim/system/water_system.py`
    -   Modify routing to collect demands before allocation
    -   Apply allocation rule at supply points (junctions before demands)
    -   Handle partial satisfaction and deficit recording
4.  **Add Junction node type** (2 days)
    -   File: `src/taqsim/node/junction.py` (new)
    -   Aggregates supply from multiple sources
    -   Distributes to multiple demands via allocation rule
    -   Replaces Splitter for demand allocation scenarios
5.  **Tests and documentation** (2 days)
    -   Priority allocation tests
    -   Deficit allocation scenarios
    -   Example notebook

**Total Effort:** 10 days

**Files to Create:** - `src/taqsim/allocation/__init__.py` - `src/taqsim/allocation/rules.py` - `src/taqsim/node/junction.py` - `tests/test_allocation/`

------------------------------------------------------------------------

### 5.2 Water Quality Module

**Objective:** Track constituent concentrations through network

**Design:**

``` python
# Constituent definition
@dataclass(frozen=True)
class Constituent:
    name: str
    decay_rate: float = 0.0  # 1/day, first-order
    settling_rate: float = 0.0  # m/day
    temperature_coefficient: float = 1.0

# Quality state
@dataclass
class WaterQuality:
    concentrations: dict[str, float]  # constituent_name -> mg/L
    temperature: float = 20.0  # Celsius

    def mix_with(self, other: "WaterQuality", self_volume: float, other_volume: float) -> "WaterQuality": ...
    def decay(self, dt: float) -> "WaterQuality": ...

# Extended edge with quality transport
@dataclass
class QualityEdge(Edge):
    travel_time: float  # days

    def transport(self, quality: WaterQuality, flow: float, t: int, dt: float) -> WaterQuality: ...
```

**Implementation Steps:**

1.  **Create quality data structures** (2 days)
    -   File: `src/taqsim/quality/constituents.py`
    -   `Constituent` dataclass with decay/settling parameters
    -   `WaterQuality` dataclass with concentration dict
    -   Mixing and decay methods
2.  **Create QualityEdge class** (3 days)
    -   File: `src/taqsim/quality/edge.py`
    -   Extend Edge with travel time
    -   Implement advection-dispersion (simplified)
    -   First-order decay during transport
3.  **Add pollution loading to Demand** (2 days)
    -   File: `src/taqsim/node/demand.py` (extend)
    -   Add `pollution_load: dict[str, float]` attribute
    -   Return flow includes pollution from use
4.  **Create WastewaterTreatment node** (3 days)
    -   File: `src/taqsim/node/wastewater.py`
    -   Removal efficiency per constituent
    -   Effluent quality calculation
5.  **Extend Storage for quality** (2 days)
    -   File: `src/taqsim/node/storage.py` (extend)
    -   Complete mixing model
    -   Settling and decay in reservoir
6.  **Quality-aware WaterSystem** (4 days)
    -   File: `src/taqsim/system/water_system.py` (extend)
    -   Track quality through network
    -   Mixing at junctions
    -   Quality events for analysis
7.  **Quality objectives** (2 days)
    -   File: `src/taqsim/objective/quality.py`
    -   Objective factories: `concentration_violation`, `load_exceedance`
8.  **Tests and documentation** (3 days)
    -   Quality transport tests
    -   Decay tests
    -   Mixing scenarios
    -   Example notebook

**Total Effort:** 21 days

**Files to Create:** - `src/taqsim/quality/__init__.py` - `src/taqsim/quality/constituents.py` - `src/taqsim/quality/edge.py` - `src/taqsim/node/wastewater.py` - `src/taqsim/objective/quality.py` - `tests/test_quality/`

------------------------------------------------------------------------

### 5.3 Groundwater Supply Module

**Objective:** Model aquifer as a supply source with pumping and SW-GW interaction

**Design:**

``` python
@dataclass
class Aquifer(BaseNode):
    storage_coefficient: float  # dimensionless
    hydraulic_conductivity: float  # m/day
    area: float  # km²
    thickness: float  # m
    initial_head: float  # m

    # Boundaries
    connected_reaches: list[str]  # Node IDs for SW-GW exchange
    pumping_wells: list[str]  # Demand node IDs

    _current_head: float = field(init=False)

    def recharge(self, amount: float, t: int) -> None: ...
    def pump(self, demand: float, t: int) -> float: ...
    def exchange_with_stream(self, stream_stage: float, t: int) -> float: ...
```

**Implementation Steps:**

1.  **Create Aquifer node** (4 days)
    -   File: `src/taqsim/node/aquifer.py`
    -   Single-cell lumped model
    -   Storage-head relationship
    -   Recharge and pumping
2.  **Surface water-groundwater exchange** (3 days)
    -   File: `src/taqsim/node/aquifer.py` (extend)
    -   Darcy's law based exchange
    -   Connected reach specification
    -   Gaining/losing stream logic
3.  **Pumping integration** (2 days)
    -   File: `src/taqsim/node/aquifer.py` (extend)
    -   Link to Demand nodes as wells
    -   Pumping capacity limits
    -   Head-dependent yield
4.  **Aquifer-specific objectives** (2 days)
    -   File: `src/taqsim/objective/groundwater.py`
    -   `head_decline`, `pumping_cost`, `sustainable_yield`
5.  **Tests and documentation** (2 days)
    -   Aquifer mass balance
    -   SW-GW exchange tests
    -   Example notebook

**Total Effort:** 13 days

**Files to Create:** - `src/taqsim/node/aquifer.py` - `src/taqsim/objective/groundwater.py` - `tests/test_groundwater/`

------------------------------------------------------------------------

### 5.4 Financial Analysis Module

**Objective:** Cost-benefit analysis for water management alternatives

**Design:**

``` python
@dataclass(frozen=True)
class CostModel:
    capital_cost: float  # $
    annual_om: float  # $/year
    variable_cost: float  # $/unit
    lifetime: int  # years
    discount_rate: float = 0.05

@dataclass(frozen=True)
class BenefitModel:
    unit_value: float  # $/unit delivered
    deficit_penalty: float  # $/unit deficit
    environmental_value: float  # $/unit environmental flow

def npv(costs: Trace, benefits: Trace, discount_rate: float) -> float: ...
```

**Implementation Steps:**

1.  **Create cost/benefit data structures** (2 days)
    -   File: `src/taqsim/financial/models.py`
    -   `CostModel`, `BenefitModel` dataclasses
    -   NPV calculation
2.  **Attach costs to nodes** (2 days)
    -   File: `src/taqsim/node/base.py` (extend)
    -   Optional `cost_model` attribute
    -   Variable cost based on throughput
3.  **Benefit calculation** (2 days)
    -   File: `src/taqsim/financial/benefits.py`
    -   Water delivery benefits
    -   Deficit penalties
    -   Environmental flow values
4.  **Financial objectives** (2 days)
    -   File: `src/taqsim/objective/financial.py`
    -   `total_npv`, `benefit_cost_ratio`, `levelized_cost`
5.  **Tests and documentation** (2 days)

**Total Effort:** 10 days

**Files to Create:** - `src/taqsim/financial/__init__.py` - `src/taqsim/financial/models.py` - `src/taqsim/financial/benefits.py` - `src/taqsim/objective/financial.py` - `tests/test_financial/`

------------------------------------------------------------------------

## 6. Implementation Priority (Water Systems)

Based on impact and effort for water systems features:

| Priority | Feature | Effort | Impact |
|-------------|-----------------|-------------|-----------------------------|
| 1 | Demand Priority System | 10 days | Critical - enables water rights modeling |
| 2 | Groundwater Supply | 13 days | Critical - enables conjunctive use |
| 3 | Water Quality Module | 21 days | Critical - environmental assessment |
| 4 | Financial Analysis | 10 days | Medium - cost-benefit comparison |

**Total Estimated Effort:** 54 days

**Recommended Approach:** 1. Start with Demand Priority System (foundation for allocation) 2. Groundwater Supply Module (extends supply options) 3. Water Quality Module (builds on node extensions) 4. Financial Analysis (builds on all above)

------------------------------------------------------------------------

## 7. Architecture Considerations

### 7.1 Maintaining Protocol-Based Design

All new features should follow TaqSim's protocol-based design:

``` python
# Good: Protocol for behavior
@runtime_checkable
class HasQuality(Protocol):
    @property
    def quality(self) -> WaterQuality: ...
    def update_quality(self, incoming: WaterQuality, volume: float) -> None: ...

# Good: Strategy for tunable behavior
@dataclass(frozen=True)
class AllocationRule(Strategy):
    __params__ = ("priority_weight",)
    __bounds__ = {"priority_weight": (0.0, 1.0)}
    priority_weight: float = 1.0
```

### 7.2 Event System Extensions

New events for extended features:

``` python
# Quality events
@dataclass(frozen=True, slots=True)
class QualityUpdated:
    concentrations: dict[str, float]
    t: int

# Allocation events
@dataclass(frozen=True, slots=True)
class AllocationDecision:
    demands: dict[str, float]  # demand_id -> allocated
    available: float
    t: int

# Groundwater events
@dataclass(frozen=True, slots=True)
class GroundwaterExchange:
    amount: float  # positive = gaining, negative = losing
    reach_id: str
    t: int
```

### 7.3 Backward Compatibility

-   All extensions should be optional
-   Existing nodes/edges work without modification
-   Quality tracking opt-in via `enable_quality=True`
-   Financial analysis opt-in via cost models

------------------------------------------------------------------------

## 8. Conclusion (Water Systems)

TaqSim provides a solid foundation for water systems modeling with its protocol-based architecture and advanced multi-objective optimization capabilities. The identified water systems gaps (priorities, groundwater supply, water quality, financial) are addressable through modular extensions that maintain existing design patterns.

**Immediate Value:** - TaqSim's optimization surpasses WEAP's manual scenario approach - Pareto front generation enables trade-off analysis impossible in WEAP - Strategy-based parameter tuning provides flexibility

**Development Roadmap (Water Systems):** - Priority system (Q1): Enables real-world allocation modeling - Groundwater + Quality (Q2): Achieves WEAP feature parity for most use cases - Financial (Q3): Enables cost-benefit comparison

------------------------------------------------------------------------

# Appendix: Hydrological Modeling (Outside TaqSim Scope)

This appendix documents WEAP's hydrological modeling capabilities, which are **outside TaqSim's current scope**. TaqSim assumes inflows are provided externally via `TimeSeries` inputs, typically from: - Observed streamflow data - External hydrological models (e.g., HBV, SWAT, VIC) - Climate scenario generators

## A.1 WEAP Hydrology Features

### A.1.1 Rainfall-Runoff Modeling

**Simplified Coefficient Method:** - Runoff = Precipitation × Runoff Coefficient - Simple, data-light approach

**Soil Moisture Method (2-bucket):** - Upper bucket: root zone with ET, runoff, interflow - Lower bucket: deep storage with baseflow - Parameters: soil water capacity, conductivity, runoff resistance

### A.1.2 Snow and Glacier Modeling

**Snow:** - Accumulation based on temperature threshold - Degree-day melt model - Energy balance approach (optional)

**Glaciers:** - Area-volume relationships - Glacier dynamics - Climate sensitivity analysis

### A.1.3 Evapotranspiration

-   Reference ET calculation (Hargreaves, Penman-Monteith)
-   Crop coefficients
-   Soil moisture stress factors

### A.1.4 Catchment Calibration

-   PEST integration for automated calibration
-   Visual and statistical assessment
-   Sensitivity analysis tools

### A.1.5 MODFLOW Integration

-   Detailed 3D groundwater simulation
-   WEAP-MODFLOW coupling for complex aquifer systems

## A.2 Hydrological Feature Comparison

| Feature              | WEAP | TaqSim | Notes                           |
|----------------------|------|--------|---------------------------------|
| Rainfall-runoff      | Yes  | No     | TaqSim uses external TimeSeries |
| Soil moisture method | Yes  | No     | Outside scope                   |
| ET calculation       | Yes  | No     | Outside scope                   |
| Interflow/baseflow   | Yes  | No     | Outside scope                   |
| Snow modeling        | Yes  | No     | Outside scope                   |
| Glacier dynamics     | Yes  | No     | Outside scope                   |
| PEST calibration     | Yes  | No     | Could integrate with TaqSim     |
| MODFLOW integration  | Yes  | No     | Outside scope                   |

## A.3 Future Consideration: Hydrological Module

If hydrological modeling becomes a requirement, a separate `taqsim.hydrology` module could be developed:

**Potential Design:**

``` python
@dataclass
class Catchment(BaseNode):
    """Hydrological catchment that generates inflows."""
    area: float  # km²
    climate: ClimateInput  # precip, temp, PET

    # Soil Moisture Method parameters
    soil_water_capacity: float  # mm
    root_zone_conductivity: float  # mm/day
    deep_conductivity: float  # mm/day

    # Snow parameters (optional)
    snow_threshold: float = 0.0  # °C
    melt_rate: float = 0.0  # mm/°C/day

    _upper_storage: float = 0.0
    _lower_storage: float = 0.0
    _snowpack: float = 0.0

    def generate(self, t: int, dt: float) -> float:
        """Generate runoff from climate forcing."""
        ...
```

**Estimated Effort:** 22 days (if implemented)

**Recommendation:** Keep hydrological modeling as an optional future extension. For most TaqSim use cases, external inflow time series are sufficient and more flexible (allowing coupling with any hydrological model).

------------------------------------------------------------------------

## A.4 Integration with External Hydrological Models

TaqSim's current design supports integration with external hydrological models:

``` python
# Example: Using HBV model output
hbv_output = pd.read_csv("hbv_runoff.csv")
inflow_ts = TimeSeries(values=hbv_output["discharge"].tolist())

river_source = Source(id="river", inflow=inflow_ts)
system.add_node(river_source)
```

This approach offers: - **Flexibility:** Use any hydrological model - **Modularity:** Keep hydrological and systems models separate - **Efficiency:** Run hydrology once, optimize allocation many times