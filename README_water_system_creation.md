# How-To Guide: Creating a Water System

This guide explains how to create and configure a water system for simulation and optimization using the classes and utilities provided in this project. It covers the main steps, node and edge types, configuration options, and provides code examples for each step. For futhrer details on the node types the reader is reffered to ['README_node_types.md](/README_node_types.md) or the source code ['water_system/nodes.py](/water_system/nodes.py).

>**Note:** The creation of the system explained below can be found in ['dummy_creator.py'](./dummy_creator.py) with input data in ['data/dummy_data/'](/data/dummy_data/).

---

## 1. Overview

A water system is modeled as a network of nodes (representing the different infrastructures, processes and functionalities within a catchment) connected by edges (rivers, canals, ...). Each node and edge can be customized to represent real-world infrastructure and water management rules.

#### Units Overview

| Quantity                | Unit         | Description                                      |
|-------------------------|--------------|--------------------------------------------------|
| Flow rate               | m³/s         | Water flow (e.g., supply, demand, outflow)       |
| Volume                  | m³           | Storage, reservoir volume, supply/demand totals  |
| Evaporation rate        | mm           | Monthly evaporation input for reservoirs         |
| Rainfall/Precipitation  | mm           | Monthly rainfall input for runoff nodes          |
| Area                    | km²          | Catchment area for runoff nodes                  |
| Water level/elevation   | m a.s.l.     | Reservoir water level (meters above sea level)   |
| Coordinates             | m (UTM)      | Easting/Northing for node locations              |
| Time step duration      | s            | Model time step (typically monthly, in seconds)  |

---

## 2. Basic Steps to Create a Water System

### Step 1: Import Required Classes

```python
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, Edge
from water_system import DeapOptimizer, WaterSystemVisualizer
from water_system.io_utils import load_optimized_parameters, load_parameters_from_file, save_optimized_parameters
```

---

### Step 2: Initialize the WaterSystem

You must specify the time step duration (`dt` in seconds), start year, and start month. The model is designed to work with a monthly timestep. The length of a timestep in seconds (dt) is used by the model in order to transform volumes to flows and vice versa. 

```python
dt = 30.44 * 24 * 3600  # Average month in seconds
my_water_system = WaterSystem(dt=dt, start_year=2020, start_month=1)
```

---

### Step 3: Add Nodes

Each node in the water system requires a unique `id` (string) and UTM coordinates (`easting`, `northing`).  
>**Note:** For a detailed description of all node types, their key parameters, and CSV file formats, see [README_node_types.md](README_node_types.md).


#### Node Types Overview

There are **six** node types available in the water system model, each with specific functionalities:

| Node Type      | Main Functionality                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------|
| **SupplyNode** | Represents water sources (e.g., river inflow). Provides inflow to the system, either as a constant or from a time series (CSV). |
| **StorageNode**| Models reservoirs. Stores water, manages releases, and can account for evaporation losses and dead storage. |
| **DemandNode** | Represents water users (agriculture, domestic, industrial). Can have consumptive and non-consumptive demands, priorities, and efficiency factors. |
| **SinkNode**   | Terminal points where water exits the system (e.g., river mouth, environmental flow). Can enforce minimum flow requirements. |
| **HydroWorks** | Represents distribution or control structures (e.g., diversions, confluences, splitters). Distributes inflow to multiple targets. |
| **RunoffNode** | Models distributed surface runoff from sub-catchments. Generates inflow based on rainfall data, catchment area, and runoff coefficient. |


### Supply Nodes

Every `WaterSystem` requires at least one `SupplyNode` or `RunoffNode`. The water supply rate in m³/s can either be defined as a constant supply rate or as a time-varying supply read from a CSV file.

Example for constant supply rate:
```python
supply1 = SupplyNode(
    id="Source1",
    easting=100,
    northing=600,
    constant_supply_rate=100,  # m³/s
    start_year=2020,
    start_month=1,
    num_time_steps=12
)
my_water_system.add_node(supply1)
```

Example for time-varying supply data from a csv file:
```python
supply2 = SupplyNode(
    id="Source2",
    easting=100,
    northing=200,
    csv_file='./data/dummy_data/supply_timeseries.csv',
    start_year=2020,
    start_month=1,
    num_time_steps=12
)
my_water_system.add_node(supply2)
```
**CSV Format:**
```csv
Date,Q
2017-01-01,20.2
2017-02-01,33.7
2017-03-01,58.0
```

---

### Runoff Nodes

RunoffNodes are used to represent distributed surface runoff from sub-catchments, allowing you to model inflow generated by rainfall over a defined area. The inflow can be calculated based on rainfall data, catchment area, and a runoff coefficient.

**Example: Runoff calculated from rainfall data**
```python
runoff = RunoffNode(
    id="SurfaceRunoff",
    easting=300,
    northing=500,
    area=50,  # km²
    runoff_coefficient=0.3,
    rainfall_csv="./data/dummy_data/rainfall_timeseries.csv", # in mm
    start_year=2020,
    start_month=1,
    num_time_steps=12
)
my_water_system.add_node(runoff)
```

**Rainfall CSV Format:**
```csv
Date,Precipitation
2020-01-01,32.5
2020-02-01,28.1
...
```
---

### Sink Nodes

Represents endpoints where water leaves the system (e.g., river mouth). Every `WaterSystem` requires at least one Sink Node and all paths within a system have to lead to a SinkNode. Sink Nodes can have a minimum flow requirement, either as a constant value or from a CSV file.

```python
# Constant minimum flow
sink1 = SinkNode(
    id="RiverMouth",
    easting=500,
    northing=200,
    constant_min_flow=10, # m³/s
    start_year=2020,
    start_month=1,
    num_time_steps=12
)
my_water_system.add_node(sink1)

# Time-varying minimum flow from CSV
sink2 = SinkNode(
    id="EnvFlow",
    easting=400,
    northing=400,
    csv_file='./data/dummy_data/sink_min_flow.csv',
    start_year=2020,
    start_month=1,
    num_time_steps=12
)
my_water_system.add_node(sink2)
```

---

### HydroWorks Nodes

Represents diversions, confluences, or other structures. These are the only nodes within a `WaterSystem` which are allowed to have more than one outgoing edge. Use HydroWorks nodes to split and distribute flows.

```python
hydrowork = HydroWorks(
    id="HydroWorks1",
    easting=300,
    northing=300
)
my_water_system.add_node(hydrowork)
```

---

### Demand Nodes

Represents water users (agriculture, domestic, industrial). Demand Nodes can have a constant demand rate or a time-varying demand rate read from a CSV file. The `non_consumptive_rate` represents the part of the water which is not consumed and returned to the system. The `priority` parameter is only used during optimization (See [README_optimization.md](/README_optimization.md))
 
The `field_efficiency` and `conveyance_efficiency` parameters represent real-world losses in water delivery and application:
- **field_efficiency** (0–1): Fraction of water applied at the field or end-use that is effectively used by crops or processes. Losses due to evaporation, deep percolation, or runoff reduce this value.
- **conveyance_efficiency** (0–1): Fraction of water delivered from the source to the field that actually reaches the field, accounting for losses in canals or distribution systems.

**How they work:**  
The model automatically adjusts the gross water demand to account for these efficiencies. The actual water withdrawn from the system is increased to ensure the net demand is met after losses:
- **Gross demand = Net demand / (field_efficiency × conveyance_efficiency)**

For example, if a demand node has a net demand of 10 m³/s, a field efficiency of 0.8, and a conveyance efficiency of 0.7, the gross demand will be:
- 10 / (0.8 × 0.7) = 17.86 m³/s

```python
demand1 = DemandNode(
    id="agriculture",
    easting=300,
    northing=400,
    csv_file='./data/dummy_data/demand_timeseries.csv',
    start_year=2020,
    start_month=1,
    num_time_steps=12,
    field_efficiency=0.8,
    conveyance_efficiency=0.7,
    priority=2
)
my_water_system.add_node(demand1)

demand2 = DemandNode(
    id="Industry",
    easting=400,
    northing=200,
    constant_demand_rate=60,   # m³/s
    non_consumptive_rate=10,   # m³/s (returns to system)
    start_year=2020,
    start_month=1,
    num_time_steps=12,
    priority=1
)
my_water_system.add_node(demand2)
```

**Demand CSV Format:**
```csv
Date,agriculture
2020-01-01,32.5
2020-02-01,28.1
...
```

>**Note:** The column header has to be set the DemandNode `id`. In the case above `agriculture` is the `id` of demand1.

---

### Storage Nodes

Represents reservoirs in a `WaterSystem`. Storage nodes store water, manage releases, and can account for evaporation and dead storage.

```python
storage = StorageNode(
    id="Reservoir",
    easting=200,
    northing=300,
    hv_file='./data/dummy_data/reservoir_hv.csv',  # Height-volume relationship
    evaporation_file='./data/dummy_data/reservoir_ev_timeseries.csv', # Monthly reservoir evaporation in mm
    start_year=2020,
    start_month=1,
    num_time_steps=12,
    initial_storage=5e6,              # m³
    dead_storage=1e5,                 # m³
    buffer_coef=0.5
)
my_water_system.add_node(storage)
```

In order to understand the function of the buffer_coefficient, the reservoir's release policy must be understood.

Each storage node (reservoir) is parameterized using a height-volume (HV) relationship, which is provided via a CSV file (`hv_file`). This file defines how the water level (elevation) in the reservoir corresponds to stored volume, allowing the model to convert between water levels and volumes for water balance calculations and evaporation estimation.

The reservoir's release policy is defined by a set of rule-curve parameters, typically found during optimization. Optionally they can also be set using the `set_release_params()` method from the `StorageNode` Class:
- **Vr**: Target monthly release volume [m³]
- **V1**: Top of buffer zone [m³]
- **V2**: Top of conservation zone [m³]

These parameters define operational zones:
- **Below dead_storage**: No release (water is not usable)
- **Buffer zone (dead_storage to V1)**: Reduced releases to conserve water
- **Conservation zone (V1 to V2)**: Normal operation, target releases
- **Above V2**: Flood control zone, increased releases to prevent overtopping

**Buffer Coefficient:**  
The `buffer_coefficient` (0–1) controls how much water is released when the reservoir is in the buffer zone (between dead storage and V1). A lower value means more conservative releases (less water released), while a higher value allows more water to be released even at low storage. This helps to prevent the reservoir from being depleted too quickly during dry periods.

### Step 4: Add Edges

Edges connect nodes and define flow capacities and optional ecological flow requirements. To initialize an edge the Source Node, Target Node and capacity have to be known. Additionally an ecological_flow rate can be defined (only relevant for optimization). In order to account for water losses along an edge optionally the lenght of an edge can be defined together with a loss_factor (0-1). If only a loss_factor is defined the length is calculated using the coordinates of the Source and Target Node. 

```python
# Edges can be defined and then added to the water system
edge1 = Edge(
    source=supply1, 
    target=storage, 
    capacity=100, # m³/s
    ecological_flow=20 # m³/s
)

my_water_system.add_edge(edge1)

# OR be added directly
my_water_system.add_edge(Edge(supply2, storage, 80))
my_water_system.add_edge(Edge(storage, hydrowork, 140, 50))
my_water_system.add_edge(Edge(hydrowork, demand1, 80))
my_water_system.add_edge(Edge(hydrowork, demand2, 80))
my_water_system.add_edge(Edge(runoff, demand1, 60))
my_water_system.add_edge(Edge(demand1, sink2, 100))
my_water_system.add_edge(Edge(demand2, sink1, 100))
```

---

## 3. Checking and Validating the Network

After adding all nodes and edges, you can check the network for consistency:

```python
my_water_system._check_network()
```

---

## 4. Example: Simple Water System

![Visual Representation of the above Water System](./data/dummy_data/simple_system_example.png)

## 5. Ready for optimization

Once your system is set up, it can be optimized (See [README_optimization.md](/README_optimization.md)).

```python
objectives = {'objective_1':[1,1,1,0,0]} 

MyProblem = DeapOptimizer(
    base_system=my_water_system,
    num_time_steps=12,  # 12 month are optimized
    objective_weights=objectives,
    ngen=100,        # Optimizing over 100 generations
    population_size=500, # A Population consists of 500 individuals
    cxpb=0.6,       # 0.6 probability for crossover
    mutpb=0.3,      # 0.3 probability for mutation 
)

# Run the optimization
results = MyProblem.optimize()
save_optimized_parameters(
    results, 
    filename="./data/dummy_data/optimization_results.json"
)

# Plot convergence and Pareto front
MyProblem.plot_convergence()
MyProblem.plot_total_objective_convergence()
```

---

## 6. Visualizing the System

```python
results = load_parameters_from_file("./data/dummy_data/optimization_results.json")
my_water_system = load_optimized_parameters(my_water_system, results)
my_water_system.simulate(time_steps=12)

vis = WaterSystemVisualizer(my_water_system, name='Example_System_Visualization')
vis.plot_network_overview()
vis.plot_minimum_flow_compliance()
vis.plot_spills()
vis.plot_reservoir_volumes()
vis.plot_system_demands_vs_inflow()
vis.plot_demand_deficit_heatmap()
vis.print_water_balance_summary()
```

>**Note:** Different Figures, Optimization results and water balance tables can be found in the [model_output](/model_output/) folder. 
