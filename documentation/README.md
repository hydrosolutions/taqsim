# Water System Simulation & Optimization

## Overview

This project provides an object-oriented Python framework for simulating and optimizing water resource systems. It models water systems as directed acyclic graphs (DAGs) of nodes (e.g., supply, demand, storage, sink, hydroworks, runoff) and edges (rivers, canals, etc.), supporting both simulation and multi-objective optimization of water allocation over multiple time steps.

---

## Features

- **Flexible Node Types:**  
  - Supply nodes (constant or time-varying inflow)
  - Demand nodes (agricultural, industrial, domestic, with priorities and efficiencies)
  - Storage nodes (reservoirs with evaporation, dead storage, rule curves)
  - Sink nodes (system outlets, environmental flows)
  - HydroWorks nodes (diversions, confluences, splitters)
  - Runoff nodes (rainfall-runoff generation)
- **Edge Representation:**  
  - Capacity constraints, optional ecological flow, losses
- **Simulation:**  
  - Water balance for each node and edge over monthly time steps
- **Optimization:**  
  - Multi-objective optimization using DEAP/NSGA-II
  - Customizable objective functions (demand deficit, spillage, minimum flows, etc.)
- **Visualization:**  
  - Network layout, time series, heatmaps, Pareto front
- **CSV I/O:**  
  - Input/output for time series, results, and water balances

---

## Project Structure

```
water_system/
│
├── __init__.py
├── water_system.py
├── edge.py
├── nodes.py
├── optimization/
│   ├── optimizer.py
│   ├── objectives.py
│   └── pareto_visualization.py
├── io_utils.py
├── visualization.py
├── validation_functions.py
data/
documentation/
model_output/
dummy_creator.py
system_creator_simple.py
system_creator_ZRB.py
...
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/hydrosolutions/water_system_simulation_optimization.git
   cd water_system_simulation_optimization
   ```

2. **Install dependencies:**
   ```sh
   pip install networkx matplotlib pandas deap seaborn numpy
   ```

---

## Usage

### 1. System Creation

See [README_system_creation_example.md](documentation/README_system_creation_example.md) for a full, step-by-step guide.

**Basic Example:**
```python
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, Edge

dt = 30.44 * 24 * 3600  # seconds in a month
my_water_system = WaterSystem(dt=dt, start_year=2020, start_month=1)

# Add nodes (see documentation for all parameters)
supply = SupplyNode(id="Source1", easting=100, northing=600, constant_supply_rate=100, start_year=2020, start_month=1, num_time_steps=12)
my_water_system.add_node(supply)
# ... add other nodes as needed

# Add edges
my_water_system.add_edge(Edge(supply, ... , capacity=100))
# ... add other edges

my_water_system._check_network()
```

### 2. Optimization

See [README_optimization.md](documentation/README_optimization.md) for details.

```python
from water_system import DeapOptimizer

objectives = {'objective_1':[1,1,1,0,0]}  # See documentation for meaning

MyProblem = DeapOptimizer(
    base_system=my_water_system,
    num_time_steps=12,
    objective_weights=objectives,
    ngen=100,
    population_size=500,
    cxpb=0.6,
    mutpb=0.2,
)

results = MyProblem.optimize()
```

### 3. Visualization

```python
from water_system import WaterSystemVisualizer

vis = WaterSystemVisualizer(my_water_system, name='Example_System_Visualization')
vis.plot_network_overview()
vis.plot_minimum_flow_compliance()
vis.plot_spills()
vis.plot_reservoir_volumes()
vis.plot_system_demands_vs_inflow()
vis.plot_demand_deficit_heatmap()
vis.print_water_balance_summary()
```

---

## Documentation

- [README_system_creation_example.md](documentation/README_system_creation_example.md): Step-by-step system creation
- [README_node_types.md](documentation/README_node_types.md): Node types and parameters
- [README_optimization.md](documentation/README_optimization.md): Optimization setup and usage

---

## Visualization

- The `visualize()` method and `WaterSystemVisualizer` class provide:
  - Network layout plots
  - Time series of storage, flows, and deficits
  - Heatmaps of demand deficits
  - Pareto front and convergence plots for optimization

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or suggestions, please open an issue or contact Tobias Siegfried at [siegfried@hydrosolutions.ch](mailto:siegfried@hydrosolutions.ch).

---