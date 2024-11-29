# Water System Simulation

## Version
Version 0.0.5

## Requirements

### Core Dependencies
```
networkx>=2.5
matplotlib>=3.3.0
pandas>=1.2.0
numpy>=1.19.0
scipy>=1.6.0
```

### Optional Dependencies
```
seaborn>=0.11.0    # Enhanced visualizations
webbrowser         # Interactive visualization viewing
```

## Overview

This project implements an object-oriented water system simulation and optimization tool using Python. It models water systems as networks of nodes and edges in a directed acyclic graph, allowing for the simulation and optimization of water flow over multiple time steps.

## Features

### Node Functionality
- Supply nodes support variable rates and CSV data import
- Storage nodes include:
  - Height-volume-area relationships
  - Evaporation losses
  - Release rules
  - Spillway handling
- Demand nodes support:
  - Time-varying demands
  - Efficiency factors
  - CSV data import

### Edge Characteristics
- Capacity constraints
- Length-based losses
- Automatic length calculation from node coordinates

### Analysis Tools
- Water balance calculations
- Flow and storage visualization
- Network layout plotting
- Interactive visualizations
- Comprehensive statistical analysis

### Data Integration
- CSV import/export support
- GIS coordinate system integration
- Time series data handling

## Project Structure
```
water_system/
├── __init__.py
├── water_system.py    # Core system management
├── structure.py       # Node type definitions
├── edge.py            # Edge class implementation
└── visualization.py   # Visualization tools
```
## Installation

1. Ensure you have Python 3.7+ installed.
2. Clone this repository:
   ```
   git clone https://github.com/hydrosolutions/water_system_simulation_optimization.git
   cd water_system_simulation_optimization
   ```
3. Install required packages:
   ```
   pip install networkx matplotlib
   ```

## Usage

Here's a basic example of how to use the water system simulation:

```python
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, Edge

# Initialize system
system = WaterSystem()

# Create nodes
supply = SupplyNode("Supply1", supply_rates=10, easting=100, northing=100)
storage = StorageNode("Storage1", capacity=1000, easting=200, northing=100)
demand = DemandNode("Demand1", demand_rates=8, easting=300, northing=100)

# Add nodes
for node in [supply, storage, demand]:
    system.add_node(node)

# Connect nodes
system.add_edge(Edge(supply, storage, capacity=15))
system.add_edge(Edge(storage, demand, capacity=10))

# Run simulation
system.simulate(time_steps=12)

# Create visualizer
vis = WaterSystemVisualizer(system)
vis.plot_network_layout()
vis.plot_storage_dynamics()
vis.print_water_balance_summary()
```


## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or suggestions, please open an issue in the GitHub repository or contact Tobias Siegfried at [siegfried@hydrosolutions.ch].
