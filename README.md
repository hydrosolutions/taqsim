# Water System Simulation

## Overview

This project implements an object-oriented water system simulation and optimization tool using Python. It models water systems as networks of nodes and edges in a directed acyclic graph, allowing for the simulation and optimization of water flow over multiple time steps.

## Features

- Modeling of various node types:
  - Supply nodes (water sources)
  - Sink nodes (water exits)
  - Demand nodes (agricultural, domestic, industrial use)
  - Structure nodes (storage, diversion, confluence)
- Edge representation of fixed connections (rivers, canals) with capacity constraints
- Simulation of water flow over specified time steps
- Visualization of the water system network and simulation results

## Project Structure

```
water_system/
│
├── __init__.py
├── water_system.py
├── structure.py
└── edge.py
```

- `__init__.py`: Initializes the package and imports main classes
- `water_system.py`: Contains the `WaterSystem` class for overall system management
- `structure.py`: Defines various node types (Supply, Sink, Demand, Storage, etc.)
- `edge.py`: Implements the `Edge` class for connections between nodes

## Installation

1. Ensure you have Python 3.7+ installed.
2. Clone this repository:
   ```
   git clone https://github.com/yourusername/water-system-simulation.git
   cd water-system-simulation
   ```
3. Install required packages:
   ```
   pip install networkx matplotlib
   ```

## Usage

Here's a basic example of how to use the water system simulation:

```python
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, Edge

# Create a water system
water_system = WaterSystem()

# Create nodes
supply = SupplyNode("Supply1", supply_rate=10)
reservoir = StorageNode("Reservoir1", capacity=1000)
demand1 = DemandNode("Demand1", demand_rate=6)
demand2 = DemandNode("Demand2", demand_rate=3)

# Add nodes to the system
for node in [supply, reservoir, demand1, demand2]:
    water_system.add_node(node)

# Create and add edges
water_system.add_edge(Edge(supply, reservoir, capacity=15))
water_system.add_edge(Edge(reservoir, demand1, capacity=8))
water_system.add_edge(Edge(reservoir, demand2, capacity=5))

# Run simulation
water_system.simulate(num_time_steps=12)

# Visualize results
water_system.visualize()
```

## Visualization

The `visualize()` method creates a hierarchical layout of the water system:
- Supply nodes are shown in blue
- Storage nodes in green
- Demand nodes in red
- Other node types in gray
- Edges are labeled with their flow values

## Future Developments

1. Implement more complex node behaviors (e.g., time-varying supply/demand)
2. Add water quality modeling capabilities
3. Incorporate optimization algorithms for water allocation
4. Develop a user interface for easier system configuration
5. Implement data import/export features for integration with other tools
6. Add support for stochastic simulations to model uncertainty

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or suggestions, please open an issue in the GitHub repository or contact Tobias Siegfried at [siegfried@hydrosolutions.ch].
