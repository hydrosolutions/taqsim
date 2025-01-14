# Water System Optimization Using Genetic Algorithms

This notebook demonstrates the optimization of a water system using genetic algorithms. We'll optimize reservoir release parameters and hydroworks distribution parameters to minimize supply deficits while considering operational constraints.

## Setup

First, let's import the required libraries and set up our parameters.

```python
import numpy as np
import matplotlib.pyplot as plt
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks
from multi_genetic_optimizer import MultiGeneticOptimizer
import webbrowser
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define optimization parameters
start_year = 2017
start_month = 1
num_time_steps = 12 * 3  # 3 years of monthly timesteps
ngen = 10  # Number of generations
pop_size = 10  # Population size
cxpb = 0.5  # Crossover probability
mutpb = 0.2  # Mutation probability
```

## Running the Optimization

Now let's run the optimization process using the genetic algorithm. This will attempt to find optimal parameters for:
- Reservoir release rules (h1, h2, w, m1, m2)
- Hydroworks distribution parameters

```python
def run_optimization(start_year, start_month, num_time_steps, ngen, pop_size, cxpb, mutpb):
    # Create optimizer instance
    optimizer = MultiGeneticOptimizer(
        create_seasonal_ZRB_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        population_size=pop_size
    )

    # Run optimization
    print("Starting optimization...")
    results = optimizer.optimize(ngen=ngen)

    # Print results
    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Final objective value: {results['objective_value']:,.0f} m³")
    
    # Plot convergence
    optimizer.plot_convergence()
    
    # Save optimization results
    save_optimized_parameters(results, 
        f"optimized_parameters_ZRB_ngen{ngen}_pop{pop_size}.json")
    
    return results

# Run optimization
results = run_optimization(start_year, start_month, num_time_steps, 
                         ngen, pop_size, cxpb, mutpb)
```

## Analyzing the Results

Let's analyze the optimized system by running a simulation with the best parameters:

```python
def analyze_optimized_system(results, start_year, start_month, num_time_steps):
    # Create and configure system with optimal parameters
    system = create_seasonal_ZRB_system(start_year, start_month, num_time_steps)
    system = load_optimized_parameters(system, results)
    
    # Run simulation
    system.simulate(num_time_steps)
    
    # Create visualizer
    vis = WaterSystemVisualizer(system, 'ZRB_system')
    
    # Generate various visualization plots
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    vis.plot_system_demands_vs_inflow()
    vis.plot_network_layout_2()
    
    # Plot release functions for reservoirs
    for res_name in ['RES-Akdarya', 'RES-Kattakurgan']:
        storage_node = system.graph.nodes[res_name]['node']
        vis.plot_release_function(storage_node)
    
    # Create interactive visualization
    html_file = vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    
    return system

# Analyze optimized system
optimized_system = analyze_optimized_system(results, start_year, start_month, num_time_steps)
```

## Interpreting the Results

The optimization results show several important aspects:

1. **Convergence**: The convergence plot shows how the optimization progressed over generations. Lower values indicate better solutions.

2. **Reservoir Parameters**: The optimized release parameters determine how each reservoir releases water based on its water level.

3. **Hydroworks Distribution**: The optimized distribution parameters determine how water is allocated among different demand nodes.

4. **System Performance**: The visualizations show:
   - Demand satisfaction and deficits
   - Water balance 
   - Network flows and connectivity
   - Reservoir operations

## Key Visualizations

The notebook generates several important visualizations:

1. **Demand Deficit Heatmap**: Shows when and where water shortages occur
2. **System Demands vs Inflow**: Compares total system demands with available water
3. **Network Layout**: Shows the spatial arrangement and connectivity of the system
4. **Release Functions**: Shows how reservoirs operate under different conditions

## Conclusions

The genetic algorithm optimization helps find a balance between:
- Meeting water demands
- Maintaining reservoir storage
- Managing system constraints
- Minimizing spillage losses

The optimized parameters provide a starting point for system operations, though they should be validated against other scenarios and constraints before implementation.

## Next Steps

Potential next steps could include:
1. Running longer optimizations (more generations, larger population)
2. Testing different objective functions
3. Analyzing system performance under different climate scenarios
4. Adding additional operational constraints
5. Comparing different optimization algorithms

