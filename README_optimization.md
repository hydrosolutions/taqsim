# How-To Guide: Running Water System Optimizaiton 

This guide explains how to use [`deap_optimization.py`](./water_system/optimization/deap_optimization.py) to optimize the water alloaction in a water resource system (such as a river basin with reservoirs and different demands). The module leverages the [DEAP](https://deap.readthedocs.io/) evolutionary computation framework to perform multi-objective or single-objective optimization of water resource systems.

In order to use the optimizaiton framework a `WaterSystem` has to be created first according to **XX**.


## 1. Introduction to the `DeapOptimizer` Class
A `DeapOptimizer`has to be initialised using the following arguments: 
- water_system: the `WaterSystem` to optimize
- start_year (int): Start year for optimization
- start_month (int): Start month for optimization (1-12)
- num_time_steps (int): Number of time steps to optimize (monthly timesteps)
- objective_weihts(dict[str, list[float]]): A dictionary mapping weights to the objective functions(keys)
- ngen (int): Number of generations for the Genetic Algorithm
- pop_size (int): Population size for the Genetic Algoritm
- cxpb (float): Crossover probability for the Genetic Algorithhm
- mutpb (float): Mutation probability for the Genetic Algorithm


```python 
from water_system import DeapOptimizer

optimizer = DeapOptimizer(
            base_system=water_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            objective_weights=objective_weights
            ngen=ngen,
            population_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
)
```

## 2. Customize your Objective Functions
In order to choose the number of objective functions and its compontents to be optimised the user has to define an objective weigth dictionary.

Example: 
```python
    objective_weights = {
        'objective_1': [1,0,0,0,0],
        'objective_2': [0,1,0,0,0],
        'objective_3': [0,0,1,0,0],
    }
```

The number of keys in the `objective_weights` dictionary (3 in the example above) determines the number of objectives. Each key maps to a list of weights, which specify how much each component contributes to that objective. The five possible components you can assign weights to are listed below. Further details and the implementation of these components can be found in [`objectives.py`](./water_system/optimization/objectives.py).

The five available Objective Function Components: 

1. **Regular Demand Deficit**  
   *Definition*: The total annual unmet water demand for regular (non-priority) users/demand nodes.  
   *Goal*: Minimize the amount of water that regular users do not receive.

2. **Priority Demand Deficit**  
   *Definition*: The total annual unmet water demand for high-priority users, such as municipal or critical infrastructure.  
   *Goal*: Minimize the deficit for users with higher priority.

3. **Spillage (Flooding)**  
   *Definition*: The total annual volume of water that is spilled from reservoirs or hydropower plants because it cannot be stored or used.  
   *Goal*: Minimize spillage in order to enforce the water system to comply with capacity constraints.

4. **Minimum Flow Deficit at Sinks**  
   *Definition*: The total annual deficit in meeting minimum required flows at sink nodes (e.g., river outlets, ecological flow requirements).  
   *Goal*: Ensure that downstream flow requirements are met.

5. **Unmet Ecological Flow at Edges**
   *Definition*: The total annual deficit in meeting ecological flow requirements on specific river reaches (edges in the network).  
   *Goal*: Minimize ecological flow violations for river health.


---

The examples below should further clarify how the objective functions for an optimization can be set up. 

**Example for a single-objective:**
```python

objective_weights = {
    'objective_1': [1, 1, 1, 1, 1],  # The order of weights from left to right reflects the order of Components (from 1 to 5). 
}
```

An optimization run with this `objective_weights` would return a single value. This value is the sum of all five components (Regular Demand Deficits, Priority Demand Deficits, Spillages, Minimum Flow Deficits at Sink Nodes, and Unmet Ecological Flows along Edges), each multiplied by its corresponding weight (in this case, 1 for all components).

**Example for a single-objective with different weights:**
```python

objective_weights = {
    'objective_1': [1, 5, 10, 0, 0],  # The order of weights from left to right reflects the order of Components (from 1 to 5). 
}
```
An optimization run with this `objective_weights` would also return a single value. However, in this case, each component is multiplied by a different weight before being summed. For example, the priority demand deficit is weighted five times higher than the regular demand deficit, and spillage is weighted ten times higher, while minimum flow deficits and unmet ecological flows are not considered (weight 0). This allows you to emphasize or ignore specific components according to your optimization priorities.

> **Note:** 
> When the weights differ from 1, the resulting objective value is a weighted sum and no longer directly represents a physical unit such as km³/a. Only when all weights are set to 1 does the objective value correspond to the sum of the original components in km³/a. If you use different weights, the value should be interpreted as a relative or composite score rather than a physical quantity.

**Example for 4 objectives (each focusing on a different aspect):**

```python
objective_weights = {
    'objective_1': [1, 0, 0, 0, 0],  # Only regular demand deficit
    'objective_2': [0, 1, 0, 0, 0],  # Only priority demand deficit
    'objective_3': [0, 0, 1, 0, 0],  # Only spillage
    'objective_4': [0, 0, 0, 1, 1],  # Minimum flow deficit at sinks and minimum unmet ecological flow along edges
}
```
- Objective 1 considers only the regular demand deficit (first component), as indicated by the weight [1, 0, 0, 0, 0].
- Objective 2 considers only the priority demand deficit (second component), with [0, 1, 0, 0, 0].
- Objective 3 focuses solely on spillage (third component), with [0, 0, 1, 0, 0].
- Objective 4 combines the minimum flow deficit at sinks and the unmet ecological flow at edges (fourth and fifth components), as shown by [0, 0, 0, 1, 1].

When you run an optimization with these weights, the algorithm will seek solutions that balance trade-offs between these four objectives. The result is a four-dimensional Pareto front: a set of optimal solutions where no single solution is best for all objectives simultaneously, but each represents a different compromise among the objectives.

Each solution on the Pareto front will have four values:

1. The total regular demand deficit,
2. The total priority demand deficit,
3. The total spillage,
4. The combined total of minimum flow deficit at sinks and unmet ecological flow at edges.

This setup allows you to analyze and visualize the trade-offs between different aspects of system performance, helping you choose a solution that best fits your management priorities.

> **Note:**  
> You can define more than four objective functions by adding additional keys to the `objective_weights` dictionary. However, the optimization framework uses the NSGA-2 selection algorithm, which is best suited for problems with 2 or 3 objectives. While NSGA-2 can handle up to 10 objectives, its performance and ability to distinguish between solutions may decrease as the number of objectives increases. For problems with more than 4 objectives, interpret results with caution and consider the limitations of the algorithm.


## 3. Set Genetic Algorithm Settings

The performance and outcome of the optimization depend strongly on the settings of the genetic algorithm. The most important parameters you need to define are:

- **Number of generations (`ngen`)**: How many iterations the algorithm will run. More generations allow for more thorough searching but increase computation time.
- **Population size (`pop_size`)**: The number of candidate solutions in each generation. Larger populations improve diversity and solution quality but require more computation per generation.
- **Crossover probability (`cxpb`)**: The likelihood (between 0 and 1) that two solutions will be combined to create new solutions. Typical values are between 0.6 and 0.9.
- **Mutation probability (`mutpb`)**: The likelihood (between 0 and 1) that a solution will be randomly changed. Typical values are between 0.01 and 0.3.

**Example:**
```python
ngen = 50           # Number of generations
pop_size = 50       # Population size
cxpb = 0.7          # Crossover probability
mutpb = 0.2         # Mutation probability
```

### Guidelines for Choosing Settings

- **Start with default values** (e.g., `ngen=50`, `pop_size=50`, `cxpb=0.7`, `mutpb=0.2`) and adjust based on your results and available computation time.
- **Increase population size** if you have many decision variables or notice premature convergence.
- **Increase number of generations** for more thorough optimization, especially for complex problems.
- **Adjust mutation probability** if the algorithm gets stuck (increase) or if solutions are too random (decrease).
- **Balance computation time**: Higher values for population size and generations will increase runtime.

> **Tip:** There is no universal best setting. The optimal configuration depends on your specific problem, the number of decision variables, and your computational resources. Experiment with different settings and monitor the convergence plots to find a good balance between solution quality and runtime.

For more advanced tuning, consider using automated hyperparameter optimization tools or running multiple experiments with different settings. A separate README file is available in this repository that explains how to use [Optuna](https://optuna.org/) to perform automated hyperparameter optimization for single-objective problems. This guide will help you set up an Optuna study to efficiently search for the best genetic algorithm parameters for your specific case.

## 4. Run Optimization

Once you have defined your objective weights, genetic algorithm settings and created a `DeapOptimizer` object you are ready to run the optimization. This is typically done by calling the `optimize` method of the `DeapOptimizer` Calss as shown in the example below.

**Example usage:**
```python
# Initialize the Optimizer
MyProblem = DeapOptimizer(
                base_system=my_water_system,
                start_year=2017,    # Start year set to 2017
                start_month=1,      # Starting in January
                num_time_steps=12,  # 12 month are optimized
                objective_weights={'objective_1':[1,1,1,0,0], 
                                   'objective_2':[0,0,0,1,1]} #Two objective functions
                ngen=50,        # Optimizing over 50 generations
                population_size=100, # A Population consists of 100 individuals
                cxpb=0.6,       # 0.6 probability for crossover
                mutpb=0.2,      # 0.2 probability for mutation 
)

# Run the optimization
results = MyProblem.optimize()

# Plot convergence and Pareto front
MyProblem.plot_convergence()
MyProblem.plot_total_objective_convergence()
```

### What Happens During Optimization

- The optimizer will evolve a population of candidate solutions over several generations.
- At each generation, solutions are evaluated according to your defined objectives.
- The best solutions are selected, recombined, and mutated to form the next generation.
- Progress and convergence are tracked and saved.

### Outputs

After the optimization completes, the `results` dictionary contains:

- **success**: Whether the optimization finished successfully.
- **message**: Status message about the run.
- **population_size**: The population size used.
- **generations**: The number of generations run.
- **crossover_probability**: The crossover probability used.
- **mutation_probability**: The mutation probability used.
- **objective_values**: The final objective value(s) for the recommended solution.
- **optimal_reservoir_parameters**: The best-found reservoir operation parameters.
- **optimal_hydroworks_parameters**: The best-found hydroworks operation parameters.
- **pareto_front**: (For multi-objective runs) The set of non-dominated (Pareto-optimal) solutions found.
- **optimizer**: The optimizer instance (for advanced use).

### Convergence Plots
The convergence plots are saved in the [model_output](./model_output/) folder. The plot_convergence() method creates a subplot for each objective function, showing its convergence over generations. The plot_total_objective_convergence() method plots the sum of all objectives (useful for multi-objective optimization).

### Save results
The `save_optimized_parameters` function allows you to save the results of your optimization run to a JSON file. The function takes a filename(str) containing a path and the optimization results dictinary as input. The saved optimization results can then be used to simulate the `WaterSystem` (further explanations in **XX**)

```python
from water_system/io_utily.py import save_optimized_parameters

save_optimized_parameters(results, filename)
```

### Visualization of Pareto Front!!

> **Tip:** Review the convergence plots and Pareto front visualizations to understand the trade-offs and performance of your solutions. If the optimization does not converge or the solutions are unsatisfactory, consider adjusting your genetic algorithm settings or objective weights.

---