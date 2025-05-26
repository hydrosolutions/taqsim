# How-To Guide: Running Water System Optimizaiton 

This guide explains how to use [`deap_optimization.py`](water_system/optimization/deap_optimization.py) to optimize the operation of a water resource system (such as a river basin with reservoirs and different demands). The module leverages the [DEAP](https://deap.readthedocs.io/) evolutionary computation framework to perform multi-objective or single-objective optimization of water resource systems.

---
In order to use the optimizaiton framework the `WaterSystem`and `DeapOptimizer` classes have to be imported from the water_system module. 

```python
from water_system import WaterSystem, DeapOptimizer
```

Further a `WaterSystem` should be created according to **XX**. The monthly water allocation then can be optimized within the defined `WaterSystem`


## 1. Prepare Your `DeapOptimizer`
A `DeapOptimizer`has to be initialised using the following 10 arguments: 
- water_system: the system to optimize
- start_year (int): Start year for optimization
- start_month (int): Start month for optimization (1-12)
- num_time_steps (int): Number of time steps to optimize (monthly timesteps)
- objective_weihts(dict[str, list[float]]): A dictionary mapping weights to the objective functions(keys)
- ngen (int): Number of generations for the Genetic Algorithm
- pop_size (int): Population size for the Genetic Algoritm
- cxpb (float): Crossover probability for the Genetic Algorithhm
- mutpb (float): Mutation probability for the Genetic Algorithm


```python 
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

## 2. Customize Objective Functions
In order to choose the number of objective functions and its compontents to be optimised the user has to define an objective weigth dictionary.

Example: 
```python
    objective_weights = {
        'objective_1': [1,0,0,0,0],
        'objective_2': [0,1,0,0,0],
        'objective_3': [0,0,1,0,0],
    }
```

The number of keys in the `objective_weights` dictionary (3 in the example above) determines the number of objectives. Each key maps to a list of weights, which specify how much each component contributes to that objective. The five possible components you can assign weights to are listed below. Further details and the implementation of these components can be found in `optimization/objectives.py`.

The five available Objective Function Components: 

1. **Regular Demand Deficit**  
   *Definition*: The total annual unmet water demand for regular (non-priority) users/demand nodes.  
   *Goal*: Minimize the amount of water that regular users do not receive.

2. **Priority Demand Deficit**  
   *Definition*: The total annual unmet water demand for high-priority users, such as municipal or critical infrastructure.  
   *Goal*: Minimize the deficit for users who must be supplied first.

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



**Example for 4 objectives (each focusing on a different aspect):**

```python
objective_weights = {
    'objective_1': [1, 0, 0, 0, 0],  # Only regular demand deficit
    'objective_2': [0, 1, 0, 0, 0],  # Only priority demand deficit
    'objective_3': [0, 0, 1, 0, 0],  # Only spillage
    'objective_4': [0, 0, 0, 1, 1],  # Only minimum flow deficit at sinks
}
```
An optimization run with these objective weights would return a four dimensional pareto front with optimal solutions. Each solution on the pareto front consists of four values, where the first one reflects the demand deficit, the second the demand deficit for priority demands, the third the spillages and the fourth objective the sum of minimul flow deficit at sink nodes and unmet ecological flow at edges. 




