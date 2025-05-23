import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import copy
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks, SinkNode
from .objectives import (
    regular_demand_deficit,
    priority_demand_deficit,
    sink_node_min_flow_deficit,
    total_spillage,
)
# --------------------- SHARED UTILITY FUNCTIONS ---------------------

def normalize_distribution(values: np.ndarray) -> np.ndarray:
    """
    Normalize a list of values to sum to 1.0 using numpy for efficiency.
    
    Args:
        values (list or ndarray): List of values to normalize
        
    Returns:
        ndarray: Normalized values that sum to 1.0
    """
    values = np.array(values)
    total = np.sum(values)
    if total == 0:
        # If all values are 0, return equal distribution
        return np.full_like(values, 1.0 / len(values))
    return values / total

def decode_individual(optimizer, individual: List[float]) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Decode individual genes into monthly reservoir and hydroworks parameters
    
    Args:
        optimizer: The optimizer instance (single or multi objective)
        individual: The individual to decode
        
    Returns:
        tuple: (reservoir_params, hydroworks_params) containing the decoded parameters
    """
    genes_per_reservoir_month = 3
    reservoir_params = {}
    hydroworks_params = {}
    
    # Convert individual to numpy array for efficient slicing
    individual = np.array(individual)
    
    # Decode reservoir parameters
    current_idx = 0
    for res_id in optimizer.reservoir_ids:
        params = {
            'Vr': [], 'V1': [], 'V2': []
        }
        
        # Get parameters for each month
        for month in range(12):
            start_idx = current_idx + month * genes_per_reservoir_month
            params['Vr'].append(individual[start_idx])
            params['V1'].append(individual[start_idx + 1])
            params['V2'].append(individual[start_idx + 2])
        
        reservoir_params[res_id] = params
        current_idx += 12 * genes_per_reservoir_month
    
    # Decode hydroworks parameters
    for hw_id in optimizer.hydroworks_ids:
        n_targets = len(optimizer.hydroworks_targets[hw_id])
        dist_params = {}
        
        for target in optimizer.hydroworks_targets[hw_id]:
            dist_params[target] = np.zeros(12)
        
        for month in range(12):
            start_idx = current_idx + month * n_targets
            end_idx = start_idx + n_targets
            
            dist_values = individual[start_idx:end_idx]
            normalized_dist = normalize_distribution(dist_values)
            
            for target, value in zip(optimizer.hydroworks_targets[hw_id], normalized_dist):
                dist_params[target][month] = value
        
        hydroworks_params[hw_id] = dist_params
        current_idx += 12 * n_targets
    
    return reservoir_params, hydroworks_params

def mutate_individual(optimizer, individual, indpb=0.5):
    """
    Custom mutation operator with enforced parameter bounds for monthly parameters
    
    Args:
        optimizer: The optimizer instance
        individual: The individual to mutate
        indpb: Independent probability for each gene to be mutated
        
    Returns:
        tuple: Containing the mutated individual
    """
    genes_per_reservoir_month = 3
    
    individual = np.array(individual)
    # Track which months need renormalization for each hydroworks
    hw_updates = set()
    
    # Mutate reservoir parameters for each month
    for res_idx, res_id in enumerate(optimizer.reservoir_ids):
        for month in range(12):
            start_idx = (res_idx * 12 * genes_per_reservoir_month) + (month * genes_per_reservoir_month)
            
            for param_idx, param_name in enumerate(['Vr', 'V1', 'V2']):
                if random.random() < indpb:
                    bounds = optimizer.reservoir_bounds[res_id][param_name]
                    value = np.random.uniform(bounds[0], bounds[1])
                    
                    # Special handling for volume relationships
                    if param_name == 'V2':
                        # Ensure V2 > V1
                        min_bound = max(bounds[0], individual[start_idx + 1])  # V1 index is 1, V2 must be > V1
                        value = np.random.uniform(min_bound, bounds[1])
                    elif param_name == 'V1':
                        # Ensure V1 < V2 and V1 > dead_storage
                        dead_storage = optimizer.base_system.graph.nodes[res_id]['node'].dead_storage
                        min_bound = max(bounds[0], dead_storage)
                        max_bound = min(bounds[1], individual[start_idx + 2])  # V2 index is 2
                        value = np.random.uniform(min_bound, max_bound)
                    
                    individual[start_idx + param_idx] = value

    # Mutate hydroworks parameters
    hw_genes_start = len(optimizer.reservoir_ids) * 12 * genes_per_reservoir_month
    current_idx = hw_genes_start
    
    for hw_idx, hw_id in enumerate(optimizer.hydroworks_ids):
        n_targets = len(optimizer.hydroworks_targets[hw_id])
        if random.random() < indpb:
            hw_updates.add(hw_id)
            for month in range(12):
                start_idx = current_idx + (month * n_targets)
                end_idx = start_idx + n_targets
                if end_idx <= len(individual):  # Check array bounds
                    individual[start_idx:end_idx] = np.random.random(n_targets)
        current_idx += 12 * n_targets  # Update index for next hydroworks

    # Normalize hydroworks distributions that were modified
    current_idx = hw_genes_start
    for hw_id in hw_updates:
        n_targets = len(optimizer.hydroworks_targets[hw_id])
        for month in range(12):
            start_idx = current_idx + (month * n_targets)
            end_idx = start_idx + n_targets
            if end_idx <= len(individual):  # Check array bounds
                individual[start_idx:end_idx] = normalize_distribution(individual[start_idx:end_idx])
        current_idx += 12 * n_targets  # Update index for next hydroworks

    # Return the creator.Individual type appropriate for the specific optimizer
    return optimizer.creator.Individual(individual.tolist()),

def crossover(optimizer, ind1, ind2):
    """
    Custom crossover that treats monthly parameters as packages.
    For each month, all parameters of a reservoir or hydrowork for that specific month
    are swapped as a complete unit between parents.
    
    Args:
        optimizer: The optimizer instance 
        ind1, ind2: The two individuals to cross
        
    Returns:
        tuple: (crossed_ind1, crossed_ind2) containing the crossed individuals
    """
    ind1, ind2 = np.array(ind1), np.array(ind2)
    
    # Handle reservoirs month by month
    genes_per_month = 3 
    num_reservoirs = len(optimizer.reservoir_ids)
    
    # For each month
    for month in range(12):
        # For each reservoir, decide whether to swap this month's parameters
        for res_idx in range(num_reservoirs):
            if random.random() < 0.5:
                # Calculate indices for this month's parameters
                start_idx = (res_idx * 12 * genes_per_month) + (month * genes_per_month)
                end_idx = start_idx + genes_per_month
                
                # Swap parameters for this month
                ind1[start_idx:end_idx], ind2[start_idx:end_idx] = \
                    ind2[start_idx:end_idx].copy(), ind1[start_idx:end_idx].copy()

    
    # Handle hydroworks month by month
    hw_start = num_reservoirs * 12 * genes_per_month
    current_idx = hw_start
    
    # For each month
    for month in range(12):
        # For each hydrowork
        hw_current_idx = current_idx
        for hw_id in optimizer.hydroworks_ids:
            n_targets = len(optimizer.hydroworks_targets[hw_id])
            
            if random.random() < 0.5:
                # Calculate indices for this month's distribution parameters
                start_idx = hw_current_idx + (month * n_targets)
                end_idx = start_idx + n_targets
                
                # Swap distribution parameters for this month
                tmp1 = ind1[start_idx:end_idx].copy()
                tmp2 = ind2[start_idx:end_idx].copy()
                
                # Normalize the distributions before swapping
                ind1[start_idx:end_idx] = normalize_distribution(tmp2)
                ind2[start_idx:end_idx] = normalize_distribution(tmp1)
            
            hw_current_idx += 12 * n_targets  # Move to next hydrowork's base index
    
    # Return individuals using the appropriate creator.Individual type
    return optimizer.creator.Individual(ind1.tolist()), optimizer.creator.Individual(ind2.tolist())

def create_individual(optimizer):
    """
    Create an individual with parameters for all reservoirs and hydroworks
    
    Args:
        optimizer: The optimizer instance
        
    Returns:
        Individual: A new individual with randomly initialized genes
    """
    genes = []

    # Add reservoir genes
    for res_id in optimizer.reservoir_ids:
        bounds = optimizer.reservoir_bounds[res_id]
        for month in range(12):
            # Generate parameters ensuring proper relationships
            Vr = np.random.uniform(bounds['Vr'][0], bounds['Vr'][1])
            
            # Ensure V1 is properly bounded
            V1_min = bounds['V1'][0]
            V1_max = bounds['V1'][1]
            V1 = np.random.uniform(V1_min, V1_max)
            
            # Ensure V2 > V1
            V2_min = max(bounds['V2'][0], V1)
            V2_max = bounds['V2'][1]
            V2 = np.random.uniform(V2_min, V2_max)
            
            genes.extend([Vr, V1, V2])

    # Add hydroworks genes
    for hw_id in optimizer.hydroworks_ids:
        n_targets = len(optimizer.hydroworks_targets[hw_id])
        for month in range(12):
            # Create random distribution that sums to 1
            dist = np.random.random(n_targets)
            normalized_dist = normalize_distribution(dist)
            genes.extend(normalized_dist)

    return optimizer.creator.Individual(genes)

# --------------------- BASE OPTIMIZER CLASS -------------------------
class DeapOptimizer:
    """
    Base class for DEAP-based water system optimizers.
    Handles setup of system, bounds, IDs, and DEAP toolbox.
    Subclasses must implement _evaluate_individual and set fitness weights.
    """
    def __init__(
        self,
        base_system: WaterSystem,
        start_year: int,
        start_month: int,
        num_time_steps: int,
        population_size: int = 50,
        ngen: int = 50,
        cxpb: float = 0.65,
        mutpb: float = 0.32,
        number_of_objectives: int = 1,
        objective_weights: dict[str,List[float]] = {
        'objective_1': [1,0,0,0],
        'objective_2': [0,1,0,0],
        'objective_3': [0,0,1,0],
        'objective_4': [0,0,0,1],
    },
    ) -> None:
        """
        Initialize the optimizer.
        
        Args:
            base_system: Water system to optimize
            start_year: Starting year for simulation
            start_month: Starting month (1-12)
            num_time_steps: Number of time steps to simulate
            population_size: Size of the population
            ngen: Number of generations
            cxpb: Crossover probability
            mutpb: Mutation probability
            weights: Fitness weights (negative for minimization)
        """
        self.base_system = base_system
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.num_years = num_time_steps / 12
        self.dt = base_system.dt

        self.population_size = population_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.num_of_objectives = number_of_objectives
        self.objective_weights = objective_weights
        
        # Extract node IDs from the system
        self.priority_demand_ids = DemandNode.high_priority_demand_ids
        self.regular_demand_ids = DemandNode.low_priority_demand_ids
        self.sink_ids = SinkNode.all_ids
        self.reservoir_ids = StorageNode.all_ids
        self.hydroworks_ids = HydroWorks.all_ids
        self.hydroworks_targets = {}

        # Reservoir bounds
        self.reservoir_bounds = {}
        for node_id in self.reservoir_ids:
            reservoir = self.base_system.graph.nodes[node_id]['node']
            total_capacity = reservoir.outflow_edge.capacity
            dead_storage = reservoir.dead_storage
            capacity = reservoir.capacity
            self.reservoir_bounds[node_id] = {
                'Vr': (0, total_capacity*self.dt),
                'V1': (dead_storage, capacity),
                'V2': (dead_storage, capacity)
            }
        
        # Get hydroworks targets
        for node_id in self.hydroworks_ids:
            hydrowork = self.base_system.graph.nodes[node_id]['node']
            self.hydroworks_targets[node_id] = list(hydrowork.outflow_edges.keys())

        # Remove any existing Fitness/Individual to avoid DEAP errors
        if 'FitnessMulti' in creator.__dict__:
            del creator.FitnessMulti
        if 'Individual' in creator.__dict__:
            del creator.Individual
            
        # Set up DEAP types
        if number_of_objectives == 1:
            weights = (-1.0,)
        elif number_of_objectives == 2:
            weights = (-1.0, -1.0)
        elif number_of_objectives == 3:
            weights = (-1.0, -1.0, -1.0)
        elif number_of_objectives == 4:
            weights = (-1.0, -1.0, -1.0, -1.0)
        creator.create("FitnessMulti", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.creator = creator

        # Configure toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", create_individual, self)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", crossover, self)
        self.toolbox.register("mutate", mutate_individual, self)
        if number_of_objectives == 1:
            # Single-objective optimization uses tournament selection
            self.toolbox.register("select", tools.selTournament, tournsize=3)
        else:
            self.toolbox.register("select", tools.selNSGA2)
        # Initialize convergence history
        self.history = {}

    def _evaluate_individual(self, individual):
        try:
            # Decode the individual's genes into parameters
            reservoir_params, hydroworks_params = decode_individual(self, individual)
            
            # Create a copy of the base system to evaluate
            system = copy.deepcopy(self.base_system)
            
            # Apply parameters to the system
            for res_id, params in reservoir_params.items():
                system.graph.nodes[res_id]['node'].set_release_params(params)
            for hw_id, params in hydroworks_params.items():
                system.graph.nodes[hw_id]['node'].set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)

            # Objective 1: Regular demand deficit
            low_priority_demand_deficit = regular_demand_deficit(system, self.regular_demand_ids, self.dt, self.num_years)
            # Objective 2: Priority demand deficit
            high_priority_demand_deficit = priority_demand_deficit(system, self.priority_demand_ids, self.dt, self.num_years)
            # Objective 3: Minimum flow deficit
            min_flow_deficit = sink_node_min_flow_deficit(system, self.sink_ids, self.dt, self.num_years)
            # Objective 4: Spillage
            flooding_volume = total_spillage(system, self.hydroworks_ids, self.reservoir_ids, self.num_years)
            

            if self.num_of_objectives == 1:
                return (self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                        + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                        + self.objective_weights['objective_1'][2]*min_flow_deficit
                        + self.objective_weights['objective_1'][3]*flooding_volume,)
            elif self.num_of_objectives == 2:
                return (self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,
                    self.objective_weights['objective_2'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_2'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_2'][2]*min_flow_deficit
                    + self.objective_weights['objective_2'][3]*flooding_volume)
            elif self.num_of_objectives == 3:
                return (self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,
                    self.objective_weights['objective_2'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_2'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_2'][2]*min_flow_deficit
                    + self.objective_weights['objective_2'][3]*flooding_volume,
                    self.objective_weights['objective_3'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_3'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_3'][2]*min_flow_deficit
                    + self.objective_weights['objective_3'][3]*flooding_volume)
            elif self.num_of_objectives == 4:
                return (self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,
                    self.objective_weights['objective_2'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_2'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_2'][2]*min_flow_deficit
                    + self.objective_weights['objective_2'][3]*flooding_volume,
                    self.objective_weights['objective_3'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_3'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_3'][2]*min_flow_deficit
                    + self.objective_weights['objective_3'][3]*flooding_volume,
                    self.objective_weights['objective_4'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_4'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_4'][2]*min_flow_deficit
                    + self.objective_weights['objective_4'][3]*flooding_volume
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'), float('inf'), float('inf'))

    def optimize(self):
        """
        Shared multi-objective genetic algorithm optimization using mu+lambda with NSGA-II selection.
        Handles any number of objectives.
        Returns:
            dict: Results of the optimization including Pareto front
        """
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Determine number of objectives from the first individual's fitness
        n_obj = len(pop[0].fitness.values)
        # Initialize history keys for all objectives
        for i in range(1, n_obj + 1):
            self.history[f'min_obj{i}'] = []
            self.history[f'avg_obj{i}'] = []
            self.history[f'std_obj{i}'] = []

        # Initialize statistics tracking for all objectives
        stats_objs = []
        for i in range(n_obj):
            stats = tools.Statistics(lambda ind, idx=i: ind.fitness.values[idx])
            stats.register("min", np.min)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats_objs.append(stats)

        # Combine statistics into a multi-statistics object
        mstats = tools.MultiStatistics(**{f'obj{i+1}': stats_objs[i] for i in range(n_obj)})

        # Create the logbook for statistics
        logbook = tools.Logbook()
        # Show gen, nevals, and only avg, min, std for each objective
        logbook.header = ['gen', 'nevals']
        for i in range(n_obj):
            logbook.header += [f'obj{i+1}_avg', f'obj{i+1}_min', f'obj{i+1}_std']

        # Record the initial statistics
        record = mstats.compile(pop)
        # Flatten the record for custom header
        flat_record = {'gen': 0, 'nevals': len(pop)}
        for i in range(n_obj):
            flat_record[f'obj{i+1}_avg'] = record[f'obj{i+1}']['avg']
            flat_record[f'obj{i+1}_min'] = record[f'obj{i+1}']['min']
            flat_record[f'obj{i+1}_std'] = record[f'obj{i+1}']['std']
        logbook.record(**flat_record)
        print(logbook.stream)

        # Begin the evolution
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals (using NSGA-II selection or tournament)
            offspring = self.toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < self.cxpb:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.mutpb:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the survivors from the combined population of parents and offspring
            pop = self.toolbox.select(pop + offspring, self.population_size)

            # Record statistics
            record = mstats.compile(pop)
            flat_record = {'gen': gen, 'nevals': len(offspring)}
            for i in range(n_obj):
                flat_record[f'obj{i+1}_avg'] = record[f'obj{i+1}']['avg']
                flat_record[f'obj{i+1}_min'] = record[f'obj{i+1}']['min']
                flat_record[f'obj{i+1}_std'] = record[f'obj{i+1}']['std']
            logbook.record(**flat_record)
            print(logbook.stream)

            # Store the history for all objectives
            for i in range(1, n_obj + 1):
                self.history[f'min_obj{i}'].append(record[f'obj{i}']['min'])
                self.history[f'avg_obj{i}'].append(record[f'obj{i}']['avg'])
                self.history[f'std_obj{i}'].append(record[f'obj{i}']['std'])

        # Extract the Pareto front
        self.pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        # Get the overall best individual based on a weighted sum of all objectives
        best_ind = min(pop, key=lambda ind: sum(ind.fitness.values) / len(ind.fitness.values))
        reservoir_params, hydroworks_params = decode_individual(self, best_ind)

        # Return results in a format similar to pymoo
        return {
            'success': True,
            'message': f"{n_obj}-objective optimization completed successfully",
            'population_size': self.population_size,
            'generations': self.ngen,
            'crossover_probability': self.cxpb,
            'mutation_probability': self.mutpb,
            'objective_values': best_ind.fitness.values,
            'optimal_reservoir_parameters': reservoir_params,
            'optimal_hydroworks_parameters': hydroworks_params,
            'pareto_front': self.pareto_front,
            'optimizer': self  # Include the optimizer instance for parameter decoding
        }

    def plot_convergence(self):
        """
        Plot convergence history for each objective as a separate subplot and save to a fixed path.
        The figure is saved at '/model_output/deap/convergence/convergence.png'.
        """
        if not self.history or not any(self.history.values()):
            print("No convergence history to plot.")
            return

        n_obj = self.num_of_objectives
        if n_obj == 0:
            print("No objectives found in convergence history.")
            return

        fig, axes = plt.subplots(n_obj, 1, figsize=(8, 4 * n_obj), sharex=True)
        if n_obj == 1:
            axes = [axes]

        generations = range(1, len(self.history.get('min_obj1', [])) + 1)

        for i in range(n_obj):
            ax = axes[i]
            min_key = f'min_obj{i+1}'
            avg_key = f'avg_obj{i+1}'
            std_key = f'std_obj{i+1}'

            min_vals = self.history.get(min_key, [])
            avg_vals = self.history.get(avg_key, [])
            std_vals = self.history.get(std_key, [])

            if not min_vals:
                continue

            ax.plot(generations, min_vals, label='Best', color='blue')
            ax.plot(generations, avg_vals, label='Average', color='orange')
            ax.fill_between(
                generations,
                np.array(avg_vals) - np.array(std_vals),
                np.array(avg_vals) + np.array(std_vals),
                color='orange', alpha=0.2, label='Std Dev'
            )
            ax.set_ylabel(f'Objective {i+1}')
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel('Generation')
        fig.suptitle('Convergence History per Objective', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Ensure directory exists
        save_path = os.path.join(os.getcwd(), 'model_output', 'deap', 'convergence')
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'convergence.png')
        plt.savefig(file_path)
        plt.close(fig)
        print(f"Convergence plot saved to {file_path}")

    def plot_total_objective_convergence(self):
        """
        Plot the convergence of the sum of all objectives over generations and save to a fixed path.
        The figure is saved at '/model_output/deap/convergence/total_objective_convergence.png'.
        """
        if not self.history or not any(self.history.values()):
            print("No convergence history to plot.")
            return

        n_obj = self.num_of_objectives
        if n_obj == 0:
            print("No objectives found in convergence history.")
            return

        min_sums = []
        avg_sums = []
        std_sums = []

        num_gens = len(self.history.get('min_obj1', []))
        generations = range(1, num_gens + 1)

        for gen in range(num_gens):
            min_sum = 0
            avg_sum = 0
            std_sum = 0
            for i in range(n_obj):
                min_vals = self.history.get(f'min_obj{i+1}', [])
                avg_vals = self.history.get(f'avg_obj{i+1}', [])
                std_vals = self.history.get(f'std_obj{i+1}', [])
                if min_vals and avg_vals and std_vals:
                    min_sum += min_vals[gen]
                    avg_sum += avg_vals[gen]
                    std_sum += std_vals[gen]
            min_sums.append(min_sum)
            avg_sums.append(avg_sum)
            std_sums.append(std_sum)

        plt.figure(figsize=(8, 5))
        plt.plot(generations, min_sums, label='Best (Sum)', color='blue')
        plt.plot(generations, avg_sums, label='Average (Sum)', color='orange')
        plt.fill_between(
            generations,
            np.array(avg_sums) - np.array(std_sums),
            np.array(avg_sums) + np.array(std_sums),
            color='orange', alpha=0.2, label='Std Dev'
        )
        plt.xlabel('Generation')
        plt.ylabel('Sum of Objectives')
        plt.title('Convergence of Total Objective (Sum of All Objectives)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Ensure directory exists
        save_path = os.path.join(os.getcwd(), 'model_output', 'deap', 'convergence')
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'total_objective_convergence.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Total objective convergence plot saved to {file_path}")

'''class DeapSingleObjectiveOptimizer(DeapOptimizer):
    """
    Single-objective optimizer using DEAP.
    Minimizes total deficit (demand shortfall + minimum flow deficit).
    """
    def __init__(self, *args, **kwargs):
        """Initialize with single-objective fitness weights (-1.0 for minimization)"""
        super().__init__(*args, weights=(-1.0,), **kwargs)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual to get the single-objective fitness value.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            tuple: Single fitness value as a 1-element tuple
        """
        try:
            # Decode the individual's genes into parameters
            reservoir_params, hydroworks_params = decode_individual(self, individual)
            
            # Create a copy of the base system to evaluate
            system = copy.deepcopy(self.base_system)
            
            # Apply parameters to the system
            for res_id, params in reservoir_params.items():
                system.graph.nodes[res_id]['node'].set_release_params(params)
            for hw_id, params in hydroworks_params.items():
                system.graph.nodes[hw_id]['node'].set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)
            
            # Calculate total penalty

            low_priority_demand_deficit = regular_demand_deficit(system, self.regular_demand_ids, self.dt, self.num_years)
            high_priority_demand_deficit = priority_demand_deficit(system, self.priority_demand_ids, self.dt, self.num_years)
            min_flow_deficit = sink_node_min_flow_deficit(system, self.sink_ids, self.dt, self.num_years)
            flooding_volume = total_spillage(system, self.hydroworks_ids, self.reservoir_ids, self.num_years)
            
            return (  self.objective_weights['objective_1'][0]*low_priority_demand_deficit 
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,)
        
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'),)


class DeapMultiObjectiveOptimizer(DeapOptimizer):
    """
    Base class for multi-objective optimizers.
    Subclasses should specify the number of objectives.
    """
    def __init__(self, *args, **kwargs):
        """Initialize with proper selection algorithm for multi-objective optimization"""
        super().__init__(*args, **kwargs)
        # Override selection to use NSGA-II for multi-objective optimization
        self.toolbox.register("select", tools.selNSGA2)

    def _evaluate_individual(self, individual):
        """
        Multi-objective evaluation. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class DeapTwoObjectiveOptimizer(DeapMultiObjectiveOptimizer):
    """
    Two-objective optimizer using DEAP.
    Objectives: demand deficit and minimum flow deficit.
    """
    def __init__(self, *args, **kwargs):
        """Initialize with two-objective fitness weights (-1.0, -1.0 for minimization)"""
        kwargs["weights"] = (-1.0, -1.0)
        super().__init__(*args, **kwargs)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual for two objectives.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            tuple: Fitness values for demand deficit and minimum flow deficit
        """
        try:
            # Decode the individual's genes into parameters
            reservoir_params, hydroworks_params = decode_individual(self, individual)
            
            # Create a copy of the base system to evaluate
            system = copy.deepcopy(self.base_system)
            
            # Apply parameters to the system
            for res_id, params in reservoir_params.items():
                system.graph.nodes[res_id]['node'].set_release_params(params)
            for hw_id, params in hydroworks_params.items():
                system.graph.nodes[hw_id]['node'].set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)

            low_priority_demand_deficit = regular_demand_deficit(system, self.regular_demand_ids, self.dt, self.num_years)
            high_priority_demand_deficit = priority_demand_deficit(system, self.priority_demand_ids, self.dt, self.num_years)
            min_flow_deficit = sink_node_min_flow_deficit(system, self.sink_ids, self.dt, self.num_years)
            flooding_volume = total_spillage(system, self.hydroworks_ids, self.reservoir_ids, self.num_years)
            
            # Convert to annual values in km³
            return (self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,
                    self.objective_weights['objective_2'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_2'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_2'][2]*min_flow_deficit
                    + self.objective_weights['objective_2'][3]*flooding_volume
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'))

class DeapThreeObjectiveOptimizer(DeapMultiObjectiveOptimizer):
    """
    Three-objective optimizer using DEAP.
    Objectives: regular demand deficit, priority demand deficit, and minimum flow deficit.
    """
    def __init__(self, *args, **kwargs):
        """Initialize with three-objective fitness weights (-1.0, -1.0, -1.0 for minimization)"""
        kwargs["weights"] = (-1.0, -1.0, -1.0)
        super().__init__(*args, **kwargs)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual for three objectives.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            tuple: Fitness values for regular demand deficit, priority demand deficit, and minimum flow deficit
        """
        try:
            # Decode the individual's genes into parameters
            reservoir_params, hydroworks_params = decode_individual(self, individual)
            
            # Create a copy of the base system to evaluate
            system = copy.deepcopy(self.base_system)
            
            # Apply parameters to the system
            for res_id, params in reservoir_params.items():
                system.graph.nodes[res_id]['node'].set_release_params(params)
            for hw_id, params in hydroworks_params.items():
                system.graph.nodes[hw_id]['node'].set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)
            
            # Objective 1: Regular demand deficit
            low_priority_demand_deficit = regular_demand_deficit(system, self.regular_demand_ids, self.dt, self.num_years)
            
            # Objective 2: Priority demand deficit
            high_priority_demand_deficit = priority_demand_deficit(system, self.priority_demand_ids, self.dt, self.num_years)
            
            # Objective 3: Minimum flow deficit
            min_flow_deficit = sink_node_min_flow_deficit(system, self.sink_ids, self.dt, self.num_years)

            # Objective 4: Flooding volume (not used in this optimizer)
            flooding_volume = total_spillage(system, self.hydroworks_ids, self.reservoir_ids, self.num_years)
            
            # Convert to annual values in km³
            return ( self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,
                    self.objective_weights['objective_2'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_2'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_2'][2]*min_flow_deficit
                    + self.objective_weights['objective_2'][3]*flooding_volume,
                    self.objective_weights['objective_3'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_3'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_3'][2]*min_flow_deficit
                    + self.objective_weights['objective_3'][3]*flooding_volume
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'), float('inf'))

class DeapFourObjectiveOptimizer(DeapMultiObjectiveOptimizer):
    """
    Four-objective optimizer using DEAP.
    Objectives: regular demand deficit, priority demand deficit, minimum flow deficit, and spillage.
    """
    def __init__(self, *args, **kwargs):
        """Initialize with four-objective fitness weights (-1.0, -1.0, -1.0, -1.0 for minimization)"""
        kwargs["weights"] = (-1.0, -1.0, -1.0, -1.0)
        super().__init__(*args, **kwargs)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual for four objectives.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            tuple: Fitness values for regular demand deficit, priority demand deficit, minimum flow deficit, and spillage
        """
        try:
            # Decode the individual's genes into parameters
            reservoir_params, hydroworks_params = decode_individual(self, individual)
            
            # Create a copy of the base system to evaluate
            system = copy.deepcopy(self.base_system)
            
            # Apply parameters to the system
            for res_id, params in reservoir_params.items():
                system.graph.nodes[res_id]['node'].set_release_params(params)
            for hw_id, params in hydroworks_params.items():
                system.graph.nodes[hw_id]['node'].set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)

            # Objective 1: Regular demand deficit
            low_priority_demand_deficit = regular_demand_deficit(system, self.regular_demand_ids, self.dt, self.num_years)
            # Objective 2: Priority demand deficit
            high_priority_demand_deficit = priority_demand_deficit(system, self.priority_demand_ids, self.dt, self.num_years)
            # Objective 3: Minimum flow deficit
            min_flow_deficit = sink_node_min_flow_deficit(system, self.sink_ids, self.dt, self.num_years)
            # Objective 4: Spillage
            flooding_volume = total_spillage(system, self.hydroworks_ids, self.reservoir_ids, self.num_years)
            
            # Convert to annual values in km³
            return (self.objective_weights['objective_1'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_1'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_1'][2]*min_flow_deficit
                    + self.objective_weights['objective_1'][3]*flooding_volume,
                    self.objective_weights['objective_2'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_2'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_2'][2]*min_flow_deficit
                    + self.objective_weights['objective_2'][3]*flooding_volume,
                    self.objective_weights['objective_3'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_3'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_3'][2]*min_flow_deficit
                    + self.objective_weights['objective_3'][3]*flooding_volume,
                    self.objective_weights['objective_4'][0]*low_priority_demand_deficit
                    + self.objective_weights['objective_4'][1]*high_priority_demand_deficit
                    + self.objective_weights['objective_4'][2]*min_flow_deficit
                    + self.objective_weights['objective_4'][3]*flooding_volume
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'), float('inf'), float('inf'))'''