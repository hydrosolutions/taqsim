import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import copy
import os
from typing import Dict, List, Tuple, Union, Optional, Any
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks, SinkNode
from .objectives import (
    regular_demand_deficit,
    priority_demand_deficit,
    min_flow_deficit,
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
        weights: tuple = (-1.0,)
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
        self.population_size = population_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.dt = base_system.dt
        
        # Extract node IDs from the system
        self.reservoir_ids = StorageNode.all_ids
        self.demand_ids = DemandNode.all_ids
        self.sink_ids = SinkNode.all_ids
        self.hydroworks_ids = HydroWorks.all_ids
        self.hydroworks_targets = {}

        # For multi-objective: identify demand types
        self.priority_demand_ids = DemandNode.high_priority_demand_ids

        self.regular_demand_ids = DemandNode.low_priority_demand_ids

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
        self.toolbox.register("select", tools.selTournament, tournsize=5)

        # Initialize convergence history
        self.history = {}

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual. Must be implemented by subclasses.
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            tuple: Fitness value(s) for the individual
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            dict: Results of the optimization
        """
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Set up statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        
        # Run genetic algorithm
        final_pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen,
            stats=stats, verbose=True
        )
        
        # Store convergence history
        self.history['min'] = logbook.select("min")
        self.history['avg'] = logbook.select("avg")
        self.history['std'] = logbook.select("std")
        
        # Get best individual
        best_ind = tools.selBest(final_pop, 1)[0]
        
        # Decode best individual parameters
        reservoir_params, hydroworks_params = decode_individual(self, best_ind)
        
        # Return results in a format similar to pymoo
        return {
            'success': True,
            'message': "Optimization completed successfully",
            'population_size': self.population_size,
            'generations': self.ngen,
            'crossover_probability': self.cxpb,
            'mutation_probability': self.mutpb,
            'objective_value': best_ind.fitness.values[0] if len(best_ind.fitness.values) == 1 else None,
            'objective_values': best_ind.fitness.values if len(best_ind.fitness.values) > 1 else None,
            'optimal_reservoir_parameters': reservoir_params,
            'optimal_hydroworks_parameters': hydroworks_params
        }

    def plot_convergence(self, save_path=None):
        """
        Plot the convergence history.
        
        Args:
            save_path: Optional path to save the plot
        """
        if 'min' in self.history and 'avg' in self.history and 'std' in self.history:
            # Single-objective
            plt.figure(figsize=(10, 6))
            gens = range(len(self.history['min']))
            
            # Plot minimum fitness
            plt.plot(gens, self.history['min'], 'b-', label='Best Fitness')
            
            # Plot average fitness with standard deviation
            plt.plot(gens, self.history['avg'], 'r-', label='Average Fitness')
            plt.fill_between(gens,
                             np.array(self.history['avg']) - np.array(self.history['std']),
                             np.array(self.history['avg']) + np.array(self.history['std']),
                             alpha=0.2, color='r')
            
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Genetic Algorithm Convergence')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Determine save path
            if save_path is None:
                directory = './model_output/deap/convergence/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = f'{directory}/convergence_pop{self.population_size}_ngen{self.ngen}_cxpb{self.cxpb}_mutpb{self.mutpb}.png'
            
            plt.savefig(save_path)
            plt.close()
            return save_path
        
        elif any(key.startswith('min_obj') for key in self.history):
            # Multi-objective
            obj_nums = sorted(set(int(k[-1]) for k in self.history if k.startswith('min_obj')))
            n_obj = len(obj_nums)
            
            plt.figure(figsize=(10, 4 * n_obj))
            gens = range(len(self.history[f'min_obj{obj_nums[0]}']))
            
            for i, obj in enumerate(obj_nums):
                plt.subplot(n_obj, 1, i + 1)
                plt.plot(gens, self.history[f'min_obj{obj}'], 'b-', label='Best Fitness')
                plt.plot(gens, self.history[f'avg_obj{obj}'], 'r-', label='Average Fitness')
                plt.fill_between(gens,
                                 np.array(self.history[f'avg_obj{obj}']) - np.array(self.history[f'std_obj{obj}']),
                                 np.array(self.history[f'avg_obj{obj}']) + np.array(self.history[f'std_obj{obj}']),
                                 alpha=0.2, color='r')
                plt.xlabel('Generation')
                plt.ylabel(f'Objective {obj}')
                plt.title(f'Objective {obj} Convergence')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            
            # Determine save path
            if save_path is None:
                directory = './model_output/deap/convergence/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = f'{directory}/multiobj_convergence_pop{self.population_size}_ngen{self.ngen}_cxpb{self.cxpb}_mutpb{self.mutpb}.png'
            
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            print("No convergence history available to plot.")
            return None


class DeapSingleObjectiveOptimizer(DeapOptimizer):
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
            total_penalty = 0
            total_penalty += regular_demand_deficit(system, self.regular_demand_ids, self.dt, self.num_years)
            total_penalty += priority_demand_deficit(system, self.priority_demand_ids, self.dt, self.num_years)
            total_penalty += min_flow_deficit(system, self.sink_ids, self.dt, self.num_years)
            total_penalty += total_spillage(system, self.hydroworks_ids, self.reservoir_ids, self.num_years)
            
            return (total_penalty,)
        
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
        
        Args:
            individual: The individual to evaluate
            
        Returns:
            tuple: Fitness values for each objective
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def optimize(self):
        """
        Run multi-objective genetic algorithm optimization using NSGA-II selection.
        Handles 2, 3, or 4 objectives with proper statistics tracking.
        
        Returns:
            dict: Results of the optimization including Pareto front
        """
        # Determine number of objectives from weights
        num_objectives = len(self.creator.FitnessMulti.weights)
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Initialize history keys for all objectives
        for i in range(1, num_objectives + 1):
            self.history[f'min_obj{i}'] = []
            self.history[f'avg_obj{i}'] = []
            self.history[f'std_obj{i}'] = []
        
        # Set up statistics tracking for each objective
        multi_stats = tools.MultiStatistics()
        for i in range(1, num_objectives + 1):
            obj_stats = tools.Statistics(lambda ind, idx=i-1: ind.fitness.values[idx])
            obj_stats.register("min", np.min)
            obj_stats.register("avg", np.mean)
            obj_stats.register("std", np.std)
            multi_stats.register(f"obj{i}", obj_stats)
        
        # Create the logbook for statistics
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + multi_stats.fields
        
        # Record the initial statistics
        record = multi_stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        print(logbook.stream)
        
        # Begin the evolution
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals (using NSGA-II selection)
            offspring = self.toolbox.select(pop, len(pop))
            
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.cxpb:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values
            
            # Apply mutation
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
            record = multi_stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
                
            # Store the history for all objectives
            for i in range(1, num_objectives + 1):
                self.history[f'min_obj{i}'].append(record[f'obj{i}']['min'])
                self.history[f'avg_obj{i}'].append(record[f'obj{i}']['avg'])
                self.history[f'std_obj{i}'].append(record[f'obj{i}']['std'])
        
        # Extract the Pareto front
        self.pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        # Get the compromise solution (weighted equally across objectives)
        if num_objectives > 1:
            if num_objectives == 2:
                # For 2 objectives, find knee point using perpendicular distance to utopia line
                f_min = np.array([min(ind.fitness.values[i] for ind in self.pareto_front) for i in range(num_objectives)])
                f_max = np.array([max(ind.fitness.values[i] for ind in self.pareto_front) for i in range(num_objectives)])

                # Normalize objectives to [0,1]
                normalized_fronts = []
                for ind in self.pareto_front:
                    normalized = (np.array(ind.fitness.values) - f_min) / (f_max - f_min)
                    normalized_fronts.append((ind, normalized))

                # Calculate distance to utopia line (y = x for minimization)
                best_dist = float('inf')
                best_ind = None

                for ind, norm_obj in normalized_fronts:
                    # Distance to line y = x is |norm_obj[0] - norm_obj[1]| / sqrt(2)
                    dist = abs(norm_obj[0] - norm_obj[1]) / np.sqrt(2)
                    # Distance to utopia point (0,0) is sqrt(norm_obj[0]^2 + norm_obj[1]^2)
                    dist_to_utopia = np.sqrt(norm_obj[0]**2 + norm_obj[1]**2)

                    # Combine both metrics
                    combined_metric = 0.5 * dist + 0.5 * dist_to_utopia

                    if combined_metric < best_dist:
                        best_dist = combined_metric
                        best_ind = ind
            else:
                # For 3+ objectives, find individual closest to utopia point in normalized space
                f_min = np.array([min(ind.fitness.values[i] for ind in self.pareto_front) for i in range(num_objectives)])
                f_max = np.array([max(ind.fitness.values[i] for ind in self.pareto_front) for i in range(num_objectives)])

                # Handle division by zero
                range_obj = f_max - f_min
                range_obj[range_obj == 0] = 1.0  # Avoid division by zero

                best_dist = float('inf')
                best_ind = None

                for ind in self.pareto_front:
                    # Normalize fitness values to [0,1]
                    norm_obj = (np.array(ind.fitness.values) - f_min) / range_obj
                    
                    # Distance to utopia point (origin in normalized space)
                    dist = np.sqrt(np.sum(norm_obj**2))
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_ind = ind
        else:
            # For single objective, just get the best individual
            best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
        
        # Decode best individual parameters
        reservoir_params, hydroworks_params = decode_individual(self, best_ind)

        # Return results in a format similar to pymoo
        return {
            'success': True,
            'message': f"{num_objectives}-objective optimization completed successfully",
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

            # Calculate number of years for annual averaging
            num_years = self.num_time_steps / 12
            
            # Objective 1: Demand deficit
            demand_deficit = 0
            for node_id in self.demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                demand_deficit += np.sum(deficit)
            
            # Objective 2: Minimum flow deficit
            min_flow_deficit = 0
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                min_flow_deficit += total_deficit_volume
            
            # Convert to annual values in km³
            return (
                float(demand_deficit) / num_years / 1e9,
                float(min_flow_deficit) / num_years / 1e9
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'))

    def optimize(self):
        """
        Run multi-objective genetic algorithm optimization using mu+lambda with NSGA-II selection.
        Returns:
            dict: Results of the optimization including Pareto front
        """
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Initialize history keys for both objectives
        for i in range(1, 3):
            self.history[f'min_obj{i}'] = []
            self.history[f'avg_obj{i}'] = []
            self.history[f'std_obj{i}'] = []            

        # Initialize statistics tracking for both objectives
        stats_obj1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_obj1.register("min", np.min)
        stats_obj1.register("avg", np.mean)
        stats_obj1.register("std", np.std)
        
        stats_obj2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_obj2.register("min", np.min)
        stats_obj2.register("avg", np.mean)
        stats_obj2.register("std", np.std)
        
        # Combine statistics into a multi-statistics object
        mstats = tools.MultiStatistics(obj1=stats_obj1, obj2=stats_obj2)
        
        # Create the logbook for statistics
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + mstats.fields
        
        # Record the initial statistics
        record = mstats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        print(logbook.stream)
        
        # Begin the evolution
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals (using NSGA-II selection)
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
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
                
            # Store the history for both objectives
            self.history['min_obj1'].append(record['obj1']['min'])
            self.history['avg_obj1'].append(record['obj1']['avg'])
            self.history['std_obj1'].append(record['obj1']['std'])
            self.history['min_obj2'].append(record['obj2']['min'])
            self.history['avg_obj2'].append(record['obj2']['avg'])
            self.history['std_obj2'].append(record['obj2']['std'])
        
        # Extract the Pareto front
        self.pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        # Get the overall best individual based on a weighted sum of both objectives
        best_ind = min(pop, key=lambda ind: sum(ind.fitness.values) / len(ind.fitness.values))
        reservoir_params, hydroworks_params = decode_individual(self, best_ind)

        # Return results in a format similar to pymoo
        return {
            'success': True,
            'message': "Two-objective optimization completed successfully",
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

            # Calculate number of years for annual averaging
            num_years = self.num_time_steps / 12
            
            # Objective 1: Regular demand deficit
            regular_demand_deficit = 0
            for node_id in self.regular_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                regular_demand_deficit += np.sum(deficit)
            
            # Objective 2: Priority demand deficit
            priority_demand_deficit = 0
            for node_id in self.priority_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                priority_demand_deficit += np.sum(deficit)
            
            # Objective 3: Minimum flow deficit
            min_flow_deficit = 0
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                min_flow_deficit += total_deficit_volume
            
            # Convert to annual values in km³
            return (
                float(regular_demand_deficit) / num_years / 1e9,
                float(priority_demand_deficit) / num_years / 1e9,
                float(min_flow_deficit) / num_years / 1e9
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'), float('inf'))

    def optimize(self):
        """
        Run multi-objective genetic algorithm optimization using mu+lambda with NSGA-II selection.
        
        Returns:
            dict: Results of the optimization including Pareto front
        """
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Initialize history keys for all objectives
        for i in range(1, 4):
            self.history[f'min_obj{i}'] = []
            self.history[f'avg_obj{i}'] = []
            self.history[f'std_obj{i}'] = []            
            
        # Initialize statistics tracking for all three objectives
        stats_obj1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_obj1.register("min", np.min)
        stats_obj1.register("avg", np.mean)
        stats_obj1.register("std", np.std)
        
        stats_obj2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_obj2.register("min", np.min)
        stats_obj2.register("avg", np.mean)
        stats_obj2.register("std", np.std)
        
        stats_obj3 = tools.Statistics(lambda ind: ind.fitness.values[2])
        stats_obj3.register("min", np.min)
        stats_obj3.register("avg", np.mean)
        stats_obj3.register("std", np.std)
        
        # Combine statistics into a multi-statistics object
        mstats = tools.MultiStatistics(obj1=stats_obj1, obj2=stats_obj2, obj3=stats_obj3)
        
        # Create the logbook for statistics
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + mstats.fields
        
        # Record the initial statistics
        record = mstats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        print(logbook.stream)
        
        # Begin the evolution
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals (using NSGA-II selection)
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
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
                
            # Store the history for all three objectives
            self.history['min_obj1'].append(record['obj1']['min'])
            self.history['avg_obj1'].append(record['obj1']['avg'])
            self.history['std_obj1'].append(record['obj1']['std'])
            self.history['min_obj2'].append(record['obj2']['min'])
            self.history['avg_obj2'].append(record['obj2']['avg'])
            self.history['std_obj2'].append(record['obj2']['std'])
            self.history['min_obj3'].append(record['obj3']['min'])
            self.history['avg_obj3'].append(record['obj3']['avg'])
            self.history['std_obj3'].append(record['obj3']['std'])
        
        # Extract the Pareto front
        self.pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        # Get the overall best individual based on a weighted sum of all three objectives
        best_ind = min(pop, key=lambda ind: sum(ind.fitness.values) / len(ind.fitness.values))
        reservoir_params, hydroworks_params = decode_individual(self, best_ind)

        # Return results in a format similar to pymoo
        return {
            'success': True,
            'message': "Three-objective optimization completed successfully",
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

            # Calculate number of years for annual averaging
            num_years = self.num_time_steps / 12
            
            # Objective 1: Regular demand deficit
            regular_demand_deficit = 0
            for node_id in self.regular_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                deficit = np.array(demand_node.unmet_demand)*self.dt
                regular_demand_deficit += np.sum(deficit)
            
            # Objective 2: Priority demand deficit
            priority_demand_deficit = 0
            for node_id in self.priority_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                deficit = np.array(demand_node.unmet_demand)*self.dt
                priority_demand_deficit += np.sum(deficit)
            
            # Objective 3: Minimum flow deficit
            min_flow_deficit = 0
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                min_flow_deficit += total_deficit_volume
            
            # Objective 4: Total spillage
            total_spillage = 0
            for node_id in self.hydroworks_ids:
                total_spillage += np.sum(system.graph.nodes[node_id]['node'].spill_register)
            for node_id in self.reservoir_ids:
                total_spillage += np.sum(system.graph.nodes[node_id]['node'].spillway_register)
            
            # Convert to annual values in km³
            return (
                float(regular_demand_deficit) / num_years / 1e9,
                float(priority_demand_deficit) / num_years / 1e9,
                float(min_flow_deficit) / num_years / 1e9,
                float(total_spillage) / num_years / 1e9
            )
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'), float('inf'), float('inf'))

    def optimize(self):
        """
        Run multi-objective genetic algorithm optimization using mu+lambda with NSGA-II selection.
        Returns:
            dict: Results of the optimization including Pareto front
        """
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Initialize history keys for all objectives
        for i in range(1, 5):
            self.history[f'min_obj{i}'] = []
            self.history[f'avg_obj{i}'] = []
            self.history[f'std_obj{i}'] = []            

        # Initialize statistics tracking for all four objectives
        stats_obj1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_obj1.register("min", np.min)
        stats_obj1.register("avg", np.mean)
        stats_obj1.register("std", np.std)
        
        stats_obj2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_obj2.register("min", np.min)
        stats_obj2.register("avg", np.mean)
        stats_obj2.register("std", np.std)
        
        stats_obj3 = tools.Statistics(lambda ind: ind.fitness.values[2])
        stats_obj3.register("min", np.min)
        stats_obj3.register("avg", np.mean)
        stats_obj3.register("std", np.std)
        
        stats_obj4 = tools.Statistics(lambda ind: ind.fitness.values[3])
        stats_obj4.register("min", np.min)
        stats_obj4.register("avg", np.mean)
        stats_obj4.register("std", np.std)
        
        # Combine statistics into a multi-statistics object
        mstats = tools.MultiStatistics(obj1=stats_obj1, obj2=stats_obj2, obj3=stats_obj3, obj4=stats_obj4)
        
        # Create the logbook for statistics
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + mstats.fields
        
        # Record the initial statistics
        record = mstats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        print(logbook.stream)
        
        # Begin the evolution
        for gen in range(1, self.ngen + 1):
            # Select the next generation individuals (using NSGA-II selection)
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
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
                
            # Store the history for all four objectives
            self.history['min_obj1'].append(record['obj1']['min'])
            self.history['avg_obj1'].append(record['obj1']['avg'])
            self.history['std_obj1'].append(record['obj1']['std'])
            self.history['min_obj2'].append(record['obj2']['min'])
            self.history['avg_obj2'].append(record['obj2']['avg'])
            self.history['std_obj2'].append(record['obj2']['std'])
            self.history['min_obj3'].append(record['obj3']['min'])
            self.history['avg_obj3'].append(record['obj3']['avg'])
            self.history['std_obj3'].append(record['obj3']['std'])
            self.history['min_obj4'].append(record['obj4']['min'])
            self.history['avg_obj4'].append(record['obj4']['avg'])
            self.history['std_obj4'].append(record['obj4']['std'])
        
        # Extract the Pareto front
        self.pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        # Get the overall best individual based on a weighted sum of all four objectives
        best_ind = min(pop, key=lambda ind: sum(ind.fitness.values) / len(ind.fitness.values))
        reservoir_params, hydroworks_params = decode_individual(self, best_ind)

        # Return results in a format similar to pymoo
        return {
            'success': True,
            'message': "Four-objective optimization completed successfully",
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