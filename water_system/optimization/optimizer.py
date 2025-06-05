import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import copy
import numpy as np
import os
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks, SinkNode
from .objectives import (
    regular_demand_deficit,
    priority_demand_deficit,
    sink_node_min_flow_deficit,
    total_spillage,
    total_unmet_ecological_flow
)
import multiprocessing
# --------------------- SHARED UTILITY FUNCTIONS ---------------------

def normalize_distribution(values: np.ndarray) -> np.ndarray:
    """
    Normalize a list or array of values so that their sum equals 1.0.

    If the sum of the input values is zero, returns an array of equal values that sum to 1.0.

    Args:
        values (np.ndarray): Array or list of values to normalize.

    Returns:
        np.ndarray: Normalized array with the same shape as input, summing to 1.0.
    """
    values = np.array(values)
    total = np.sum(values)
    if total == 0:
        # If all values are 0, return equal distribution
        return np.full_like(values, 1.0 / len(values))
    return values / total

def decode_individual(
    reservoir_ids, hydroworks_ids, hydroworks_targets, individual
):
    """
    Decode a flat list of genes into structured reservoir and hydroworks parameters.

    Args:
        reservoir_ids (list): List of reservoir node IDs.
        hydroworks_ids (list): List of hydroworks node IDs.
        hydroworks_targets (dict): Mapping from hydroworks ID to list of target node IDs.
        individual (list or np.ndarray): Flat list of genes representing an individual.

    Returns:
        tuple: (reservoir_params, hydroworks_params)
            - reservoir_params (dict): Reservoir release parameters per month.
            - hydroworks_params (dict): Hydroworks distribution parameters per month.
    """
    genes_per_reservoir_month = 3
    reservoir_params = {}
    hydroworks_params = {}

    individual = np.array(individual)
    current_idx = 0
    for res_id in reservoir_ids:
        params = {'Vr': [], 'V1': [], 'V2': []}
        for month in range(12):
            start_idx = current_idx + month * genes_per_reservoir_month
            params['Vr'].append(individual[start_idx])
            params['V1'].append(individual[start_idx + 1])
            params['V2'].append(individual[start_idx + 2])
        reservoir_params[res_id] = params
        current_idx += 12 * genes_per_reservoir_month

    for hw_id in hydroworks_ids:
        n_targets = len(hydroworks_targets[hw_id])
        dist_params = {}
        for target in hydroworks_targets[hw_id]:
            dist_params[target] = np.zeros(12)
        for month in range(12):
            start_idx = current_idx + month * n_targets
            end_idx = start_idx + n_targets
            dist_values = individual[start_idx:end_idx]
            total = np.sum(dist_values)
            if total == 0:
                normalized_dist = np.full_like(dist_values, 1.0 / len(dist_values))
            else:
                normalized_dist = dist_values / total
            for target, value in zip(hydroworks_targets[hw_id], normalized_dist):
                dist_params[target][month] = value
        hydroworks_params[hw_id] = dist_params
        current_idx += 12 * n_targets

    return reservoir_params, hydroworks_params

# --------------------- OPTIMIZER CLASS -----------------------------
class DeapOptimizer:
    """
    DEAP-based multi-objective optimizer for water system management.

    Handles setup of the water system, genetic algorithm configuration, and optimization process.
    Supports multiple objectives and parallel evaluation.

    Attributes:
        base_system (WaterSystem): The water system to optimize.
        num_time_steps (int): Number of time steps in the simulation.
        population_size (int): Number of individuals in the population.
        ngen (int): Number of generations to run the algorithm.
        cxpb (float): Probability of crossover.
        mutpb (float): Probability of mutation.
        objective_weights (dict): Mapping of objective names to their weight vectors.
        num_of_objectives (int): Number of objectives in the optimization.
        history (dict): Stores convergence statistics for each objective.
    """
    def __init__(
        self,
        base_system: WaterSystem,
        num_time_steps: int,
        population_size: int,
        ngen: int,
        cxpb: float,
        mutpb: float,
        objective_weights: dict[str, list[int]]
    ) -> None:
        """
        Initialize the DEAP optimizer for the water system.

        Args:
            base_system (WaterSystem): The water system to optimize.
            num_time_steps (int): Number of time steps to simulate.
            population_size (int): Size of the population. Defaults to 50.
            ngen (int): Number of generations. Defaults to 50.
            cxpb (float): Crossover probability. Defaults to 0.65.
            mutpb (float): Mutation probability. Defaults to 0.32.
            objective_weights (dict): Weights for each objective. Defaults to 4 objectives.
        """
        self.base_system = base_system
        self.num_time_steps = num_time_steps
        self.num_years = num_time_steps / 12
        self.dt = base_system.dt

        self.population_size = population_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.objective_weights = objective_weights
        self.num_of_objectives = len(objective_weights)
        
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

        self._setup_toolbox()

    def _setup_toolbox(self):
        """
        Set up the DEAP toolbox, including custom individual and fitness classes,
        genetic operators, and selection method.

        This method also initializes the multiprocessing pool and convergence history.
        """
        # Remove any existing Fitness/Individual to avoid DEAP errors
        if 'FitnessMulti' in creator.__dict__:
            del creator.FitnessMulti
        if 'Individual' in creator.__dict__:
            del creator.Individual

        # Set up DEAP types
        if self.num_of_objectives == 1:
            weights = (-1.0,)
        elif self.num_of_objectives == 2:
            weights = (-1.0, -1.0)
        elif self.num_of_objectives == 3:
            weights = (-1.0, -1.0, -1.0)
        elif self.num_of_objectives == 4:
            weights = (-1.0, -1.0, -1.0, -1.0)
        elif self.num_of_objectives > 4:
            weights = (-1.0,) * self.num_of_objectives

        creator.create("FitnessMulti", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.creator = creator

        # Configure toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Do NOT register "evaluate" with self._evaluate_individual for multiprocessing!
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate_individual)
        if self.num_of_objectives == 1:
            self.toolbox.register("select", tools.selTournament, tournsize=3)
        else:
            self.toolbox.register("select", tools.selNSGA2)
        # Do NOT register a parallel map function here; use self.pool.map directly in optimize
        self.pool = multiprocessing.Pool()
        # Initialize convergence history
        self.history = {}

    def create_individual(self):
        """
        Create a new individual with randomly initialized genes for all reservoirs and hydroworks.

        Returns:
            Individual: A DEAP Individual object with genes for all parameters.
        """
        genes = []

        # Add reservoir genes
        for res_id in self.reservoir_ids:
            bounds = self.reservoir_bounds[res_id]
            for month in range(12):
                Vr = np.random.uniform(bounds['Vr'][0], bounds['Vr'][1])
                V1_min = bounds['V1'][0]
                V1_max = bounds['V1'][1]
                V1 = np.random.uniform(V1_min, V1_max)
                V2_min = max(bounds['V2'][0], V1)
                V2_max = bounds['V2'][1]
                V2 = np.random.uniform(V2_min, V2_max)
                genes.extend([Vr, V1, V2])

        # Add hydroworks genes
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            for month in range(12):
                dist = np.random.random(n_targets)
                normalized_dist = normalize_distribution(dist)
                genes.extend(normalized_dist)

        return self.creator.Individual(genes)

    def crossover(self, ind1, ind2):
        """
        Custom crossover operator for individuals.

        Swaps all parameters for each month as a package between two individuals,
        separately for reservoirs and hydroworks. Hydroworks distributions are normalized after swapping.

        Args:
            ind1 (Individual): First parent individual.
            ind2 (Individual): Second parent individual.

        Returns:
            tuple: (child1, child2) - Two new individuals after crossover.
        """
        ind1, ind2 = np.array(ind1), np.array(ind2)
        
        # Handle reservoirs month by month
        genes_per_month = 3 
        num_reservoirs = len(self.reservoir_ids)
        
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
            for hw_id in self.hydroworks_ids:
                n_targets = len(self.hydroworks_targets[hw_id])
                
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
        return self.creator.Individual(ind1.tolist()), self.creator.Individual(ind2.tolist())

    def mutate_individual(self, individual, indpb=0.5):
        """
        Custom mutation operator for individuals.

        Mutates reservoir and hydroworks parameters with independent probability,
        enforcing parameter bounds and relationships (e.g., V2 > V1).
        Hydroworks distributions are renormalized after mutation.

        Args:
            individual (Individual): The individual to mutate.
            indpb (float, optional): Probability of mutating each gene. Defaults to 0.5.

        Returns:
            tuple: (mutated_individual,)
        """
        genes_per_reservoir_month = 3

        individual = np.array(individual)
        # Track which months need renormalization for each hydroworks
        hw_updates = set()

        # Mutate reservoir parameters for each month
        for res_idx, res_id in enumerate(self.reservoir_ids):
            for month in range(12):
                start_idx = (res_idx * 12 * genes_per_reservoir_month) + (month * genes_per_reservoir_month)

                for param_idx, param_name in enumerate(['Vr', 'V1', 'V2']):
                    if random.random() < indpb:
                        bounds = self.reservoir_bounds[res_id][param_name]
                        value = np.random.uniform(bounds[0], bounds[1])

                        # Special handling for volume relationships
                        if param_name == 'V2':
                            # Ensure V2 > V1
                            min_bound = max(bounds[0], individual[start_idx + 1])  # V1 index is 1, V2 must be > V1
                            value = np.random.uniform(min_bound, bounds[1])
                        elif param_name == 'V1':
                            # Ensure V1 < V2 and V1 > dead_storage
                            dead_storage = self.base_system.graph.nodes[res_id]['node'].dead_storage
                            min_bound = max(bounds[0], dead_storage)
                            max_bound = min(bounds[1], individual[start_idx + 2])  # V2 index is 2
                            value = np.random.uniform(min_bound, max_bound)

                        individual[start_idx + param_idx] = value

        # Mutate hydroworks parameters
        hw_genes_start = len(self.reservoir_ids) * 12 * genes_per_reservoir_month
        current_idx = hw_genes_start

        for hw_idx, hw_id in enumerate(self.hydroworks_ids):
            n_targets = len(self.hydroworks_targets[hw_id])
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
            n_targets = len(self.hydroworks_targets[hw_id])
            for month in range(12):
                start_idx = current_idx + (month * n_targets)
                end_idx = start_idx + n_targets
                if end_idx <= len(individual):  # Check array bounds
                    individual[start_idx:end_idx] = normalize_distribution(individual[start_idx:end_idx])
            current_idx += 12 * n_targets  # Update index for next hydroworks

        # Return the creator.Individual type appropriate for the specific optimizer
        return self.creator.Individual(individual.tolist()),

    def optimize(self):
        """
        Run the multi-objective genetic algorithm optimization using NSGA-II selection.

        Handles parallel evaluation, statistics tracking, and Pareto front extraction.

        Returns:
            dict: Results of the optimization, including:
                - 'success' (bool): Whether optimization completed successfully.
                - 'message' (str): Status message.
                - 'population_size' (int): Population size used.
                - 'generations' (int): Number of generations run.
                - 'crossover_probability' (float): Crossover probability used.
                - 'mutation_probability' (float): Mutation probability used.
                - 'objective_values' (tuple): Fitness values of the best individual.
                - 'optimal_reservoir_parameters' (dict): Best reservoir parameters.
                - 'optimal_hydroworks_parameters' (dict): Best hydroworks parameters.
                - 'pareto_front' (list): List of Pareto-optimal individuals.
                - 'optimizer' (DeapOptimizer): Reference to the optimizer instance.
        """
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)

        # Prepare static arguments for evaluation (everything except the individual)
        static_eval_args = (
            self.base_system,
            self.num_time_steps,
            self.num_years,
            self.dt,
            self.objective_weights,
            self.num_of_objectives,
            self.reservoir_ids,
            self.reservoir_bounds,
            self.hydroworks_ids,
            self.hydroworks_targets,
            self.regular_demand_ids,
            self.priority_demand_ids,
            self.sink_ids
        )

        # Evaluate initial population in parallel
        fitnesses = list(self.pool.map(
            evaluate_individual,
            [static_eval_args + (ind,) for ind in pop]
        ))
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
        logbook.header = ['gen', 'nevals']
        for i in range(n_obj):
            logbook.header += [f'obj{i+1}_avg', f'obj{i+1}_min', f'obj{i+1}_std']

        # Record the initial statistics
        record = mstats.compile(pop)
        flat_record = {'gen': 0, 'nevals': len(pop)}
        for i in range(n_obj):
            flat_record[f'obj{i+1}_avg'] = record[f'obj{i+1}']['avg']
            flat_record[f'obj{i+1}_min'] = record[f'obj{i+1}']['min']
            flat_record[f'obj{i+1}_std'] = record[f'obj{i+1}']['std']
        logbook.record(**flat_record)
        print(logbook.stream)

        # Begin the evolution
        for gen in range(1, self.ngen + 1):
            offspring = self.toolbox.select(pop, len(pop))
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
            if invalid_ind:
                fitnesses = list(self.pool.map(
                    evaluate_individual,
                    [static_eval_args + (ind,) for ind in invalid_ind]
                ))
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
        reservoir_params, hydroworks_params = decode_individual(self.reservoir_ids, self.hydroworks_ids, self.hydroworks_targets, best_ind)

        self.pool.close()
        self.pool.join()
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
        Plot and save the convergence history for each objective.

        Each objective is plotted in a separate subplot, showing best, average, and standard deviation
        per generation. The plot is saved to '/model_output/optimization/convergence.png'.
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
        save_path = os.path.join(os.getcwd(), 'model_output', 'optimization')
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'convergence.png')
        plt.savefig(file_path)
        plt.close(fig)
        print(f"Convergence plot saved to {file_path}")

    def plot_total_objective_convergence(self):
        """
        Plot and save the convergence of the sum of all objectives over generations.

        Shows best, average, and standard deviation of the total objective value per generation.
        The plot is saved to '/model_output/optimization/total_objective_convergence.png'.
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
        save_path = os.path.join(os.getcwd(), 'model_output', 'optimization')
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, 'total_objective_convergence.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Total objective convergence plot saved to {file_path}")

# --------------------- EVALUATION FUNCTION -----------------------------
# This function is used to evaluate an individual in parallel.
# It is outside the DeapOptimizer class to allow multiprocessing.
def evaluate_individual(args):
    """
    Evaluate a single individual for the genetic algorithm (for multiprocessing).

    Decodes the individual's genes, applies them to a copy of the water system,
    runs the simulation, and computes all objectives. Returns a tuple of weighted
    objective values.

    Args:
        args (tuple): All static and individual-specific arguments required for evaluation.

    Returns:
        tuple: Weighted objective values for the individual.
    """
    (
        base_system,
        num_time_steps,
        num_years,
        dt,
        objective_weights,
        num_of_objectives,
        reservoir_ids,
        reservoir_bounds,
        hydroworks_ids,
        hydroworks_targets,
        regular_demand_ids,
        priority_demand_ids,
        sink_ids,
        individual
    ) = args


    try:
        # Decode the individual's genes into parameters
        reservoir_params, hydroworks_params = decode_individual(
            reservoir_ids, hydroworks_ids, hydroworks_targets, individual
        )

        # Create a copy of the base system to evaluate
        system = copy.deepcopy(base_system)

        # Apply parameters to the system
        for res_id, params in reservoir_params.items():
            system.graph.nodes[res_id]['node'].set_release_params(params)
        for hw_id, params in hydroworks_params.items():
            system.graph.nodes[hw_id]['node'].set_distribution_parameters(params)

        # Run simulation
        system.simulate(num_time_steps)

        # Objective 1: Regular demand deficit
        low_priority_demand_deficit = regular_demand_deficit(system, regular_demand_ids, dt, num_years)
        # Objective 2: Priority demand deficit
        high_priority_demand_deficit = priority_demand_deficit(system, priority_demand_ids, dt, num_years)
        # Objective 3: Minimum flow deficit
        sink_min_flow_deficit = sink_node_min_flow_deficit(system, sink_ids, dt, num_years)
        # Objective 4: Spillage
        flooding_volume = total_spillage(system, hydroworks_ids, reservoir_ids, num_years)
        # Objective 5: Unmet ecological flow at edges
        edge_ecological_flow_deficit = total_unmet_ecological_flow(system, dt, num_years)

        base_objectives = np.array([
            low_priority_demand_deficit,
            high_priority_demand_deficit,
            flooding_volume,
            sink_min_flow_deficit,
            edge_ecological_flow_deficit
        ])

        # Prepare the weights for each objective as a list of arrays
        weights_list = []
        for i in range(1, num_of_objectives + 1):
            weights = np.array(objective_weights[f'objective_{i}'])
            weights_list.append(weights)

        # Calculate the weighted sum for each objective
        results = tuple(np.dot(weights, base_objectives) for weights in weights_list)
        return results

    except Exception as e:
        print(f"Error evaluating individual (static): {str(e)}")
        return (float('inf'),) * num_of_objectives
