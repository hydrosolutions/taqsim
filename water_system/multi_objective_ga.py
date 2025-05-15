import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks, SinkNode
import copy
import os

class MultiObjectiveOptimizer:
    """
    Two-objective genetic algorithm optimizer for water systems with multiple
    reservoirs and hydroworks nodes.
    """
    def __init__(self, base_system, start_year, start_month, num_time_steps, ngen=50, population_size=50, cxpb=0.9, mutpb=0.5):
        self.base_system = base_system 
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.population_size = population_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.dt = base_system.dt  # Time step in seconds
        self.reservoir_ids = []
        self.hydroworks_ids = []
        self.hydroworks_targets = {}

        # Dictionary to store reservoir-specific bounds
        self.reservoir_bounds = {}
        
        # Identify reservoirs and hydroworks in system
        for node_id, node_data in base_system.graph.nodes(data=True):
            if isinstance(node_data['node'], StorageNode):
                self.reservoir_ids.append(node_id)
                reservoir = node_data['node']
                
                # Calculate total outflow capacity using numpy
                total_capacity = reservoir.outflow_edge.capacity
                
                # Set volume-based bounds for the new parameterization
                dead_storage = reservoir.dead_storage
                capacity = reservoir.capacity
                
                # Set reservoir-specific bounds
                self.reservoir_bounds[node_id] = {
                    'Vr': (0, total_capacity*self.dt),              # Target monthly release volume
                    'V1': (dead_storage, capacity),  # Top of buffer zone
                    'V2': (dead_storage, capacity)  # Top of conservation zone
                }

            elif isinstance(node_data['node'], HydroWorks):
                self.hydroworks_ids.append(node_id)
                # Get target nodes for each hydroworks
                self.hydroworks_targets[node_id] = list(node_data['node'].outflow_edges.keys())
        
        # Set up DEAP multi-objective genetic algorithm components
        # Remove any existing FitnessMulti and Individual if they exist
        if 'FitnessMulti' in creator.__dict__:
            del creator.FitnessMulti
        if 'Individual' in creator.__dict__:
            del creator.Individual
            
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self._setup_genetic_operators()
        
        # Store convergence history for both objectives
        self.history = {'min_obj1': [], 'avg_obj1': [], 'std_obj1': [],
                       'min_obj2': [], 'avg_obj2': [], 'std_obj2': [], 
                       'min_obj3': [], 'avg_obj3': [], 'std_obj3': []}
        
        # Store Pareto front
        self.pareto_front = []
        
    def _normalize_distribution(self, values):
        """
        Normalize a list of values to sum to 1.0 using numpy for efficiency.
        
        Args:
            values (list): List of values to normalize
            
        Returns:
            list: Normalized values that sum to 1.0
        """
        values = np.array(values)
        total = np.sum(values)
        if total == 0:
            # If all values are 0, return equal distribution
            return np.full_like(values, 1.0 / len(values))
        return values / total
    
    def _setup_genetic_operators(self):
        """Configure genetic algorithm operators for multiple structures with dynamic bounds"""

        # Create individual and population
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selNSGA2)  # Use NSGA-II selection for multi-objective

    def _mutate_individual(self, individual, indpb=0.5):
        """Custom mutation operator with enforced parameter bounds for monthly parameters"""
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
                    individual[start_idx:end_idx] = self._normalize_distribution(individual[start_idx:end_idx])
            current_idx += 12 * n_targets  # Update index for next hydroworks

        return creator.Individual(individual.tolist()),

    def _crossover(self, ind1, ind2):
        """
        Custom crossover that treats monthly parameters as packages.
        For each month, all parameters of a reservoir or hydrowork for that specific month
        are swapped as a complete unit between parents.
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
                    ind1[start_idx:end_idx] = self._normalize_distribution(tmp2)
                    ind2[start_idx:end_idx] = self._normalize_distribution(tmp1)
                
                hw_current_idx += 12 * n_targets  # Move to next hydrowork's base index
        
        return creator.Individual(ind1.tolist()), creator.Individual(ind2.tolist())

    def _evaluate_individual(self, individual):
        """
        Evaluate multi-objective fitness with three objectives:
        1. Minimize unmet demand for regular demand nodes (weight = 1)
        2. Minimize unmet demand for high-priority demand nodes (weight = 1000)
        3. Minimize deficit in minimum flow requirements at sink nodes
        """
        try:
            # Decode parameters and configure water system
            reservoir_params, hydroworks_params = self._decode_individual(individual)
            
            # Create and configure water system
            system = copy.deepcopy(self.base_system)

            # Set parameters for all reservoirs
            for res_id, params in reservoir_params.items():
                reservoir_node = system.graph.nodes[res_id]['node']
                reservoir_node.set_release_params(params)
            
            # Set parameters for all hydroworks
            for hw_id, params in hydroworks_params.items():
                hydroworks_node = system.graph.nodes[hw_id]['node']
                hydroworks_node.set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)
            
            # Initialize objective values
            regular_demand_deficit = 0  # Objective 1: Regular demand nodes (weight = 1)
            priority_demand_deficit = 0  # Objective 2: High-priority demand nodes (weight = 1000)
            minflow_deficit = 0  # Objective 3: Minimum flow requirements at sink nodes
            
            # Calculate penalties for each component
            for node_id, node_data in system.graph.nodes(data=True):
                node = node_data['node']
                
                if isinstance(node, DemandNode):
                    # Get demand data
                    demand = np.array([node.demand_rates[t] for t in range(self.num_time_steps)])
                    satisfied = np.array(node.satisfied_demand_total)
                    deficit = (demand - satisfied) * system.dt
                    
                    # Separate nodes based on their weight
                    if node.weight > 1:  # High-priority node
                        priority_demand_deficit += np.sum(deficit)
                    elif node.weight == 1:  # Regular node
                        regular_demand_deficit += np.sum(deficit)

                
                elif isinstance(node, SinkNode):
                    # Objective 3: Calculate minimum flow deficit
                    min_flow_volume = sum(deficit * system.dt for deficit in node.flow_deficits)
                    minflow_deficit += min_flow_volume  # Use the weight if needed
            
            # Handle spills (distribute evenly among objectives or based on priority)
            spill_penalty = 0
            for node_id, node_data in system.graph.nodes(data=True):
                node = node_data['node']
                
                if hasattr(node, 'spillway_register'):
                    spill_penalty += np.sum(node.spillway_register)
                
                elif hasattr(node, 'spill_register'):
                    spill_penalty += np.sum(node.spill_register)
            
            # Place Spill penalty to priority objective
            priority_demand_deficit += spill_penalty
            
            
            return (regular_demand_deficit, priority_demand_deficit, minflow_deficit)
        
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'), float('inf'), float('inf'))
    
    def _decode_individual(self, individual):
        """Decode individual genes into monthly reservoir and hydroworks parameters"""
        genes_per_reservoir_month = 3
        reservoir_params = {}
        hydroworks_params = {}
        
        # Convert individual to numpy array for efficient slicing
        individual = np.array(individual)
        
        # Decode reservoir parameters
        current_idx = 0
        for res_id in self.reservoir_ids:
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
        
        # Decode hydroworks parameters (unchanged)
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            dist_params = {}
            
            for target in self.hydroworks_targets[hw_id]:
                dist_params[target] = np.zeros(12)
            
            for month in range(12):
                start_idx = current_idx + month * n_targets
                end_idx = start_idx + n_targets
                
                dist_values = individual[start_idx:end_idx]
                normalized_dist = self._normalize_distribution(dist_values)
                
                for target, value in zip(self.hydroworks_targets[hw_id], normalized_dist):
                    dist_params[target][month] = value
            
            hydroworks_params[hw_id] = dist_params
            current_idx += 12 * n_targets
        
        return reservoir_params, hydroworks_params
   
    def _create_individual(self):
        """Create an individual with parameters for all reservoirs and hydroworks"""
        genes = []

        # Add reservoir genes
        for res_id in self.reservoir_ids:
            bounds = self.reservoir_bounds[res_id]
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
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            for month in range(12):
                # Create random distribution that sums to 1
                dist = np.random.random(n_targets)
                normalized_dist = self._normalize_distribution(dist)
                genes.extend(normalized_dist)

        return creator.Individual(genes)

    def optimize(self):
        """Run multi-objective genetic algorithm optimization using mu+lambda with NSGA-II selection for three objectives"""
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
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
        best_ind = min(pop, key=lambda ind: (ind.fitness.values[0] + ind.fitness.values[1] + ind.fitness.values[2])/3)
        reservoir_params, hydroworks_params = self._decode_individual(best_ind)

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
            'optimizer': self  # Include the optimizer instance
        }

    def plot_convergence(self):
        """Plot convergence history for all three objectives"""
        plt.figure(figsize=(15, 15))
        
        # Plot regular demand objective convergence
        plt.subplot(3, 1, 1)
        gens = range(len(self.history['min_obj1']))
        
        plt.plot(gens, self.history['min_obj1'], 'b-', label='Best Fitness')
        plt.plot(gens, self.history['avg_obj1'], 'r-', label='Average Fitness')
        plt.fill_between(gens, 
                        np.array(self.history['avg_obj1']) - np.array(self.history['std_obj1']),
                        np.array(self.history['avg_obj1']) + np.array(self.history['std_obj1']),
                        alpha=0.2, color='r')
        
        plt.xlabel('Generation')
        plt.ylabel('Regular Demand Deficit [m³]')
        plt.title('Regular Demand Deficit Convergence')
        plt.legend()
        plt.grid(True)
        
        # Plot priority demand objective convergence
        plt.subplot(3, 1, 2)
        
        plt.plot(gens, self.history['min_obj2'], 'b-', label='Best Fitness')
        plt.plot(gens, self.history['avg_obj2'], 'r-', label='Average Fitness')
        plt.fill_between(gens, 
                        np.array(self.history['avg_obj2']) - np.array(self.history['std_obj2']),
                        np.array(self.history['avg_obj2']) + np.array(self.history['std_obj2']),
                        alpha=0.2, color='r')
        
        plt.xlabel('Generation')
        plt.ylabel('Priority Demand Deficit [m³]')
        plt.title('Priority Demand Deficit Convergence')
        plt.legend()
        plt.grid(True)
        
        # Plot minimum flow objective convergence
        plt.subplot(3, 1, 3)
        
        plt.plot(gens, self.history['min_obj3'], 'b-', label='Best Fitness')
        plt.plot(gens, self.history['avg_obj3'], 'r-', label='Average Fitness')
        plt.fill_between(gens, 
                        np.array(self.history['avg_obj3']) - np.array(self.history['std_obj3']),
                        np.array(self.history['avg_obj3']) + np.array(self.history['std_obj3']),
                        alpha=0.2, color='r')
        
        plt.xlabel('Generation')
        plt.ylabel('Minimum Flow Deficit [m³]')
        plt.title('Minimum Flow Deficit Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        # Ensure the directory exists
        filename = './model_output/optimisation/deap/three_objective_convergence_pop{self.population_size}_ngen{self.ngen}_cxpb{self.cxpb}_mutpb{self.mutpb}.png'
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename)
        
