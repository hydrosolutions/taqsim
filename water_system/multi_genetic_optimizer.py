import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks, SinkNode
import copy

class MultiGeneticOptimizer:
    """
    Enhanced genetic algorithm optimizer for water systems with multiple reservoirs
    and hydroworks nodes.
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
                
                # Get water level bounds from HVA data
                min_level = reservoir.dead_storage_level
                max_level = reservoir.hva_data['max_waterlevel']
                
                # Calculate total outflow capacity using numpy
                total_capacity = np.sum([edge.capacity for edge in reservoir.outflow_edges.values()])
                
                # Set reservoir-specific bounds
                self.reservoir_bounds[node_id] = {
                    'h1': (min_level, max_level),  # Full range for h1
                    'h2': (min_level, max_level),  # Full range for h2
                    'w': (0, total_capacity),      # From 0 to total outflow capacity
                    'm1': (1.47, 1.57),            # Standard slopes
                    'm2': (1.47, 1.57)             # Standard slopes
                }

            elif isinstance(node_data['node'], HydroWorks):
                self.hydroworks_ids.append(node_id)
                # Get target nodes for each hydroworks
                self.hydroworks_targets[node_id] = list(node_data['node'].outflow_edges.keys())
        
        # Set up DEAP genetic algorithm components
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))  # Minimize objective
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self._setup_genetic_operators()
        
        # Store convergence history
        self.history = {'min': [], 'avg': [], 'std': []}
        
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
    
    def _bound_parameter(self, value, bounds):
        """
        Helper method to ensure a parameter stays within bounds using numpy for efficiency.
        
        Args:
            value (float): Parameter value to check
            bounds (tuple): (min, max) bounds for the parameter
            
        Returns:
            float: Value clamped to bounds
        """
        return np.clip(value, bounds[0], bounds[1])

    def _setup_genetic_operators(self):
        """Configure genetic algorithm operators for multiple structures with dynamic bounds"""
        # Create attributes for each reservoir's parameters
        for res_id, bounds in self.reservoir_bounds.items():
            for param, (low, high) in bounds.items():
                self.toolbox.register(
                    f"reservoir_{res_id}_{param}",
                    random.uniform,
                    low,
                    high
                )

        # Create individual and population
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=5)

    def _mutate_individual(self, individual, indpb=0.5):
        """Custom mutation operator with enforced parameter bounds using numpy for efficiency"""
        genes_per_reservoir = 5  # 5 parameters per reservoir
        
        # Convert individual to numpy array for vectorized operations
        individual = np.array(individual)
        
        # Track which hydroworks need renormalization
        hw_updates = set()
        
        # Reservoir parameter mutation
        for res_idx, res_id in enumerate(self.reservoir_ids):
            start_idx = res_idx * genes_per_reservoir
            end_idx = start_idx + genes_per_reservoir
            for param_idx, param_name in enumerate(['h1', 'h2', 'w', 'm1', 'm2']):
                if random.random() < indpb:
                    bounds = self.reservoir_bounds[res_id][param_name]
                    value = np.random.uniform(bounds[0], bounds[1])
                    
                    # Special handling for h2 to maintain h1 < h2
                    if param_name == 'h2':
                        min_bound = max(bounds[0], individual[start_idx] + 0.1)
                        value = np.random.uniform(min_bound, bounds[1])
                    elif param_name == 'h1':
                        max_bound = min(bounds[1], individual[start_idx + 1] - 0.1)
                        value = np.random.uniform(bounds[0], max_bound)
                    
                    # Enforce bounds for all parameters
                    individual[start_idx + param_idx] = self._bound_parameter(value, bounds)
        
        # Hydroworks parameter mutation
        hw_genes_start = len(self.reservoir_ids) * genes_per_reservoir
        for hw_idx, hw_id in enumerate(self.hydroworks_ids):
            n_targets = len(self.hydroworks_targets[hw_id])
            start_idx = hw_genes_start + sum(len(self.hydroworks_targets[hw]) for hw in self.hydroworks_ids[:hw_idx])
            end_idx = start_idx + n_targets
            if random.random() < indpb:
                hw_updates.add(hw_id)
                individual[start_idx:end_idx] = np.random.random(n_targets)
        
        # Normalize hydroworks distributions that were modified
        for hw_id in hw_updates:
            n_targets = len(self.hydroworks_targets[hw_id])
            start_idx = hw_genes_start + sum(len(self.hydroworks_targets[hw]) for hw in self.hydroworks_ids[:self.hydroworks_ids.index(hw_id)])
            end_idx = start_idx + n_targets
            individual[start_idx:end_idx] = self._normalize_distribution(individual[start_idx:end_idx])
        
        return creator.Individual(individual.tolist()),

    def _crossover(self, ind1, ind2):
        """
        Custom crossover that treats each reservoir and hydrowork as a complete package.
        Each reservoir (with its 5 parameters) or hydrowork (with its distribution parameters)
        is swapped as a complete unit between parents.
        """
        ind1, ind2 = np.array(ind1), np.array(ind2)
        
        # Handle reservoirs as packages
        genes_per_reservoir = 5  # h1, h2, w, m1, m2
        num_reservoirs = len(self.reservoir_ids)
        
        # Create masks for crossover
        reservoir_mask = np.random.rand(num_reservoirs) < 0.5
        
        for res_idx in range(num_reservoirs):
            if reservoir_mask[res_idx]:
                start_idx = res_idx * genes_per_reservoir
                end_idx = start_idx + genes_per_reservoir
                ind1[start_idx:end_idx], ind2[start_idx:end_idx] = \
                    ind2[start_idx:end_idx], ind1[start_idx:end_idx]
        
        # Handle hydroworks as packages
        hw_start = num_reservoirs * genes_per_reservoir
        current_idx = hw_start
        
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            hw_end = current_idx + n_targets
            
            if random.random() < 0.5:
                ind1[current_idx:hw_end], ind2[current_idx:hw_end] = \
                    ind2[current_idx:hw_end], ind1[current_idx:hw_end]
            
            current_idx = hw_end
        
        # Convert back to list and return as DEAP individuals
        return creator.Individual(ind1.tolist()), creator.Individual(ind2.tolist())

    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual with bound checking"""
        genes_per_reservoir = 5  # 5 parameters per reservoir
        
        try:
            # Decode parameters and continue with evaluation
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
            
            total_penalty = 0
            
            # Calculate weighted demand deficits
            for node_id, node_data in system.graph.nodes(data=True):
                node = node_data['node']
                
                if isinstance(node, DemandNode):
                    demand = np.array([node.get_demand_rate(t) for t in range(self.num_time_steps)])
                    satisfied = np.array(node.satisfied_demand_total)
                    deficit = (demand - satisfied) * system.dt
                    weighted_deficit = deficit * node.weight
                    total_penalty += np.sum(weighted_deficit)
                
                elif isinstance(node, SinkNode):
                    total_deficit_volume = node.get_total_deficit_volume(system.dt)
                    total_penalty += total_deficit_volume * node.weight
                    
                elif hasattr(node, 'spillway_register'):
                    total_penalty += 100.0 * np.sum(node.spillway_register)
                
                elif hasattr(node, 'spill_register'):
                    total_penalty += 100.0 * np.sum(node.spill_register)
            
            return (total_penalty,)
        
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'),)

    def _decode_individual(self, individual):
        genes_per_reservoir = 5  # 5 parameters per reservoir
        reservoir_params = {}
        hydroworks_params = {}
        
        # Decode reservoir parameters using numpy slicing for efficiency
        individual = np.array(individual)
        for res_idx, res_id in enumerate(self.reservoir_ids):
            start_idx = res_idx * genes_per_reservoir
            params = {
                'h1': individual[start_idx],
                'h2': individual[start_idx + 1],
                'w': individual[start_idx + 2],
                'm1': individual[start_idx + 3],
                'm2': individual[start_idx + 4]
            }
            reservoir_params[res_id] = params
        
        # Calculate start index for hydroworks genes
        hw_start = len(self.reservoir_ids) * genes_per_reservoir
        
        # Decode hydroworks parameters for each month
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            dist_params = {}
            
            # Initialize arrays for each target
            for target in self.hydroworks_targets[hw_id]:
                dist_params[target] = np.zeros(12)
            
            # Fill in monthly values for each target
            for month in range(12):
                start_idx = hw_start + month * n_targets
                end_idx = start_idx + n_targets
                
                # Get and normalize distribution parameters for this month
                dist_values = individual[start_idx:end_idx]
                normalized_dist = self._normalize_distribution(dist_values)
                
                # Assign to each target
                for target, value in zip(self.hydroworks_targets[hw_id], normalized_dist):
                    dist_params[target][month] = value
            
            hydroworks_params[hw_id] = dist_params
            hw_start += 12 * n_targets  # Move to next hydroworks node
            
        return reservoir_params, hydroworks_params
    
    def _create_individual(self):
        """Create an individual with parameters for all reservoirs and hydroworks"""
        genes = []

        for res_id in self.reservoir_ids:
            # Get bounds for this reservoir
            bounds = self.reservoir_bounds[res_id]
            h1 = np.random.uniform(bounds['h1'][0], bounds['h1'][1])
            h2 = np.random.uniform(h1, bounds['h2'][1])
            genes.extend([h1, h2])
            # Generate remaining parameters with their bounds
            for param in ['w', 'm1', 'm2']:
                value = np.random.uniform(bounds[param][0], bounds[param][1])
                genes.append(value)

        # Add genes for hydroworks with monthly parameters
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            for month in range(12):  # Create parameters for each month
                raw_dist = np.random.random(n_targets)
                normalized_dist = self._normalize_distribution(raw_dist)
                genes.extend(normalized_dist)

        return creator.Individual(genes)

    def optimize(self):
        """Run genetic algorithm optimization with parameter validation"""
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Initialize statistics tracking
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
        reservoir_params, hydroworks_params = self._decode_individual(best_ind)

        return {
            'success': True,
            'message': "Optimization completed successfully",
            'population_size': self.population_size,
            'generations': self.ngen,
            'crossover_probability': self.cxpb,
            'mutation_probability': self.mutpb,
            'objective_value': best_ind.fitness.values[0],
            'optimal_reservoir_parameters': reservoir_params,
            'optimal_hydroworks_parameters': hydroworks_params
        }

    def plot_convergence(self):
        """Plot convergence history"""
        plt.figure(figsize=(10, 6))
        gens = range(len(self.history['min']))
        
        plt.plot(gens, self.history['min'], 'b-', label='Best Fitness')
        plt.plot(gens, self.history['avg'], 'r-', label='Average Fitness')
        plt.fill_between(gens, 
                        np.array(self.history['avg']) - np.array(self.history['std']),
                        np.array(self.history['avg']) + np.array(self.history['std']),
                        alpha=0.2, color='r')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Total Deficit)')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'GA_experiments/convergence_pop{self.population_size}_ngen{self.ngen}_cxpb{self.cxpb}_mutpb{self.mutpb}.png')
        plt.close()
