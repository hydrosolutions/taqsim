import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from scoop import futures
import random
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks

class MultiGeneticOptimizer:
    """
    Enhanced genetic algorithm optimizer for water systems with multiple reservoirs
    and hydroworks nodes.
    """
    def __init__(self, system_creator, start_year, start_month, num_time_steps, population_size=50):
        self.system_creator = system_creator
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.population_size = population_size
        
        # Get initial system to identify reservoirs and hydroworks
        test_system = self.system_creator(self.start_year, self.start_month, self.num_time_steps)  # Create test system to analyze structure
        self.reservoir_ids = []
        self.hydroworks_ids = []
        self.hydroworks_targets = {}

        # Dictionary to store reservoir-specific bounds
        self.reservoir_bounds = {}
        
        # Identify reservoirs and hydroworks in system
        for node_id, node_data in test_system.graph.nodes(data=True):
            if isinstance(node_data['node'], StorageNode):
                self.reservoir_ids.append(node_id)
                # Get bounds from HVA data and outflow capacity
                reservoir = node_data['node']
                
                # Get water level bounds from HVA data
                min_level = reservoir.hva_data['min_waterlevel']
                max_level = reservoir.hva_data['max_waterlevel']
                mid_level = min_level + (max_level - min_level) / 2

                # Calculate total outflow capacity
                total_capacity = sum(edge.capacity for edge in reservoir.outflow_edges.values())
                
                # Set reservoir-specific bounds
                self.reservoir_bounds[node_id] = {
                    'h1': (min_level, mid_level),  # Full range for h1
                    'h2': (mid_level, max_level),  # Full range for h2
                    'w': (0, total_capacity),      # From 0 to total outflow capacity
                    'm1': (1.47, 1.57),            # Standard slopes
                    'm2': (1.47, 1.57)             # Standard slopes
                }

            elif isinstance(node_data['node'], HydroWorks):
                self.hydroworks_ids.append(node_id)
                # Get target nodes for each hydroworks
                self.hydroworks_targets[node_id] = list(
                    node_data['node'].outflow_edges.keys()
                )
        
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
        Normalize a list of values to sum to 1.0
        
        Args:
            values (list): List of values to normalize
            
        Returns:
            list: Normalized values that sum to 1.0
        """
        total = sum(values)
        if total == 0:
            # If all values are 0, return equal distribution
            return [1.0/len(values)] * len(values)
        return [v/total for v in values]
    
    def _bound_parameter(self, value, bounds):
        """
        Helper method to ensure a parameter stays within bounds.
        
        Args:
            value (float): Parameter value to check
            bounds (tuple): (min, max) bounds for the parameter
            
        Returns:
            float: Value clamped to bounds
        """
        return max(bounds[0], min(bounds[1], value))

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
        """
        # Create attribute for hydroworks distribution
        self.toolbox.register(
            "hydroworks_dist",
            random.random
        )
        """
        # Create individual and population
        #self.toolbox.register("map", futures.map)
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=5)

    def _mutate_individual(self, individual, indpb=0.2):
        """Custom mutation operator with enforced parameter bounds"""
        genes_per_reservoir = 5  # 5 parameters per reservoir
        
        # Track which hydroworks need renormalization
        hw_updates = set()
        
        for i in range(len(individual)):
            if random.random() < indpb:
                if i < len(self.reservoir_ids) * genes_per_reservoir:
                    # Reservoir parameter mutation
                    res_idx = i // genes_per_reservoir
                    res_id = self.reservoir_ids[res_idx]
                    param_idx = i % genes_per_reservoir
                    param_name = ['h1', 'h2', 'w', 'm1', 'm2'][param_idx]
                    bounds = self.reservoir_bounds[res_id][param_name]
                    
                    # Generate new value
                    value = random.uniform(bounds[0], bounds[1])
                    
                    # Special handling for h2 to maintain h1 < h2
                    if param_name == 'h2':
                        h1_idx = i - 1  # Index of corresponding h1 value
                        min_bound = max(bounds[0], individual[h1_idx] + 0.1)
                        value = random.uniform(min_bound, bounds[1])
                    elif param_name == 'h1':
                        h2_idx = i + 1  # Index of corresponding h2 value
                        max_bound = min(bounds[1], individual[h2_idx] - 0.1)
                        value = random.uniform(bounds[0], max_bound)
                    
                    # Enforce bounds for all parameters
                    individual[i] = self._bound_parameter(value, bounds)
                    
                else:
                    # Hydroworks parameter mutation
                    hw_genes_start = len(self.reservoir_ids) * genes_per_reservoir
                    offset = i - hw_genes_start
                    
                    # Find which hydroworks this gene belongs to
                    hw_idx = 0
                    gene_count = 0
                    for hw_id in self.hydroworks_ids:
                        n_targets = len(self.hydroworks_targets[hw_id])
                        if gene_count + n_targets > offset:
                            hw_updates.add(hw_id)
                            break
                        gene_count += n_targets
                        hw_idx += 1
                    
                    # Mutate the gene
                    individual[i] = random.random()
        
        # Normalize hydroworks distributions that were modified
        gene_idx = len(self.reservoir_ids) * genes_per_reservoir
        for hw_id in self.hydroworks_ids:
            if hw_id in hw_updates:
                n_targets = len(self.hydroworks_targets[hw_id])
                dist_values = individual[gene_idx:gene_idx + n_targets]
                normalized_dist = self._normalize_distribution(dist_values)
                for j, value in enumerate(normalized_dist):
                    individual[gene_idx + j] = value
            gene_idx += len(self.hydroworks_targets[hw_id])
        return individual,

    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual with bound checking"""
        # Verify all parameters are within bounds before evaluation
        genes_per_reservoir = 5  # 5 parameters per reservoir
        
        try:
            # Check reservoir parameters
            for res_idx, res_id in enumerate(self.reservoir_ids):
                start_idx = res_idx * genes_per_reservoir
                for param_idx, param_name in enumerate(['h1', 'h2', 'w', 'm1', 'm2']):
                    bounds = self.reservoir_bounds[res_id][param_name]
                    i = start_idx + param_idx
                    individual[i] = self._bound_parameter(individual[i], bounds)
            
            # Decode parameters and continue with evaluation
            reservoir_params, hydroworks_params = self._decode_individual(individual)
            
            # Create and configure water system
            system = self.system_creator(self.start_year, self.start_month, self.num_time_steps)
            
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
            
            # Calculate weighted deficits using node weights
            total_weighted_deficit = 0
            for node_id, node_data in system.graph.nodes(data=True):
                if isinstance(node_data['node'], DemandNode):
                    node = node_data['node']
                    for t in range(self.num_time_steps):
                        demand = node.get_demand_rate(t)
                        satisfied = node.satisfied_demand[t]
                        deficit = (demand - satisfied) * system.dt
                        weighted_deficit = deficit * node.weight  # Use node's weight attribute
                        total_weighted_deficit += weighted_deficit
            
                # Calculate total spills with double weight
                total_spills = 0
                for node_id, node_data in system.graph.nodes(data=True):
                    # Add reservoir spills
                    if hasattr(node_data['node'], 'spillway_register'):
                        total_spills += 100.0 * sum(node_data['node'].spillway_register) 
                    
                    # Add hydroworks spills
                    if hasattr(node_data['node'], 'spill_register'):
                        total_spills += 100.0 * sum(node_data['node'].spill_register)
                
                # Combined objective (weighted deficits + weighted spills)
                total_objective = total_weighted_deficit + total_spills
            
            return (total_objective,)
     
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'),)

    def _decode_individual(self, individual):
        genes_per_reservoir = 5  # 5 parameters per reservoir
        reservoir_params = {}
        hydroworks_params = {}
        
        # Decode reservoir parameters
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
        
        # Decode hydroworks parameters - single annual distribution per target
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            start_idx = hw_start
            end_idx = start_idx + n_targets
            
            # Get and normalize distribution parameters
            dist_values = individual[start_idx:end_idx]
            normalized_dist = self._normalize_distribution(dist_values)
            
            # Create parameter dictionary
            dist_params = {}
            for target, value in zip(self.hydroworks_targets[hw_id], normalized_dist):
                dist_params[target] = value
            
            hydroworks_params[hw_id] = dist_params
            hw_start += n_targets
            
            return reservoir_params, hydroworks_params
    
    def _create_individual(self):
        """Create an individual with parameters for all reservoirs and hydroworks"""
        genes = []
        
        # Add genes for each reservoir 
        for res_id in self.reservoir_ids:
            for param in ['h1', 'h2', 'w', 'm1', 'm2']:
                bounds = self.reservoir_bounds[res_id][param]
                value = getattr(self.toolbox, f"reservoir_{res_id}_{param}")()
                value = self._bound_parameter(value, bounds)
                genes.append(value)
    
        # Add genes for hydroworks
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            raw_dist = [random.random() for _ in range(n_targets)]
            normalized_dist = self._normalize_distribution(raw_dist)
            genes.extend(normalized_dist)
        
        return creator.Individual(genes)

    def optimize(self, ngen=50, cxpb=0.9, mutpb=0.1):
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
            pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
            stats=stats, verbose=True
        )
        
        # Store convergence history
        self.history['min'] = logbook.select("min")
        self.history['avg'] = logbook.select("avg")
        self.history['std'] = logbook.select("std")
        
        # Get best individual
        best_ind = tools.selBest(final_pop, 1)[0]
        reservoir_params, hydroworks_params = self._decode_individual(best_ind)
        """
        # Validate and print decoded parameters
        print("\nOptimal Parameters:")
        for res_id, params in reservoir_params.items():
            print(f"\n{res_id}:")
            for param_name, values in params.items():
                print(f"{param_name}: {[f'{v:.3f}' for v in values]}")
                
                # Verify bounds
                bounds = self.reservoir_bounds[res_id][param_name]
                if any(v < bounds[0] or v > bounds[1] for v in values):
                    print(f"Warning: Some {param_name} values outside bounds {bounds}")
        """
        return {
            'success': True,
            'message': "Optimization completed successfully",
            'population_size': self.population_size,
            'generations': ngen,
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
        
        plt.savefig('convergence.png')
        plt.close()
