import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from water_system import StorageNode, DemandNode, HydroWorks, SinkNode, WaterSystem
import copy
import os
from typing import List, Dict, Tuple, Union, Optional

class SingleObjectiveOptimizer:
    """
    Enhanced genetic algorithm optimizer for water systems with multiple reservoirs
    and hydroworks nodes.
    """
    def __init__(
        self,
        base_system: WaterSystem,
        start_year: int,
        start_month: int,
        num_time_steps: int,
        ngen: int = 50,
        population_size: int = 50,
        cxpb: float = 0.65,
        mutpb: float = 0.32
    ) -> None:
        self.base_system = base_system 
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.population_size = population_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.dt = base_system.dt  # Time step in seconds
        self.reservoir_ids = StorageNode.all_ids 
        self.demand_ids = DemandNode.all_ids
        self.sink_ids = SinkNode.all_ids
        self.hydroworks_ids = HydroWorks.all_ids
        self.hydroworks_targets = {}

        # Dictionary to store reservoir-specific bounds
        self.reservoir_bounds = {}
        
        # Identify reservoirs and hydroworks in system
        for node_id in self.reservoir_ids:
            reservoir = self.base_system.graph.nodes[node_id]['node']
            
            # Calculate total outflow capacity using numpy
            total_capacity = reservoir.outflow_edge.capacity
            
            # Set volume-based bounds for the new parameterization
            dead_storage = reservoir.dead_storage
            capacity = reservoir.capacity
            
            # Set reservoir-specific bounds
            self.reservoir_bounds[node_id] = {
                'Vr': (0, total_capacity*self.dt),# Target monthly release volume
                'V1': (dead_storage, capacity),  # Top of buffer zone
                'V2': (dead_storage, capacity)  # Top of conservation zone
            }

        for node_id in self.hydroworks_ids:
            hydrowork = self.base_system.graph.nodes[node_id]['node']
            self.hydroworks_targets[node_id] = list(hydrowork.outflow_edges.keys())
        
        # Set up DEAP genetic algorithm components
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))  # Minimize objective
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self._setup_genetic_operators()
        
        # Store convergence history
        self.history = {'min': [], 'avg': [], 'std': []}
        
    def _normalize_distribution(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize a list of values to sum to 1.0.
        
        Args:
            values (list): List of values to normalize
            
        Returns:
            list: Normalized values that sum to 1.0
        """
        npvalue = np.array(values)
        total = np.sum(npvalue)
        if total == 0:
            # If all values are 0, return equal distribution
            return np.full_like(npvalue, 1.0 / len(npvalue))
        return npvalue / total
    
    def _setup_genetic_operators(self) -> None:
        """Configure genetic algorithm operators for multiple structures with dynamic bounds"""

        # Create individual and population
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=5)

    def _mutate_individual(self, individuals: List[float], indpb: float=0.5):
        """Custom mutation operator with enforced parameter bounds for monthly parameters"""
        genes_per_reservoir_month = 3
        
        individual = np.array(individuals)
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
                        individual[start_idx:end_idx] = self._normalize_distribution(individual[start_idx:end_idx])
            current_idx += 12 * n_targets  # Update index for next hydroworks

        return creator.Individual(individual.tolist()),

    def _crossover(self, ind_one:List[float], ind_two:List[float]):
        """
        Custom crossover that treats monthly parameters as packages.
        For each month, all parameters of a reservoir or hydrowork for that specific month
        are swapped as a complete unit between parents.
        """
        ind1, ind2 = np.array(ind_one), np.array(ind_two)
        
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

    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate fitness of an individual with bound checking"""        
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

            for node_id in self.demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                total_penalty += np.sum(deficit)
            
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                total_penalty += total_deficit_volume
            
            '''# Calculate weighted demand deficits
            for node_id, node_data in system.graph.nodes(data=True):
                node = node_data['node']
                
                if isinstance(node, DemandNode):
                    demand = np.array([node.demand_rates[t] for t in range(self.num_time_steps)])
                    satisfied = np.array(node.satisfied_demand_total)
                    deficit = (demand - satisfied) * system.dt
                    weighted_deficit = deficit * node.weight
                    total_penalty += np.sum(weighted_deficit)
                
                elif isinstance(node, SinkNode):
                    total_deficit_volume = sum(deficit * system.dt for deficit in node.flow_deficits)
                    total_penalty += total_deficit_volume * node.weight
                    
                elif hasattr(node, 'spillway_register'):
                    total_penalty += 10.0 * np.sum(node.spillway_register)
                
                elif hasattr(node, 'spill_register'):
                    total_penalty += 10.0 * np.sum(node.spill_register)'''
            
            total_penalty = float(total_penalty)
            return (total_penalty,)
        
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return (float('inf'),)

    def _decode_individual(self, indiv: List[float]) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, np.ndarray]]]:

        """Decode individual genes into monthly reservoir and hydroworks parameters"""
        genes_per_reservoir_month = 3
        reservoir_params = {}
        hydroworks_params = {}
        
        # Convert individual to numpy array for efficient slicing
        individual = np.array(indiv)
        
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

    def optimize(self) -> Dict[str, Union[bool, str, int, float, Dict[str, Dict[str, List[float]]]]]:
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

    def plot_convergence(self)-> None:
        """Plot convergence history"""
        directory = './model_output/optimisation'
        if not os.path.exists(directory):
            os.makedirs(directory)

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
        plt.savefig(f'./model_output/optimisation/convergence_pop{self.population_size}_ngen{self.ngen}_cxpb{self.cxpb}_mutpb{self.mutpb}.png')
        plt.close()
