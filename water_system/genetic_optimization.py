import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from water_system import WaterSystem, StorageNode, DemandNode, HydroWorks

class GeneticReleaseOptimizer:
    """
    Genetic algorithm optimizer for water system release parameters and hydroworks distribution.
    """
    def __init__(self, system_creator, num_time_steps, population_size=50):
        self.system_creator = system_creator
        self.num_time_steps = num_time_steps
        self.population_size = population_size
        
        # Parameter bounds
        self.bounds = {
            'h1': (500, 505),  # Min/max water levels for h1
            'h2': (505, 510),  # Min/max water levels for h2
            'w': (5, 80),     # Min/max base release rates
            'm1': (1.5, 1.57),   # Min/max slopes for low level
            'm2': (1.5, 1.57),   # Min/max slopes for high level
            'dist': (0, 1)     # Min/max distribution parameters
        }
        
        # Set up DEAP genetic algorithm components
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize objective
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self._setup_genetic_operators()
        
        # Store convergence history
        self.history = {'min': [], 'avg': [], 'std': []}
        
    def _setup_genetic_operators(self):
        """Configure genetic algorithm operators"""
        # Create attributes for each parameter type
        self.toolbox.register("h1", random.uniform, self.bounds['h1'][0], self.bounds['h1'][1])
        self.toolbox.register("h2", random.uniform, self.bounds['h2'][0], self.bounds['h2'][1])
        self.toolbox.register("w", random.uniform, self.bounds['w'][0], self.bounds['w'][1])
        self.toolbox.register("m1", random.uniform, self.bounds['m1'][0], self.bounds['m1'][1])
        self.toolbox.register("m2", random.uniform, self.bounds['m2'][0], self.bounds['m2'][1])
        self.toolbox.register("dist", random.uniform, self.bounds['dist'][0], self.bounds['dist'][1])
        
        # Create individual and population
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _create_individual(self):
        """Create an individual with all monthly parameters"""
        # Create lists of 12 monthly values for each parameter
        h1_values = [self.toolbox.h1() for _ in range(12)]
        h2_values = [self.toolbox.h2() for _ in range(12)]
        w_values = [self.toolbox.w() for _ in range(12)]
        m1_values = [self.toolbox.m1() for _ in range(12)]
        m2_values = [self.toolbox.m2() for _ in range(12)]
        
        # Create distribution parameters for hydroworks (must sum to 1)
        dist_values = []
        for _ in range(12):
            d1 = self.toolbox.dist()
            dist_values.extend([d1, 1-d1])  # Second value ensures sum of 1
            
        # Combine all parameters
        individual = creator.Individual(h1_values + h2_values + w_values + 
                                     m1_values + m2_values + dist_values)
        return individual
    
    def _decode_individual(self, individual):
        """Convert individual's genes into parameter dictionaries"""
        # Split individual into parameter groups
        h1 = individual[0:12]
        h2 = individual[12:24]
        w = individual[24:36]
        m1 = individual[36:48]
        m2 = individual[48:60]
        dist = individual[60:]  # Distribution parameters
        
        # Create reservoir release parameters
        release_params = {
            'h1': h1,
            'h2': h2,
            'w': w,
            'm1': m1,
            'm2': m2
        }
        
        # Create hydroworks distribution parameters
        distribution_params = {
            'Demand1': dist[::2],     # Even indices
            'Demand2': dist[1::2]     # Odd indices
        }
        
        return release_params, distribution_params
    
    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual"""
        # Decode parameters
        release_params, distribution_params = self._decode_individual(individual)
        
        try:
            # Create and configure water system
            system = self.system_creator(self.num_time_steps)
            
            # Set release parameters for reservoir
            reservoir_node = system.graph.nodes['Reservoir']['node']
            reservoir_node.set_release_params(release_params)
            
            # Set distribution parameters for hydroworks
            hydroworks_node = system.graph.nodes['Hydroworks']['node']
            hydroworks_node.set_distribution_parameters(distribution_params)
            
            # Run simulation
            system.simulate(self.num_time_steps)
            
            # Calculate total deficit (objective to minimize)
            total_deficit = 0
            for node_id, node_data in system.graph.nodes(data=True):
                if isinstance(node_data['node'], DemandNode):
                    node = node_data['node']
                    for t in range(self.num_time_steps):
                        demand = node.get_demand_rate(t)
                        satisfied = node.satisfied_demand[t]
                        deficit = (demand - satisfied) * system.dt
                        total_deficit += deficit
            
            return (total_deficit,)  # Return as tuple for DEAP
            
        except Exception as e:
            print(f"Error evaluating individual: Individual NOT fit: ({str(e)})")
            return (float('inf'),)  # Penalize invalid solutions
    
    def _mutate_individual(self, individual, indpb=0.1):
        """Custom mutation operator"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # Determine parameter type based on index
                if i < 12:  # h1
                    individual[i] = random.uniform(self.bounds['h1'][0], self.bounds['h1'][1])
                elif i < 24:  # h2
                    individual[i] = random.uniform(self.bounds['h2'][0], self.bounds['h2'][1])
                elif i < 36:  # w
                    individual[i] = random.uniform(self.bounds['w'][0], self.bounds['w'][1])
                elif i < 48:  # m1
                    individual[i] = random.uniform(self.bounds['m1'][0], self.bounds['m1'][1])
                elif i < 60:  # m2
                    individual[i] = random.uniform(self.bounds['m2'][0], self.bounds['m2'][1])
                else:  # distribution parameters
                    # Maintain sum of 1 for each month's distribution
                    month_idx = (i - 60) // 2
                    base_idx = 60 + month_idx * 2
                    d1 = random.uniform(self.bounds['dist'][0], self.bounds['dist'][1])
                    individual[base_idx] = d1
                    individual[base_idx + 1] = 1 - d1
                    
        return individual,
    
    def optimize(self, ngen=50, cxpb=0.2, mutpb=0.5):
        """Run genetic algorithm optimization"""
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
        release_params, distribution_params = self._decode_individual(best_ind)
        
        return {
            'success': True,
            'message': "Optimization completed successfully",
            'population_size': self.population_size,
            'generations': ngen,
            'objective_value': best_ind.fitness.values[0],
            'optimal_parameters': release_params,
            'optimal_distribution': distribution_params
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
