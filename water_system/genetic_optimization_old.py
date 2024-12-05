import random
from deap import base, creator, tools, algorithms
import numpy as np
from water_system import WaterSystem, StorageNode
import matplotlib.pyplot as plt

class GeneticReleaseOptimizer:
    """
    Optimizes reservoir release parameters using genetic algorithms.
    """
    
    def __init__(self, create_system_func, num_time_steps, population_size=50):
        self.create_system_func = create_system_func
        self.num_time_steps = num_time_steps
        self.population_size = population_size
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.generation_count = 0
        
        # Initialize system and get reservoir data
        system = self.create_system_func(num_time_steps)
        self.reservoir = None
        for _, node_data in system.graph.nodes(data=True):
            if isinstance(node_data['node'], StorageNode):
                self.reservoir = node_data['node']
                break
                
        if self.reservoir is None:
            raise ValueError("No reservoir found in system")
            
        # Store system constraints
        self.min_level = self.reservoir.hva_data['min_waterlevel']
        self.max_level = self.reservoir.hva_data['max_waterlevel']
        self.max_capacity = sum(edge.capacity for edge in self.reservoir.outflow_edges.values())
        self.level_range = self.max_level - self.min_level
        
        # Set up genetic algorithm components
        self._setup_genetic_algorithm()
        
    def _setup_genetic_algorithm(self):
        """Initialize DEAP genetic algorithm components."""
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Define parameter generation with guaranteed valid initial values
        def create_h1():
            return random.uniform(
                self.min_level + 0.2*self.level_range, 
                self.max_level - 0.2*self.level_range
            )
            
        def create_h2(h1_min=None):
            if h1_min is None:
                h1_min = self.min_level + 0.2*self.level_range
            return random.uniform(
                h1_min + 1,
                self.max_level - 0.1*self.level_range
            )
            
        def create_w():
            return random.uniform(
                5.0,  # Minimum reasonable release
                self.max_capacity * 0.8  # Maximum safe release
            )
            
        def create_m():
            return random.uniform(1.45, 1.57)  # Reasonable slope range
        
        # Register parameter generators
        self.toolbox.register("h1", create_h1)
        self.toolbox.register("h2", create_h2)
        self.toolbox.register("w", create_w)
        self.toolbox.register("m1", create_m)
        self.toolbox.register("m2", create_m)
        
        # Create valid individual ensuring h1 < h2
        def create_valid_individual():
            h1 = create_h1()
            h2 = create_h2(h1)
            w = create_w()
            m1 = create_m()
            m2 = create_m()
            return [h1, h2, w, m1, m2]
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                            create_valid_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def crossover(self, ind1, ind2):
        """Custom crossover ensuring valid offspring."""
        # Standard two-point crossover
        child1, child2 = list(ind1), list(ind2)
        if random.random() < 0.5:
            # Crossover h1 and h2 together to maintain their relationship
            if random.random() < 0.5:
                child1[0], child2[0] = child2[0], child1[0]
                child1[1], child2[1] = child2[1], child1[1]
            # Crossover other parameters independently
            if random.random() < 0.5:
                child1[2], child2[2] = child2[2], child1[2]
            if random.random() < 0.5:
                child1[3], child2[3] = child2[3], child1[3]
            if random.random() < 0.5:
                child1[4], child2[4] = child2[4], child1[4]
                
        # Ensure h1 < h2 in offspring
        for child in [child1, child2]:
            if child[0] >= child[1]:
                mid = (child[0] + child[1]) / 2
                spread = random.uniform(1, 3)
                child[0] = mid - spread/2
                child[1] = mid + spread/2
                
        return creator.Individual(child1), creator.Individual(child2)
        
    def mutate_individual(self, individual):
        """Custom mutation operator that ensures valid parameters."""
        # Copy individual
        mutant = list(individual)
        
        # Mutation probabilities
        probs = [0.2, 0.2, 0.3, 0.2, 0.2]  # Higher probability for w
        
        # Mutate each parameter with its probability
        for i, prob in enumerate(probs):
            if random.random() < prob:
                if i == 0:  # h1
                    mutant[0] = random.uniform(
                        self.min_level + 0.2*self.level_range,
                        min(mutant[1] - 1, self.max_level - 0.3*self.level_range)
                    )
                elif i == 1:  # h2
                    mutant[1] = random.uniform(
                        max(mutant[0] + 1, self.min_level + 0.3*self.level_range),
                        self.max_level - 0.1*self.level_range
                    )
                elif i == 2:  # w
                    current = mutant[2]
                    delta = random.gauss(0, current * 0.2)  # 20% standard deviation
                    mutant[2] = max(5.0, min(self.max_capacity * 0.8, current + delta))
                else:  # m1, m2
                    current = mutant[i]
                    delta = random.gauss(0, 0.1)
                    mutant[i] = max(0.2, min(1.57, current + delta))
        
        return creator.Individual(mutant),
        
    def evaluate_individual(self, individual):
        """Evaluate fitness of an individual (parameter set)."""
        h1, h2, w, m1, m2 = individual
        
        # Basic constraint check
        if h1 >= h2 or w > self.max_capacity:
            return (1e10,)  # High but finite penalty
            
        system = self.create_system_func(self.num_time_steps)
        
        try:
            # Set release parameters
            for _, node_data in system.graph.nodes(data=True):
                if isinstance(node_data['node'], StorageNode):
                    node_data['node'].set_release_params({
                        'h1': float(h1),
                        'h2': float(h2),
                        'w': float(w),
                        'm1': float(m1),
                        'm2': float(m2)
                    })
            
            # Run simulation
            system.simulate(self.num_time_steps)
            
            # Calculate metrics
            total_unmet_demand = 0
            total_demand = 0
            storage_violations = 0
            
            for _, node_data in system.graph.nodes(data=True):
                node = node_data['node']
                if hasattr(node, 'get_demand_rate'):
                    for t in range(self.num_time_steps):
                        demand = node.get_demand_rate(t)
                        satisfied = node.satisfied_demand[t]
                        total_demand += demand * system.dt
                        total_unmet_demand += max(0, demand - satisfied) * system.dt
                elif isinstance(node, StorageNode):
                    for storage in node.storage:
                        if storage < 0.1 * node.capacity or storage > 0.95 * node.capacity:
                            storage_violations += 1
            
            # Calculate fitness with penalties
            fitness = total_unmet_demand
            
            # Add penalties for storage violations
            if storage_violations > 0:
                fitness += storage_violations * 1e5
            
            # Add small regularization terms
            fitness += (abs(m1) + abs(m2)) * 100  # Prefer smaller slopes
            fitness += (w / self.max_capacity) * 1000  # Prefer smaller releases
            
            # Update best solution if improved
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual[:]
                print(f"\nGeneration {self.generation_count}: New best solution found:")
                print(f"  Unmet demand: {total_unmet_demand:,.0f} m³ ({100*total_unmet_demand/total_demand:.1f}%)")
                print(f"  Parameters: h1={h1:.1f}, h2={h2:.1f}, w={w:.1f}, m1={m1:.3f}, m2={m2:.3f}")
                print(f"  Storage violations: {storage_violations}")
            
            return (fitness,)
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return (1e10,)
            
    def optimize(self, ngen=50):
        """Run genetic algorithm optimization."""
        print("\nStarting genetic algorithm optimization...")
        print(f"Population size: {self.population_size}")
        print(f"Number of generations: {ngen}")
        
        # Initialize population
        pop = self.toolbox.population(n=self.population_size)
        
        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of Fame to keep best individuals
        hof = tools.HallOfFame(5)
        
        try:
            # Run optimization
            for gen in range(ngen):
                self.generation_count = gen
                
                # Select and clone the next generation individuals
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                        
                for mutant in offspring:
                    if random.random() < 0.2:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                        
                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    
                # Replace population
                pop[:] = offspring
                
                # Update hall of fame
                hof.update(pop)
                
                # Gather statistics
                record = stats.compile(pop)
                self.fitness_history.append(record['min'])
                
                # Print progress
                if gen % 10 == 0:
                    print(f"Generation {gen}: Best fitness = {record['min']:.0f}")
                    
        except Exception as e:
            print(f"Optimization interrupted: {str(e)}")
            
        finally:
            # Format results using best individual found
            if self.best_individual is not None:
                best_params = {
                    'h1': self.best_individual[0],
                    'h2': self.best_individual[1],
                    'w': self.best_individual[2],
                    'm1': self.best_individual[3],
                    'm2': self.best_individual[4]
                }
            else:
                best_params = {param: None for param in ['h1', 'h2', 'w', 'm1', 'm2']}
            
            return {
                'success': self.best_individual is not None,
                'optimal_parameters': best_params,
                'objective_value': self.best_fitness,
                'message': "Genetic optimization completed",
                'generations': self.generation_count + 1,
                'population_size': self.population_size,
                'hall_of_fame': hof
            }
        
    def plot_convergence(self):
        """Plot optimization convergence history."""
        if not self.fitness_history:
            print("No optimization history available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Unmet Demand)')
        plt.title('Genetic Algorithm Convergence History')
        plt.grid(True)
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()