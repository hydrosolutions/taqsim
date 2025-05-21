import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional
import os

# For pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

# Import your water system components
from water_system import StorageNode, DemandNode, HydroWorks, SinkNode, WaterSystem, RunoffNode

class WaterSystemCallback(Callback):
    """
    Callback to track optimization progress for visualization
    """
    def __init__(self) -> None:
        super().__init__()
        self.data["min_obj"] = []
        self.data["avg_obj"] = []
        self.data["std_obj"] = []
    
    def notify(self, algorithm):
        # Get fitness values of all individuals
        fitness_values = algorithm.pop.get("F")
        
        # Store statistics
        self.data["min_obj"].append(np.min(fitness_values))
        self.data["avg_obj"].append(np.mean(fitness_values))
        self.data["std_obj"].append(np.std(fitness_values))

class PymooProblemSingleObjective(Problem):
    """
    Pymoo Problem formulation for water system optimization
    """
    def __init__(self, 
                 base_system: WaterSystem,
                 start_year: int,
                 start_month: int,
                 num_time_steps: int):
        
        self.base_system = base_system
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.dt = base_system.dt  # Time step in seconds
        
        # Get IDs from the system
        self.reservoir_ids = StorageNode.all_ids
        self.demand_ids = DemandNode.all_ids
        self.sink_ids = SinkNode.all_ids
        self.hydroworks_ids = HydroWorks.all_ids
        self.hydroworks_targets = {}
        
        # Dictionary to store reservoir-specific bounds
        self.reservoir_bounds = {}
        
        # Calculate number of variables
        n_vars = 0
        
        # For reservoirs: 3 parameters per month per reservoir
        n_vars += len(self.reservoir_ids) * 12 * 3
        
        # For hydroworks: Get targets and count parameters
        for node_id in self.hydroworks_ids:
            hydrowork = self.base_system.graph.nodes[node_id]['node']
            targets = list(hydrowork.outflow_edges.keys())
            self.hydroworks_targets[node_id] = targets
            # For each hydrowork: One parameter per target per month
            n_vars += len(targets) * 12
            
        # Store bounds for variables
        xl, xu = [], []  # Lower and upper bounds
        
        # Set up reservoir parameters bounds
        for node_id in self.reservoir_ids:
            reservoir = self.base_system.graph.nodes[node_id]['node']
            
            # Calculate total outflow capacity
            total_capacity = reservoir.outflow_edge.capacity
            
            # Set volume-based bounds for parameterization
            dead_storage = reservoir.dead_storage
            capacity = reservoir.capacity
            
            # Store bounds for later use
            self.reservoir_bounds[node_id] = {
                'Vr': (0, total_capacity * self.dt),  # Target monthly release volume
                'V1': (dead_storage, capacity),       # Top of buffer zone
                'V2': (dead_storage, capacity)        # Top of conservation zone
            }
            
            # Add bounds for each month and parameter
            for month in range(12):
                # For Vr (release volume)
                xl.append(0)
                xu.append(total_capacity * self.dt)
                
                # For V1 (buffer zone)
                xl.append(dead_storage)
                xu.append(capacity)
                
                # For V2 (conservation zone)
                xl.append(dead_storage)
                xu.append(capacity)
        
        # Set up hydroworks distribution parameters bounds
        for node_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[node_id])
            # For each month and target
            for month in range(12):
                for _ in range(n_targets):
                    xl.append(0.0)
                    xu.append(1.0)
        
        # Initialize the problem with bounds
        super().__init__(n_var=n_vars, 
                        n_obj=1,
                        n_constr=0,
                        xl=np.array(xl),
                        xu=np.array(xu))
    
    def _normalize_distribution(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize distribution values to sum to 1.0
        """
        total = np.sum(values)
        if total == 0:
            return np.full_like(values, 1.0 / len(values))
        return values / total
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the fitness of a batch of solutions
        """
        # Handle batch evaluation
        n_individuals = x.shape[0]
        f = np.zeros(n_individuals)
        
        for i in range(n_individuals):
            f[i] = self._evaluate_individual(x[i])
        
        out["F"] = np.array(f)
    
    def _evaluate_individual(self, x):
        """
        Evaluate a single individual
        """
        # Decode parameters
        reservoir_params, hydroworks_params = self._decode_individual(x)
        
        try:
            # Create and configure water system
            import copy
            system = copy.deepcopy(self.base_system)
            
            # Set parameters for reservoirs
            for res_id, params in reservoir_params.items():
                reservoir_node = system.graph.nodes[res_id]['node']
                reservoir_node.set_release_params(params)
            
            # Set parameters for hydroworks
            for hw_id, params in hydroworks_params.items():
                hydroworks_node = system.graph.nodes[hw_id]['node']
                hydroworks_node.set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)
            
            total_penalty = 0
            
            # Calculate penalties for demand nodes
            for node_id in self.demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                total_penalty += np.sum(deficit)* demand_node.weight
            
            # Calculate penalties for sink nodes
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                deficit = np.array([sink_node.flow_deficits[t] for t in range(self.num_time_steps)])
                required= np.array([sink_node.min_flows[t] for t in range(self.num_time_steps)])
                total_penalty += np.sum(deficit)*system.dt * sink_node.weight
            
            # Penalties for hydroworks spills
            for node_id in self.hydroworks_ids:
                hydroworks_node = system.graph.nodes[node_id]['node']
                total_penalty += 10.0 * np.sum(hydroworks_node.spill_register)
            
            # Penalties for reservoir spillage
            for node_id in self.reservoir_ids:
                reservoir_node = system.graph.nodes[node_id]['node']
                total_penalty += 10.0 * np.sum(reservoir_node.spillway_register)
            
            return float(float(total_penalty)/(self.num_time_steps / 12) / 1e9)  # Convert to km3/year
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return float('inf')
    
    def _decode_individual(self, x):
        """
        Decode individual genes into reservoir and hydroworks parameters
        """
        genes_per_reservoir_month = 3
        reservoir_params = {}
        hydroworks_params = {}
        
        # Convert to numpy array for efficient slicing
        x = np.array(x)
        
        # Decode reservoir parameters
        current_idx = 0
        for res_id in self.reservoir_ids:
            params = {
                'Vr': [], 'V1': [], 'V2': []
            }
            
            # Get parameters for each month
            for month in range(12):
                # Ensure proper volume relationships
                start_idx = current_idx + month * genes_per_reservoir_month
                vr = x[start_idx]
                v1 = x[start_idx + 1]
                v2 = x[start_idx + 2]
                
                # Ensure V1 < V2 (conservation > buffer)
                if v1 >= v2:
                    v2 = v1 * 1.01  # Make V2 slightly larger
                
                # Ensure dead_storage < V1
                dead_storage = self.base_system.graph.nodes[res_id]['node'].dead_storage
                if v1 <= dead_storage:
                    v1 = dead_storage * 1.01
                
                # Ensure V2 <= capacity
                capacity = self.base_system.graph.nodes[res_id]['node'].capacity
                if v2 > capacity:
                    v2 = capacity
                
                params['Vr'].append(vr)
                params['V1'].append(v1)
                params['V2'].append(v2)
            
            reservoir_params[res_id] = params
            current_idx += 12 * genes_per_reservoir_month
        
        # Decode hydroworks parameters
        for hw_id in self.hydroworks_ids:
            n_targets = len(self.hydroworks_targets[hw_id])
            dist_params = {}
            
            # Initialize distribution parameters for each target
            for target in self.hydroworks_targets[hw_id]:
                dist_params[target] = np.zeros(12)
            
            # Process each month
            for month in range(12):
                start_idx = current_idx + month * n_targets
                end_idx = start_idx + n_targets
                
                # Get distribution values for this month
                dist_values = x[start_idx:end_idx]
                
                # Normalize distribution to sum to 1
                normalized_dist = self._normalize_distribution(dist_values)
                
                # Assign values to targets
                for i, target in enumerate(self.hydroworks_targets[hw_id]):
                    dist_params[target][month] = normalized_dist[i]
            
            hydroworks_params[hw_id] = dist_params
            current_idx += 12 * n_targets
        
        return reservoir_params, hydroworks_params

class PymooProblemTwoObjective(PymooProblemSingleObjective):
    """
    Multi-objective extension of the PymooProblemSingleObjective
    """
    def __init__(self, 
                 base_system: WaterSystem,
                 start_year: int,
                 start_month: int,
                 num_time_steps: int):
        
        # Use the same initialization as the single-objective version,
        # but change n_obj to 2 (or 3 based on your needs)
        super().__init__(base_system, start_year, start_month, num_time_steps)
        
        # Override n_obj for multi-objective optimization
        self.n_obj = 2  # Modify as needed (e.g., demand deficit and min flow deficit)
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate multi-objective fitness for a batch of solutions
        """
        n_individuals = x.shape[0]
        f = np.zeros((n_individuals, self.n_obj))
        
        for i in range(n_individuals):
            f[i, :] = self._evaluate_individual_multi(x[i])
        
        out["F"] = f
    
    def _evaluate_individual_multi(self, x):
        """
        Multi-objective evaluation of an individual
        Returns array of objective values
        """
        # Decode parameters
        reservoir_params, hydroworks_params = self._decode_individual(x)
        
        try:
            # Create and configure water system
            import copy
            system = copy.deepcopy(self.base_system)
            
            # Set parameters for reservoirs
            for res_id, params in reservoir_params.items():
                reservoir_node = system.graph.nodes[res_id]['node']
                reservoir_node.set_release_params(params)
            
            # Set parameters for hydroworks
            for hw_id, params in hydroworks_params.items():
                hydroworks_node = system.graph.nodes[hw_id]['node']
                hydroworks_node.set_distribution_parameters(params)
            
            # Run simulation
            system.simulate(self.num_time_steps)

            # Calculate the number of years
            num_years = self.num_time_steps / 12
            
            # Objective 1: Demand deficit
            demand_deficit = 0
            for node_id in self.demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                demand_deficit += np.sum(deficit) #* demand_node.weight
            
            # Objective 2: Minimum flow deficit
            min_flow_deficit = 0
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                min_flow_deficit += total_deficit_volume #* sink_node.weight
            
            # Return both objectives
            return [float(demand_deficit)/num_years/1e9, float(min_flow_deficit)/num_years/1e9]
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return [float('inf'), float('inf')]

class PymooProblemThreeObjective(PymooProblemTwoObjective):
    """
    Three-objective extension focusing on different types of demands
    (if needed for your specific application)
    """
    def __init__(self, 
                 base_system: WaterSystem,
                 start_year: int,
                 start_month: int,
                 num_time_steps: int):
        
        super().__init__(base_system, start_year, start_month, num_time_steps)
        self.n_obj = 3  # Three objectives
        
        # Identify high priority demand nodes (those with weight > 1)
        self.priority_demand_ids = [node_id for node_id in self.demand_ids 
                                    if self.base_system.graph.nodes[node_id]['node'].weight > 1]
        
        # Regular priority demands
        self.regular_demand_ids = [node_id for node_id in self.demand_ids 
                                  if self.base_system.graph.nodes[node_id]['node'].weight == 1]
    
    def _evaluate_individual_multi(self, x):
        """
        Three-objective evaluation
        """
        # Decode parameters
        reservoir_params, hydroworks_params = self._decode_individual(x)
        
        try:
            # Create and configure water system
            import copy
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

            # Calculate the number of years
            num_years = self.num_time_steps / 12
            
            # Objective 1: Regular priority demand deficit
            regular_demand_deficit = 0
            for node_id in self.regular_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                regular_demand_deficit += np.sum(deficit)
            
            # Objective 2: High priority demand deficit
            priority_demand_deficit = 0
            for node_id in self.priority_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                priority_demand_deficit += np.sum(deficit) #* demand_node.weight
            
            # Objective 3: Minimum flow deficit
            min_flow_deficit = 0
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                min_flow_deficit += total_deficit_volume #* sink_node.weight
            
            return [float(regular_demand_deficit)/num_years/1e9, float(priority_demand_deficit)/num_years/1e9, float(min_flow_deficit)/num_years/1e9]
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return [float('inf'), float('inf'), float('inf')]

class PymooProblemFourObjective(PymooProblemTwoObjective):
    """
    Four-objective extension focusing on different types of demands
    (if needed for your specific application)
    """
    def __init__(self, 
                 base_system: WaterSystem,
                 start_year: int,
                 start_month: int,
                 num_time_steps: int):
        
        super().__init__(base_system, start_year, start_month, num_time_steps)
        self.n_obj = 4  # Four objectives
        
        # Identify high priority demand nodes (those with weight > 1)
        self.priority_demand_ids = [node_id for node_id in self.demand_ids 
                                    if self.base_system.graph.nodes[node_id]['node'].weight > 1]
        
        # Regular priority demands
        self.regular_demand_ids = [node_id for node_id in self.demand_ids 
                                  if self.base_system.graph.nodes[node_id]['node'].weight == 1]
    
    def _evaluate_individual_multi(self, x):
        """
        Three-objective evaluation
        """
        # Decode parameters
        reservoir_params, hydroworks_params = self._decode_individual(x)
        
        try:
            # Create and configure water system
            import copy
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

            # Calculate the number of years
            num_years = self.num_time_steps /12
            
            # Objective 1: Regular priority demand deficit
            regular_demand_deficit = 0
            for node_id in self.regular_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                regular_demand_deficit += np.sum(deficit)
            
            # Objective 2: High priority demand deficit
            priority_demand_deficit = 0
            for node_id in self.priority_demand_ids:
                demand_node = system.graph.nodes[node_id]['node']
                demand = np.array([demand_node.demand_rates[t] for t in range(self.num_time_steps)])
                satisfied = np.array(demand_node.satisfied_demand_total)
                deficit = (demand - satisfied) * system.dt
                priority_demand_deficit += np.sum(deficit) #* demand_node.weight
            
            # Objective 3: Minimum flow deficit
            min_flow_deficit = 0
            for node_id in self.sink_ids:
                sink_node = system.graph.nodes[node_id]['node']
                total_deficit_volume = sum(deficit * system.dt for deficit in sink_node.flow_deficits)
                min_flow_deficit += total_deficit_volume #* sink_node.weight
            
            # Objective 4: Total hydrowork and reservoir spillage
            total_spillage = 0
            for node_id in self.hydroworks_ids:
                hydroworks_node = system.graph.nodes[node_id]['node']
                total_spillage += np.sum(hydroworks_node.spill_register)
            for node_id in self.reservoir_ids:
                reservoir_node = system.graph.nodes[node_id]['node']
                total_spillage += np.sum(reservoir_node.spillway_register)
            
            return [float(regular_demand_deficit)/num_years/1e9, float(priority_demand_deficit)/num_years/1e9, float(min_flow_deficit)/num_years/1e9, float(total_spillage)/num_years/1e9]
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return [float('inf'), float('inf'), float('inf'), float('inf')]


class PymooSingleObjectiveOptimizer:
    """
    Single-objective water system optimizer using pymoo
    """
    def __init__(
        self,
        base_system: WaterSystem,
        start_year: int,
        start_month: int,
        num_time_steps: int,
        n_gen: int = 50,
        pop_size: int = 50,
    ) -> None:
        self.base_system = base_system
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.n_gen = n_gen
        self.pop_size = pop_size
        
        # Create problem instance
        self.problem = PymooProblemSingleObjective(
            base_system=base_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps
        )
        
        # Create callback for tracking progress
        self.callback = WaterSystemCallback()
        
        # Then use it in the GA algorithm setup
        self.algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

    def optimize(self) -> Dict[str, Union[bool, str, int, float, Dict[str, Dict[str, List[float]]]]]:
        """
        Run the optimization process
        """
        # Set termination criteria
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=0.0001,
            period=25,
            n_max_gen=self.n_gen,
            n_max_evals=self.n_gen * self.pop_size
        )
        
        # Run the optimization
        try:
            print("Starting optimization...")
            result = minimize(
                problem=self.problem,
                algorithm=self.algorithm,
                termination=termination,
                seed=42,
                save_history=True,
                callback=self.callback,
                verbose=True
            )
            
            print(f"Optimization completed! Best fitness: {result.F[0]:.4f}")
            
            # Get best individual
            best_x = result.X
            
            # Decode the best individual
            reservoir_params, hydroworks_params = self.problem._decode_individual(best_x)
            
            return {
                'success': True,
                'message': f"Optimization completed successfully after {result.algorithm.n_gen} generations",
                'population_size': self.pop_size,
                'generations': result.algorithm.n_gen,
                'objective_value': float(result.F[0]),
                'optimal_reservoir_parameters': reservoir_params,
                'optimal_hydroworks_parameters': hydroworks_params
            }
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return {
                'success': False,
                'message': f"Optimization failed: {str(e)}",
                'population_size': self.pop_size,
                'generations': 0,
                'objective_value': float('inf'),
                'optimal_reservoir_parameters': {},
                'optimal_hydroworks_parameters': {}
            }
    
    def plot_convergence(self) -> None:
        """
        Plot convergence history after optimization
        """
        directory = './model_output/pymoo/convergence'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure(figsize=(10, 6))
        gens = range(len(self.callback.data["min_obj"]))
        
        plt.plot(gens, self.callback.data["min_obj"], 'b-', label='Best Fitness')
        plt.plot(gens, self.callback.data["avg_obj"], 'r-', label='Average Fitness')
        plt.fill_between(gens, 
                        np.array(self.callback.data["avg_obj"]) - np.array(self.callback.data["std_obj"]),
                        np.array(self.callback.data["avg_obj"]) + np.array(self.callback.data["std_obj"]),
                        alpha=0.2, color='r')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Total Deficit)')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./model_output/pymoo/convergence/pymoo_convergence_pop{self.pop_size}_ngen{self.n_gen}.png')
        plt.close()
    

class PymooMultiObjectiveOptimizer:
    """
    Multi-objective water system optimizer using pymoo
    """
    def __init__(
        self,
        base_system: WaterSystem,
        start_year: int,
        start_month: int,
        num_time_steps: int,
        n_gen: int = 50,
        pop_size: int = 50,
        num_objectives: int = 2  # Default to 2 objectives
    ) -> None:
        self.base_system = base_system
        self.start_year = start_year
        self.start_month = start_month
        self.num_time_steps = num_time_steps
        self.n_gen = n_gen
        self.pop_size = pop_size
        
        # Create problem instance based on number of objectives
        if num_objectives == 2:
            self.problem = PymooProblemTwoObjective(
                base_system=base_system,
                start_year=start_year,
                start_month=start_month,
                num_time_steps=num_time_steps
            )
        elif num_objectives == 3:
            self.problem = PymooProblemThreeObjective(
                base_system=base_system,
                start_year=start_year,
                start_month=start_month,
                num_time_steps=num_time_steps
            )
        elif num_objectives == 4:
            self.problem = PymooProblemFourObjective(
                base_system=base_system,
                start_year=start_year,
                start_month=start_month,
                num_time_steps=num_time_steps
            )
        else:
            raise ValueError(f"Unsupported number of objectives: {num_objectives}")
        
        # Create callback for tracking progress
        self.callback = WaterSystemCallback()
        
        if num_objectives ==0: 
            # create the reference directions to be used for the optimization
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)

            # Use NSGA3 for 4 objectives
            self.algorithm = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )
        else:
            # Use NSGA2 for 2 or 3 objectives
            self.algorithm = NSGA2(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )

    
    def optimize(self) -> Dict[str, Union[bool, str, int, float, Dict[str, Dict[str, List[float]]]]]:
        """
        Run the multi-objective optimization process
        """
        # Set termination criteria
        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=0.0001,
            period=25,
            n_max_gen=self.n_gen,
            n_max_evals=self.n_gen * self.pop_size
        )
        
        # Run the optimization
        try:
            print("Starting multi-objective optimization...")
            result = minimize(
                problem=self.problem,
                algorithm=self.algorithm,
                termination=termination,
                seed=42,
                save_history=True,
                callback=self.callback,
                verbose=True
            )
            
            print(f"Optimization completed! Found {len(result.F)} Pareto optimal solutions.")
            
            # Get recommended solution (using compromise programming approach)
            # For this example, we'll pick the solution closest to the ideal point
            ideal_point = np.min(result.F, axis=0)
            nadir_point = np.max(result.F, axis=0)
            
            # Normalize objectives to [0,1] range
            F_norm = (result.F - ideal_point) / (nadir_point - ideal_point)
            
            # Calculate Euclidean distance to ideal point (origin in normalized space)
            distances = np.sqrt(np.sum(F_norm**2, axis=1))
            
            # Get index of solution closest to ideal point
            compromise_idx = np.argmin(distances)
            compromise_solution = result.X[compromise_idx]
            
            # Decode the chosen solution
            reservoir_params, hydroworks_params = self.problem._decode_individual(compromise_solution)
            
            # Create result structure with Pareto front
            pareto_front = []
            for i, (x, f) in enumerate(zip(result.X, result.F)):
                # Create solution object similar to DEAP format
                class Solution:
                    def __init__(self, x, fitness):
                        self.X = x
                        self.fitness = fitness
                
                sol = Solution(x, f)
                pareto_front.append(sol)
            
            return {
                'success': True,
                'message': f"Multi-objective optimization completed successfully after {result.algorithm.n_gen} generations",
                'population_size': self.pop_size,
                'generations': result.algorithm.n_gen,
                'objective_values': tuple(float(val) for val in result.F[compromise_idx]),
                'optimal_reservoir_parameters': reservoir_params,
                'optimal_hydroworks_parameters': hydroworks_params,
                'pareto_front': pareto_front,
                'optimizer': self.problem  # For pareto front processing
            }
            
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return {
                'success': False,
                'message': f"Optimization failed: {str(e)}",
                'population_size': self.pop_size,
                'generations': 0,
                'objective_values': tuple([float('inf')] * self.problem.n_obj),
                'optimal_reservoir_parameters': {},
                'optimal_hydroworks_parameters': {},
                'pareto_front': []
            }
    
    def plot_convergence(self) -> None:
        """
        Plot convergence history after optimization
        """
        directory = './model_output/pymoo/convergence'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Plot for convergence of minimum values per objective
        plt.figure(figsize=(10, 6))
        gens = range(len(self.callback.data["min_obj"]))
        
        plt.plot(gens, self.callback.data["min_obj"], 'b-', label='Best Aggregate Fitness')
        plt.plot(gens, self.callback.data["avg_obj"], 'r-', label='Average Aggregate Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Multi-Objective Convergence')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./model_output/pymoo/convergence/pymoo_mo_convergence_pop{self.pop_size}_ngen{self.n_gen}.png')
        plt.close()
        
        # Plot Pareto front for 2-objective case
        if self.problem.n_obj == 2:
            try:
                # Run optimization once more to get result object
                result = minimize(
                    problem=self.problem,
                    algorithm=self.algorithm,
                    termination=DefaultMultiObjectiveTermination(n_max_gen=self.n_gen),
                    seed=42,
                    verbose=False
                )
                
                plt.figure(figsize=(10, 6))
                plt.scatter(result.F[:, 0], result.F[:, 1], s=30, facecolors='none', edgecolors='blue')
                
                # Mark reference (ideal) point
                ideal = np.min(result.F, axis=0)
                plt.plot(ideal[0], ideal[1], 'r*', markersize=12, label="Ideal Point")
                
                # Highlight compromise solution
                ideal_point = np.min(result.F, axis=0)
                nadir_point = np.max(result.F, axis=0)
                F_norm = (result.F - ideal_point) / (nadir_point - ideal_point)
                distances = np.sqrt(np.sum(F_norm**2, axis=1))
                compromise_idx = np.argmin(distances)
                
                plt.plot(result.F[compromise_idx, 0], result.F[compromise_idx, 1], 'go', 
                        markersize=10, label="Compromise Solution")
                
                plt.xlabel('Demand Deficit')
                plt.ylabel('Min Flow Deficit')
                plt.title('Pareto Front')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'./model_output/pymoo/convergence/pymoo_pareto_front_pop{self.pop_size}_ngen{self.n_gen}.png')
                plt.close()
            except Exception as e:
                print(f"Could not plot Pareto front: {str(e)}")
