from scipy.optimize import minimize
import numpy as np
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge

class ReleaseOptimizer:
    """
    Optimizer for reservoir release function parameters to minimize unsupplied demand.
    """
    
    def __init__(self, create_system_func, num_time_steps):
        """
        Initialize the optimizer.
        
        Args:
            create_system_func: Function that creates and returns a water system
            num_time_steps (int): Number of time steps to simulate
        """
        self.create_system_func = create_system_func
        self.num_time_steps = num_time_steps
        self.best_params = None
        self.best_objective = float('inf')
        
        # Initialize system and get reservoir data
        system = self.create_system_func(self.num_time_steps)
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
        
    def objective_function(self, x):
        """
        Calculate total unsupplied demand for given parameters.
        
        Args:
            x: Array of parameters [h1, h2, w, m1, m2]
            
        Returns:
            float: Total unsupplied demand volume
        """
        # Extract and validate parameters
        h1, h2, w, m1, m2 = x
        
        # Create new system with these parameters
        system = self.create_system_func(self.num_time_steps)
        
        # Update release parameters for the reservoir
        for node_id, node_data in system.graph.nodes(data=True):
            node = node_data['node']
            if isinstance(node, StorageNode):
                node.set_release_params({
                    'h1': float(h1),
                    'h2': float(h2),
                    'w': float(w),
                    'm1': float(m1),
                    'm2': float(m2)
                })
        
        # Run simulation with error handling
        try:
            system.simulate(self.num_time_steps)
            
            # Calculate total unmet demand
            total_unmet_demand = 0
            total_demand = 0
            
            for node_id, node_data in system.graph.nodes(data=True):
                node = node_data['node']
                if isinstance(node, DemandNode):
                    for t in range(self.num_time_steps):
                        demand = node.get_demand_rate(t)
                        satisfied = node.satisfied_demand[t]
                        total_demand += demand * system.dt
                        total_unmet_demand += (demand - satisfied) * system.dt
            
            # Add penalty for release parameters near bounds
            penalty = 0
            if h1 < self.min_level + 1 or h1 > self.max_level - 1:
                penalty += 1e5
            if h2 < self.min_level + 1 or h2 > self.max_level - 1:
                penalty += 1e5
            if w < 1 or w > self.max_capacity - 1:
                penalty += 1e5
                
            objective_value = total_unmet_demand + penalty
            
            # Update best solution if this is better
            if objective_value < self.best_objective:
                self.best_objective = objective_value
                self.best_params = x.copy()
                print(f"New best solution found: {objective_value:,.0f} m³ "
                      f"(Unmet: {total_unmet_demand:,.0f} m³, {100*total_unmet_demand/total_demand:.1f}%)")
                
            return objective_value
            
        except Exception as e:
            print(f"Simulation failed with parameters {x}: {str(e)}")
            return 1e10
    
    def constraint_h1_h2(self, x):
        """Ensure h1 < h2 with minimum separation"""
        h1, h2, _, _, _ = x
        return h2 - h1 - 1  # Minimum 1m separation
    
    def constraint_w_capacity(self, x):
        """Ensure w doesn't exceed maximum outflow capacity"""
        _, _, w, _, _ = x
        return self.max_capacity - w
    
    def constraint_h1_range(self, x):
        """Ensure h1 is within valid range"""
        h1, _, _, _, _ = x
        return min(h1 - self.min_level, self.max_level - h1)
    
    def constraint_h2_range(self, x):
        """Ensure h2 is within valid range"""
        _, h2, _, _, _ = x
        return min(h2 - self.min_level, self.max_level - h2)
        
    def optimize(self, initial_guess=None):
        """
        Run the optimization to find optimal release parameters.
        
        Args:
            initial_guess: Initial parameter values [h1, h2, w, m1, m2]
            
        Returns:
            dict: Optimization results
        """
        # Set bounds for parameters
        level_range = self.max_level - self.min_level
        bounds = [
            (self.min_level + 0.1*level_range, self.max_level - 0.1*level_range),  # h1
            (self.min_level + 0.2*level_range, self.max_level - 0.05*level_range), # h2
            (1.0, self.max_capacity * 0.95),    # w
            (0.1, 1.5),                         # m1
            (0.1, 1.5)                          # m2
        ]
        
        # Set default initial guess if none provided
        if initial_guess is None:
            initial_guess = [
                self.min_level + 0.3 * level_range,  # h1
                self.min_level + 0.7 * level_range,  # h2
                0.3 * self.max_capacity,             # w
                0.785,                               # m1 (≈ π/4)
                0.785                                # m2 (≈ π/4)
            ]
        
        # Define constraints
        constraints = [
            {'type': 'ineq', 'fun': self.constraint_h1_h2},
            {'type': 'ineq', 'fun': self.constraint_w_capacity},
            {'type': 'ineq', 'fun': self.constraint_h1_range},
            {'type': 'ineq', 'fun': self.constraint_h2_range}
        ]
        
        print("\nStarting optimization...")
        print(f"Initial parameters: h1={initial_guess[0]:.1f}, h2={initial_guess[1]:.1f}, "
              f"w={initial_guess[2]:.1f}, m1={initial_guess[3]:.3f}, m2={initial_guess[4]:.3f}")
        print(f"Bounds: {bounds}")
        
        # Try multiple starting points if first optimization fails
        for attempt in range(3):
            if attempt > 0:
                print(f"\nRetrying optimization (attempt {attempt + 1})...")
                # Modify initial guess for subsequent attempts
                initial_guess = [
                    self.min_level + (0.2 + 0.1*attempt) * level_range,  # h1
                    self.min_level + (0.6 + 0.1*attempt) * level_range,  # h2
                    (0.2 + 0.1*attempt) * self.max_capacity,             # w
                    0.5 + 0.2*attempt,                                   # m1
                    0.5 + 0.2*attempt                                    # m2
                ]
            
            # Run optimization
            result = minimize(
                self.objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                break
        
        # Format results
        optimized_params = {
            'h1': result.x[0],
            'h2': result.x[1],
            'w': result.x[2],
            'm1': result.x[3],
            'm2': result.x[4]
        }
        
        # Validate final parameters
        validation_system = self.create_system_func(self.num_time_steps)
        for node_id, node_data in validation_system.graph.nodes(data=True):
            if isinstance(node_data['node'], StorageNode):
                try:
                    node_data['node'].set_release_params(optimized_params)
                    validation_system.simulate(self.num_time_steps)
                except Exception as e:
                    print(f"\nWarning: Final parameters failed validation: {str(e)}")
                    result.success = False
                break
        
        return {
            'success': result.success,
            'optimal_parameters': optimized_params,
            'objective_value': result.fun,
            'message': result.message,
            'iterations': result.nit,
            'best_objective': self.best_objective,
            'best_parameters': {
                'h1': self.best_params[0],
                'h2': self.best_params[1],
                'w': self.best_params[2],
                'm1': self.best_params[3],
                'm2': self.best_params[4]
            } if self.best_params is not None else None
        }