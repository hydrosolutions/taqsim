"""
This module extends the WaterSystem class with optimization capabilities using linear programming.
It uses PuLP as the linear programming framework to optimize water allocation in the system.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import pulp

from .water_system import WaterSystem
from .structure import SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode


class OptimizedWaterSystem(WaterSystem):
    """
    Extends the WaterSystem class with optimization capabilities using linear programming.
    
    Attributes:
        All attributes from WaterSystem
        optimization_results (dict): Stores the results of the most recent optimization
    """
    
    def __init__(self, dt: float = 2629800, start_year: int = 2017, start_month: int = 1) -> None:
        """
        Initialize an OptimizedWaterSystem instance.
        
        Args:
            dt (float): The length of each time step in seconds. Defaults to one month (2629800 seconds).
            start_year (int): The starting year for the simulation.
            start_month (int): The starting month (1-12) for the simulation.
        """
        super().__init__(dt, start_year, start_month)
        self.optimization_results = {}
        
    def optimize(self, time_steps: int, objective_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize the water system for a specified number of time steps using linear programming.
        
        Args:
            time_steps (int): The number of time steps to optimize.
            objective_weights (Dict[str, float], optional): Weights for different objectives in the optimization.
                Possible keys: 'demand', 'min_flow', 'storage', 'losses'.
                
        Returns:
            Dict[str, Any]: Results of the optimization, including objective value and decision variables.
        """
        if not self.has_been_checked:
            self._check_network()
            
        # Set default weights if not provided
        if objective_weights is None:
            objective_weights = {
                'demand': 1.0,  # Weight for demand shortfalls
                'min_flow': 1.0,  # Weight for minimum flow violations
                'storage': 0.1,  # Weight for final storage volumes
                'losses': 0.1    # Weight for system losses
            }
            
        # Initialize the LP problem
        problem = pulp.LpProblem("WaterAllocationOptimization", pulp.LpMinimize)
        
        # Extract all nodes and edges from the graph
        sorted_nodes = list(nx.topological_sort(self.graph))
        all_edges = list(self.graph.edges())
        
        # Create decision variables
        variables = self._create_decision_variables(time_steps, all_edges, sorted_nodes)
        
        # Add constraints
        self._add_constraints(problem, variables, time_steps, sorted_nodes, all_edges)
        
        # Set objective function
        objective = self._create_objective_function(variables, objective_weights, time_steps)
        problem += objective
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=False)  # Use CBC solver with reduced output
        problem.solve(solver)
        
        # Check solution status
        if problem.status != pulp.LpStatusOptimal:
            print(f"Warning: Optimization did not reach optimal solution. Status: {pulp.LpStatus[problem.status]}")
        
        # Process and store results
        self.optimization_results = self._process_optimization_results(problem, variables, time_steps)
        
        # Apply optimized flows to the water system
        self._apply_optimized_flows(variables, time_steps)
        
        # Run a verification simulation
        self.simulate(time_steps)
        
        return self.optimization_results
    
    def _create_decision_variables(self, time_steps: int, edges: List[Tuple[str, str]], 
                               nodes: List[str]) -> Dict[str, Any]:
        """
        Create decision variables for the optimization problem.
        
        Args:
            time_steps (int): Number of time steps to optimize.
            edges (List[Tuple[str, str]]): List of edges in the water system.
            nodes (List[str]): List of nodes in the water system.
            
        Returns:
            Dict[str, Any]: Dictionary containing all decision variables.
        """
        variables = {
            'edge_flow': {},      # Flow on each edge at each time step
            'storage': {},        # Storage volume in reservoirs at each time step
            'unmet_demand': {},   # Unmet demand at each demand node at each time step
            'min_flow_deficit': {}  # Minimum flow deficit at each sink node at each time step
        }
        
        # Create edge flow variables
        for source, target in edges:
            edge_data = self.graph.edges[source, target]
            edge_obj = edge_data['edge']
            capacity = edge_obj.capacity
            
            for t in range(time_steps):
                var_name = f"flow_{source}_{target}_{t}"
                variables['edge_flow'][(source, target, t)] = pulp.LpVariable(
                    var_name, lowBound=0, upBound=capacity, cat=pulp.LpContinuous
                )
        
        # Create storage variables for reservoir nodes
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, StorageNode):
                # Variable for each time step + initial storage
                for t in range(time_steps + 1):  # +1 for final storage
                    var_name = f"storage_{node_id}_{t}"
                    variables['storage'][(node_id, t)] = pulp.LpVariable(
                        var_name, lowBound=node.dead_storage, upBound=node.capacity, cat=pulp.LpContinuous
                    )
        
        # Create unmet demand variables
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, DemandNode):
                for t in range(time_steps):
                    var_name = f"unmet_demand_{node_id}_{t}"
                    variables['unmet_demand'][(node_id, t)] = pulp.LpVariable(
                        var_name, lowBound=0, cat=pulp.LpContinuous
                    )
        
        # Create minimum flow deficit variables
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, SinkNode):
                for t in range(time_steps):
                    var_name = f"min_flow_deficit_{node_id}_{t}"
                    variables['min_flow_deficit'][(node_id, t)] = pulp.LpVariable(
                        var_name, lowBound=0, cat=pulp.LpContinuous
                    )
        
        return variables
    
    def _add_constraints(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                       time_steps: int, nodes: List[str], edges: List[Tuple[str, str]]) -> None:
        """
        Add constraints to the optimization problem.
        
        Args:
            problem (pulp.LpProblem): The PuLP problem to add constraints to.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            nodes (List[str]): List of nodes in the water system.
            edges (List[Tuple[str, str]]): List of edges in the water system.
        """
        # Add flow continuity constraints (mass balance) for each node
        self._add_flow_continuity_constraints(problem, variables, time_steps, nodes, edges)
        
        # Add storage continuity constraints for reservoir nodes
        self._add_storage_continuity_constraints(problem, variables, time_steps, nodes)
        
        # Add demand satisfaction constraints
        self._add_demand_constraints(problem, variables, time_steps, nodes)
        
        # Add minimum flow constraints for sink nodes
        self._add_minimum_flow_constraints(problem, variables, time_steps, nodes)
        
        # Add distribution constraints for HydroWorks nodes
        self._add_distribution_constraints(problem, variables, time_steps, nodes)
    
    def _add_flow_continuity_constraints(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                                     time_steps: int, nodes: List[str], edges: List[Tuple[str, str]]) -> None:
        """
        Add flow continuity constraints ensuring mass balance at each node.
        
        Args:
            problem (pulp.LpProblem): The PuLP problem to add constraints to.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            nodes (List[str]): List of nodes in the water system.
            edges (List[Tuple[str, str]]): List of edges in the water system.
        """
        for t in range(time_steps):
            for node_id in nodes:
                node_data = self.graph.nodes[node_id]
                node = node_data['node']
                
                # Skip storage nodes (handled separately)
                if isinstance(node, StorageNode):
                    continue
                
                # Get inflow edges to this node
                in_edges = [(u, v) for u, v in edges if v == node_id]
                
                # Get outflow edges from this node
                out_edges = [(u, v) for u, v in edges if u == node_id]
                
                # For SupplyNode and RunoffNode:
                if isinstance(node, SupplyNode):
                    # Supply rate is the input
                    supply_rate = node.supply_rates[t] if t < len(node.supply_rates) else 0
                    
                    # Total outflow must equal supply rate
                    if out_edges:
                        outflow_sum = sum(variables['edge_flow'][(node_id, v, t)] for _, v in out_edges)
                        problem += (outflow_sum == supply_rate, f"Supply_continuity_{node_id}_{t}")
                
                elif isinstance(node, RunoffNode):
                    # Calculate runoff for this time step
                    rainfall = node.rainfall_data[t] if t < len(node.rainfall_data) else 0
                    runoff_rate = node.calculate_runoff(rainfall, self.dt)
                    
                    # Total outflow must equal runoff rate
                    if out_edges:
                        outflow_sum = sum(variables['edge_flow'][(node_id, v, t)] for _, v in out_edges)
                        problem += (outflow_sum == runoff_rate, f"Runoff_continuity_{node_id}_{t}")
                
                # For SinkNode:
                elif isinstance(node, SinkNode):
                    # Total inflow must meet or exceed minimum flow requirement
                    if in_edges:
                        inflow_sum = sum(variables['edge_flow'][(u, node_id, t)] for u, _ in in_edges)
                        min_flow = node.min_flows[t] if t < len(node.min_flows) else 0
                        
                        # Define relationship between inflow, minimum flow, and deficit
                        problem += (inflow_sum + variables['min_flow_deficit'][(node_id, t)] >= min_flow, 
                                   f"MinFlow_requirement_{node_id}_{t}")
                
                # For DemandNode:
                elif isinstance(node, DemandNode):
                    if in_edges and out_edges:
                        # Total inflow
                        inflow_sum = sum(variables['edge_flow'][(u, node_id, t)] for u, _ in in_edges)
                        
                        # Total outflow (non-consumptive portion)
                        outflow_sum = sum(variables['edge_flow'][(node_id, v, t)] for _, v in out_edges)
                        
                        # Demand for this time step
                        demand = node.demand_rates[t] if t < len(node.demand_rates) else 0
                        
                        # Non-consumptive demand portion
                        non_consumptive = node.non_consumptive_rate
                        
                        # Consumptive demand
                        consumptive_demand = demand - non_consumptive
                        
                        # Constraint: inflow = consumptive use + outflow + unmet demand
                        problem += (inflow_sum == outflow_sum + (consumptive_demand - variables['unmet_demand'][(node_id, t)]),
                                   f"Demand_balance_{node_id}_{t}")
                        
                        # Cannot supply more than demand (avoid over-supply)
                        problem += (variables['unmet_demand'][(node_id, t)] <= consumptive_demand, 
                                   f"Max_unmet_{node_id}_{t}")
                
                # For HydroWorks (distribution points):
                elif isinstance(node, HydroWorks):
                    if in_edges and out_edges:
                        # Total inflow
                        inflow_sum = sum(variables['edge_flow'][(u, node_id, t)] for u, _ in in_edges)
                        
                        # Total outflow
                        outflow_sum = sum(variables['edge_flow'][(node_id, v, t)] for _, v in out_edges)
                        
                        # Inflow should equal outflow (no losses at HydroWorks except spillage)
                        problem += (inflow_sum == outflow_sum, f"HydroWorks_balance_{node_id}_{t}")
                        
                        # Set outflow constraints based on distribution parameters
                        # This is handled in a separate method
    
    def _add_storage_continuity_constraints(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                                        time_steps: int, nodes: List[str]) -> None:
        """
        Add storage continuity constraints for reservoir nodes.
        
        Args:
            problem (pulp.LpProblem): The PuLP problem to add constraints to.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            nodes (List[str]): List of nodes in the water system.
        """
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, StorageNode):
                # Set initial storage from node's current storage
                problem += (variables['storage'][(node_id, 0)] == node.storage[-1], 
                           f"Initial_storage_{node_id}")
                
                # Get inflow and outflow edges
                in_edges = list(self.graph.in_edges(node_id))
                out_edges = list(self.graph.out_edges(node_id))
                
                for t in range(time_steps):
                    # Calculate evaporation loss for this time step
                    evap_rate = node.evaporation_rates[t] if t < len(node.evaporation_rates) else 0
                    
                    # Simplified evaporation calculation - in a real implementation, you'd need 
                    # a linearized approximation of the nonlinear relationship
                    evap_loss = evap_rate * self.dt * 0.001  # Simplified estimate, not exact
                    
                    # Sum of inflows
                    inflow_sum = sum(variables['edge_flow'][(u, node_id, t)] for u, _ in in_edges)
                    
                    # Sum of outflows
                    outflow_sum = sum(variables['edge_flow'][(node_id, v, t)] for _, v in out_edges)
                    
                    # Storage continuity equation: next_storage = current_storage + inflows - outflows - evaporation
                    problem += (variables['storage'][(node_id, t+1)] == 
                                variables['storage'][(node_id, t)] + inflow_sum * self.dt - outflow_sum * self.dt - evap_loss,
                               f"Storage_continuity_{node_id}_{t}")
                    
                    # Release policy constraints (simplification of the actual policy)
                    # This is a simplified linear approximation of the more complex buffer-based release policy
                    if t % 12 < len(node.release_params.get('V1', [])):  # If release parameters are set
                        current_month = t % 12
                        Vr = node.release_params['Vr'][current_month] if 'Vr' in node.release_params else 0
                        max_release_rate = Vr / self.dt
                        
                        # Limit outflow based on a simplified linear release policy
                        if out_edges:
                            problem += (outflow_sum <= max_release_rate, f"Max_release_{node_id}_{t}")
    
    def _add_demand_constraints(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                             time_steps: int, nodes: List[str]) -> None:
        """
        Add constraints related to demand satisfaction.
        
        Args:
            problem (pulp.LpProblem): The PuLP problem to add constraints to.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            nodes (List[str]): List of nodes in the water system.
        """
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, DemandNode):
                for t in range(time_steps):
                    # Demand satisfaction is handled in the flow continuity constraints
                    # Additional constraints could be added here if needed
                    pass
    
    def _add_minimum_flow_constraints(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                                   time_steps: int, nodes: List[str]) -> None:
        """
        Add constraints related to minimum flow requirements at sink nodes.
        
        Args:
            problem (pulp.LpProblem): The PuLP problem to add constraints to.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            nodes (List[str]): List of nodes in the water system.
        """
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, SinkNode):
                # Handled in flow continuity constraints
                pass
    
    def _add_distribution_constraints(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                                   time_steps: int, nodes: List[str]) -> None:
        """
        Add constraints related to flow distribution at HydroWorks nodes.
        
        Args:
            problem (pulp.LpProblem): The PuLP problem to add constraints to.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            nodes (List[str]): List of nodes in the water system.
        """
        for node_id in nodes:
            node_data = self.graph.nodes[node_id]
            node = node_data['node']
            
            if isinstance(node, HydroWorks) and node.distribution_params:
                in_edges = list(self.graph.in_edges(node_id))
                out_edges = list(self.graph.out_edges(node_id))
                
                for t in range(time_steps):
                    current_month = t % 12
                    
                    # Calculate total inflow
                    inflow_sum = sum(variables['edge_flow'][(u, node_id, t)] for u, _ in in_edges)
                    
                    # Apply distribution parameters to each outflow
                    for _, target in out_edges:
                        if target in node.distribution_params:
                            # Get distribution parameter for this edge and current month
                            dist_param = node.distribution_params[target][current_month]
                            
                            # Outflow should be proportional to distribution parameter
                            # Allow some flexibility by using inequality instead of strict equality
                            flow_var = variables['edge_flow'][(node_id, target, t)]
                            target_flow = inflow_sum * dist_param
                            
                            # Flow should be close to target flow (within a small tolerance)
                            problem += (flow_var <= target_flow * 1.01, f"Dist_upper_{node_id}_{target}_{t}")
                            problem += (flow_var >= target_flow * 0.99, f"Dist_lower_{node_id}_{target}_{t}")
    
    def _create_objective_function(self, variables: Dict[str, Any], weights: Dict[str, float], 
                               time_steps: int) -> pulp.LpAffineExpression:
        """
        Create the objective function for the optimization problem.
        
        Args:
            variables (Dict[str, Any]): Dictionary of decision variables.
            weights (Dict[str, float]): Weights for different components of the objective.
            time_steps (int): Number of time steps.
            
        Returns:
            pulp.LpAffineExpression: The objective function to minimize.
        """
        objective_terms = []
        
        # Minimize unmet demand (weighted by demand node weights)
        for (node_id, t), var in variables['unmet_demand'].items():
            node = self.graph.nodes[node_id]['node']
            node_weight = node.weight if hasattr(node, 'weight') else 1.0
            objective_terms.append(weights['demand'] * node_weight * var)
        
        # Minimize minimum flow deficits (weighted by sink node weights)
        for (node_id, t), var in variables['min_flow_deficit'].items():
            node = self.graph.nodes[node_id]['node']
            node_weight = node.weight if hasattr(node, 'weight') else 1.0
            objective_terms.append(weights['min_flow'] * node_weight * var)
        
        # Maximize final storage in reservoirs (negative coefficient for maximization)
        for (node_id, t), var in variables['storage'].items():
            if t == time_steps:  # Final storage only
                objective_terms.append(-weights['storage'] * var)
        
        # Minimize losses (more complex, simplified implementation)
        # In a full implementation, you would need to model and include:
        # - Edge transmission losses
        # - Reservoir evaporation
        # - Spillage from reservoirs and HydroWorks
        
        return pulp.lpSum(objective_terms)
    
    def _process_optimization_results(self, problem: pulp.LpProblem, variables: Dict[str, Any], 
                                  time_steps: int) -> Dict[str, Any]:
        """
        Process the results of the optimization.
        
        Args:
            problem (pulp.LpProblem): The solved PuLP problem.
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
            
        Returns:
            Dict[str, Any]: The processed optimization results.
        """
        results = {
            'status': pulp.LpStatus[problem.status],
            'objective_value': pulp.value(problem.objective),
            'edge_flows': {},
            'storage_levels': {},
            'unmet_demands': {},
            'min_flow_deficits': {}
        }
        
        # Extract edge flows
        for (source, target, t), var in variables['edge_flow'].items():
            if (source, target) not in results['edge_flows']:
                results['edge_flows'][(source, target)] = []
            results['edge_flows'][(source, target)].append(pulp.value(var))
        
        # Extract storage levels
        for (node_id, t), var in variables['storage'].items():
            if node_id not in results['storage_levels']:
                results['storage_levels'][node_id] = []
            if t < time_steps:  # Don't include final storage in the time series
                results['storage_levels'][node_id].append(pulp.value(var))
        
        # Extract unmet demands
        for (node_id, t), var in variables['unmet_demand'].items():
            if node_id not in results['unmet_demands']:
                results['unmet_demands'][node_id] = []
            results['unmet_demands'][node_id].append(pulp.value(var))
        
        # Extract minimum flow deficits
        for (node_id, t), var in variables['min_flow_deficit'].items():
            if node_id not in results['min_flow_deficits']:
                results['min_flow_deficits'][node_id] = []
            results['min_flow_deficits'][node_id].append(pulp.value(var))
        
        return results
    
    def _apply_optimized_flows(self, variables: Dict[str, Any], time_steps: int) -> None:
        """
        Apply the optimized flows to the water system.
        
        This method updates the water system with the optimized flow values,
        allowing the system to be simulated using these values.
        
        Args:
            variables (Dict[str, Any]): Dictionary of decision variables.
            time_steps (int): Number of time steps.
        """
        # Apply optimized flows to edges
        for (source, target, t), var in variables['edge_flow'].items():
            edge = self.graph.edges[source, target]['edge']
            if t < len(edge.flow_after_losses):
                edge.flow_after_losses[t] = pulp.value(var)
            else:
                edge.flow_after_losses.append(pulp.value(var))
            
            # Set flow_before_losses to match (simplified)
            if t < len(edge.flow_before_losses):
                edge.flow_before_losses[t] = pulp.value(var)
            else:
                edge.flow_before_losses.append(pulp.value(var))
        
        # Apply optimized storage levels to storage nodes
        for (node_id, t), var in variables['storage'].items():
            if t < time_steps:  # Don't apply final storage
                node = self.graph.nodes[node_id]['node']
                if t < len(node.storage):
                    node.storage[t] = pulp.value(var)
                else:
                    node.storage.append(pulp.value(var))
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Generate a summary of the optimization results.
        
        Returns:
            pd.DataFrame: Summary of optimization results.
        """
        if not self.optimization_results:
            return pd.DataFrame()
        
        # Basic summary statistics
        summary = {
            'Objective Value': self.optimization_results['objective_value'],
            'Status': self.optimization_results['status']
        }
        
        # Calculate demand satisfaction statistics
        total_demand = 0
        total_supplied = 0
        
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            
            if isinstance(node, DemandNode):
                if node_id in self.optimization_results['unmet_demands']:
                    unmet_demand = sum(self.optimization_results['unmet_demands'][node_id])
                    node_demand = sum(node.demand_rates[:len(self.optimization_results['unmet_demands'][node_id])])
                    
                    total_demand += node_demand
                    total_supplied += (node_demand - unmet_demand)
        
        if total_demand > 0:
            summary['Demand Satisfaction (%)'] = (total_supplied / total_demand) * 100
        
        # Calculate minimum flow satisfaction statistics
        total_min_flow = 0
        total_deficit = 0
        
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            
            if isinstance(node, SinkNode):
                if node_id in self.optimization_results['min_flow_deficits']:
                    deficit = sum(self.optimization_results['min_flow_deficits'][node_id])
                    min_flow = sum(node.min_flows[:len(self.optimization_results['min_flow_deficits'][node_id])])
                    
                    total_min_flow += min_flow
                    total_deficit += deficit
        
        if total_min_flow > 0:
            summary['Min Flow Satisfaction (%)'] = ((total_min_flow - total_deficit) / total_min_flow) * 100
        
        return pd.DataFrame([summary])
    
    def compare_with_simulation(self, time_steps: int) -> pd.DataFrame:
        """
        Compare optimized results with a standard simulation.
        
        Args:
            time_steps (int): Number of time steps to simulate.
            
        Returns:
            pd.DataFrame: Comparison of optimized vs. simulated results.
        """
        # Create a copy of the current system for simulation
        import copy
        sim_system = copy.deepcopy(self)
        
        # Run standard simulation
        sim_system.simulate(time_steps)
        
        # Get water balance for both systems
        opt_balance = self.get_water_balance()
        sim_balance = sim_system.get_water_balance()
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Metric': [
                'Total Demand (m³)',
                'Supplied Demand (m³)',
                'Unmet Demand (m³)',
                'Demand Satisfaction (%)',
                'Min Flow Requirements (m³)',
                'Min Flow Deficits (m³)',
                'Min Flow Satisfaction (%)',
                'Total Storage Change (m³)',
                'Total Losses (m³)'
            ]
        })
        
        # Calculate metrics for optimized system
        opt_metrics = {
            'Total Demand (m³)': opt_balance['demands'].sum(),
            'Supplied Demand (m³)': opt_balance['supplied consumptive demand'].sum() + 
                                   opt_balance['supplied non consumptive demand'].sum(),
            'Unmet Demand (m³)': opt_balance['unmet demand'].sum(),
            'Min Flow Requirements (m³)': opt_balance['sink min flow requirement'].sum(),
            'Min Flow Deficits (m³)': opt_balance['sink min flow deficit'].sum(),
            'Total Storage Change (m³)': opt_balance['storage_change'].sum(),
            'Total Losses (m³)': opt_balance['edge losses'].sum() + 
                               opt_balance['reservoir ET losses'].sum() + 
                               opt_balance['reservoir spills'].sum() + 
                               opt_balance['hydroworks spills'].sum()
        }
        
        opt_metrics['Demand Satisfaction (%)'] = (
            (opt_metrics['Supplied Demand (m³)'] / opt_metrics['Total Demand (m³)']) * 100 
            if opt_metrics['Total Demand (m³)'] > 0 else 100
        )
        
        opt_metrics['Min Flow Satisfaction (%)'] = (
            ((opt_metrics['Min Flow Requirements (m³)'] - opt_metrics['Min Flow Deficits (m³)']) / 
             opt_metrics['Min Flow Requirements (m³)']) * 100
            if opt_metrics['Min Flow Requirements (m³)'] > 0 else 100
        )
        
        # Calculate metrics for simulated system
        sim_metrics = {
            'Total Demand (m³)': sim_balance['demands'].sum(),
            'Supplied Demand (m³)': sim_balance['supplied consumptive demand'].sum() + 
                                   sim_balance['supplied non consumptive demand'].sum(),
            'Unmet Demand (m³)': sim_balance['unmet demand'].sum(),
            'Min Flow Requirements (m³)': sim_balance['sink min flow requirement'].sum(),
            'Min Flow Deficits (m³)': sim_balance['sink min flow deficit'].sum(),
            'Total Storage Change (m³)': sim_balance['storage_change'].sum(),
            'Total Losses (m³)': sim_balance['edge losses'].sum() + 
                               sim_balance['reservoir ET losses'].sum() + 
                               sim_balance['reservoir spills'].sum() + 
                               sim_balance['hydroworks spills'].sum()
        }
        
        sim_metrics['Demand Satisfaction (%)'] = (
            (sim_metrics['Supplied Demand (m³)'] / sim_metrics['Total Demand (m³)']) * 100 
            if sim_metrics['Total Demand (m³)'] > 0 else 100
        )
        
        sim_metrics['Min Flow Satisfaction (%)'] = (
            ((sim_metrics['Min Flow Requirements (m³)'] - sim_metrics['Min Flow Deficits (m³)']) / 
             sim_metrics['Min Flow Requirements (m³)']) * 100
            if sim_metrics['Min Flow Requirements (m³)'] > 0 else 100
        )
        
        # Add metrics to comparison dataframe
        comparison['Optimized'] = [opt_metrics[metric] for metric in comparison['Metric']]
        comparison['Simulated'] = [sim_metrics[metric] for metric in comparison['Metric']]
        comparison['Improvement (%)'] = [
            ((opt_metrics[metric] - sim_metrics[metric]) / sim_metrics[metric] * 100)
            if metric in ['Demand Satisfaction (%)', 'Min Flow Satisfaction (%)']
            else ((sim_metrics[metric] - opt_metrics[metric]) / sim_metrics[metric] * 100)
            if metric in ['Unmet Demand (m³)', 'Min Flow Deficits (m³)', 'Total Losses (m³)']
            else 0
            for metric in comparison['Metric']
        ]
        
        return comparison.round(2)