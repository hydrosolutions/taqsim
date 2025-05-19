"""
ZRB System Optimization with Pyomo

This module transforms the ZRB water system model to use Pyomo for linear programming optimization.
The optimization determines optimal water allocation across the network to maximize benefits.
"""

from typing import Callable, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Import original system components for reference
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, Edge
from ZRB_system_creator import create_ZRB_system
from deap_example import load_optimized_parameters, load_parameters_from_file

class ZRBOptimizationModel:
    """
    Represents the ZRB water system as a Pyomo optimization model.
    This class transforms the original simulation-based water system to a
    constraint-based optimization model for optimal water allocation.
    
    Attributes:
        model (pyo.ConcreteModel): The Pyomo model instance
        dt (float): Time step duration in seconds
        num_time_steps (int): Number of time steps in the optimization period
        start_year (int): Starting year of the optimization period
        start_month (int): Starting month of the optimization period
        scenario (str): Climate scenario name (if applicable)
        nodes (Dict): Dictionary of node information
        edges (Dict): Dictionary of edge information 
    """
    
    def __init__(self, 
                start_year: int,
                start_month: int,
                num_time_steps: int,
                dt: float = 30.44 * 24 * 3600,  # Default to monthly time step
                system_type: str = "baseline",
                scenario: str = '',
                period: str = '',
                agr_scenario: str = '',
                efficiency: str = ''):
        """
        Initialize a new ZRB optimization model.
        
        Args:
            start_year (int): Start year for optimization
            start_month (int): Start month for optimization (1-12)
            num_time_steps (int): Number of time steps to optimize
            dt (float): Time step duration in seconds
            system_type (str): "baseline" or "scenario" 
            scenario (str): Climate scenario (e.g., 'ssp126')
            period (str): Time period (e.g., '2041-2070')
            agr_scenario (str): Agricultural scenario
            efficiency (str): Efficiency scenario (e.g., 'improved_efficiency')
        """
        self.model = pyo.ConcreteModel(name="ZRB_Optimization")
        self.dt = dt
        self.num_time_steps = num_time_steps
        self.start_year = start_year
        self.start_month = start_month
        self.system_type = system_type
        self.scenario = scenario
        self.period = period
        self.agr_scenario = agr_scenario
        self.efficiency = efficiency
        
        # Dictionaries to store node and edge information
        self.nodes = {}
        self.edges = []
        
        # Reference to the original water system (for data import)
        self.original_system = None
        print('b')
        
        # Initialize the optimization model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the core components of the Pyomo optimization model."""
        model = self.model
        # Define time steps set
        model.T = pyo.RangeSet(0, self.num_time_steps-1)
        # Create reference to the original system for data import
        self._create_original_system()
        # Create node and edge structures for optimization
        self._create_node_structures()
        self._create_edge_structures()
        
        # Initialize decision variables, constraints, and objective
        self._create_decision_variables()
        print('c')
        self._create_constraints()
        print('d')

        self._create_objective()
        print('e')
    
    def _create_original_system(self):
        """
        Create the original water system model to extract data for optimization.
        This is used only for initializing data, not for simulation.
        """
        
        self.original_system = create_ZRB_system(
            start_year=self.start_year,
            start_month=self.start_month,
            num_time_steps=self.num_time_steps,
            system_type=self.system_type,
            scenario=self.scenario,
            period=self.period,
            agr_scenario=self.agr_scenario,
            efficiency=self.efficiency
        )
    
    def _create_node_structures(self):
        """
        Extract node information from the original system and create
        corresponding structures for the optimization model.
        """
        # Process each node in the original system
        for node_id, node_data in self.original_system.graph.nodes(data=True):
            node = node_data['node']
            node_type = type(node).__name__
            
            # Store node information based on type
            if isinstance(node, SupplyNode):
                self.nodes[node_id] = {
                    'type': 'supply',
                    'supply_rates': node.supply_rates,
                    'easting': node.easting,
                    'northing': node.northing
                }
            
            elif isinstance(node, StorageNode):
                self.nodes[node_id] = {
                    'type': 'storage',
                    'capacity': node.capacity,
                    'dead_storage': node.dead_storage,
                    'initial_storage': node.storage[0],
                    'evaporation_rates': node.evaporation_rates,
                    'easting': node.easting,
                    'northing': node.northing,
                    'hv_data': node.hv_data,
                    'buffer_coef': node.buffer_coef
                }
                
                # Handle release parameters if they exist
                if hasattr(node, 'release_params') and node.release_params:
                    self.nodes[node_id]['release_params'] = node.release_params
            
            elif isinstance(node, DemandNode):
                self.nodes[node_id] = {
                    'type': 'demand',
                    'demand_rates': node.demand_rates,
                    'non_consumptive_rate': node.non_consumptive_rate,
                    'weight': node.weight,
                    'field_efficiency': node.field_efficiency,
                    'conveyance_efficiency': node.conveyance_efficiency,
                    'easting': node.easting,
                    'northing': node.northing
                }
            
            elif isinstance(node, SinkNode):
                self.nodes[node_id] = {
                    'type': 'sink',
                    'min_flows': node.min_flows,
                    'weight': node.weight,
                    'easting': node.easting,
                    'northing': node.northing
                }
            
            elif isinstance(node, HydroWorks):
                self.nodes[node_id] = {
                    'type': 'hydroworks',
                    'easting': node.easting,
                    'northing': node.northing
                }
                
                # Handle distribution parameters if they exist
                if hasattr(node, 'distribution_params') and node.distribution_params:
                    self.nodes[node_id]['distribution_params'] = node.distribution_params
            
            elif isinstance(node, RunoffNode):
                self.nodes[node_id] = {
                    'type': 'runoff',
                    'area': node.area,
                    'runoff_coefficient': node.runoff_coefficient,
                    'rainfall_data': node.rainfall_data,
                    'easting': node.easting,
                    'northing': node.northing
                }
    
    def _create_edge_structures(self):
        """
        Extract edge information from the original system and create
        corresponding structures for the optimization model.
        """
        # Process each edge in the original system
        for source_id, target_id, edge_data in self.original_system.graph.edges(data=True):
            edge = edge_data['edge']
            
            self.edges.append({
                'source': source_id,
                'target': target_id,
                'capacity': edge.capacity,
                'length': edge.length,
                'loss_factor': edge.loss_factor
            })
    
    def _create_decision_variables(self):
        """
        Create decision variables for the optimization model.
        
        Variables include:
        - flow_before_losses: Flow entering each edge before losses
        - flow_after_losses: Flow exiting each edge after losses
        - storage: Storage volume for each storage node
        - release: Release from each storage node
        - supplied_demand: Demand supplied for each demand node
        - deficit: Minimum flow deficit for each sink node
        """
        model = self.model
        
        # Create index sets for different node types
        model.supply_nodes = pyo.Set(initialize=[n for n, data in self.nodes.items() 
                                              if data['type'] == 'supply'])
        model.storage_nodes = pyo.Set(initialize=[n for n, data in self.nodes.items() 
                                               if data['type'] == 'storage'])
        model.demand_nodes = pyo.Set(initialize=[n for n, data in self.nodes.items() 
                                              if data['type'] == 'demand'])
        model.sink_nodes = pyo.Set(initialize=[n for n, data in self.nodes.items() 
                                            if data['type'] == 'sink'])
        model.hydroworks_nodes = pyo.Set(initialize=[n for n, data in self.nodes.items() 
                                                  if data['type'] == 'hydroworks'])
        model.runoff_nodes = pyo.Set(initialize=[n for n, data in self.nodes.items() 
                                              if data['type'] == 'runoff'])
        
        # Create edge index set
        model.edges = pyo.Set(initialize=[(e['source'], e['target']) for e in self.edges])
        
        # Flow variables for edges
        model.flow_before_losses = pyo.Var(model.edges, model.T, 
                                         domain=pyo.NonNegativeReals)
        model.flow_after_losses = pyo.Var(model.edges, model.T, 
                                        domain=pyo.NonNegativeReals)
        
        # Storage variables
        model.storage = pyo.Var(model.storage_nodes, model.T, 
                              domain=pyo.NonNegativeReals)
        model.release = pyo.Var(model.storage_nodes, model.T, 
                              domain=pyo.NonNegativeReals)
        
        # Demand satisfaction variables
        model.supplied_consumptive_demand = pyo.Var(model.demand_nodes, model.T, 
                                                  domain=pyo.NonNegativeReals)
        model.supplied_non_consumptive_demand = pyo.Var(model.demand_nodes, model.T, 
                                                      domain=pyo.NonNegativeReals)
        
        # Deficit variables for minimum flow requirements
        model.deficit = pyo.Var(model.sink_nodes, model.T, 
                              domain=pyo.NonNegativeReals)
        
        # Spill variables for reservoirs and hydroworks
        model.reservoir_spill = pyo.Var(model.storage_nodes, model.T, 
                                      domain=pyo.NonNegativeReals)
        model.hydroworks_spill = pyo.Var(model.hydroworks_nodes, model.T, 
                                       domain=pyo.NonNegativeReals)
    
    def _create_constraints(self):
        """
        Create constraints for the optimization model.
        
        Constraints include:
        - Mass balance at nodes
        - Edge capacity limits
        - Storage capacity limits
        - Minimum flow requirements
        - Demand satisfaction limits
        - Edge loss relationships
        """
        model = self.model
        
        # 1. a) Edge Capacity Constraints
        def edge_capacity_rule(model, source, target, t):
            edge_data = next(e for e in self.edges if e['source'] == source and e['target'] == target)
            return model.flow_before_losses[source, target, t] <= edge_data['capacity']
        
        model.edge_capacity_constraint = pyo.Constraint(
            model.edges, model.T, rule=edge_capacity_rule
        )
        # 1. b) Edge flow positivity constraint
        def flow_positivity_rule(model, source, target, t):
            return model.flow_before_losses[source, target, t] >= 0
        
        model.flow_positivity_constraint = pyo.Constraint(
            model.edges, model.T, rule=flow_positivity_rule
        )

        # 2. Edge Loss Constraints
        def edge_loss_rule(model, source, target, t):
            edge_data = next(e for e in self.edges if e['source'] == source and e['target'] == target)
            loss_factor = edge_data['loss_factor']
            length = edge_data['length']
            total_loss_fraction = 1 - (1 - loss_factor)**length
            
            # Limit total loss fraction to maximum of 1 (100%)
            total_loss_fraction = min(total_loss_fraction, 1.0)
            
            return model.flow_after_losses[source, target, t] == \
                   model.flow_before_losses[source, target, t] * (1 - total_loss_fraction)
        
        model.edge_loss_constraint = pyo.Constraint(
            model.edges, model.T, rule=edge_loss_rule
        )
        # 3. Mass Balance Constraints for SupplyNodes
        def supply_mass_balance_rule(model, node, t):
            # Get outgoing edges
            outgoing_edges = [(source, target) for source, target in model.edges if source == node]
            
            # Get supply rate from data
            supply_rate = self.nodes[node]['supply_rates'][t]
            
            # Total outflow must equal supply rate
            return sum(model.flow_before_losses[source, target, t] for source, target in outgoing_edges) == supply_rate
        
        model.supply_mass_balance_constraint = pyo.Constraint(
            model.supply_nodes, model.T, rule=supply_mass_balance_rule
        )
        # 4. Mass Balance Constraints for RunoffNodes
        def runoff_mass_balance_rule(model, node, t):
            # Get outgoing edges
            outgoing_edges = [(source, target) for source, target in model.edges if source == node]
            
            # Calculate runoff from rainfall data
            rainfall = self.nodes[node]['rainfall_data'][t]
            area = self.nodes[node]['area']
            runoff_coefficient = self.nodes[node]['runoff_coefficient']
            
            # Convert rainfall from mm to m³/s
            runoff_rate = (rainfall / 1000) * area * 1e6 * runoff_coefficient / self.dt
            
            # Total outflow must equal runoff rate
            return sum(model.flow_before_losses[source, target, t] for source, target in outgoing_edges) <= runoff_rate
        
        model.runoff_mass_balance_constraint = pyo.Constraint(
            model.runoff_nodes, model.T, rule=runoff_mass_balance_rule
        )

        # 5. Mass Balance Constraints for StorageNodes
        def storage_mass_balance_rule(model, node, t):
            # Get incoming and outgoing edges
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            outgoing_edges = [(source, target) for source, target in model.edges if source == node]
            
            # Calculate total inflow after losses
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            
            
            '''# Calculate evaporation losses
            evaporation_rate = self.nodes[node]['evaporation_rates'][t]
            # Simplified evaporation calculation - in a full model, this would depend on surface area
            # which depends on storage volume, making it nonlinear
            estimated_evaporation = evaporation_rate / 1000 * 1e6  # Convert mm to m³'''
            
            # Storage balance equation
            if t == 0:
                previous_storage = self.nodes[node]['initial_storage']
            else:
                previous_storage = model.storage[node, t-1]
            

            return model.storage[node, t] == previous_storage + \
                   (total_inflow - model.release[node,t]) * self.dt - \
                   model.reservoir_spill[node, t]
        
        model.storage_mass_balance_constraint = pyo.Constraint(
            model.storage_nodes, model.T, rule=storage_mass_balance_rule
        )

        # 6. Storage Capacity Constraints
        def storage_capacity_rule(model, node, t):
            capacity = self.nodes[node]['capacity']
            return model.storage[node, t] <= capacity
        
        model.storage_capacity_constraint = pyo.Constraint(
            model.storage_nodes, model.T, rule=storage_capacity_rule
        )
        
        # 7. Dead Storage Constraints
        def dead_storage_rule(model, node, t):
            dead_storage = self.nodes[node]['dead_storage']
            return model.storage[node, t] >= dead_storage
        
        model.dead_storage_constraint = pyo.Constraint(
            model.storage_nodes, model.T, rule=dead_storage_rule
        )

        # 8. Mass Balance Constraints for DemandNodes
        # (a) Consumptive demand can't exceed total inflow
        def demand_consumptive_constraint_rule(model, node, t):
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            return model.supplied_consumptive_demand[node, t] <= total_inflow

        model.demand_consumptive_constraint = pyo.Constraint(
            model.demand_nodes, model.T, rule=demand_consumptive_constraint_rule
        )

        # (b) Total demand satisfaction (consumptive + non-consumptive) can't exceed total inflow
        def demand_total_constraint_rule(model, node, t):
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            return (model.supplied_consumptive_demand[node, t] +
                    model.supplied_non_consumptive_demand[node, t]) <= total_inflow

        model.demand_total_constraint = pyo.Constraint(
            model.demand_nodes, model.T, rule=demand_total_constraint_rule
        )

        # (c) Outflow equals inflow minus consumptive use
        def demand_outflow_constraint_rule(model, node, t):
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            outgoing_edges = [(source, target) for source, target in model.edges if source == node]
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            return sum(model.flow_before_losses[source, target, t] for source, target in outgoing_edges) == (
                total_inflow - model.supplied_consumptive_demand[node, t]
            )

        model.demand_outflow_constraint = pyo.Constraint(
            model.demand_nodes, model.T, rule=demand_outflow_constraint_rule
        )

        # 9. Demand Satisfaction Limits
        # (a) Consumptive demand limit
        def demand_consumptive_limit_rule(model, node, t):
            demand_rate = self.nodes[node]['demand_rates'][t]
            non_consumptive_rate = self.nodes[node]['non_consumptive_rate']
            consumptive_rate = demand_rate - non_consumptive_rate
            return model.supplied_consumptive_demand[node, t] <= consumptive_rate

        model.demand_consumptive_limit_constraint = pyo.Constraint(
            model.demand_nodes, model.T, rule=demand_consumptive_limit_rule
        )

        # (b) Non-consumptive demand limit
        def demand_non_consumptive_limit_rule(model, node, t):
            non_consumptive_rate = self.nodes[node]['non_consumptive_rate']
            return model.supplied_non_consumptive_demand[node, t] <= non_consumptive_rate

        model.demand_non_consumptive_limit_constraint = pyo.Constraint(
            model.demand_nodes, model.T, rule=demand_non_consumptive_limit_rule
        )

        # 10. Mass Balance Constraints for SinkNodes
        # (a) Deficit must be non-negative
        def sink_deficit_nonneg_rule(model, node, t):
            return model.deficit[node, t] >= 0

        model.sink_deficit_nonneg_constraint = pyo.Constraint(
            model.sink_nodes, model.T, rule=sink_deficit_nonneg_rule
        )

        # (b) Deficit must be at least the shortfall
        def sink_deficit_shortfall_rule(model, node, t):
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            min_flow = self.nodes[node]['min_flows'][t]
            return model.deficit[node, t] >= min_flow - total_inflow

        model.sink_deficit_shortfall_constraint = pyo.Constraint(
            model.sink_nodes, model.T, rule=sink_deficit_shortfall_rule
        )
        
        # 11. Mass Balance Constraints for HydroWorks
        def hydroworks_mass_balance_rule(model, node, t):
            # Get incoming and outgoing edges
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            outgoing_edges = [(source, target) for source, target in model.edges if source == node]
            
            # Calculate total inflow after losses
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            
            # Total outflow plus spill must equal total inflow
            return sum(model.flow_before_losses[source, target, t] 
                      for source, target in outgoing_edges) + \
                   model.hydroworks_spill[node, t] / self.dt == total_inflow
        
        model.hydroworks_mass_balance_constraint = pyo.Constraint(
            model.hydroworks_nodes, model.T, rule=hydroworks_mass_balance_rule
        )
        
        # 12. Hydroworks Distribution Constraints (if parameters exist)
        def hydroworks_distribution_rule(model, node, t):
            # Skip if node doesn't have distribution parameters
            if 'distribution_params' not in self.nodes[node]:
                return pyo.Constraint.Skip
            
            # Get distribution parameters
            dist_params = self.nodes[node]['distribution_params']
            
            # Get outgoing edges
            outgoing_edges = [(source, target) for source, target in model.edges if source == node]
            
            # Get incoming edges
            incoming_edges = [(source, target) for source, target in model.edges if target == node]
            
            # Calculate total inflow
            total_inflow = sum(model.flow_after_losses[source, target, t] for source, target in incoming_edges)
            
            # Create constraints list
            constraints = []
            
            # Current month for parameter selection
            month = t % 12
            
            for target_id in dist_params:
                # Find the edge to this target
                edge = next((e for e in outgoing_edges if e[1] == target_id), None)
                
                if edge:
                    # Calculate the allowed flow based on distribution parameters
                    param = dist_params[target_id][month]
                    target_flow = total_inflow * param
                    
                    # Get edge capacity
                    edge_data = next(e for e in self.edges if e['source'] == edge[0] and e['target'] == edge[1])
                    capacity = edge_data['capacity']
                    
                    # Flow should be less than or equal to target flow or capacity
                    constraints.append(model.flow_before_losses[edge] <= min(target_flow, capacity))
            
            return constraints
        
        model.hydroworks_distribution_constraint = pyo.Constraint(
            model.hydroworks_nodes, model.T, rule=hydroworks_distribution_rule
        )
    
    def _create_objective(self):
        """
        Create the objective function for the optimization model.
        
        The objective maximizes a weighted sum of:
        - Demand satisfaction (positive contribution)
        - Minimum flow satisfaction (positive contribution)
        - Storage levels to desired targets (positive contribution)
        
        while minimizing:
        - Flow deficits at sink nodes (negative contribution)
        - Deviations from target storage levels (negative contribution)
        """
        model = self.model
        
        def objective_rule(model):
            # 1. Demand satisfaction benefit (weighted by priority)
            demand_benefit = sum(
                self.nodes[node]['weight'] * 
                (model.supplied_consumptive_demand[node, t] + model.supplied_non_consumptive_demand[node, t]) / 
                self.nodes[node]['demand_rates'][t] if self.nodes[node]['demand_rates'][t] > 0 else 0
                for node in model.demand_nodes
                for t in model.T
            )
            
            # 2. Minimum flow satisfaction penalty (weighted by priority)
            flow_deficit_penalty = sum(
                self.nodes[node]['weight'] * model.deficit[node, t]
                for node in model.sink_nodes
                for t in model.T
            )
            
            # 3. Storage benefit (reward higher storage levels)
            storage_benefit = sum(
                0.05 * model.storage[node, t] / self.nodes[node]['capacity']
                for node in model.storage_nodes
                for t in model.T
            )
            
            # 4. Spill penalties
            spill_penalty = sum(
                10 * model.reservoir_spill[node, t]
                for node in model.storage_nodes
                for t in model.T
            ) + sum(
                10 * model.hydroworks_spill[node, t]
                for node in model.hydroworks_nodes
                for t in model.T
            )
            
            # Combined objective (maximize benefits, minimize penalties)
            return demand_benefit + storage_benefit - flow_deficit_penalty - spill_penalty
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    def solve(self, solver='glpk', options=None):
        """
        Solve the optimization model using the specified solver.
        
        Args:
            solver (str): Name of the solver to use ('glpk', 'cbc', 'gurobi', etc.)
            options (dict): Solver-specific options
            
        Returns:
            dict: Solution status and objective value
        """
        # Initialize solver
        opt = SolverFactory(solver)
        
        # Apply solver options if provided
        if options:
            for key, value in options.items():
                opt.options[key] = value
        
        # Solve the model
        results = opt.solve(self.model, tee=True)
        
        # Check solution status
        if results.solver.status == pyo.SolverStatus.ok and \
           results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Optimal solution found.")
            return {
                'status': 'optimal',
                'objective_value': pyo.value(self.model.objective)
            }
        elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            print("Model is infeasible.")
            return {
                'status': 'infeasible',
                'objective_value': None
            }
        else:
            print(f"Solution status: {results.solver.status}, Termination condition: {results.solver.termination_condition}")
            return {
                'status': str(results.solver.status),
                'termination_condition': str(results.solver.termination_condition),
                'objective_value': pyo.value(self.model.objective) if hasattr(self.model, 'objective') else None
            }
    
    def get_results(self):
        """
        Extract and organize the optimization results.
        
        Returns:
            dict: Dictionary containing the optimized values for all variables
        """
        model = self.model
        results = {}
        
        # Extract edge flows
        results['flows'] = {
            'before_losses': {
                (source, target, t): pyo.value(model.flow_before_losses[source, target, t])
                for source, target in model.edges
                for t in model.T
            },
            'after_losses': {
                (source, target, t): pyo.value(model.flow_after_losses[source, target, t])
                for source, target in model.edges
                for t in model.T
            }
        }
        
        # Extract storage values
        results['storage'] = {
            (node, t): pyo.value(model.storage[node, t])
            for node in model.storage_nodes
            for t in model.T
        }
        results['release'] = {
            (node, t): pyo.value(model.release[node, t])
            for node in model.storage_nodes
            for t in model.T
        }
        
        # Extract demand satisfaction
        results['demand'] = {
            'consumptive': {
                (node, t): pyo.value(model.supplied_consumptive_demand[node, t])
                for node in model.demand_nodes
                for t in model.T
            },
            'non_consumptive': {
                (node, t): pyo.value(model.supplied_non_consumptive_demand[node, t])
                for node in model.demand_nodes
                for t in model.T
            }
        }
        
        
        # Extract minimum flow deficits
        results['deficits'] = {
            (node, t): pyo.value(model.deficit[node, t])
            for node in model.sink_nodes
            for t in model.T
        }
        
        # Extract spills
        results['spills'] = {
            'reservoir': {
                (node, t): pyo.value(model.reservoir_spill[node, t])
                for node in model.storage_nodes
                for t in model.T
            },
            'hydroworks': {
                (node, t): pyo.value(model.hydroworks_spill[node, t])
                for node in model.hydroworks_nodes
                for t in model.T
            }
        }        
        return results
    
    def compare_with_simulation(self):
        """
        Compare optimization results with simulation-based approach.
        
        This method creates and runs the original simulation model,
        then compares key performance metrics with the optimization results.
        
        Returns:
            dict: Comparison metrics between optimization and simulation
        """
        
        sim_system = create_ZRB_system(
            start_year=self.start_year,
            start_month=self.start_month,
            num_time_steps=self.num_time_steps,
            system_type=self.system_type,
            scenario=self.scenario,
            period=self.period,
            agr_scenario=self.agr_scenario,
            efficiency=self.efficiency
        )
        loaded_results = load_parameters_from_file(f"./data/simplified_ZRB/parameter/euler_singleobjective_params_simplified_ZRB_100_3000_0.65_0.32.json")
        sim_system = load_optimized_parameters(sim_system, loaded_results)
        # Run simulation
        sim_system.simulate(self.num_time_steps)
        
        # Get optimization results
        opt_results = self.get_results()
        
        # Initialize comparison metrics
        comparison = {
            'demand_satisfaction': {
                'optimization': 0,
                'simulation': 0
            },
            'min_flow_deficits': {
                'optimization': 0,
                'simulation': 0
            },
            'storage_utilization': {
                'optimization': 0,
                'simulation': 0
            }
        }
        
        # Compare demand satisfaction
        opt_demand_satisfaction = 0
        total_demand = 0
        
        for node in self.model.demand_nodes:
            for t in self.model.T:
                node_demand = self.nodes[node]['demand_rates'][t]
                total_demand += node_demand
                
                opt_supplied = (opt_results['demand']['consumptive'].get((node, t), 0) + 
                               opt_results['demand']['non_consumptive'].get((node, t), 0))
                opt_demand_satisfaction += opt_supplied
        
        # Get simulation demand satisfaction
        sim_demand_nodes = [n for n, data in sim_system.graph.nodes(data=True) 
                          if isinstance(data['node'], DemandNode)]
        
        sim_demand_satisfaction = 0
        for node_id in sim_demand_nodes:
            node = sim_system.graph.nodes[node_id]['node']
            sim_demand_satisfaction += sum(
                node.satisfied_consumptive_demand[t] + node.satisfied_non_consumptive_demand[t]
                for t in range(self.num_time_steps)
            )
        
        # Calculate percentages
        if total_demand > 0:
            comparison['demand_satisfaction']['optimization'] = opt_demand_satisfaction / total_demand * 100
            comparison['demand_satisfaction']['simulation'] = sim_demand_satisfaction / total_demand * 100
        
        # Compare minimum flow deficits
        opt_deficits = sum(opt_results['deficits'].values())
        
        sim_sink_nodes = [n for n, data in sim_system.graph.nodes(data=True) 
                        if isinstance(data['node'], SinkNode)]
        
        sim_deficits = 0
        for node_id in sim_sink_nodes:
            node = sim_system.graph.nodes[node_id]['node']
            sim_deficits += sum(node.flow_deficits)
        
        comparison['min_flow_deficits']['optimization'] = opt_deficits
        comparison['min_flow_deficits']['simulation'] = sim_deficits
        
        # Compare storage utilization
        opt_storage = 0
        total_capacity = 0
        
        for node in self.model.storage_nodes:
            capacity = self.nodes[node]['capacity']
            total_capacity += capacity * self.num_time_steps
            
            for t in self.model.T:
                opt_storage += opt_results['storage'].get((node, t), 0)
        
        sim_storage_nodes = [n for n, data in sim_system.graph.nodes(data=True) 
                           if isinstance(data['node'], StorageNode)]
        
        sim_storage = 0
        for node_id in sim_storage_nodes:
            node = sim_system.graph.nodes[node_id]['node']
            sim_storage += sum(node.storage[1:])  # Skip initial storage
        
        if total_capacity > 0:
            comparison['storage_utilization']['optimization'] = opt_storage / total_capacity * 100
            comparison['storage_utilization']['simulation'] = sim_storage / total_capacity * 100
        
        return comparison
    
    def print_optimization_water_balance(self):
        """
        Print a summary of the water balance based on optimization results only.
        This does not rely on any simulation or external scripts.
        """
        results = self.get_results()
        num_years = self.num_time_steps / 12 if self.num_time_steps >= 12 else 1

        print("\nOptimization Water Balance Summary")
        print("=" * 50)

        # Source volumes
        print("\nSource Volumes")
        print("-" * 20)
        total_supply = 0
        for node in self.model.supply_nodes:
            supply = sum(self.nodes[node]['supply_rates'][t] for t in self.model.T)*self.dt
            total_supply += supply
            print(f"{node:20s}: {supply/num_years:15,.0f} m³/a")
        total_runoff = 0
        for node in self.model.runoff_nodes:
            runoff = sum(self.nodes[node]['rainfall_data'][t] / 1000 * self.nodes[node]['area'] * 1e6 * self.nodes[node]['runoff_coefficient'] / self.dt for t in self.model.T)*self.dt
            total_runoff += runoff
            print(f"{node:20s}: {runoff/num_years:15,.0f} m³/a")
        total_source = total_supply + total_runoff
        print(f"{'Total Source':20s}: {total_source/num_years:15,.0f} m³/a (100%)")

        # Demand satisfaction
        print("\nDemand Satisfaction")
        print("-" * 20)
        total_demand = 0
        total_satisfied = 0
        total_unmet = 0
        for node in self.model.demand_nodes:
            for t in self.model.T:
                demand = self.nodes[node]['demand_rates'][t]
                cons = results['demand']['consumptive'].get((node, t), 0)
                non_cons = results['demand']['non_consumptive'].get((node, t), 0)
                satisfied = cons + non_cons
                total_demand += demand*self.dt
                total_satisfied += satisfied*self.dt
                total_unmet += max(0, demand - satisfied)*self.dt
        print(f"Total demand:         {total_demand/num_years:15,.0f} m³/a")
        print(f"Satisfied demand:     {total_satisfied/num_years:15,.0f} m³/a")
        print(f"Unmet demand:         {total_unmet/num_years:15,.0f} m³/a")

        # Sink outflows and deficits
        print("\nSink Nodes")
        print("-" * 20)
        total_sink_outflow = 0
        total_sink_deficit = 0
        for node in self.model.sink_nodes:
            node_outflow = 0
            node_deficit = 0
            for t in self.model.T:
                node_deficit += results['deficits'].get((node, t), 0)*self.dt
                # Outflow is inflow minus deficit
                incoming_edges = [(source, target) for source, target in self.model.edges if target == node]
                inflow = sum(results['flows']['after_losses'].get((source, target, t), 0) for source, target in incoming_edges)
                node_outflow += max(0, inflow - results['deficits'].get((node, t), 0))*self.dt
            total_sink_outflow += node_outflow
            total_sink_deficit += node_deficit
            print(f"{node:20s} Outflow: {node_outflow/num_years:15,.0f} m³/a, Deficit: {node_deficit/num_years:15,.0f} m³/a")

        # Storage changes
        print("\nStorage Nodes")
        print("-" * 20)
        total_storage_change = 0
        for node in self.model.storage_nodes:
            initial = results['storage'].get((node, 0), 0)
            final = results['storage'].get((node, self.num_time_steps-1), 0)
            release = sum(results['release'].get((node, t), 0) for t in self.model.T)
            change = final - initial
            total_storage_change += change
            total_release = release * self.dt
            print(f"{node:20s} ΔStorage: {change:15,.0f} m³")
            print(f"{node:20s} Release: {total_release/num_years:15,.0f} m³/a")

        # Spills
        print("\nSpills")
        print("-" * 20)
        total_reservoir_spill = 0
        for node in self.model.storage_nodes:
            node_spill = sum(results['spills']['reservoir'].get((node, t), 0) for t in self.model.T)
            total_reservoir_spill += node_spill*self.dt
            print(f"{node:20s} Reservoir Spill: {node_spill/num_years:15,.0f} m³/a")
        total_hw_spill = 0
        for node in self.model.hydroworks_nodes:
            node_spill = sum(results['spills']['hydroworks'].get((node, t), 0) for t in self.model.T)
            total_hw_spill += node_spill*self.dt
            print(f"{node:20s} HydroWorks Spill: {node_spill/num_years:15,.0f} m³/a")

        # Water balance check
        print("\nWater Balance Check")
        print("-" * 20)
        total_out = total_satisfied + total_sink_outflow + total_reservoir_spill + total_hw_spill
        print(f"Total In:   {total_source/num_years:15,.0f} m³/a")
        print(f"Total Out:  {total_out/num_years:15,.0f} m³/a")
        print(f"ΔStorage:   {total_storage_change/num_years:15,.0f} m³/a")
        balance = total_source - total_out - total_storage_change
        print(f"Balance residual: {balance/num_years:15,.0f} m³/a")
        print("=" * 50)

    def visualize_results(self):
        """
        Visualize the optimization results using matplotlib.
        
        Creates plots for:
        - Demand satisfaction vs. time
        - Storage volumes vs. time
        - Flow deficits vs. time
        - Comparison with simulation results
        
        Returns:
            dict: Dictionary of matplotlib figure objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Get optimization results
        results = self.get_results()
        
        # Initialize plot containers
        plots = {}
        
        # Plot demand satisfaction
        fig1 = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        ax1 = fig1.add_subplot(gs[0])
        ax2 = fig1.add_subplot(gs[1])
        
        # Group demand satisfaction by time step
        time_steps = list(range(self.num_time_steps))
        
        # Prepare data for demand nodes
        for node in self.model.demand_nodes:
            demand_vals = [self.nodes[node]['demand_rates'][t] for t in time_steps]
            consumptive_vals = [results['demand']['consumptive'].get((node, t), 0) for t in time_steps]
            non_consumptive_vals = [results['demand']['non_consumptive'].get((node, t), 0) for t in time_steps]
            
            # Calculate deficit
            deficit_vals = [max(0, demand - cons - non_cons) for demand, cons, non_cons 
                          in zip(demand_vals, consumptive_vals, non_consumptive_vals)]
            
            # Plot demands and satisfaction
            ax1.plot(time_steps, demand_vals, '--', label=f"{node} Demand")
            ax1.plot(time_steps, [c + nc for c, nc in zip(consumptive_vals, non_consumptive_vals)], 
                    '-', label=f"{node} Satisfied")
            
            # Plot deficit
            ax2.plot(time_steps, deficit_vals, '-', label=f"{node} Deficit")
        
        ax1.set_title('Demand Satisfaction Over Time')
        ax1.set_ylabel('Flow Rate (m³/s)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Demand Deficit Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Deficit (m³/s)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plots['demand'] = fig1
        
        # Plot storage volumes
        fig2 = plt.figure(figsize=(12, 6))
        ax = fig2.add_subplot(111)
        
        for node in self.model.storage_nodes:
            capacity = self.nodes[node]['capacity']
            dead_storage = self.nodes[node]['dead_storage']
            
            # Get storage values over time
            storage_vals = [results['storage'].get((node, t), 0) for t in time_steps]
            
            # Plot storage, capacity, and dead storage
            ax.plot(time_steps, storage_vals, '-', label=f"{node} Storage")
            ax.axhline(y=capacity, linestyle='--', color='gray', label=f"{node} Capacity")
            ax.axhline(y=dead_storage, linestyle=':', color='red', label=f"{node} Dead Storage")
        
        ax.set_title('Reservoir Storage Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Storage Volume (m³)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plots['storage'] = fig2
        
        # Plot minimum flow deficits
        fig3 = plt.figure(figsize=(12, 6))
        ax = fig3.add_subplot(111)
        
        for node in self.model.sink_nodes:
            # Get minimum flow requirements
            min_flows = [self.nodes[node]['min_flows'][t] for t in time_steps]
            
            # Get deficit values
            deficit_vals = [results['deficits'].get((node, t), 0) for t in time_steps]
            
            # Plot minimum flow and deficit
            ax.plot(time_steps, min_flows, '--', label=f"{node} Min Flow")
            ax.plot(time_steps, deficit_vals, '-', label=f"{node} Deficit")
        
        ax.set_title('Minimum Flow Deficits Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Flow Rate (m³/s)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plots['deficits'] = fig3
        
        # Compare with simulation if available
        try:
            comparison = self.compare_with_simulation()
            
            fig4 = plt.figure(figsize=(10, 8))
            gs = GridSpec(3, 1)
            
            ax1 = fig4.add_subplot(gs[0])
            ax2 = fig4.add_subplot(gs[1])
            ax3 = fig4.add_subplot(gs[2])
            
            # Plot demand satisfaction comparison
            labels = ['Optimization', 'Simulation']
            demand_vals = [comparison['demand_satisfaction']['optimization'],
                        comparison['demand_satisfaction']['simulation']]
            
            ax1.bar(labels, demand_vals)
            ax1.set_title('Demand Satisfaction (%)')
            ax1.set_ylim(0, 100)
            for i, v in enumerate(demand_vals):
                ax1.text(i, v + 2, f"{v:.1f}%", ha='center')
            
            # Plot deficit comparison
            deficit_vals = [comparison['min_flow_deficits']['optimization'],
                          comparison['min_flow_deficits']['simulation']]
            
            ax2.bar(labels, deficit_vals)
            ax2.set_title('Total Minimum Flow Deficits (m³/s)')
            for i, v in enumerate(deficit_vals):
                ax2.text(i, v + 2, f"{v:.1f}", ha='center')
            
            # Plot storage utilization comparison
            storage_vals = [comparison['storage_utilization']['optimization'],
                          comparison['storage_utilization']['simulation']]
            
            ax3.bar(labels, storage_vals)
            ax3.set_title('Storage Utilization (%)')
            ax3.set_ylim(0, 100)
            for i, v in enumerate(storage_vals):
                ax3.text(i, v + 2, f"{v:.1f}%", ha='center')
            
            plt.tight_layout()
            plots['comparison'] = fig4
            
        except Exception as e:
            print(f"Error creating comparison plot: {str(e)}")
        
        import seaborn as sns

        # Demand deficit heatmap (absolute)
        try:
            demand_nodes = list(self.model.demand_nodes)
            time_steps = list(range(self.num_time_steps))
            deficit_data = pd.DataFrame(index=demand_nodes, columns=time_steps)

            for node in demand_nodes:
                for t in time_steps:
                    demand = self.nodes[node]['demand_rates'][t]
                    cons = results['demand']['consumptive'].get((node, t), 0)
                    non_cons = results['demand']['non_consumptive'].get((node, t), 0)
                    deficit = max(0, demand - cons - non_cons)
                    deficit_data.at[node, t] = deficit

            plt.figure(figsize=(16, 6))
            sns.heatmap(deficit_data.astype(float), cmap='YlOrRd', annot=False, cbar_kws={'label': 'Deficit [m³/s]'})
            plt.title('Demand Deficit Heatmap (Absolute)')
            plt.xlabel('Time Step')
            plt.ylabel('Demand Node')
            plt.tight_layout()
            fig_heatmap = plt.gcf()
            plots['deficit_heatmap'] = fig_heatmap
        except Exception as e:
            print(f"Error creating demand deficit heatmap: {str(e)}")

        try:
            pct_data = deficit_data.copy()
            for node in demand_nodes:
                for t in time_steps:
                    demand = self.nodes[node]['demand_rates'][t]
                    pct_data.at[node, t] = (deficit_data.at[node, t] / demand * 100) if demand > 0 else 0

            plt.figure(figsize=(16, 6))
            sns.heatmap(pct_data.astype(float), cmap='YlOrRd', annot=False, cbar_kws={'label': 'Deficit [%]'})
            plt.title('Demand Deficit Heatmap (Percentage)')
            plt.xlabel('Time Step')
            plt.ylabel('Demand Node')
            plt.tight_layout()
            fig_pct_heatmap = plt.gcf()
            plots['deficit_heatmap_pct'] = fig_pct_heatmap
        except Exception as e:
            print(f"Error creating percentage deficit heatmap: {str(e)}")

        try: 
            self.print_optimization_water_balance()
        except Exception as e:
            print(f"Error printing water balance: {str(e)}")

        return plots

def create_and_solve_ZRB_optimization(start_year: int, start_month: int, num_time_steps: int, 
                                     system_type: str = "baseline", scenario: str = '',
                                     period: str = '', agr_scenario: str = '', 
                                     efficiency: str = '', solver: str = 'glpk'):
    """
    Create and solve an optimization model for the ZRB water system.
    
    Args:
        start_year (int): Start year for optimization
        start_month (int): Start month for optimization (1-12)
        num_time_steps (int): Number of time steps to optimize
        system_type (str): "baseline" or "scenario"
        scenario (str): Climate scenario (e.g., 'ssp126')
        period (str): Time period (e.g., '2041-2070')
        agr_scenario (str): Agricultural scenario
        efficiency (str): Efficiency scenario (e.g., 'improved_efficiency')
        solver (str): Solver to use ('glpk', 'cbc', 'gurobi', etc.)
        
    Returns:
        tuple: Optimization model and results
    """
    print('a')
    # Create the optimization model
    model = ZRBOptimizationModel(
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        system_type=system_type,
        scenario=scenario,
        period=period,
        agr_scenario=agr_scenario,
        efficiency=efficiency
    )
    
    # Solve the model
    solution = model.solve(solver=solver)
    
    # Get and return results
    if solution['status'] == 'optimal':
        results = model.get_results()
        return model, results
    else:
        print(f"Failed to find optimal solution. Status: {solution['status']}")
        return model, None


# Example usage for running the optimization
if __name__ == "__main__":
    # Example parameters
    START_YEAR = 2017
    START_MONTH = 1
    NUM_TIME_STEPS = 12  # Run for one year
    SYSTEM_TYPE = "simplified_ZRB"
    
    # Create and solve the optimization model
    opt_model, results = create_and_solve_ZRB_optimization(
        start_year=START_YEAR,
        start_month=START_MONTH,
        num_time_steps=NUM_TIME_STEPS,
        system_type=SYSTEM_TYPE,
        solver='glpk'  # Change to other solvers as needed
    )
    
    # Compare with simulation
    if results:
        comparison = opt_model.compare_with_simulation()
        print("\nComparison with Simulation:")
        print(f"Demand Satisfaction: Optimization {comparison['demand_satisfaction']['optimization']:.2f}%, "
              f"Simulation {comparison['demand_satisfaction']['simulation']:.2f}%")
        print(f"Min Flow Deficits: Optimization {comparison['min_flow_deficits']['optimization']:.2f}, "
              f"Simulation {comparison['min_flow_deficits']['simulation']:.2f}")
        print(f"Storage Utilization: Optimization {comparison['storage_utilization']['optimization']:.2f}%, "
              f"Simulation {comparison['storage_utilization']['simulation']:.2f}%")
        
        # Visualize results
        plots = opt_model.visualize_results()
        for name, fig in plots.items():
            fig.savefig(f"ZRB_optimization_{name}.png")
            print(f"Saved plot to ZRB_optimization_{name}.png")