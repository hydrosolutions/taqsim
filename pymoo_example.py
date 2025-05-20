from typing import Union, Dict, List
import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from water_system import (WaterSystem, SupplyNode, StorageNode, DemandNode, 
                          SinkNode, HydroWorks, RunoffNode, Edge, 
                          PymooSingleObjectiveOptimizer, PymooMultiObjectiveOptimizer, 
                          ParetoFrontDashboard, ParetoFrontDashboard4D)
# Import from your original script to get the system creator
from ZRB_system_creator import create_ZRB_system

def run_pymoo_optimization(
    system_creator,
    start_year: int = 2017,
    start_month: int = 1,
    num_time_steps: int = 12,
    system_type: str = 'baseline',
    scenario: str = '',
    period: str = '',
    agr_scenario: str = '',
    efficiency: str = '',
    n_gen: int = 50,
    pop_size: int = 50,
    crossover_prob: float = 0.65,
    mutation_prob: float = 0.32
) -> Dict[str, Union[Dict, List]]:
    """
    Run single-objective optimization for the ZRB water system using pymoo.
    
    Args:
        system_creator: Function to create the water system
        start_year: Start year for simulation
        start_month: Start month for simulation (1-12)
        num_time_steps: Number of time steps to simulate
        system_type: "baseline" or "scenario"
        scenario: Climate scenario (e.g., 'ssp126')
        period: Time period (e.g., '2041-2070')
        agr_scenario: Agricultural scenario
        efficiency: Efficiency scenario
        n_gen: Number of generations
        pop_size: Population size
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        
    Returns:
        dict: Results of the optimization
    """
    print(f"Creating {system_type} water system...")
    ZRB_system = system_creator(
        start_year, start_month, num_time_steps,
        system_type, scenario, period, agr_scenario, efficiency
    )
    
    print(f"Initializing pymoo optimizer with pop_size={pop_size}, n_gen={n_gen}...")
    optimizer = PymooSingleObjectiveOptimizer(
        base_system=ZRB_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        n_gen=n_gen,
        pop_size=pop_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob
    )
    
    print("Starting optimization process...")
    start_time = time.time()
    results = optimizer.optimize()
    end_time = time.time()
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print("-" * 50)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Crossover probability: {results['crossover_probability']}")
    print(f"Mutation probability: {results['mutation_probability']}")
    print(f"Final objective value: {results['objective_value']:,.0f} m³")
    
    print("\nOptimal Reservoir Parameters:")
    for res_id, params in results['optimal_reservoir_parameters'].items():
        print(f"\n{res_id}:")
        for param, values in params.items():
            print(f"{param}: [", end="")
            print(", ".join(f"{v:.3f}" for v in values), end="")
            print("]")
        
    print("\nOptimal Hydroworks Parameters:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for hw_id, params in results['optimal_hydroworks_parameters'].items():
        print(f"\n{hw_id}:")
        for target, values in params.items():
            print(f"{target}: [", end="")
            print(", ".join(f"{v:.3f}" for v in values), end="")
            print("]")
    
    # Save optimization results
    output_dir = f"./model_output/optimisation/pymoo/{system_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    if system_type == 'scenario':
        filename = f"{output_dir}/pymoo_params_{scenario}_{period}_{agr_scenario}_{efficiency}_{n_gen}_{pop_size}.json"
    elif system_type == 'baseline':
        filename = f"{output_dir}/pymoo_params_{n_gen}_{pop_size}.json"
    else:
        filename = f"{output_dir}/pymoo_params_{system_type}_{n_gen}_{pop_size}.json"
    
    save_optimized_parameters(results, filename)
    # Generate convergence plot
    optimizer.plot_convergence()

    print(f"Optimization results saved to {filename}")
    
    return results


def run_pymoo_multi_objective(
    system_creator,
    start_year: int = 2017,
    start_month: int = 1,
    num_time_steps: int = 12,
    system_type: str = 'baseline',
    scenario: str = '',
    period: str = '',
    agr_scenario: str = '',
    efficiency: str = '',
    n_gen: int = 50,
    pop_size: int = 100,
    crossover_prob: float = 0.65,
    mutation_prob: float = 0.32,
    num_objectives: int = 2
) -> Dict[str, Union[Dict, List]]:
    """
    Run multi-objective optimization for the ZRB water system using pymoo.
    
    Args:
        system_creator: Function to create the water system
        start_year: Start year for simulation
        start_month: Start month for simulation (1-12)
        num_time_steps: Number of time steps to simulate
        system_type: "baseline" or "scenario"
        scenario: Climate scenario (e.g., 'ssp126')
        period: Time period (e.g., '2041-2070')
        agr_scenario: Agricultural scenario
        efficiency: Efficiency scenario
        n_gen: Number of generations
        pop_size: Population size
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        num_objectives: Number of objectives (2 or 3)
        
    Returns:
        dict: Results of the optimization
    """
    print(f"Creating {system_type} water system...")
    ZRB_system = system_creator(
        start_year, start_month, num_time_steps,
        system_type, scenario, period, agr_scenario, efficiency
    )
    
    print(f"Initializing pymoo multi-objective optimizer with {num_objectives} objectives...")
    optimizer = PymooMultiObjectiveOptimizer(
        base_system=ZRB_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        n_gen=n_gen,
        pop_size=pop_size,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        num_objectives=num_objectives
    )
    
    print("Starting multi-objective optimization process...")
    start_time = time.time()
    results = optimizer.optimize()
    end_time = time.time()
    
    print(f"\nMulti-objective optimization completed in {end_time - start_time:.2f} seconds")
    print("-" * 60)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Crossover probability: {results['crossover_probability']}")
    print(f"Mutation probability: {results['mutation_probability']}")
    
    print(f"\nRecommended solution objective values:")
    if len(results['objective_values']) == 2:
        print(f"  - Demand deficit: {results['objective_values'][0]:,.0f} m³")
        print(f"  - Min flow deficit: {results['objective_values'][1]:,.0f} m³")
    elif len(results['objective_values']) == 3:
        print(f"  - Regular demand deficit: {results['objective_values'][0]:,.0f} m³")
        print(f"  - Priority demand deficit: {results['objective_values'][1]:,.0f} m³")
        print(f"  - Min flow deficit: {results['objective_values'][2]:,.0f} m³")
    
    print(f"\nNumber of Pareto-optimal solutions: {len(results['pareto_front'])}")
    
    
    # Save optimization results
    output_dir = f"./model_output/optimisation/pymoo/{system_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    if system_type == 'scenario':
        filename = f"{output_dir}/pymoo_mo{num_objectives}_{scenario}_{period}_{agr_scenario}_{efficiency}_{n_gen}_{pop_size}.json"
    elif system_type == 'baseline':
        filename = f"{output_dir}/pymoo_mo{num_objectives}_{n_gen}_{pop_size}.json"
    else:
        filename = f"{output_dir}/pymoo_mo{num_objectives}_{system_type}_{n_gen}_{pop_size}.json"
    
    save_optimized_parameters(results, filename)
    # Generate convergence and Pareto front plots
    optimizer.plot_convergence()

    print(f"Multi-objective optimization results saved to {filename}")
    
    return results


def save_optimized_parameters(optimization_results, filename):
    """
    Save optimized parameters to a file for later use.
    
    Args:
        optimization_results: Results from the optimizer
        filename: Path to save the parameters
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Convert all numeric values to floats for JSON serialization
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(x) for x in obj]
        elif isinstance(obj, (int, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):  # Handle numpy arrays
            return [float(x) for x in obj]
        elif isinstance(obj, tuple):  # Handle tuples (for multi-objective values)
            return [float(x) for x in obj]
        return obj
    
    # Check if we have multi-objective results
    is_multi_objective = 'objective_values' in optimization_results
    
    # Prepare basic data for saving
    save_data = {
        'is_multi_objective': is_multi_objective,
        'population_size': optimization_results.get('population_size', 0),
        'generations': optimization_results.get('generations', 0),
        'crossover_probability': optimization_results.get('crossover_probability', 0),
        'mutation_probability': optimization_results.get('mutation_probability', 0),
    }
    
    # Add the recommended solution
    save_data['recommended_solution'] = {
        'reservoir_parameters': convert_to_float(optimization_results['optimal_reservoir_parameters']),
        'hydroworks_parameters': convert_to_float(optimization_results['optimal_hydroworks_parameters'])
    }
    
    # Add objective values based on format
    if is_multi_objective:
        objective_values = optimization_results['objective_values']
        save_data['recommended_solution']['objective_values'] = convert_to_float(objective_values)
        
        # Add the individual objective values separately
        if len(objective_values) == 2:
            save_data['recommended_solution']['demand_deficit'] = float(objective_values[0])
            save_data['recommended_solution']['minflow_deficit'] = float(objective_values[1])
        elif len(objective_values) == 3:
            save_data['recommended_solution']['demand_deficit'] = float(objective_values[0])
            save_data['recommended_solution']['priority_demand_deficit'] = float(objective_values[1])
            save_data['recommended_solution']['minflow_deficit'] = float(objective_values[2])
        
        # If pareto front exists, save all solutions
        if 'pareto_front' in optimization_results and optimization_results['pareto_front']:
            pareto_solutions = []
            
            # Process each solution in the Pareto front
            for i, sol in enumerate(optimization_results['pareto_front']):
                solution_values = sol.fitness if hasattr(sol, 'fitness') else sol.F
                
                # Create a solution entry
                solution = {
                    'id': i,
                    'objective_values': convert_to_float(solution_values)
                }
                
                # Individual objective values
                if len(solution_values) == 2:
                    solution['demand_deficit'] = float(solution_values[0])
                    solution['minflow_deficit'] = float(solution_values[1])
                elif len(solution_values) == 3:
                    solution['demand_deficit'] = float(solution_values[0])
                    solution['priority_demand_deficit'] = float(solution_values[1])
                    solution['minflow_deficit'] = float(solution_values[2])
                
                # Add to pareto solutions
                pareto_solutions.append(solution)
            
            # Add all Pareto solutions to the save data
            save_data['pareto_solutions'] = pareto_solutions
            save_data['num_pareto_solutions'] = len(pareto_solutions)
    else:
        # Single objective case
        save_data['recommended_solution']['objective_value'] = float(optimization_results.get('objective_value', 0))
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)


if __name__ == "__main__":
    
    
    '''# Example of running the pymoo single-objective optimization
    print("\n=== PYMOO SINGLE-OBJECTIVE OPTIMIZATION ===\n")
    start = datetime.now()
    
    results_so = run_pymoo_optimization(
        create_ZRB_system,
        start_year=2017, 
        start_month=1, 
        num_time_steps=12*6,  # Reduced for faster runtime in example
        system_type='simplified_ZRB',
        n_gen=20,              # Reduced for faster runtime in example
        pop_size=10,           # Reduced for faster runtime in example
        crossover_prob=0.65,
        mutation_prob=0.32
    )
    
    print(f"\nSingle-objective optimization completed in {datetime.now() - start}")'''
    
    # Example of running pymoo multi-objective optimization
    print("\n=== PYMOO MULTI-OBJECTIVE OPTIMIZATION ===\n")
    start = datetime.now()
    
    results_mo = run_pymoo_multi_objective(
        create_ZRB_system,
        start_year=2017, 
        start_month=1, 
        num_time_steps=12*2,  # Reduced for faster runtime in example
        system_type='simplified_ZRB',
        n_gen=10,              # Reduced for faster runtime in example
        pop_size=30,           # Reduced for faster runtime in example
        crossover_prob=0.65,
        mutation_prob=0.32,
        num_objectives=4       # Using 2 objectives for simplicity
    )
    
    with open("./model_output/optimisation/pymoo/simplified_ZRB/pymoo_mo4_simplified_ZRB_10_30.json", "r") as f:
        data = json.load(f)
    
    pareto_solutions = data.get('pareto_solutions', [])
    
    # Create dashboard
    dashboard = ParetoFrontDashboard4D(
        pareto_solutions=pareto_solutions,
        output_dir="./model_output/dashboard/baseline"
    )
    
    # Generate all visualizations
    dashboard.generate_full_report()
    
    print(f"Dashboard created at {dashboard.output_dir}/index.html")

    print(f"\nMulti-objective optimization completed in {datetime.now() - start}")
