from typing import Union, Dict, List
import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from water_system import (PymooSingleObjectiveOptimizer, PymooMultiObjectiveOptimizer, 
                          ParetoFrontDashboard3D, ParetoFrontDashboard4D)
# Import from your original script to get the system creator
from system_creator_ZRB import create_simplified_ZRB_system
from system_creator_simple import create_simple_system

def run_pymoo_optimization(
    system_creator,
    start_year: int,
    start_month: int,
    num_time_steps: int,
    n_gen: int,
    pop_size: int,
    num_objectives: int
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
        num_objectives: Number of objectives (2 or 3)
        
    Returns:
        dict: Results of the optimization
    """
    print(f"Creating water system...")
    ZRB_system = system_creator(
        start_year, start_month, num_time_steps
    )
    
    print(f"Initializing pymoo optimizer with {num_objectives} objectives...")
    if num_objectives == 1:
        optimizer = PymooSingleObjectiveOptimizer(
            base_system=ZRB_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            n_gen=n_gen,
            pop_size=pop_size
        )
    else:
        optimizer = PymooMultiObjectiveOptimizer(
            base_system=ZRB_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            n_gen=n_gen,
            pop_size=pop_size,
            num_objectives=num_objectives
        )
    
    print("Starting objective optimization process...")
    start_time = time.time()
    results = optimizer.optimize()
    end_time = time.time()
    
    print(f"\nMulti-objective optimization completed in {end_time - start_time:.2f} seconds")
    print("-" * 60)
    print(f"Success:        {results['success']}")
    print(f"Message:        {results['message']}")
    print(f"Population size:{results['population_size']}")
    print(f"Generations:    {results['generations']}")
    
    print(f"\nRecommended solution objective values:")
    print(f"  - Objective 1:    {results['objective_values'][0]:,.3f} km³/a")
    if len(results['objective_values']) >= 2:
        print(f"  - Objective 2:    {results['objective_values'][1]:,.3f} km³/a")
    if len(results['objective_values']) >= 3:
        print(f"  - Objective 3:    {results['objective_values'][2]:,.3f} km³/a")
    if len(results['objective_values']) >= 4:
        print(f"  - Objective 4:    {results['objective_values'][3]:,.3f} km³/a")
    
    if num_objectives >= 2:
        print(f"\nNumber of Pareto-optimal solutions: {len(results['pareto_front'])}")
    
    
    # Save optimization results
    output_dir = f"./model_output/pymoo/parameter/"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/pymoo_mo{num_objectives}_simplified_ZRB_{n_gen}_{pop_size}.json"
    
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
    
    start = datetime.now()
    ngen = 10
    popsize = 10
    num_obj = 4

    
    results_mo = run_pymoo_optimization(
        create_simplified_ZRB_system,
        start_year=2017, 
        start_month=1, 
        num_time_steps=12*6,  # Reduced for faster runtime in example
        n_gen=ngen,              # Reduced for faster runtime in example
        pop_size=popsize,           # Reduced for faster runtime in example
        num_objectives=num_obj       # Using 2 objectives for simplicity
    )
    
    if num_obj > 2:
        with open(f"./model_output/pymoo/parameter/pymoo_mo{num_obj}_simplified_ZRB_{ngen}_{popsize}.json", "r") as f:
            data = json.load(f)
        
        pareto_solutions = data.get('pareto_solutions', [])
        
    if num_obj == 3:
        dashboard = ParetoFrontDashboard3D(
            pareto_solutions=pareto_solutions,
            output_dir="./model_output/pymoo/pareto_front"
        )
        # Generate all visualizations
        dashboard.generate_full_report()

    elif num_obj == 4:
        # Create 3D dashboard
        dashboard = ParetoFrontDashboard4D(
            pareto_solutions=pareto_solutions,
            output_dir="./model_output/pymoo/pareto_front"
        )
        # Generate all visualizations
        dashboard.generate_full_report()
    

    print(f"\nMulti-objective optimization completed in {datetime.now() - start}")
