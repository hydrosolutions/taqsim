from typing import Callable, Dict, List, Optional, Union
import numpy as np
import os
import json
from water_system import (WaterSystem, WaterSystemVisualizer, 
                          DeapSingleObjectiveOptimizer, DeapTwoObjectiveOptimizer,
                          DeapThreeObjectiveOptimizer, DeapFourObjectiveOptimizer,
                          ParetoFrontDashboard3D, ParetoFrontDashboard4D)
from water_system.optimization.deap_optimization import decode_individual
from datetime import datetime
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_intermediate_values, plot_timeline, plot_slice, plot_edf
from system_creator_ZRB import create_simplified_ZRB_system

def save_optimized_parameters(optimization_results: Dict[str, Union[Dict, List]], filename: str) -> None:
    """
    Save optimized parameters to a file for later use.

    Args:
        optimization_results (dict): Results from the optimizer
        filename (str): Path to save the parameters
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
    is_multi_objective = isinstance(optimization_results.get('objective_values', None), (tuple, list))

    # Prepare basic data for saving
    save_data = {
        'is_multi_objective': is_multi_objective,
        'population_size': optimization_results.get('population_size', 0),
        'generations': optimization_results.get('generations', 0),
        'crossover_probability': optimization_results.get('crossover_probability', 0),
        'mutation_probability': optimization_results.get('mutation_probability', 0),
    }

    # Add the recommended solution (weighted best)
    save_data['recommended_solution'] = {
        'reservoir_parameters': convert_to_float(optimization_results['optimal_reservoir_parameters']),
        'hydroworks_parameters': convert_to_float(optimization_results['optimal_hydroworks_parameters'])
    }

    # Add objective values based on format
    if is_multi_objective:
        obj_values = convert_to_float(optimization_results['objective_values'])
        save_data['recommended_solution']['objective_values'] = obj_values

        # Add each objective value separately with generic keys
        if isinstance(obj_values, (list, tuple)):
            for i, val in enumerate(obj_values):
                save_data['recommended_solution'][f'objective_{i+1}'] = float(val)

        # If pareto front exists, save all solutions
        if 'pareto_front' in optimization_results and optimization_results['pareto_front']:
            pareto_solutions = []
            for i, ind in enumerate(optimization_results['pareto_front']):
                # Try to decode parameters if possible, fallback to None if not available
                try:
                    reservoir_params, hydroworks_params = decode_individual.__get__(None, type(optimization_results['optimizer']))(optimization_results['optimizer'], ind)
                except Exception:
                    reservoir_params, hydroworks_params = None, None

                # Create a solution entry
                solution = {
                    'id': i,
                    'objective_values': convert_to_float(ind.fitness.values),
                    'reservoir_parameters': convert_to_float(reservoir_params),
                    'hydroworks_parameters': convert_to_float(hydroworks_params)
                }
                # Add each objective value separately
                for j, val in enumerate(ind.fitness.values):
                    solution[f'objective_{j+1}'] = float(val)
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

    print(f"Optimization results saved to {filename}")
    if is_multi_objective and 'pareto_solutions' in save_data:
        print(f"Saved {save_data['num_pareto_solutions']} Pareto-optimal solutions")

def run_optimization(
    system_creator: Callable[..., WaterSystem],
    start_year: int,
    start_month: int,
    num_time_steps: int,
    ngen: int,
    pop_size:int,
    cxpb: float,
    mutpb: float, 
    number_of_objectives: int,
    objective_weights: dict[str, list[float]],
) -> Dict[str, Union[Dict, List]]:
    """
    Run a multi-objective optimization for the ZRB water system.
    
    Args:
        system_creator (function): Function to create the water system
        start_year (int): Start year for simulation
        start_month (int): Start month for simulation (1-12)
        num_time_steps (int): Number of time steps to simulate
        system_type (str): "baseline" or "scenario"
        scenario (str): Climate scenario (e.g., 'ssp126') - only used for scenario simulations
        period (str): Time period (e.g., '2041-2070') - only used for scenario simulations
        agr_scenario (str): Agricultural scenario - only used for scenario simulations
        efficiency (str): Efficiency scenario (e.g., 'improved_efficiency') - only used for scenario simulations
        ngen (int): Number of generations for the optimizer
        pop_size (int): Population size for the optimizer
        cxpb (float): Crossover probability
        mutpb (float): Mutation probability
        
    Returns:
        dict: Results of the optimization
    """

    # Create the base water system
    water_system = system_creator(start_year, start_month, num_time_steps)

    if number_of_objectives == 1:
        # Initialize the single-objective optimizer
        optimizer = DeapSingleObjectiveOptimizer(
            base_system=water_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            ngen=ngen,
            population_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            number_of_objectives=number_of_objectives,
            objective_weights=objective_weights
        )
    
    elif number_of_objectives == 2:
        optimizer = DeapTwoObjectiveOptimizer(
            base_system=water_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            ngen=ngen,
            population_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb, 
            number_of_objectives=number_of_objectives,
            objective_weights=objective_weights
        )

    elif number_of_objectives == 3:
        # Initialize the multi-objective optimizer
        optimizer = DeapThreeObjectiveOptimizer(
            base_system=water_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            ngen=ngen,
            population_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            number_of_objectives=number_of_objectives,
            objective_weights=objective_weights
        )
    
    elif number_of_objectives == 4:
        # Initialize the multi-objective optimizer
        optimizer = DeapFourObjectiveOptimizer(
            base_system=water_system,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            ngen=ngen,
            population_size=pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            number_of_objectives=number_of_objectives,
            objective_weights=objective_weights
        )

    else:
        raise ValueError("number_of_objectives must be 1, 2, 3, or 4.")
    # Run the optimization
    results = optimizer.optimize()

    # Plot convergence and Pareto front
    optimizer.plot_convergence()
    optimizer.plot_total_objective_convergence()

    # Print optimization results
    print("\nMulti-Objective Optimization Results:")
    print("-" * 60)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations:     {results['generations']}")
    print(f"Crossover probability:  {results['crossover_probability']}")
    print(f"Mutation probability:   {results['mutation_probability']}")
    print(f"Final objective values: {results['objective_values']}")
    print(f"  - Objective 1:    {results['objective_values'][0]:,.3f} km³/a")
    if number_of_objectives >= 2:
        print(f"  - Objective 2:    {results['objective_values'][1]:,.3f} km³/a")
    if number_of_objectives >= 3:
        print(f"  - Objective 3:    {results['objective_values'][2]:,.3f} km³/a")
    if number_of_objectives == 4:
        print(f"  - Objective 4:    {results['objective_values'][3]:,.3f} km³/a")
    
    # Print the number of solutions in the Pareto front
    print(f"\nNumber of non-dominated solutions: {len(results['pareto_front'])}")
    
    return results

# Run the sample tests
if __name__ == "__main__":

    start = datetime.now()
    
    number_of_objectives = 4
    number_of_generations = 20
    population_size = 50


    if number_of_objectives == 1:
        objective_weights ={
            'objective_1': [1.0,1.0,1.0,1.0]
        }
    if number_of_objectives == 2:
        objective_weights ={
            'objective_1': [1.0,1.0,0.0,0.0],
            'objective_2': [0.0,0.0,1.0,1.0]
        }
    if number_of_objectives == 3:
        objective_weights ={
            'objective_1': [1.0,1.0,0.0,0.0],
            'objective_2': [0.0,0.0,1.0,0.0],
            'objective_3': [0.0,0.0,0.0,1.0]
        }
    if number_of_objectives == 4:
        objective_weights ={
            'objective_1': [1.0,0.0,0.0,0.0],
            'objective_2': [0.0,1.0,0.0,0.0],
            'objective_3': [0.0,0.0,1.0,0.0],
            'objective_4': [0.0,0.0,0.0,1.0]
        }

    # Example of running the multi-objective optimization for a baseline system
    results = run_optimization(
        create_simplified_ZRB_system,
        start_year=2017, 
        start_month=1, 
        num_time_steps=12*6,
        ngen=number_of_generations, 
        pop_size=population_size, 
        cxpb=0.65, 
        mutpb=0.32,
        number_of_objectives=number_of_objectives,
        objective_weights=objective_weights
    )

    
    save_optimized_parameters(results, f"./model_output/deap/parameter/multiobjective_params.json")
    if number_of_objectives == 3:
        dashboard = ParetoFrontDashboard3D(
            pareto_solutions=results['pareto_front'],
            output_dir=f"./model_output/deap/pareto_front/",
        )
        dashboard.generate_full_report()
    if number_of_objectives == 4:
        # Create dashboard for the Pareto front
        dashboard = ParetoFrontDashboard4D(
            pareto_solutions=results['pareto_front'],
            output_dir=f"./model_output/deap/pareto_front/",
        )      
        # Generate all visualizations
        dashboard.generate_full_report()

    optunastudy = False
    if optunastudy:
        # Making an Optuna study
        def objective(trial):
            # Define the hyperparameters to optimize
            ngen = trial.suggest_int("ngen", 1, 10)
            pop_size = trial.suggest_int("pop_size", 5, 30)
            cxpb = trial.suggest_float("cxpb", 0.5, 1.0)
            mutpb = trial.suggest_float("mutpb", 0.1, 0.5)

            # Run the optimization
            results = run_optimization(
                create_simplified_ZRB_system,
                start_year=2017, 
                start_month=1, 
                num_time_steps=12*6,
                ngen=ngen, 
                pop_size=pop_size, 
                cxpb=cxpb, 
                mutpb=mutpb,
                number_of_objectives=1
            )

            return results['objective_value']

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)

        # Print the best parameters and fitness value
        print("Best Parameters:", study.best_params)
        print("Best Objective Value:", study.best_value)

        # save the study
        study_name = "ZRB_study"
        study_file = f"{study_name}.pkl"
        study.trials_dataframe().to_csv(f"{study_name}.csv")
        study.trials_dataframe().to_pickle(study_file)

        # create the study directory
        if not os.path.exists("model_output/optuna"):
            os.makedirs("model_output/optuna")
        #save the plots
        plot_optimization_history(study).write_html(f"model_output/optuna/{study_name}_history.html")
        plot_param_importances(study).write_html(f"model_output/optuna/{study_name}_importances.html")
        plot_contour(study).write_html(f"model_output/optuna/{study_name}_contour.html")
        plot_timeline(study).write_html(f"model_output/optuna/{study_name}_timeline.html")
        plot_slice(study).write_html(f"model_output/optuna/{study_name}_slice.html")
        plot_edf(study).write_html(f"model_output/optuna/{study_name}_edf.html")


    end = datetime.now()
    print(f"Execution time: {end - start}")
