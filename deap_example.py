from typing import Callable, Dict, List, Optional, Union
import numpy as np
import os
import json
from water_system import (WaterSystem, WaterSystemVisualizer, 
                          DeapSingleObjectiveOptimizer, DeapThreeObjectiveOptimizer, 
                          ParetoFrontDashboard3D)
from water_system.deap_optimization import decode_individual
from datetime import datetime
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_intermediate_values, plot_timeline, plot_slice, plot_edf
from ZRB_system_creator import create_ZRB_system


def load_optimized_parameters(system: WaterSystem,optimization_results: Dict[str, Union[Dict, List]]) -> WaterSystem:
    """
    Load optimized parameters into an existing water system.
    
    Args:
        system (WaterSystem): The water system to update
        optimization_results (dict): Results from the optimizer containing:
            - optimal_reservoir_parameters (dict): Parameters for each reservoir
            - optimal_hydroworks_parameters (dict): Parameters for each hydroworks
            
    Returns:
        WaterSystem: Updated system with optimized parameters
    """
    try:
        # Check if the results have the recommended_solution structure or direct parameters
        if 'recommended_solution' in optimization_results:
            # Use the recommended solution structure
            reservoir_params = optimization_results['recommended_solution']['reservoir_parameters'] 
            hydroworks_params = optimization_results['recommended_solution']['hydroworks_parameters']
        else:
            # Use the direct structure from the optimizer
            reservoir_params = optimization_results['optimal_reservoir_parameters']
            hydroworks_params = optimization_results['optimal_hydroworks_parameters']
        
        # Load reservoir parameters
        for res_id, params in reservoir_params.items():
            reservoir_node = system.graph.nodes[res_id]['node']
            if not hasattr(reservoir_node, 'set_release_params'):
                raise ValueError(f"Node {res_id} does not appear to be a StorageNode")
            reservoir_node.set_release_params(params)
            print(f"Successfully updated parameters for reservoir {res_id}")
            
        # Load hydroworks parameters
        for hw_id, params in hydroworks_params.items():
            hydroworks_node = system.graph.nodes[hw_id]['node']
            if not hasattr(hydroworks_node, 'set_distribution_parameters'):
                raise ValueError(f"Node {hw_id} does not appear to be a HydroWorks")
            hydroworks_node.set_distribution_parameters(params)
            print(f"Successfully updated parameters for hydroworks {hw_id}")
            
        return system
        
    except Exception as e:
        raise ValueError(f"Failed to load optimized parameters: {str(e)}")  

def load_parameters_from_file(filename: str) -> Dict[str, Union[Dict, List]]:
    """
    Load previously saved optimized parameters from a file.
    
    Args:
        filename (str): Path to the parameter file
        
    Returns:
        dict: Dictionary containing the optimization results
    """
    import json
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract relevant data from the JSON structure
    return {
        'success': True,
        'message': "Parameters loaded from file",
        'recommended_solution': {
            'reservoir_parameters': data['recommended_solution']['reservoir_parameters'],
            'hydroworks_parameters': data['recommended_solution']['hydroworks_parameters']
        }
    }

def save_optimized_parameters(optimization_results: Dict[str, Union[Dict, List]],filename: str) -> None:
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
    is_multi_objective = isinstance(optimization_results.get('objective_values', None), tuple)
    
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
        save_data['recommended_solution']['objective_values'] = convert_to_float(optimization_results['objective_values'])
        # Add the individual objective values separately
        save_data['recommended_solution']['demand_deficit'] = float(optimization_results['objective_values'][0])
        save_data['recommended_solution']['minflow_deficit'] = float(optimization_results['objective_values'][1])
        
        # If pareto front exists, save all solutions
        if 'pareto_front' in optimization_results and optimization_results['pareto_front']:
            pareto_solutions = []
            
            # Process each solution in the Pareto front
            for i, ind in enumerate(optimization_results['pareto_front']):
                # Decode this individual's parameters
                reservoir_params, hydroworks_params = decode_individual.__get__(None, DeapThreeObjectiveOptimizer)(optimization_results['optimizer'], ind)
                
                # Create a solution entry
                solution = {
                    'id': i,
                    'objective_values': convert_to_float(ind.fitness.values),
                    'demand_deficit': float(ind.fitness.values[0]),
                    'priority_demand_deficit': float(ind.fitness.values[1]),
                    'minflow_deficit': float(ind.fitness.values[2]),
                    'reservoir_parameters': convert_to_float(reservoir_params),
                    'hydroworks_parameters': convert_to_float(hydroworks_params)
                }
                
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
    start_year: int = 2017,
    start_month: int = 1,
    num_time_steps: int = 12,
    system_type: str = 'baseline',
    scenario: str = '',
    period: str = '',
    agr_scenario: str = '',
    efficiency: str = '',
    ngen: int = 100,
    pop_size: int = 2000,
    cxpb: float = 0.5,
    mutpb: float = 0.2
) -> Dict[str, Union[Dict, List]]:
    """
    Run a single-objective optimization for the ZRB water system.
    
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
        
    ZRB_system = system_creator(start_year, start_month, num_time_steps,system_type, scenario, period, agr_scenario, efficiency)

    
    optimizer = DeapSingleObjectiveOptimizer(
        base_system=ZRB_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        ngen=ngen,
        population_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb
    )

    results = optimizer.optimize()

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Corss-over probability: {results['crossover_probability']}")
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

    
    if system_type == 'scenario':
        save_optimized_parameters(results, f"./model_output/optimisation/deap/{system_type}/singleobjective_params_{scenario}_{period}_{agr_scenario}_{efficiency}_{ngen}_{pop_size}_{cxpb}_{mutpb}.json")
    elif system_type == 'baseline':
        save_optimized_parameters(results, f"./model_output/optimisation/deap/{system_type}/singleobjective_params_{ngen}_{pop_size}_{cxpb}_{mutpb}.json")
    else:
        save_optimized_parameters(results, f"./model_output/optimisation/deap/{system_type}/singleobjective_params_{system_type}_{ngen}_{pop_size}_{cxpb}_{mutpb}.json")

    optimizer.plot_convergence()   
    '''ZRB_system = load_optimized_parameters(ZRB_system, results)
    ZRB_system.simulate(num_time_steps)

    vis=WaterSystemVisualizer(ZRB_system, name=f'ZRB_optimization_{system_type}')
    vis.plot_reservoir_dynamics()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.print_water_balance_summary()
    vis.plot_demand_deficit_heatmap()
    vis.plot_network_overview()
    vis.plot_minimum_flow_compliance()'''

    return results

def run_multi_objective_optimization(
    system_creator: Callable[..., WaterSystem],
    start_year: int = 2017,
    start_month: int = 1,
    num_time_steps: int = 12,
    system_type: str = 'baseline',
    scenario: str = '',
    period: str = '',
    agr_scenario: str = '',
    efficiency: str = '',
    ngen: int = 100,
    pop_size: int = 100,
    cxpb: float = 0.5,
    mutpb: float = 0.2
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
    water_system = system_creator(start_year, start_month, num_time_steps, system_type, scenario, period, agr_scenario, efficiency)
    
    # Initialize the multi-objective optimizer
    optimizer = DeapThreeObjectiveOptimizer(
        base_system=water_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        ngen=ngen,
        population_size=pop_size,
        cxpb=cxpb,
        mutpb=mutpb
    )

    # Run the optimization
    results = optimizer.optimize()

    # Plot convergence and Pareto front
    optimizer.plot_convergence()

    # Print optimization results
    print("\nMulti-Objective Optimization Results:")
    print("-" * 60)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Crossover probability: {results['crossover_probability']}")
    print(f"Mutation probability: {results['mutation_probability']}")
    print(f"Final objective values: {results['objective_values']}")
    print(f"  - Demand deficit: {results['objective_values'][0]:,.0f} m³")
    print(f"  - Min flow deficit: {results['objective_values'][1]:,.0f} m³")
    
    # Print the number of solutions in the Pareto front
    print(f"\nNumber of non-dominated solutions: {len(results['pareto_front'])}")
    
    # Print optimal reservoir parameters
    print("\nOptimal Reservoir Parameters:")
    for res_id, params in results['optimal_reservoir_parameters'].items():
        print(f"\n{res_id}:")
        for param, values in params.items():
            print(f"{param}: [", end="")
            print(", ".join(f"{v:.3f}" for v in values), end="")
            print("]")
        
    # Print optimal hydroworks parameters
    print("\nOptimal Hydroworks Parameters:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for hw_id, params in results['optimal_hydroworks_parameters'].items():
        print(f"\n{hw_id}:")
        for target, values in params.items():
            print(f"{target}: [", end="")
            print(", ".join(f"{v:.3f}" for v in values), end="")
            print("]")
    
    return results

def run_simulation(
    system_creator: Callable[..., WaterSystem],
    optimization_results: Dict[str, Union[Dict, List]],
    start_year: int,
    start_month: int,
    num_time_steps: int,
    system_type: str = 'baseline',
    scenario: str = '',
    period: str = '',
    agr_scenario: str = '',
    efficiency: str = ''
) -> WaterSystem:
    """
    Run a simulation for the ZRB water system using optimized parameters.
    
    Args:
        system_creator (function): Function to create the water system
        optimization_results (dict): Results from the optimizer
        start_year (int): Start year for simulation
        start_month (int): Start month for simulation (1-12)
        num_time_steps (int): Number of time steps to simulate
        system_type (str): "baseline" or "scenario"
        scenario (str): Climate scenario (e.g., 'ssp126') - only used for scenario simulations
        period (str): Time period (e.g., '2041-2070') - only used for scenario simulations
        agr_scenario (str): Agricultural scenario - only used for scenario simulations
        efficiency (str): Efficiency scenario (e.g., 'improved_efficiency') - only used for scenario simulations
        
    Returns:
        WaterSystem: Simulated water system
    """
    # Create new system
    system = system_creator(start_year, start_month, num_time_steps,system_type, scenario, period, agr_scenario, efficiency)
    
    # Load optimized parameters
    system = load_optimized_parameters(system, optimization_results)
    print("Optimized parameters loaded successfully")
    # Run simulation
    system.simulate(num_time_steps)
    print("Simulation complete")

    vis=WaterSystemVisualizer(system, name=f'ZRB_simulation_{system_type}')
    vis.plot_network_overview()
    vis.plot_minimum_flow_compliance()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    print("Visualizations complete")
    
    return system

# Run the sample tests
if __name__ == "__main__":

    start = datetime.now()


    optimization = True
    simulation = False
    multiobjective = False
    optunastudy = False

    
    if optimization: 
        # Example of running the optimization for a baseline system
        results = run_optimization(
            create_ZRB_system,
            start_year=2017, 
            start_month=1, 
            num_time_steps=12*2,
            system_type = 'simplified_ZRB',
            scenario = '', 
            period = '', 
            agr_scenario= ' ', 
            efficiency = ' ', 
            ngen=5, 
            pop_size=100, 
            cxpb=0.98, 
            mutpb= 0.22
        )
        
        # Example of running the optimization for a future scenario
        '''results = run_optimization(
            create_ZRB_system,
            start_year=2041, 
            start_month=1, 
            num_time_steps=12*30,
            system_type = 'scenario',
            scenario = 'ssp126',
            period = '2041-2070',
            agr_scenario = 'diversification', 
            efficiency = 'improved_efficiency', # or 'noeff' 
            ngen=10, 
            pop_size=30, 
            cxpb=0.65, 
            mutpb= 0.32
        )''' 

    if simulation:
        # Example of running the simulation with optimized parameters for a simplified ZRB system
        loaded_results = load_parameters_from_file(f"./data/simplified_ZRB/parameter/euler_singleobjective_params_simplified_ZRB_100_2500_0.98_0.22.json")
        system = run_simulation(
            create_ZRB_system,
            loaded_results,
            start_year=2017,
            start_month=1,
            num_time_steps=12,
            system_type = 'simplified_ZRB', 
            scenario = '', 
            period = '',
            agr_scenario = '', 
            efficiency = ''
        )

        """# Example of running the simulation with optimized parameters for a baseline system
        loaded_results = load_parameters_from_file(f"./data/baseline/parameter/2025-05-08_euler_multiobjective_params.json")
        
        system = run_simulation(
            create_ZRB_system,
            loaded_results,
            start_year=2017,
            start_month=1,
            num_time_steps=12*6,
            system_type = 'baseline', 
            scenario = '',
            period = '',
            agr_scenario = '', 
            efficiency = ''
        )"""
        '''#Example of running the simulation with optimized parameters for a future scenario
        system = run_simulation(
            create_ZRB_system,
            loaded_results,
            start_year=2041,
            start_month=1,
            num_time_steps=12*30,
            system_type = 'scenario', 
            scenario = 'ssp126',
            period = '2041-2070',
            agr_scenario = 'diversification', 
            efficiency = 'improved_efficiency' # or 'noeff'
        )'''

    if multiobjective:
        # Example of running the multi-objective optimization for a baseline system
        results = run_multi_objective_optimization(
            create_ZRB_system,
            start_year=2017, 
            start_month=1, 
            num_time_steps=12*6,
            system_type = 'baseline',
            scenario = '', 
            period = '', 
            agr_scenario= '', 
            efficiency = '', 
            ngen=10, 
            pop_size=30, 
            cxpb=0.65, 
            mutpb=0.32
        )


        save_optimized_parameters(results, f"./model_output/optimisation/deap/multiobjective_params.json")

        # Create dashboard for the Pareto front
        dashboard = ParetoFrontDashboard3D(
            pareto_solutions=results['pareto_front'],
            output_dir=f"./model_output/dashboard/baseline",
        )
        
        # Generate all visualizations
        dashboard.generate_full_report()

        with open("./model_output/optimisation/deap/multiobjective_params.json", "r") as f:
            data = json.load(f)
        
        pareto_solutions = data.get('pareto_solutions', [])
        
        # Create dashboard
        dashboard = ParetoFrontDashboard3D(
            pareto_solutions=pareto_solutions,
            output_dir="./model_output/dashboard/baseline"
        )
        
        # Generate all visualizations
        dashboard.generate_full_report()
        
        print(f"Dashboard created at {dashboard.output_dir}/index.html")
   
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
                create_ZRB_system,
                start_year=2017, 
                start_month=1, 
                num_time_steps=12*6,
                system_type = 'simplified_ZRB',
                scenario = '', 
                period = '', 
                agr_scenario= '', 
                efficiency = '', 
                ngen=ngen, 
                pop_size=pop_size, 
                cxpb=cxpb, 
                mutpb=mutpb
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
    

    ##################################
    ### Options for Code Profiling ###
    ##################################
    '''
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Stop profiling
    profiler.disable()

    # Save profiling stats
    #profile_output = 'cprofile_stats.prof'
    #profiler.dump_stats(profile_output)

    # Display profiling stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')  # Sort by cumulative time
    stats.print_stats(20)  # Print top 20 functions
    print(stream.getvalue())
    '''
