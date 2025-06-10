import os
import json
import numpy as np
from typing import Callable, Dict, List, Optional, Union
from water_system.optimization.optimizer import decode_individual
from water_system import WaterSystem

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
        'population_size': optimization_results.get('population_size', 0),
        'generations': optimization_results.get('generations', 0),
        'crossover_probability': optimization_results.get('crossover_probability', 0),
        'mutation_probability': optimization_results.get('mutation_probability', 0),
        'num_pareto_solutions': len(optimization_results.get('pareto_front', [])),
    }
    # If pareto front exists, save all solutions
    if 'pareto_front' in optimization_results and optimization_results['pareto_front']:
        pareto_solutions = []
        optimizer = optimization_results.get('optimizer', None)
        # Prepare decode_individual arguments if optimizer is available
        if optimizer is not None:
            reservoir_ids = optimizer.reservoir_ids
            hydroworks_ids = optimizer.hydroworks_ids
            hydroworks_targets = optimizer.hydroworks_targets
        else:
            reservoir_ids = hydroworks_ids = hydroworks_targets = None

        for i, ind in enumerate(optimization_results['pareto_front']):
            # Try to decode parameters if possible, fallback to None if not available
            try:
                if optimizer is not None:
                    reservoir_params, hydroworks_params = decode_individual(
                        reservoir_ids, hydroworks_ids, hydroworks_targets, ind
                    )
                else:
                    reservoir_params, hydroworks_params = None, None
            except Exception:
                reservoir_params, hydroworks_params = None, None

            # Create a solution entry
            solution = {
                'id': i,
                'objective_values': convert_to_float(ind.fitness.values),
            }
            # Add each objective value separately
            for j, val in enumerate(ind.fitness.values):
                solution[f'objective_{j+1}'] = float(val)

            solution['reservoir_parameters']= convert_to_float(reservoir_params)
            solution['hydroworks_parameters']= convert_to_float(hydroworks_params)
            pareto_solutions.append(solution)

        # Add all Pareto solutions to the save data
        save_data['pareto_solutions'] = pareto_solutions

    # Save to file
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Optimization results saved to {filename}")
    if is_multi_objective and 'pareto_solutions' in save_data:
        print(f"Saved {save_data['num_pareto_solutions']} Pareto-optimal solutions")

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
        dict: Dictionary containing the pareto solutions and their parameters
    """
    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Parameter file {filename} does not exist.")
    # Load the JSON data from the file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Check if the file contains Pareto solutions
    if 'pareto_solutions' not in data:
        raise ValueError(f"File {filename} does not contain Pareto solutions..")

    # Find the solution with the given id
    pareto_solutions = data['pareto_solutions']
    return pareto_solutions