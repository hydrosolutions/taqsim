from typing import Callable, Dict, List, Optional, Union
from water_system import WaterSystem, WaterSystemVisualizer
from datetime import datetime
from system_creator_ZRB import create_simplified_ZRB_system


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
    system = system_creator(start_year, start_month, num_time_steps)
    
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

    # Example of running the simulation with optimized parameters for a simplified ZRB system
    loaded_results = load_parameters_from_file(f"./data/simplified_ZRB/parameter/test5.json")
    system = run_simulation(
        create_simplified_ZRB_system,
        loaded_results,
        start_year=2017,
        start_month=1,
        num_time_steps=12*6,
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
    
    end = datetime.now()
    print(f"Execution time: {end - start}")
    

    