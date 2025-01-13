import csv
import math
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import os
import json
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer, MultiGeneticOptimizer

def create_test_system(start_year, start_month, num_time_steps):
    """
    Creates a test water system with a seasonal supply, a large reservoir,
    a seasonal demand, and a sink node. The system runs for 10 years with monthly time steps.

    Returns:
        WaterSystem: The configured water system for testing.
    """
    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    num_time_steps = num_time_steps
    system = WaterSystem(dt=dt)
    start_year=start_year
    start_month=start_month

    # Define reservoir release parameters
    release_params = {
        'h1': 504.899,
        'h2': 507.215,
        'w':  13.462,
        'm1': 1.540,
        'm2': 1.527
    }
    # Create nodes
    supply = SupplyNode("Source", supply_rates=generate_seasonal_supply(num_time_steps), easting=0, northing=1000)
    hydrowork1 = HydroWorks("Hydroworks1", easting=200, northing=1000)
    reservoir = StorageNode("Reservoir", hva_file='./data/Kattakurgan_H_V_A.csv', initial_storage=2e8, easting=500, northing=1000, evaporation_file='./data/Reservoir_ET_2010_2023.csv', 
                             start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, release_params=release_params)
    hydrowork2 = HydroWorks("Hydroworks2", easting=1000, northing=1000)
    demand1 = DemandNode("Demand1", easting=1600, northing=1200, csv_file='./data/ETblue/monthly_ETblue_Kattaqorgon_17to22.csv', 
                         start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, field_efficiency=0.5, weight=1.0)
    demand2 = DemandNode("Demand2", easting=2000, northing=800, csv_file='./data/ETblue/monthly_ETblue_Pastdargom_17to22.csv', 
                         start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, field_efficiency=0.5, weight=1.0)
    demand3 = DemandNode("Demand3", easting=2400, northing=1200, csv_file='./data/ETblue/monthly_ETblue_Ishtixon_17to22.csv', 
                         start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, field_efficiency=0.5, weight=1.0)
    sink = SinkNode("RiverMouth", easting=3000, northing=1000)

    # Add nodes to the system
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(hydrowork1)
    system.add_node(hydrowork2)
    system.add_node(demand1)
    system.add_node(demand2)
    system.add_node(demand3)
    system.add_node(sink)

    # Connect nodes with edges
    system.add_edge(Edge(supply, hydrowork1, capacity=100))  # 100 m³/s max flow from supply to reservoir
    system.add_edge(Edge(hydrowork1, reservoir, capacity=80))   # 80 m³/s max flow from reservoir to demand
    system.add_edge(Edge(hydrowork1, demand1, capacity=80))   # 80 m³/s max flow from reservoir to demand
    system.add_edge(Edge(reservoir, hydrowork2, capacity=80))   # 80 m³/s max flow from reservoir to demand
    system.add_edge(Edge(hydrowork2, demand1, capacity=50))   # 50 m³/s max flow from hydrowork to demand
    system.add_edge(Edge(hydrowork2, demand2, capacity=50))   # 50 m³/s max flow from hydrowork to demand
    system.add_edge(Edge(demand1, demand3, capacity=50))   # 50 m³/s max flow from hydrowork to demand
    system.add_edge(Edge(demand3, sink, capacity=130))        # 50 m³/s max flow of excess to sink
    system.add_edge(Edge(demand2, sink, capacity=50))        # 50 m³/s max flow of excess to sink

    # Set monthly distribution parameters for edges
    hydrowork1.set_distribution_parameters({
        'Demand1': 0.5,
        'Reservoir': 0.5
    })
    hydrowork2.set_distribution_parameters({
        'Demand1': 0.5,
        'Demand2': 0.5
    })
    return system

def save_water_balance_to_csv(water_system, filename):
    """
    Save the water balance table of a water system to a CSV file.
    
    Args:
    water_system (WaterSystem): The water system to save the balance for.
    filename (str): The name of the CSV file to save to.
    """
    balance_table = water_system.get_water_balance()
    balance_table.to_csv(filename, index=False)
    print(f"Water balance table saved to {filename}")

def generate_seasonal_supply(num_time_steps):
    """
    Generates a list of seasonal supply rates.

    Args:
        num_time_steps (int): The number of time steps to generate supply for.

    Returns:
        list: A list of supply rates for each time step.
    """
    base_supply = 40  # m³/s
    amplitude = 15    # m³/s
    supply_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = math.sin(2 * math.pi * (month-2) / 12)
        supply_rate = base_supply + amplitude * seasonal_factor
        supply_rates.append(max(0, supply_rate))  # Ensure non-negative supply
    return supply_rates

def load_optimized_parameters(system, optimization_results):
    """
    Load optimized parameters into an existing water system.
    
    Args:
        system (WaterSystem): The water system to update
        optimization_results (dict): Results from the MultiGeneticOptimizer containing:
            - optimal_reservoir_parameters (dict): Parameters for each reservoir
            - optimal_hydroworks_parameters (dict): Parameters for each hydroworks
            
    Returns:
        WaterSystem: Updated system with optimized parameters
    """
    try:
        # Load reservoir parameters
        for res_id, params in optimization_results['optimal_reservoir_parameters'].items():
            reservoir_node = system.graph.nodes[res_id]['node']
            if not hasattr(reservoir_node, 'set_release_params'):
                raise ValueError(f"Node {res_id} does not appear to be a StorageNode")
            reservoir_node.set_release_params(params)
            print(f"Successfully updated parameters for reservoir {res_id}")
            
        # Load hydroworks parameters
        for hw_id, params in optimization_results['optimal_hydroworks_parameters'].items():
            hydroworks_node = system.graph.nodes[hw_id]['node']
            if not hasattr(hydroworks_node, 'set_distribution_parameters'):
                raise ValueError(f"Node {hw_id} does not appear to be a HydroWorks")
            hydroworks_node.set_distribution_parameters(params)
            print(f"Successfully updated parameters for hydroworks {hw_id}")
            
        return system
        
    except Exception as e:
        raise ValueError(f"Failed to load optimized parameters: {str(e)}")

def save_optimized_parameters(optimization_results, filename):
    """
    Save optimized parameters to a file for later use.
    
    Args:
        optimization_results (dict): Results from the MultiGeneticOptimizer
        filename (str): Path to save the parameters
    """
    
    # Convert all numeric values to floats for JSON serialization
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(x) for x in obj]
        elif isinstance(obj, (int, float)):
            return float(obj)
        return obj
    
    # Prepare data for saving
    save_data = {
        'objective_value': float(optimization_results['objective_value']),
        'reservoir_parameters': convert_to_float(optimization_results['optimal_reservoir_parameters']),
        'hydroworks_parameters': convert_to_float(optimization_results['optimal_hydroworks_parameters'])
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Optimization results saved to {filename}")

def load_parameters_from_file(filename):
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
    
    return {
        'success': True,
        'message': "Parameters loaded from file",
        'objective_value': data['objective_value'],
        'optimal_reservoir_parameters': data['reservoir_parameters'],
        'optimal_hydroworks_parameters': data['hydroworks_parameters']
    }

def run_system_with_optimized_parameters(system_creator, optimization_results, 
                                       start_year, start_month, num_time_steps):
    """
    Create and run a water system using optimized parameters.
    
    Args:
        system_creator (function): Function that creates the water system
        optimization_results (dict): Results from optimization or loaded from file
        start_year (int): Start year for simulation
        start_month (int): Start month (1-12)
        num_time_steps (int): Number of time steps to simulate
        
    Returns:
        WaterSystem: Simulated system with optimized parameters
    """
    # Create new system
    system = system_creator(start_year, start_month, num_time_steps)
    
    # Load optimized parameters
    system = load_optimized_parameters(system, optimization_results)
    
    # Run simulation
    system.simulate(num_time_steps)
    
    return system

def run_sample_tests(start_year=2017, start_month=1, num_time_steps=12):

    num_time_steps = num_time_steps
    start_year=start_year
    start_month=start_month

    print("\n" + "="*50 + "\n")
    # Test: Seasonal Reservoir. Fully seasonal system.
    test_system = create_test_system(start_year, start_month, num_time_steps)
    print("Running test system")
    test_system.simulate(num_time_steps)
    
    # Generate water balance table
    balance_table = test_system.get_water_balance()
    balance_table.to_csv("balance_table_test_system.csv", index=False)
    print("\nWater balance table saved to 'balance_table_test_system.csv'")

    # Visualize the system
    print("Test system visualization:")
    vis=WaterSystemVisualizer(test_system, 'test_system')
    vis.plot_network_layout()
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    storage_node = test_system.graph.nodes['Reservoir']['node']
    vis.plot_release_function(storage_node)
    vis.plot_reservoir_dynamics()
    vis.plot_storage_dynamics()
    vis.plot_storage_waterbalance(storage_node)
    vis.plot_demand_satisfaction()
    
    html_file=vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')    

def run_optimization(start_year=2017, start_month=1, num_time_steps=12, ngen=100, popsize=200, cxpb=0.8, mutpb=0.2):
    optimizer = MultiGeneticOptimizer(
        create_test_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        ngen=ngen,
        population_size=popsize, 
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
    print(f"Final objective value: {results['objective_value']:,.0f} m³")
    
    print("\nOptimal Reservoir Parameters:")
    for res_id, params in results['optimal_reservoir_parameters'].items():
        print(f"\n{res_id}:")
        for param, values in params.items():
            print(f"{param}: ", end="")
            print(f"{values:.3f}")
        
    print("\nOptimal Hydroworks Parameters:")
    for hw_id, params in results['optimal_hydroworks_parameters'].items():
        print(f"\n{hw_id}:")
        for target, values in params.items():
            print(f"{target}: ", end="")
            print(f"{values:.3f}")

    optimizer.plot_convergence()
    return results

# Run the sample tests
if __name__ == "__main__":
    start_year=2017
    start_month=1
    num_time_steps=12*3
    ngen=10
    popsize=200
    cxpb=0.8
    mutpb=0.2


    #run_sample_tests(start_year, start_month, num_time_steps)
    results=run_optimization(start_year, start_month, num_time_steps, ngen, popsize, cxpb, mutpb)
    
    # Save optimization results
    #save_optimized_parameters(results, f"optimized_parameters_test_system_ngen{ngen}_pop{popsize}.json")

    """
    loaded_results = load_parameters_from_file("optimized_parameters_test_system_ngen10_pop200.json")
    # Run system with optimized parameters
    optimized_system = run_system_with_optimized_parameters(
        create_test_system,  # Your system creator function
        loaded_results,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps
    )

    # Visualize the optimized system
    print("Optimized system visualization:")
    vis=WaterSystemVisualizer(optimized_system, 'optimized_test_system')
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    vis.plot_storage_dynamics()
    vis.plot_reservoir_dynamics()
    vis.plot_network_layout()
    vis.plot_demand_satisfaction()  
    vis.plot_system_demands_vs_inflow()
    
    # Get the storage node from the system's graph
    storage_node = optimized_system.graph.nodes['Reservoir']['node']
    vis.plot_release_function(storage_node)
    

    html_file=vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    """