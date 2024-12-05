import csv
import math
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import os
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer, ReleaseOptimizer, GeneticReleaseOptimizer

def create_test_system(num_time_steps):
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
    start_year=2017
    start_month=1

    # Define reservoir release parameters
    release_params = {
        'h1': [504.899, 503.459, 501.389, 504.954, 503.371, 503.185, 504.311, 500.305, 500.915, 504.497, 502.594, 503.628],
        'h2': [507.215, 507.855, 506.230, 508.566, 508.976, 505.482, 506.568, 506.742, 505.111, 508.261, 507.864, 506.210],
        'w': [15.214, 16.653, 6.254, 14.200, 24.522, 9.554, 66.175, 62.287, 72.130, 45.774, 75.487, 13.462],
        'm1': [1.511, 1.557, 1.534, 1.560, 1.565, 1.541, 1.524, 1.567, 1.559, 1.514, 1.530, 1.540],
        'm2': [1.512, 1.553, 1.558, 1.551, 1.529, 1.522, 1.556, 1.515, 1.560, 1.533, 1.539, 1.527]
    }
    # Create nodes
    supply = SupplyNode("Source", supply_rates=generate_seasonal_supply(num_time_steps), easting=0, northing=1000)
    reservoir = StorageNode("Reservoir", hva_file='./data/Kattakurgan_H_V_A.csv', initial_storage=2e8, easting=500, northing=1000, evaporation_file='./data/Reservoir_ET_2010_2023.csv', 
                             start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, release_params=release_params)
    hydrowork = HydroWorks("Hydroworks", easting=1000, northing=1000)
    demand1 = DemandNode("Demand1", easting=2000, northing=1200, csv_file='./data/ETblue/monthly_ETblue_Kattaqorgon_17to22.csv', start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, field_efficiency=0.5)
    demand2 = DemandNode("Demand2", easting=2000, northing=800, csv_file='./data/ETblue/monthly_ETblue_Pastdargom_17to22.csv', start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, field_efficiency=0.5)
    sink = SinkNode("RiverMouth", easting=3000, northing=1000)

    # Add nodes to the system
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(hydrowork)
    system.add_node(demand1)
    system.add_node(demand2)
    system.add_node(sink)

    # Connect nodes with edges
    system.add_edge(Edge(supply, reservoir, capacity=100))  # 100 m³/s max flow from supply to reservoir
    system.add_edge(Edge(reservoir, hydrowork, capacity=80))   # 80 m³/s max flow from reservoir to demand
    system.add_edge(Edge(hydrowork, demand1, capacity=50))   # 50 m³/s max flow from hydrowork to demand
    system.add_edge(Edge(hydrowork, demand2, capacity=50))   # 50 m³/s max flow from hydrowork to demand
    system.add_edge(Edge(demand1, sink, capacity=50))        # 50 m³/s max flow of excess to sink
    system.add_edge(Edge(demand2, sink, capacity=50))        # 50 m³/s max flow of excess to sink

    # Set monthly distribution parameters for edges
    hydrowork.set_distribution_parameters({
        'Demand1': [0.874, 0.890, 0.356, 0.485, 0.392, 0.581, 0.370, 0.493, 0.308, 0.953, 0.847, 0.942],
        'Demand2': [0.126, 0.110, 0.644, 0.515, 0.608, 0.419, 0.630, 0.507, 0.692, 0.047, 0.153, 0.058]
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
    base_supply = 20  # m³/s
    amplitude = 15    # m³/s
    supply_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = math.sin(2 * math.pi * (month-2) / 12)
        supply_rate = base_supply + amplitude * seasonal_factor
        supply_rates.append(max(0, supply_rate))  # Ensure non-negative supply
    return supply_rates

def generate_seasonal_demand(num_time_steps):
    """
    Generates a list of seasonal demand rates.

    Args:
        num_time_steps (int): The number of time steps to generate demand for.

    Returns:
        list: A list of demand rates for each time step.
    """
    base_demand = 20  # m³/s
    amplitude = 15    # m³/s
    demand_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = -math.cos(2 * math.pi * month / 12)  # Peak in summer, trough in winter
        demand_rate = base_demand + amplitude * seasonal_factor
        demand_rates.append(max(0, demand_rate))  # Ensure non-negative demand
    return demand_rates

def run_sample_tests():

    num_time_steps = 12*3  # 1 year with monthly time steps

    print("\n" + "="*50 + "\n")
    # Test: Seasonal Reservoir. Fully seasonal system.
    test_system = create_test_system(num_time_steps)
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
    vis.plot_release_function(storage_node, months=[3,4,5,6])
    vis.plot_reservoir_dynamics()
    vis.plot_storage_dynamics()
    vis.plot_edge_flow_summary()
    vis.plot_edge_flows()
    vis.plot_water_balance_debug(storage_node)
    vis.plot_storage_waterbalance(storage_node)
    #vis.plot_monthly_waterbalance(storage_node)
    vis.plot_hydroworks_flows()
    vis.plot_demand_satisfaction()
    
    html_file=vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')    

def run_optimization():
    optimizer = GeneticReleaseOptimizer(
        create_test_system,
        num_time_steps=12*3,
        population_size=20
    )

    results = optimizer.optimize(ngen=100)

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Final objective value: {results['objective_value']:,.0f} m³")
    
    print("\nOptimal Release Parameters:")
    for param, values in results['optimal_parameters'].items():
        print(f"{param}: ", end="")
        print([f"{v:.3f}" for v in values])
        
    print("\nOptimal Distribution Parameters:")
    for demand, values in results['optimal_distribution'].items():
        print(f"{demand}: ", end="")
        print([f"{v:.3f}" for v in values])

    optimizer.plot_convergence()

# Run the sample tests
if __name__ == "__main__":
  run_sample_tests()
  #run_optimization()