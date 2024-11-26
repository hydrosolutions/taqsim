import csv
import math
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import os
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer

def debug_hydroworks_flow():
    """
    Stress testing water balances on nodes with multiple inputs and outputs. 
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt=dt)

    supply1 = SupplyNode("Supply1", default_supply_rate=50)
    supply2 = SupplyNode("Supply2", default_supply_rate=16)
    hydroworks = HydroWorks("HydroWorks")
    sink = SinkNode("Sink")

    system.add_node(supply1)
    system.add_node(supply2)
    system.add_node(hydroworks)
    system.add_node(sink)

    system.add_edge(Edge(supply1, hydroworks, capacity=50))
    system.add_edge(Edge(supply2, hydroworks, capacity=20))
    system.add_edge(Edge(hydroworks, sink, capacity=100))

    return system

def create_complex_system():
    """
    Creates a more complex water system with multiple supplies, storages, and demands.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    num_time_steps = 36 # 3 years of monthly data
    system = WaterSystem(dt)

    supply1 = SupplyNode("MountainSupply", supply_rates=generate_seasonal_supply(num_time_steps), easting=1, northing=3.5)
    supply2 = SupplyNode("ValleySupply", supply_rates=generate_seasonal_supply(num_time_steps), easting=1, northing=2.5)
    
    reservoir1 = StorageNode("MountainReservoir", csv_path='./data/Akdarya_H_V_A.csv', easting=2, northing=2)
    reservoir2 = StorageNode("ValleyReservoir", csv_path='./data/Akdarya_H_V_A.csv', easting=2, northing=4)
    
    hydrowork1 = HydroWorks("HydroWork1", easting=1.5, northing=3)
    hydrowork2 = HydroWorks("HydroWork2", easting=2.5, northing=2)
    hydrowork3 = HydroWorks("HydroWork3", easting=2.5, northing=4)

    agriculture1 = DemandNode("Farmland1", demand_rates=generate_seasonal_demand(num_time_steps), easting=3, northing=1)
    agriculture2 = DemandNode("Farmland2", demand_rates=generate_seasonal_demand(num_time_steps), easting=3, northing=4)
    urban1 = DemandNode("City1", demand_rates=10,easting=3, northing=2)
    urban2 = DemandNode("City2", demand_rates=10, easting=3, northing=5)
    industry = DemandNode("IndustrialPark", demand_rates=10, easting=3, northing=3)
    sink = SinkNode("RiverMouth", easting=4, northing=3)

    nodes = [supply1,supply2, hydrowork1, hydrowork2, hydrowork3, reservoir1, reservoir2, agriculture1, agriculture2, 
            urban1, urban2, industry, sink]
    for node in nodes:
        system.add_node(node)

    system.add_edge(Edge(supply1, hydrowork1, capacity=100))
    system.add_edge(Edge(supply2, hydrowork1, capacity=100))
    system.add_edge(Edge(hydrowork1, reservoir1, capacity=80))
    system.add_edge(Edge(hydrowork1, reservoir2, capacity=120))
    system.add_edge(Edge(reservoir1, hydrowork2, capacity=80))
    system.add_edge(Edge(hydrowork2, agriculture1, capacity=30))
    system.add_edge(Edge(hydrowork2, urban1, capacity=20))
    system.add_edge(Edge(reservoir2, hydrowork3, capacity=120))
    system.add_edge(Edge(hydrowork3, agriculture2, capacity=30))
    system.add_edge(Edge(hydrowork3, urban2, capacity=15))
    system.add_edge(Edge(hydrowork3, industry, capacity=15))
    system.add_edge(Edge(agriculture1, sink, capacity=30))
    system.add_edge(Edge(agriculture2, sink, capacity=30))
    system.add_edge(Edge(urban1, sink, capacity=20))
    system.add_edge(Edge(urban2, sink, capacity=15))
    system.add_edge(Edge(industry, sink, capacity=15))

    return system

def create_seasonal_reservoir_system():
    """
    Creates a test water system with a seasonal supply, a large reservoir,
    a seasonal demand, and a sink node. The system runs for 10 years with monthly time steps.

    Returns:
        WaterSystem: The configured water system for testing.
    """
    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    num_time_steps = 12 * 10  # 10 years of monthly data
    system = WaterSystem(dt=dt)

    # Create nodes
    supply = SupplyNode("MountainSource", supply_rates=generate_seasonal_supply(num_time_steps), easting=0, northing=0)
    reservoir = StorageNode("LargeReservoir", csv_path='./data/Akdarya_H_V_A.csv', initial_storage=5e7, easting=1, northing=0.5)  # 1 billion m³ capacity, start half full
    demand = DemandNode("SeasonalDemand", demand_rates=generate_seasonal_demand(num_time_steps), easting=2, northing=0.5)
    sink = SinkNode("RiverMouth", easting=3, northing=1)

    reservoir.get_volume_from_level(2)

    # Add nodes to the system
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(demand)
    system.add_node(sink)

    # Connect nodes with edges
    system.add_edge(Edge(supply, reservoir, capacity=100))  # 100 m³/s max flow from supply to reservoir
    system.add_edge(Edge(reservoir, demand, capacity=40))   # 80 m³/s max flow from reservoir to demand
    system.add_edge(Edge(demand, sink, capacity=50))        # 50 m³/s max flow of excess to sink

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

def plot_water_balance_time_series(water_system, filename):
    """
    Create and save a single plot for the entire water system with dual y-axes.
    
    Args:
    water_system (WaterSystem): The water system to plot.
    filename (str): The name of the PNG file to save to.
    columns_to_plot (list): Optional list of column names to plot. If None, all columns are plotted.
    """
    balance_table = water_system.get_water_balance()

    
    columns_to_plot = [col for col in balance_table.columns if col != 'time_step']

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title('Water System Simulation Results')

    colors = list(mcolors.TABLEAU_COLORS.values())  # Use predefined Tableau colors
    line_styles = ['-', '--', ':', '-.']

    color_index = 0
    for column in columns_to_plot:
        if column == 'time_step' or column == 'storage_start' or column == 'storage_end' or column == 'demands':
            continue

        color = colors[color_index % len(colors)]
        line_style = line_styles[color_index % len(line_styles)]
        color_index += 1

        # Plot other columns on the left y-axis
        ax1.plot(balance_table['time_step'], balance_table[column], 
        color=color, linestyle=line_style, label=column)

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Volume (m³)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Water system plot saved to {filename}")

def generate_seasonal_supply(num_time_steps):
    """
    Generates a list of seasonal supply rates.

    Args:
        num_time_steps (int): The number of time steps to generate supply for.

    Returns:
        list: A list of supply rates for each time step.
    """
    base_supply = 50  # m³/s
    amplitude = 30    # m³/s
    supply_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = math.sin(2 * math.pi * month / 12)
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
    base_demand = 40  # m³/s
    amplitude = 20    # m³/s
    demand_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = -math.cos(2 * math.pi * month / 12)  # Peak in summer, trough in winter
        demand_rate = base_demand + amplitude * seasonal_factor
        demand_rates.append(max(0, demand_rate))  # Ensure non-negative demand
    return demand_rates

def run_sample_tests():
    
    print("\n" + "="*50 + "\n")
    
    # Test: Complex System. This is a complex system to test many to many connections.
    complex_system = create_complex_system()
    print("Complex System Test:")
    num_time_steps = 36
    complex_system.simulate(num_time_steps)
    plot_water_balance_time_series(complex_system, "ts_plot_complex_system.png")
    save_water_balance_to_csv(complex_system, "balance_table_complex_system.csv")
    vis=WaterSystemVisualizer(complex_system, 'complex')
    vis.plot_network_layout()
    vis.plot_demand_deficit_heatmap()
    vis.plot_demand_satisfaction()
    vis.plot_water_levels()
    vis.print_water_balance_summary()
    vis.plot_edge_flow_summary()
    
    html_file=vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    """
    print("\n" + "="*50 + "\n")

    # Test: Seasonal Reservoir. Fully seasonal system.
    seasonal_system = create_seasonal_reservoir_system()
    num_time_steps = 2*12  # 2 years of monthly data

    print("Running Seasonal Reservoir Test")

    seasonal_system.simulate(num_time_steps)
    # Generate water balance table
    balance_table = seasonal_system.get_water_balance()
    balance_table.to_csv("balance_table_seasonal_reservoir_system.csv", index=False)
    print("\nWater balance table saved to 'balance_table_seasonal_reservoir_system.csv'")

    # Visualize the system
    plot_water_balance_time_series(seasonal_system, "ts_plot_seasonal_reservoir_system.png")
    print("System layout visualization saved to 'seasonal_reservoir_test_layout.png'")
    vis=WaterSystemVisualizer(seasonal_system, 'seasonal_reservoir')
    vis.print_water_balance_summary()
    vis.plot_water_levels()

    html_file=vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    """
    
# Run the sample tests
if __name__ == "__main__":
  run_sample_tests()