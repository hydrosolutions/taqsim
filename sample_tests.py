import csv
import math
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def create_most_simple_system():
    """
    Creates the most simple system with one supply, one demand, and one sink node 
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt=dt)

    supply = SupplyNode("Source", default_supply_rate=100,easting=1,northing=1)
    agriculture = DemandNode("Agriculture", demand_rates=80,easting=2, northing=1)
    outflow = SinkNode("Sink",easting=3,northing=1)

    system.add_node(supply)
    system.add_node(agriculture)
    system.add_node(outflow)

    e1=Edge(supply, agriculture, capacity=100)
    e2=Edge(agriculture, outflow, capacity=50)

    for edge in [e1, e2]:
        system.add_edge(edge)

    return system

def create_hydroworks_system_3_diversions():
    """
    Creates a system with one supply, three HydroWorks nodes, and one sink node
    """

    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)

    supplyA = SupplyNode("River A", default_supply_rate=60, easting=1, northing=1)
    supplyB = SupplyNode("River B", default_supply_rate=30, easting=1, northing=2)
    hydroworks1 = HydroWorks("HydroWorks1", easting=2, northing=1.5)
    hydroworks2 = HydroWorks("HydroWorks2", easting=3, northing=2)
    demandsite = DemandNode("Agriculture", demand_rates=20, easting=3, northing=1)
    hydroworks4 = HydroWorks("HydroWorks4", easting=4, northing=1.5)
    outflow = SinkNode("Outflow", easting=5, northing=1.5)

    system.add_node(supplyA)
    system.add_node(supplyB)
    system.add_node(hydroworks1)
    system.add_node(hydroworks2)
    system.add_node(demandsite)
    system.add_node(hydroworks4)
    system.add_node(outflow)

    system.add_edge(Edge(supplyA, hydroworks1, capacity=100, length=10, loss_factor=0.01))
    system.add_edge(Edge(supplyB, hydroworks1, capacity=100))
    system.add_edge(Edge(hydroworks1, hydroworks2, capacity=60, length=10, loss_factor=0.01))
    system.add_edge(Edge(hydroworks2, hydroworks4, capacity=50, length=10, loss_factor=0.01))
    system.add_edge(Edge(hydroworks1, demandsite, capacity=40))
    system.add_edge(Edge(demandsite, hydroworks4, capacity=50, length=10, loss_factor=0.01))
    system.add_edge(Edge(hydroworks4, outflow, capacity=100))

    return system

def create_hydroworks_system_2_diversions():
    """
    Creates a system with one supply, two HydroWorks nodes, and one sink node
    """

    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)

    supplyA = SupplyNode("River A", default_supply_rate=60, easting=1, northing=1)
    supplyB = SupplyNode("River B", default_supply_rate=30, easting=1, northing=2)
    hydroworks1 = HydroWorks("HydroWorks1", easting=2, northing=1.5)
    hydroworks2 = HydroWorks("HydroWorks2", easting=3, northing=1.5)
    demandsite = DemandNode("Agriculture", demand_rates=20, easting=2.5, northing=1)
    outflow = SinkNode("Outflow", easting=4, northing=1.5)

    system.add_node(supplyA)
    system.add_node(supplyB)
    system.add_node(hydroworks1)
    system.add_node(hydroworks2)
    system.add_node(demandsite)
    system.add_node(outflow)

    system.add_edge(Edge(supplyA, hydroworks1, capacity=100))
    system.add_edge(Edge(supplyB, hydroworks1, capacity=100))
    system.add_edge(Edge(hydroworks1, hydroworks2, capacity=60))
    system.add_edge(Edge(hydroworks1, demandsite, capacity=40))
    system.add_edge(Edge(demandsite, hydroworks2, capacity=50))
    system.add_edge(Edge(hydroworks2, outflow, capacity=100))

    return system

def create_simple_system():
    """
    Creates a simple water system with one supply, one storage, two demands, and one sink.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)

    supply = SupplyNode("Source", default_supply_rate=100,easting=1,northing=1)
    reservoir = StorageNode("MainReservoir", capacity=500, easting=2, northing=1)
    agriculture = DemandNode("Agriculture", demand_rates=60, easting=3,northing=2)
    urban = DemandNode("Urban", demand_rates=30,easting=3,northing=0)
    sink = SinkNode("Sink",easting=4,northing=1)

    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(agriculture)
    system.add_node(urban)
    system.add_node(sink)

    system.add_edge(Edge(supply, reservoir, capacity=120))
    system.add_edge(Edge(reservoir, agriculture, capacity=70))
    system.add_edge(Edge(reservoir, urban, capacity=40))
    system.add_edge(Edge(agriculture, sink, capacity=100))
    system.add_edge(Edge(urban, sink, capacity=50))

    return system

def create_simple_system_with_diversion():
    """
    Creates a simple water system with one supply, one storage, two demands, and one sink.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)

    supply = SupplyNode("Source", default_supply_rate=100, easting=1, northing=1)
    reservoir = StorageNode("MainReservoir", capacity=500, easting=2, northing=1)
    diversion = HydroWorks("Diversion", easting=3, northing=2)
    urban = DemandNode("Urban", demand_rates=30, easting=3, northing=0)
    sink = SinkNode("Sink", easting=4, northing=1)

    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(diversion)
    system.add_node(urban)
    system.add_node(sink)

    system.add_edge(Edge(supply, reservoir, capacity=120))
    system.add_edge(Edge(reservoir, diversion, capacity=70, length=4, loss_factor=0.1))
    system.add_edge(Edge(reservoir, urban, capacity=40))
    system.add_edge(Edge(diversion, sink, capacity=100, length=2, loss_factor=0.2))
    system.add_edge(Edge(urban, sink, capacity=50))

    return system

def create_complex_system():
    """
    Creates a more complex water system with multiple supplies, storages, and demands.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)

    supply1 = SupplyNode("MountainSupply", default_supply_rate=150, easting=1, northing=2)
    supply2 = SupplyNode("RiverSupply", default_supply_rate=100, easting=1, northing=4)
    reservoir1 = StorageNode("MountainReservoir", capacity=1000, easting=2, northing=2)
    reservoir2 = StorageNode("ValleyReservoir", capacity=800, easting=2, northing=4)
    agriculture1 = DemandNode("Farmland1", demand_rates=80, easting=3, northing=1)
    agriculture2 = DemandNode("Farmland2", demand_rates=70, easting=3, northing=4)
    urban1 = DemandNode("City1", demand_rates=50,easting=3, northing=2)
    urban2 = DemandNode("City2", demand_rates=40, easting=3, northing=5)
    industry = DemandNode("IndustrialPark", demand_rates=30, easting=3, northing=3)
    sink = SinkNode("RiverMouth", easting=4, northing=3)

    nodes = [supply1, supply2, reservoir1, reservoir2, agriculture1, agriculture2, 
            urban1, urban2, industry, sink]
    for node in nodes:
        system.add_node(node)

    system.add_edge(Edge(supply1, reservoir1, capacity=160))
    system.add_edge(Edge(supply2, reservoir2, capacity=120))
    system.add_edge(Edge(reservoir1, agriculture1, capacity=90))
    system.add_edge(Edge(reservoir1, urban1, capacity=60))
    system.add_edge(Edge(reservoir2, agriculture2, capacity=80))
    system.add_edge(Edge(reservoir2, urban2, capacity=50))
    system.add_edge(Edge(reservoir2, industry, capacity=40))
    system.add_edge(Edge(agriculture1, sink, capacity=100))
    system.add_edge(Edge(agriculture2, sink, capacity=100))
    system.add_edge(Edge(urban1, sink, capacity=70))
    system.add_edge(Edge(urban2, sink, capacity=60))
    system.add_edge(Edge(industry, sink, capacity=50))

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
    supply = SupplyNode("MountainSource", supply_rates=generate_seasonal_supply(num_time_steps), easting=1, northing=1)
    reservoir = StorageNode("LargeReservoir", capacity=1e9, initial_storage=5e8, easting=2, northing=1)  # 1 billion m³ capacity, start half full
    demand = DemandNode("SeasonalDemand", demand_rates=generate_seasonal_demand(num_time_steps), easting=3, northing=1)
    sink = SinkNode("RiverMouth", easting=4, northing=1)

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
    balance_table = water_system.get_water_balance_table()
    balance_table.to_csv(filename, index=False)
    print(f"Water balance table saved to {filename}")

def plot_water_balance_time_series(water_system, filename, columns_to_plot=None):
    """
    Create and save a single plot for the entire water system with dual y-axes.
    
    Args:
    water_system (WaterSystem): The water system to plot.
    filename (str): The name of the PNG file to save to.
    columns_to_plot (list): Optional list of column names to plot. If None, all columns are plotted.
    """
    balance_table = water_system.get_water_balance_table()

    if columns_to_plot is None:
        columns_to_plot = [col for col in balance_table.columns if col != 'TimeStep']

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()  # Create a second y-axis

    plt.title('Water System Simulation Results')

    colors = list(mcolors.TABLEAU_COLORS.values())  # Use predefined Tableau colors
    line_styles = ['-', '--', ':', '-.']

    color_index = 0
    for column in columns_to_plot:
        if column == 'TimeStep':
            continue

        color = colors[color_index % len(colors)]
        line_style = line_styles[color_index % len(line_styles)]
        color_index += 1

        if 'Storage' in column:
            # Plot storage volume on the right y-axis
            ax2.plot(balance_table['TimeStep'], balance_table[column], 
            color=color, linestyle=line_style, label=column)
        elif 'ExcessVolume' in column:
            ax2.plot(balance_table['TimeStep'], balance_table[column], 
            color=color, linestyle=line_style, label=column)
        else:
            # Plot other columns on the left y-axis
            ax1.plot(balance_table['TimeStep'], balance_table[column], 
            color=color, linestyle=line_style, label=column)

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Flow Rate (m³/s)')
    ax2.set_ylabel('Volume (m³)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')

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

    # Test: Super Simple System. This is a simple linear system with one source, one demand site, and one sink
    super_simple_system = create_most_simple_system()
    print("Super simple system test:")
    num_time_steps = 12
    super_simple_system.simulate(num_time_steps)

    # Visualize the system
    columns_to_plot = [
    "Source_Outflow",
    "Agriculture_Inflow",
    "Agriculture_SatisfiedDemand",
    "Agriculture_Outflow",
    "Sink_Inflow"
    ]

    plot_water_balance_time_series(super_simple_system, "ts_plot_super_simple_system.png", columns_to_plot=columns_to_plot)
    save_water_balance_to_csv(super_simple_system, "balance_table_super_simple_system.csv")
    vis=WaterSystemVisualizer(super_simple_system, 'super_simple')
    vis.plot_network_layout()
    
    print("\n" + "="*50 + "\n")

    # Test: Simple System. This is a system with one source, one reservoir, two demand sites connected to the reservoir and return flows to one sink.
    simple_system = create_simple_system()
    print("Simple System Test:")
    num_time_steps = 12
    simple_system.simulate(num_time_steps)

    # Visualize the system
    columns_to_plot = [
    "Source_Outflow",
    "MainReservoir_Outflow",
    "Agriculture_Inflow",
    "Urban_Inflow",
    "Agriculture_SatisfiedDemand",
    "Urban_SatisfiedDemand",
    "Agriculture_Outflow",
    "Urban_Outflow",
    "Sink_Inflow"
    ]

    plot_water_balance_time_series(simple_system, "ts_plot_simple_system.png", columns_to_plot=columns_to_plot)
    save_water_balance_to_csv(simple_system, "balance_table_simple_system.csv")
    vis=WaterSystemVisualizer(simple_system, 'simple')
    vis.plot_network_layout()
    
    print("\n" + "="*50 + "\n")

    # Test: HydroWorks System with 3 Diversions. Two sources feeding into a hydroworks diverting water to a third hydroworks. Fromt the first hydroworks, there is a brnaching to a demand node from which return flow ends in the third hydroworks. Total water from the third hydroworks flows into the sink.
    hydroworks_system_3_diversions = create_hydroworks_system_3_diversions()
    print("HydroWorks System Test:")
    num_time_steps = 12
    hydroworks_system_3_diversions.simulate(num_time_steps)

    visualizer=WaterSystemVisualizer(hydroworks_system_3_diversions, 'HW_3_diversions')
    visualizer.plot_node_flows(['HydroWorks1', 'Agriculture'])
    visualizer.plot_network_layout()

    plot_water_balance_time_series(hydroworks_system_3_diversions, "ts_plot_hydroworks_3_diversions_system.png")
    save_water_balance_to_csv(hydroworks_system_3_diversions, "balance_table_hydroworks_3_diversions_system.csv")
    
    print("\n" + "="*50 + "\n")

   # Test: Simple System with Diversion. Same as the simple system but with the agricultural node replaced by a hydroworks node.
    simple_system_with_diversion = create_simple_system_with_diversion()
    print("Simple System with Diversion Test:")
    num_time_steps = 12
    simple_system_with_diversion.simulate(num_time_steps)

    # Visualize the system
    columns_to_plot = [
    "Source_Outflow",
    "MainReservoir_Outflow",
    "Diversion_Inflow",
    "Urban_Inflow",
    "Urban_SatisfiedDemand",
    "Diversion_Outflow",
    "Urban_Outflow",
    "Sink_Inflow"
    ]

    plot_water_balance_time_series(simple_system_with_diversion, "ts_plot_simple_system_with_diversion.png", columns_to_plot=columns_to_plot)
    save_water_balance_to_csv(simple_system_with_diversion, "balance_table_simple_system_with_diversion.csv")
    vis=WaterSystemVisualizer(simple_system_with_diversion, 'simple_w_diversion')
    vis.plot_network_layout()
    
    print("\n" + "="*50 + "\n")

    # Test: HydroWorks System with 2 Diversions. Two sources feeding into a hydrowokrs diversion water to a ag demand site and a second diversion. Return flow from the ag demand site enters that second diversion for the total flow to end up in one sink.
    hydroworks_system_2_diversions = create_hydroworks_system_2_diversions()
    print("HydroWorks System Test:")
    num_time_steps = 12
    hydroworks_system_2_diversions.simulate(num_time_steps)
    plot_water_balance_time_series(hydroworks_system_2_diversions, "ts_plot_hydroworks_2_diversions_system.png")
    save_water_balance_to_csv(hydroworks_system_2_diversions, "balance_table_hydroworks_2_diversions_system.csv")
    vis=WaterSystemVisualizer(hydroworks_system_2_diversions, 'HW_2_diversion')
    vis.plot_network_layout()

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
    
    print("\n" + "="*50 + "\n")

    # Test: Seasonal Reservoir. Fully seasonal system.
    seasonal_system = create_seasonal_reservoir_system()
    num_time_steps = 12 * 2  # 2 years of monthly data

    print("Running Seasonal Reservoir Test")

    seasonal_system.simulate(num_time_steps)
    # Generate water balance table
    balance_table = seasonal_system.get_water_balance_table()
    balance_table.to_csv("balance_table_seasonal_reservoir_system.csv", index=False)
    print("\nWater balance table saved to 'balance_table)_seasonal_reservoir_system.csv'")

    # Visualize the system
    columns_to_plot = [
    "LargeReservoir_Inflow",
    "LargeReservoir_Outflow",
    "LargeReservoir_Storage",
    "LargeReservoir_ExcessVolume",
    "SeasonalDemand_Demand",
    "SeasonalDemand_SatisfiedDemand"
    ]

    plot_water_balance_time_series(seasonal_system, "ts_plot_seasonal_reservoir_system.png", columns_to_plot)
    print("System layout visualization saved to 'seasonal_reservoir_test_layout.png'")
    vis=WaterSystemVisualizer(seasonal_system, 'seasonal')
    vis.plot_network_layout()
    
# Run the sample tests
if __name__ == "__main__":
  run_sample_tests()