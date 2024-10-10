import csv
import math
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge

def create_most_simple_system():
    """
    Creates the most simple system with one supply, one demand, and one sink node 
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt=dt)

    supply = SupplyNode("Inflow", default_supply_rate=100)
    agriculture = DemandNode("Agriculture", demand_rates=80)
    outflow = SinkNode("Outflow")

    system.add_node(supply)
    system.add_node(agriculture)
    system.add_node(outflow)

    system.add_edge(Edge(supply, agriculture, capacity=100))
    system.add_edge(Edge(agriculture, outflow, capacity=50))

    return system

def create_hydroworks_system():
    """
    Creates a system with one supply, three HydroWorks nodes, and one sink node
    """

    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)

    supplyA = SupplyNode("River A", default_supply_rate=100)
    supplyB = SupplyNode("River B", default_supply_rate=50)
    hydroworks1 = HydroWorks("HydroWorks1")
    hydroworks2 = HydroWorks("HydroWorks2")
    demandsite = DemandNode("Agriculture", demand_rates=40)
    hydroworks4 = HydroWorks("HydroWorks4")
    outflow = SinkNode("Outflow")

    system.add_node(supplyA)
    system.add_node(supplyB)
    system.add_node(hydroworks1)
    system.add_node(hydroworks2)
    system.add_node(demandsite)
    system.add_node(hydroworks4)
    system.add_node(outflow)

    system.add_edge(Edge(supplyA, hydroworks1, capacity=100))
    system.add_edge(Edge(supplyB, hydroworks1, capacity=100))
    system.add_edge(Edge(hydroworks1, hydroworks2, capacity=50))
    system.add_edge(Edge(hydroworks2, hydroworks4, capacity=50))
    system.add_edge(Edge(hydroworks1, demandsite, capacity=50))
    system.add_edge(Edge(demandsite, hydroworks4, capacity=50))
    system.add_edge(Edge(hydroworks4, outflow, capacity=100))

    return system

def create_simple_system():
    """
    Creates a simple water system with one supply, one storage, two demands, and one sink.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)
    
    supply = SupplyNode("MainSupply", default_supply_rate=100)
    reservoir = StorageNode("MainReservoir", capacity=500)
    agriculture = DemandNode("Agriculture", demand_rates=60)
    urban = DemandNode("Urban", demand_rates=30)
    sink = SinkNode("Sink")
    
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

def create_complex_system():
    """
    Creates a more complex water system with multiple supplies, storages, and demands.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)
    
    supply1 = SupplyNode("MountainSupply", default_supply_rate=150)
    supply2 = SupplyNode("RiverSupply", default_supply_rate=100)
    reservoir1 = StorageNode("MountainReservoir", capacity=1000)
    reservoir2 = StorageNode("ValleyReservoir", capacity=800)
    agriculture1 = DemandNode("Farmland1", demand_rates=80)
    agriculture2 = DemandNode("Farmland2", demand_rates=70)
    urban1 = DemandNode("City1", demand_rates=50)
    urban2 = DemandNode("City2", demand_rates=40)
    industry = DemandNode("IndustrialPark", demand_rates=30)
    sink = SinkNode("RiverMouth")
    
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

def create_drought_system():
    """
    Creates a system to test drought conditions with variable supply.
    """
    dt = 30.44 * 24 * 3600 # Average month in seconds
    system = WaterSystem(dt)
    
    # Alternating between normal (100) and drought (20) conditions
    variable_supply = [100 if i % 2 == 0 else 20 for i in range(50)]
    supply = SupplyNode("VariableSupply", supply_rates=variable_supply)
    reservoir = StorageNode("EmergencyReservoir", capacity=500)
    city = DemandNode("DroughtCity", demand_rates=80)
    agriculture = DemandNode("DroughtAgriculture", demand_rates=40)
    sink = SinkNode("Sink")
    
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(city)
    system.add_node(agriculture)
    system.add_node(sink)
    
    system.add_edge(Edge(supply, reservoir, capacity=150))
    system.add_edge(Edge(reservoir, city, capacity=90))
    system.add_edge(Edge(reservoir, agriculture, capacity=50))
    system.add_edge(Edge(city, sink, capacity=100))
    system.add_edge(Edge(agriculture, sink, capacity=60))
    
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
    supply = SupplyNode("MountainSource", supply_rates=generate_seasonal_supply(num_time_steps))
    reservoir = StorageNode("LargeReservoir", capacity=1e9, initial_storage=5e8)  # 1 billion m³ capacity, start half full
    demand = DemandNode("SeasonalDemand", demand_rates=generate_seasonal_demand(num_time_steps))
    sink = SinkNode("RiverMouth")

    # Add nodes to the system
    system.add_node(supply)
    system.add_node(reservoir)
    system.add_node(demand)
    system.add_node(sink)

    # Connect nodes with edges
    system.add_edge(Edge(supply, reservoir, capacity=100))  # 100 m³/s max flow from supply to reservoir
    system.add_edge(Edge(reservoir, demand, capacity=80))   # 80 m³/s max flow from reservoir to demand
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

def plot_water_system(water_system, filename, columns_to_plot=None):
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
    
    # Test super simple system
    super_simple_system = create_most_simple_system()
    print("Super simple system test:")
    num_time_steps = 12
    super_simple_system.simulate(num_time_steps)
    plot_water_system(super_simple_system, "super_simple_time_series_plot.png")
    save_water_balance_to_csv(super_simple_system, "super_simple_system_balance.csv")
    super_simple_system.visualize("super_simple_system.png", display=False)

    print("\n" + "="*50 + "\n")

    # Test simple system
    simple_system = create_simple_system()
    print("Simple System Test:")
    num_time_steps = 12
    simple_system.simulate(num_time_steps)
    save_water_balance_to_csv(simple_system, "simple_system_balance.csv")
    simple_system.visualize("simple_system.png", display=False)

    print("\n" + "="*50 + "\n")

    # Test complex system
    complex_system = create_complex_system()
    print("Complex System Test:")
    num_time_steps = 36
    complex_system.simulate(num_time_steps)
    save_water_balance_to_csv(complex_system, "complex_system_balance.csv")
    complex_system.visualize("complex_system.png", display=False)

    print("\n" + "="*50 + "\n")

    # Test drought system
    drought_system = create_drought_system()
    print("Drought System Test:")
    num_time_steps = 120
    drought_system.simulate(num_time_steps)
    save_water_balance_to_csv(drought_system, "drought_system_balance.csv")
    drought_system.visualize("drought_system.png", display=False)

    print("\n" + "="*50 + "\n")

    # Test HydroWorks system
    hydroworks_system = create_hydroworks_system()
    print("HydroWorks System Test:")
    num_time_steps = 12
    hydroworks_system.simulate(num_time_steps)
    save_water_balance_to_csv(hydroworks_system, "hydroworks_system_balance.csv")
    hydroworks_system.visualize("hydroworks_system_balance.png",display=False)

    print("\n" + "="*50 + "\n")

    # Test seasonal reservoir
    seasonal_system = create_seasonal_reservoir_system()
    num_time_steps = 12 * 2  # 2 years of monthly data
    
    print("Running Seasonal Reservoir Test")
    
    seasonal_system.simulate(num_time_steps)
    
     # Extract and print results
    reservoir_node = next(data['node'] for _, data in seasonal_system.graph.nodes(data=True) if isinstance(data['node'], StorageNode))
    demand_node = next(data['node'] for _, data in seasonal_system.graph.nodes(data=True) if isinstance(data['node'], DemandNode))
    
    print("\nReservoir Storage Levels (every 12 months):")
    for year in range(10):
        month = year * 12
        storage = reservoir_node.get_storage(month)
        print(f"Year {year + 1}: {storage:,.0f} m³")
    
    print("\nDemand Satisfaction (every 12 months):")
    for year in range(10):
        month = year * 12
        satisfied = demand_node.get_satisfied_demand(month)
        total_demand = demand_node.get_demand_rate(month)
        satisfaction_rate = (satisfied / total_demand) * 100 if total_demand > 0 else 100
        print(f"Year {year + 1}: {satisfaction_rate:.2f}% ({satisfied:.2f} / {total_demand:.2f} m³/s)")
    
    # Generate water balance table
    balance_table = seasonal_system.get_water_balance_table()
    balance_table.to_csv("seasonal_reservoir_test_balance.csv", index=False)
    print("\nWater balance table saved to 'seasonal_reservoir_test_balance.csv'")
    
    # Visualize the system
    columns_to_plot = [
        "LargeReservoir_Inflow",
        "LargeReservoir_Outflow",
        "LargeReservoir_Storage",
        "SeasonalDemand_Demand",
        "SeasonalDemand_SatisfiedDemand"
    ]
    plot_water_system(seasonal_system, "seasonal_reservoir_test_time_series.png", columns_to_plot)
    seasonal_system.visualize("seasonal_reservoir_test_layout.png", display=False)
    print("System layout visualization saved to 'seasonal_reservoir_test_layout.png'")


# Run the sample tests
if __name__ == "__main__":
    run_sample_tests()