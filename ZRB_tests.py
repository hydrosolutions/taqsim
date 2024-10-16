import csv
import math
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge

def create_seasonal_ZRB_system():
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
    supply = SupplyNode("MountainSource", supply_rates=import_supply_data("data/Inflow_Rovatkhodzha_monthly_2010_2023_ts.csv", 2010, 1, num_time_steps), easting=381835,northing=4374682)
    # HydroWorks Nodes
    HW_Ravadhoza = HydroWorks("HW-Ravadhoza", easting=363094.43,northing=4377810.64)
    HW_AkKaraDarya = HydroWorks("HW-AkKaraDarya", easting=333156.64,northing=4395650.43)
    HW_dargom = HydroWorks("HW-Dargom", easting=317022.3, northing=4391315.3)
    HW_Tuyatortor = HydroWorks("HW-Tuyatortor", easting=369358.0,northing=4397125.3)
    HW_Damkodzha = HydroWorks("HW_Damkodzha", easting=284720.68, northing=4417759.40)
    HW_AkDarya = HydroWorks("HW_AkDarya", easting=310395,northing=4420338)
    HW_Narpay = HydroWorks("HW-Narpay", easting=270403.55,northing=4424501.92)
    HW_Confluence=HydroWorks("HW-Confluence", easting=239889.6,northing=4433214.0)
    HW_Karmana=HydroWorks("HW-Karmana", easting=209334.3,northing=4448118.7)
    # Demand Nodes
    D1 = DemandNode("D1", demand_rates=generate_seasonal_demand(num_time_steps),easting=289406,northing=4398839)
    D2 = DemandNode("D2", demand_rates=generate_seasonal_demand(num_time_steps),easting=344919,northing=4403848)
    D3 = DemandNode("D3", demand_rates=generate_seasonal_demand(num_time_steps),easting=300105,northing=4415798)
    D4 = DemandNode("D4", demand_rates=generate_seasonal_demand(num_time_steps),easting=233633,northing=4424363)
    D5 = DemandNode("D5", demand_rates=generate_seasonal_demand(num_time_steps),easting=198042,northing=4446583)
    D6 = DemandNode("D6", demand_rates=generate_seasonal_demand(num_time_steps),easting=182974,northing=4461910)
    
    # Reservoir
    RES_Kattakurgan =StorageNode("RES-Kattakurgan",easting=265377.2,northing= 4414217.5, capacity=1e8, initial_storage=5e7)
    RES_AkDarya = StorageNode("RES-AkDarya", easting= 274383.7,northing=4432954.7, capacity=4.55e8, initial_storage=4e8)
    # Sink Nodes
    sink_tuyatortor = SinkNode("TuyaTortor", easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("EskiAnkhor", easting=286019.5,northing=4384078.7)
    sink_downstream = SinkNode("Outflow-Navoi", easting=176079,northing=4454183)

    # Add nodes to the system
    system.add_node(supply)
    system.add_node(HW_Ravadhoza)
    system.add_node(HW_AkKaraDarya)
    system.add_node(HW_dargom)
    system.add_node(HW_Tuyatortor)
    system.add_node(HW_Damkodzha)
    system.add_node(HW_AkDarya)
    system.add_node(HW_Narpay)
    system.add_node(HW_Confluence)
    system.add_node(HW_Karmana)
    system.add_node(RES_Kattakurgan)
    system.add_node(RES_AkDarya)
    system.add_node(sink_tuyatortor)
    system.add_node(sink_eskiankhor)
    system.add_node(sink_downstream)
    system.add_node(D1)
    system.add_node(D2)
    system.add_node(D3)
    system.add_node(D4)
    system.add_node(D5)
    system.add_node(D6)


    # Connect nodes with edges
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=400))
    system.add_edge(Edge(HW_Ravadhoza, HW_Tuyatortor, capacity=100))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=150))  
    system.add_edge(Edge(HW_Ravadhoza, HW_dargom, capacity=150))
    
    system.add_edge(Edge(HW_Tuyatortor, sink_tuyatortor, capacity=100))  
    system.add_edge(Edge(HW_dargom, sink_eskiankhor, capacity=150))

    system.add_edge(Edge(HW_AkKaraDarya, HW_AkDarya, capacity=50))  
    system.add_edge(Edge(HW_AkKaraDarya, HW_Damkodzha, capacity=100))

    system.add_edge(Edge(HW_AkDarya, RES_AkDarya, capacity=50))
    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=50))

    system.add_edge(Edge(HW_Damkodzha, RES_Kattakurgan, capacity=100))
    system.add_edge(Edge(RES_Kattakurgan, HW_Narpay, capacity=70))
    system.add_edge(Edge(HW_Narpay, HW_Confluence, capacity=70))

    system.add_edge(Edge(HW_Confluence, HW_Karmana, capacity=150))
    system.add_edge(Edge(HW_Karmana, sink_downstream, capacity=150))

    system.add_edge(Edge(HW_dargom, D1, capacity=40))
    system.add_edge(Edge(D1, HW_Damkodzha, capacity=40))
    system.add_edge(Edge(HW_Tuyatortor, D2, capacity=40))
    system.add_edge(Edge(D2, HW_AkDarya, capacity=40))
    system.add_edge(Edge(HW_AkKaraDarya, D3, capacity=40))
    system.add_edge(Edge(D3, HW_Confluence, capacity=40))
    system.add_edge(Edge(HW_Narpay, D4, capacity=40))
    system.add_edge(Edge(D4, HW_Karmana, capacity=40))
    system.add_edge(Edge(HW_Karmana, D5, capacity=40))
    system.add_edge(Edge(D5,sink_downstream, capacity=40))
    system.add_edge(Edge(HW_Karmana, D6, capacity=40))
    
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
    base_supply = 175  # m³/s
    amplitude = 125    # m³/s
    supply_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = -math.cos(2 * math.pi * (month+2) / 12)
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
    amplitude = 20    # m³/s
    demand_rates = []
    for t in range(num_time_steps):
        month = t % 12
        seasonal_factor = -math.cos(2 * math.pi * month / 12)  # Peak in summer, trough in winter
        demand_rate = base_demand + amplitude * seasonal_factor
        demand_rates.append(max(0, demand_rate))  # Ensure non-negative demand
    return demand_rates

def import_supply_data(csv_file, start_year, start_month, num_time_steps):
    # Read the CSV file into a pandas DataFrame
    supply = pd.read_csv(csv_file, parse_dates=['Date'])
    
    # Filter the DataFrame to find the start point
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    end_date = start_date+pd.DateOffset(months=num_time_steps)
    supply = supply[(supply['Date'] >= start_date) & (supply["Date"]< end_date)]
    
    
    # Check if the start date can be found in the data
    if supply["Date"].iloc[0] != start_date:
        print(f"Warning: No data found starting from {start_year}-{start_month:02d}. Please check the input date.")
        raise SystemExit
        return []
    
    # Extract the required number of Q values
    supply_values = supply['Q'].tolist()
    # Check if enough time steps are available
    if len(supply_values) < num_time_steps:
        print(f"Warning: Only {len(supply_values)} time steps are available starting from {start_year}-{start_month:02d}, but {num_time_steps} were requested.")
        raise SystemExit
    return supply_values

def run_sample_tests():

    print("\n" + "="*50 + "\n")

    # Test: Super Simple System. This is a simple linear system with one source, one demand site, and one sink
    ZRB_system = create_seasonal_ZRB_system()
    print("ZRB system test:")
    num_time_steps = 48
    ZRB_system.simulate(num_time_steps)

    # Extract and print results
    reservoir_node = next(data['node'] for _, data in ZRB_system.graph.nodes(data=True) if isinstance(data['node'], StorageNode))
    demand_node = next(data['node'] for _, data in ZRB_system.graph.nodes(data=True) if isinstance(data['node'], DemandNode))

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

    # Visualize the system
    columns_to_plot = [
        "HW-Ravadhoza_Inflow",
        "D1_Inflow",
        "D1_Outflow",
        "D1_Demand",
        "D1_SatisfiedDemand",
        "RES-Kattakurgan_Inflow",
        "RES-Kattakurgan_Outflow",
        "RES-Kattakurgan_Storage",
        "RES-Kattakurgan_ExcessVolume",
    ]

    plot_water_balance_time_series(ZRB_system, "ts_plot_ZRB_system.png", columns_to_plot=columns_to_plot)
    save_water_balance_to_csv(ZRB_system, "balance_table_ZRB_system.csv")
    ZRB_system.visualize("nw_plot_ZRB_system.png", display=False)


# Run the sample tests
if __name__ == "__main__":
  run_sample_tests()
  
 