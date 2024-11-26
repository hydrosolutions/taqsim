import csv
import math
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer

def create_seasonal_ZRB_system():
    """
    Creates a test water system with a seasonal supply, a large reservoir,
    a seasonal demand, and a sink node. The system runs for 10 years with monthly time steps.

    Returns:
        WaterSystem: The configured water system for testing.
    """
    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    num_time_steps = 12 * 2  # 5 years of monthly data
    start_year = 2017
    start_month = 1
    system = WaterSystem(dt=dt)

    # Create nodes
    supply = SupplyNode("MountainSource", easting=381835,northing=4374682, csv_file="data/Inflow_Rovatkhodzha_monthly_2010_2023_ts.csv", start_year= start_year, start_month=start_month, num_time_steps=num_time_steps)
    # HydroWorks Nodes
    HW_Ravadhoza = HydroWorks("HW-Ravadhoza", easting=363094.43,northing=4377810.64)
    HW_AkKaraDarya = HydroWorks("HW-AkKaraDarya", easting=333156.64,northing=4395650.43)
    HW_Damkodzha = HydroWorks("HW_Damkodzha", easting=284720.68, northing=4417759.40)
    HW_Narpay = HydroWorks("HW-Narpay", easting=270403.55,northing=4424501.92)
    HW_Confluence=HydroWorks("HW-Confluence", easting=239889.6,northing=4433214.0)
    HW_Karmana=HydroWorks("HW-Karmana", easting=209334.3,northing=4448118.7)
    # Demand Nodes
    D1 = DemandNode("Irrigation-Dargom", demand_rates=generate_seasonal_demand(num_time_steps),easting=289406,northing=4398839)
    D2 = DemandNode("Irrigation-Mirzapay", demand_rates=generate_seasonal_demand(num_time_steps),easting=344919,northing=4403848)
    D3 = DemandNode("Irrigation-Akdarya", demand_rates=generate_seasonal_demand(num_time_steps),easting=314014,northing=4413771)
    D4 = DemandNode("Irrigation-Narpay", demand_rates=generate_seasonal_demand(num_time_steps),easting=233633,northing=4424363)
    D5 = DemandNode("Irrigation-MiankalToss", demand_rates=generate_seasonal_demand(num_time_steps),easting=234815,northing=4443998)
    D6 = DemandNode("Irrigation-Karmana-Konimex", demand_rates=generate_seasonal_demand(num_time_steps),easting=183378,northing=4462461)
    D7 = DemandNode("Navoi-Powerplant", demand_rates=25,easting=186146.3,northing=4451659.3)
    # Reservoir
    RES_Kattakurgan =StorageNode("RES-Kattakurgan",hva_file='./data/Kattakurgan_H_V_A.csv',easting=265377.2,northing= 4414217.5, initial_storage=4e7,
                                 evaporation_file='./data/Reservoir_ET_2010_2023.csv', start_year=start_year, start_month=start_month, num_time_steps=num_time_steps)
    RES_AkDarya = StorageNode("RES-Akdarya", hva_file='./data/Akdarya_H_V_A.csv' ,easting= 274383.7,northing=4432954.7, initial_storage=5e6,
                              evaporation_file='./data/Reservoir_ET_2010_2023.csv', start_year=start_year, start_month=start_month, num_time_steps=num_time_steps)
    
    print('Reservoir Information: ')
    print(f'Akdarya: {RES_AkDarya.get_reservoir_info()}')
    print(f'Kattakurgan: {RES_Kattakurgan.get_reservoir_info()}')
    
    # Sink Nodes
    sink_tuyatortor = SinkNode("TuyaTortor", easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("EskiAnkhor", easting=286019.5,northing=4384078.7)
    sink_downstream = SinkNode("Sink-Navoi", easting=176079,northing=4454183)

    # Add nodes to the system
    supply_node = [supply]  # List of supply nodes
    reservoir = [RES_Kattakurgan, RES_AkDarya]  # List of reservoir nodes
    hydroworks = [HW_Ravadhoza, HW_AkKaraDarya, HW_Damkodzha, HW_Narpay, HW_Confluence, HW_Karmana]  # List of agricultural demand nodes
    demand = [D1, D2, D3, D4, D5, D6, D7]  # List of domestic demand nodes
    sink_node = [sink_tuyatortor, sink_eskiankhor, sink_downstream]  # List of sink nodes

    # Iterate through each category and add nodes to the system
    for node in supply_node + reservoir + hydroworks + demand + sink_node:
        system.add_node(node)


    # Connect nodes with edges
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=885))
    system.add_edge(Edge(HW_Ravadhoza, sink_tuyatortor, capacity=50))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=885))  
    system.add_edge(Edge(HW_Ravadhoza, sink_eskiankhor, capacity=60))
    system.add_edge(Edge(HW_Ravadhoza, D2, capacity=125, length=268.6))
    system.add_edge(Edge(HW_Ravadhoza, D1, capacity=40, length=205))
 
    system.add_edge(Edge(HW_AkKaraDarya, HW_Damkodzha, capacity=550))
    system.add_edge(Edge(HW_AkKaraDarya, D3, capacity=230, length=79.9))

    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=20))
    system.add_edge(Edge(HW_Damkodzha, RES_Kattakurgan, capacity=100))
    system.add_edge(Edge(HW_Damkodzha, HW_Narpay, capacity=80))
    system.add_edge(Edge(HW_Damkodzha, HW_Confluence, capacity=350))
    system.add_edge(Edge(RES_Kattakurgan, HW_Narpay, capacity=20))
    
    system.add_edge(Edge(HW_Narpay, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_Narpay, D4, capacity=80, length=53.8))
    
    system.add_edge(Edge(HW_Confluence, HW_Karmana, capacity=600))
    
    system.add_edge(Edge(HW_Karmana, sink_downstream, capacity=550))
    system.add_edge(Edge(HW_Damkodzha, D5, capacity=54, length=102.4))
    system.add_edge(Edge(HW_Karmana, D6, capacity=60))
    system.add_edge(Edge(HW_Karmana, D7, capacity=30))

    # Demand return flows
    system.add_edge(Edge(D1, HW_Damkodzha, capacity=40))
    system.add_edge(Edge(D2, RES_AkDarya, capacity=40))
    system.add_edge(Edge(D3, RES_AkDarya, capacity=230))
    system.add_edge(Edge(D4, HW_Karmana, capacity=40))
    system.add_edge(Edge(D5,HW_Karmana, capacity=40))
    system.add_edge(Edge(D6,sink_downstream, capacity=40))
    system.add_edge(Edge(D7,sink_downstream, capacity=40))    

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

def run_sample_tests():

    print("\n" + "="*50 + "\n")

    # Test: Super Simple System. This is a simple linear system with one source, one demand site, and one sink
    ZRB_system = create_seasonal_ZRB_system()
    print("ZRB system test:")
    num_time_steps = 2*12
    ZRB_system.simulate(num_time_steps)

    vis_ZRB=WaterSystemVisualizer(ZRB_system, 'ZRB')
    vis_ZRB.plot_network_layout()
    

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

    plot_water_balance_time_series(ZRB_system, "ts_plot_ZRB_system.png")
    save_water_balance_to_csv(ZRB_system, "balance_table_ZRB_system.csv")

# Run the sample tests
if __name__ == "__main__":
  run_sample_tests()
  
 