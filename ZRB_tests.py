import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import os
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer

def create_seasonal_ZRB_system(start_year, start_month, num_time_steps):
    """
    Creates a test water system with a seasonal supply, a large reservoir,
    a seasonal demand, and a sink node. The system runs for 10 years with monthly time steps.

    Returns:
        WaterSystem: The configured water system for testing.
    """
    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
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
    
    # Creates demand nodes from csv file DemandNode_Info.csv (columns: name,utm_easting,utm_northing,longitude,latitude,csv_path)
    Demand_info = pd.read_csv('./data/DemandNode_Info.csv', sep=',')
    demand_nodes = []
    for index, row in Demand_info.iterrows():
        demand_node = DemandNode(
            row['name'],
            easting=row['utm_easting'],
            northing=row['utm_northing'],
            csv_file=f"./data/ETblue/{row['csv_path']}",
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            field_efficiency=0.75,
            conveyance_efficiency=0.65
        )
        demand_nodes.append(demand_node)
    
    # Unpack demand nodes into individual variables based on Demand_info['name'] row
    for index, row in Demand_info.iterrows():
        globals()[row['name']] = demand_nodes[index]
        print(f"Created demand node: {row['name']}")

    # Demand Thermal powerplant Navoi (25 m³/s)
    Powerplant = DemandNode("Navoi-Powerplant", demand_rates=25,easting=186146.3,northing=4451659.3)
    Jizzakh = DemandNode("Eski Tuyatortor", easting=376882.3,northing=4401307.9, csv_file='./data/Sink_Eski Tuyatortor_monthly_2000_2022.csv', start_year=start_year, start_month=start_month, num_time_steps=num_time_steps)
    Kashkadarya = DemandNode("Eski Ankhor", easting=272551,northing=4371872, csv_file='./data/Sink_Eski Ankhor_monthly_2000_2022.csv', start_year=start_year, start_month=start_month, num_time_steps=num_time_steps)

    # Reservoir
    release_params_kattakurgan = {
        'h1': 500.0,
        'h2': 511.0,
        'w': 5.0,
        'm1': 1.5,
        'm2': 1.5,
    }
    release_params_akdarya = {
        'h1': 486.0,
        'h2': 496.0,
        'w': 10.0,
        'm1': 1.5,
        'm2': 1.5,
    }
    RES_Kattakurgan =StorageNode("RES-Kattakurgan",hva_file='./data/Kattakurgan_H_V_A.csv',easting=265377.2,northing= 4414217.5, initial_storage=4e8,
                                 evaporation_file='./data/Reservoir_ET_2010_2023.csv', start_year=start_year, start_month=start_month, 
                                 num_time_steps=num_time_steps, release_params=release_params_kattakurgan)
    RES_AkDarya = StorageNode("RES-Akdarya", hva_file='./data/Akdarya_H_V_A.csv' ,easting= 274383.7,northing=4432954.7, initial_storage=6e7, 
                              evaporation_file='./data/Reservoir_ET_2010_2023.csv', start_year=start_year, start_month=start_month, 
                              num_time_steps=num_time_steps, release_params=release_params_akdarya)

    
    # Sink Nodes
    sink_tuyatortor = SinkNode("Jizzakh", easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("Kashkadarya", easting=272551,northing=4361872)
    sink_downstream = SinkNode("Sink-Navoi", easting=153771,northing=4454402)

    # Add nodes to the system
    supply_node = [supply]  # List of supply nodes
    reservoir = [RES_Kattakurgan, RES_AkDarya]  # List of reservoir nodes
    hydroworks = [HW_Ravadhoza, HW_AkKaraDarya, HW_Damkodzha, HW_Narpay, HW_Confluence, HW_Karmana]  # List of agricultural demand nodes
    demand_node = [Bulungur, Ishtixon, Jomboy, Karmana, Kattaqorgon, Narpay, Navbahor, Nurobod, Oqdaryo, Pastdargom, Paxtachi, Payariq, Samarqand, Toyloq, Urgut, Xatirchi, Powerplant, Jizzakh, Kashkadarya]  # List of demand nodes
    sink_node = [sink_tuyatortor, sink_eskiankhor, sink_downstream]  # List of sink nodes

    # Iterate through each category and add nodes to the system
    for node in supply_node + demand_node + reservoir + hydroworks + sink_node:
        system.add_node(node)

    # Connect nodes with edges
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=1350))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=850))
   
    # Supply for Bulungur, Jomboy and Payriq (and Jizzakh-Region)
    system.add_edge(Edge(HW_Ravadhoza, Bulungur, capacity=125, length=50.1))
    system.add_edge(Edge(Bulungur, Jomboy, capacity=70, length=152.8))
    system.add_edge(Edge(HW_Ravadhoza, Jizzakh, capacity=55, length=97.7))
    system.add_edge(Edge(Jizzakh, sink_tuyatortor, capacity=55))
    system.add_edge(Edge(Jomboy, Payariq, capacity=70, length=97.7))
    # Supply for Toyloq, Urgut, Samarqand
    system.add_edge(Edge(HW_Ravadhoza, Toyloq, capacity=80, length=32.6))
    system.add_edge(Edge(HW_Ravadhoza, Urgut, capacity=125, length=99.0))
    system.add_edge(Edge(HW_Ravadhoza, Nurobod, capacity=80, length=42.6))
    system.add_edge(Edge(Toyloq, Samarqand, capacity=80, length=42.6))
    system.add_edge(Edge(Urgut, Samarqand, capacity=125))
    system.add_edge(Edge(Samarqand, Pastdargom, capacity=205, length=280.5))
    system.add_edge(Edge(Nurobod, Kashkadarya, capacity=80))
    system.add_edge(Edge(Kashkadarya, sink_eskiankhor, capacity=60))
    system.add_edge(Edge(Pastdargom, HW_Damkodzha, capacity=205))
    # HW_AkKaraDarya
    system.add_edge(Edge(HW_AkKaraDarya, Oqdaryo, capacity=230, length=64.3))
    system.add_edge(Edge(HW_AkKaraDarya, HW_Damkodzha, capacity=550))
    system.add_edge(Edge(Oqdaryo, Payariq, capacity=50))
    system.add_edge(Edge(Payariq, Ishtixon, capacity=100, length=63.0))
    system.add_edge(Edge(Ishtixon, RES_AkDarya, capacity=230))
    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=20))

    # Damkodzha
    system.add_edge(Edge(HW_Damkodzha, RES_Kattakurgan, capacity=100))
    system.add_edge(Edge(HW_Damkodzha, HW_Narpay, capacity=80))
    system.add_edge(Edge(HW_Damkodzha, HW_Confluence, capacity=350))
    system.add_edge(Edge(HW_Damkodzha, Kattaqorgon, capacity=54, length=159.9))
    system.add_edge(Edge(Kattaqorgon, Xatirchi, capacity=94))
    system.add_edge(Edge(Xatirchi, HW_Karmana, capacity=94))

    system.add_edge(Edge(RES_Kattakurgan, HW_Narpay, capacity=20))

    # HW_Narpay
    system.add_edge(Edge(HW_Narpay, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_Narpay, Narpay, capacity=80, length=53.3))
    system.add_edge(Edge(HW_Narpay, Kattaqorgon, capacity=40))
    system.add_edge(Edge(Narpay, Paxtachi, capacity=80, length=78.9))
    # Downstream
    system.add_edge(Edge(Paxtachi, Karmana, capacity=800))
    system.add_edge(Edge(Karmana, sink_downstream  , capacity=80))
    system.add_edge(Edge(HW_Confluence, HW_Karmana, capacity=400))

    # HW_Karmana
    system.add_edge(Edge(HW_Karmana, Navbahor, capacity=45))
    system.add_edge(Edge(Navbahor, sink_downstream, capacity=45))
    system.add_edge(Edge(HW_Karmana, sink_downstream, capacity=300))
    system.add_edge(Edge(HW_Karmana, Powerplant, capacity=35))
    system.add_edge(Edge(Powerplant, sink_downstream, capacity=35))
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

def run_sample_tests():

    print("\n" + "="*50 + "\n")

    # Definition of simulation period
    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 3  # 3 years of monthly data
    ZRB_system = create_seasonal_ZRB_system(start_year, start_month, num_time_steps)

    print("ZRB system simulation:")
    ZRB_system.simulate(num_time_steps)
    print("Simulation complete")
    
    print('ZRB system visualization:')
    save_water_balance_to_csv(ZRB_system, "balance_table_ZRB_system.csv")
    vis_ZRB=WaterSystemVisualizer(ZRB_system, 'ZRB')
    vis_ZRB.print_water_balance_summary()
    vis_ZRB.plot_reservoir_dynamics()
    vis_ZRB.plot_network_layout()

    # Get the storage node from the system's graph
    storage_node = ZRB_system.graph.nodes['RES-Akdarya']['node']
    vis_ZRB.plot_release_function(storage_node)
    storage_node = ZRB_system.graph.nodes['RES-Kattakurgan']['node']
    vis_ZRB.plot_release_function(storage_node)
    
    vis_ZRB.plot_demand_deficit_heatmap()
    vis_ZRB.plot_storage_dynamics()
    vis_ZRB.plot_network_layout()

    html_file = vis_ZRB.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')

    print("Visualizations complete")

# Run the sample tests
if __name__ == "__main__":
  run_sample_tests()
  