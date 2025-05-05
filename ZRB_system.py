import pandas as pd
import numpy as np
import webbrowser
import os
import json
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer, MultiGeneticOptimizer
import ctypes
import cProfile
import pstats
import io

def create_ZRB_system_baseline(start_year, start_month, num_time_steps, scenario='', period='', agr_scenario='', efficiency=''):

    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)

    # Create nodes
    supply = SupplyNode("MountainSource", easting=381835,northing=4374682, csv_file="data/baseline/inflow/inflow_ravatkhoza_2010-2023_monthly.csv", start_year= start_year, start_month=start_month, num_time_steps=num_time_steps)
    # HydroWorks Nodes
    HW_Ravadhoza = HydroWorks("HW-Ravatkhoza", easting=363094.43,northing=4377810.64)
    HW_AkKaraDarya = HydroWorks("HW-AkKaraDarya", easting=333156.64,northing=4395650.43)
    HW_Damkodzha = HydroWorks("HW-Damkhoza", easting=284720.68, northing=4417759.40)
    HW_Narpay = HydroWorks("HW-Narpay", easting=270403.55,northing=4424501.92)
    HW_Confluence=HydroWorks("HW-Confluence", easting=239889.6,northing=4433214.0)
    HW_Karmana=HydroWorks("HW-Karmana", easting=209334.3,northing=4448118.7)
    HW_EskiAnkhor=HydroWorks("HW-EskiAnkhor", easting=315015,northing=4390976)
    
    # Creates demand nodes from csv file DemandNode_Info.csv (columns: name,easting,northing,field_efficiency,conveyance_efficiency,weight)
    Demand_info = pd.read_csv('./data/baseline/config/demand_nodes_config.csv', sep=',')
    demand_nodes = []
    for index, row in Demand_info.iterrows():
        demand_node = DemandNode(
            row['name'],
            easting=row['easting'],
            northing=row['northing'],
            csv_file=f"./data/baseline/demand/demand_all_districts_2017-2022_monthly.csv",
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            field_efficiency=row['field_efficiency'],
            conveyance_efficiency=row['conveyance_efficiency'],
            weight=row['weight'],
        )
        demand_nodes.append(demand_node)
    
    # Unpack demand nodes into individual variables based on Demand_info['name'] row
    for index, row in Demand_info.iterrows():
        globals()[row['name']] = demand_nodes[index]
        #print(f"Created demand node: {row['name']}")

    # Demand Thermal powerplant Navoi (25 m³/s)
    Powerplant = DemandNode("Navoiy-Powerplant", demand_rates=25, non_consumptive_rate=17, easting=186146.3,northing=4454459.3, weight=1000)

    # Reservoir
    release_params_kattakurgan = {
        'Vr': 400000000,
        'V1': 300000000,
        'V2': 500000000
    }
    release_params_akdarya = {
        'Vr': 50000000,
        'V1': 3000000,
        'V2': 60000000
    }

    RES_Kattakurgan =StorageNode("RES-Kattakurgan",hv_file='./data/baseline/reservoir/reservoir_kattakurgan_hv.csv',easting=265377.2,northing= 4414217.5, initial_storage=3e8,
                                 evaporation_file='./data/baseline/reservoir/reservoir_evaporation_2017-2022_monthly.csv', start_year=start_year, start_month=start_month, 
                                 num_time_steps=num_time_steps, release_params=release_params_kattakurgan, dead_storage=32e5, buffer_coef=0.2)
    RES_AkDarya = StorageNode("RES-Akdarya", hv_file='./data/baseline/reservoir/reservoir_akdarya_hv.csv' ,easting= 274383.7,northing=4432954.7, initial_storage=4e7, 
                              evaporation_file='./data/baseline/reservoir/reservoir_evaporation_2017-2022_monthly.csv', start_year=start_year, start_month=start_month, 
                              num_time_steps=num_time_steps, release_params=release_params_akdarya, dead_storage=14e5, buffer_coef=0.2)
    
    # Sink Nodes
    sink_tuyatortor = SinkNode("Sink-Jizzakh", min_flow_csv_file='./data/baseline/min_flow/min_flow_jizzakh_2000-2022_monthly.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("Sink-Kashkadarya",min_flow_csv_file='./data/baseline/min_flow/min_flow_kashkadarya_2000-2022_monthly.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=272551,northing=4361872)
    sink_downstream = SinkNode("Sink-Navoi", min_flow_csv_file='./data/baseline/min_flow/min_flow_navoi_1968-2022_monthly.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=153771,northing=4454402)

    # Add nodes to the system
    supply_node = [supply]  # List of supply nodes
    reservoir = [RES_Kattakurgan, RES_AkDarya]  # List of reservoir nodes
    hydroworks = [HW_EskiAnkhor, HW_Ravadhoza, HW_AkKaraDarya, HW_Damkodzha, HW_Narpay, HW_Confluence, HW_Karmana]  # List of agricultural demand nodes
    demand_node = [Bulungur, Ishtixon, Jomboy, Karmana, Kattaqorgon, Narpay, Navbahor, Nurobod, Oqdaryo, Pastdargom, Paxtachi, Payariq, Samarqand, Toyloq, Urgut, Xatirchi, Powerplant]  # List of demand nodes
    sink_node = [sink_tuyatortor, sink_eskiankhor, sink_downstream]  # List of sink nodes

    # Iterate through each category and add nodes to the system
    for node in supply_node + demand_node + reservoir + hydroworks + sink_node:
        system.add_node(node)

    # Add Edges to the system
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=1230))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=885))

    # Supply for Bulungur, Jomboy and Payriq (and Jizzakh-Region)
    system.add_edge(Edge(HW_Ravadhoza, Bulungur, capacity=45))
    system.add_edge(Edge(HW_Ravadhoza, Jomboy, capacity=60))
    system.add_edge(Edge(Bulungur, Jomboy, capacity=45))
    system.add_edge(Edge(Jomboy, Payariq, capacity=105))
    system.add_edge(Edge(HW_Ravadhoza, sink_tuyatortor, capacity=35))


    # Supply for Toyloq, Urgut, Samarqand
    system.add_edge(Edge(HW_Ravadhoza, Toyloq, capacity=80))
    system.add_edge(Edge(Toyloq, Samarqand, capacity=80))
    system.add_edge(Edge(HW_Ravadhoza, Urgut, capacity=125))
    system.add_edge(Edge(Urgut, Samarqand, capacity=125))
    system.add_edge(Edge(Samarqand, HW_EskiAnkhor, capacity=205))
    system.add_edge(Edge(HW_EskiAnkhor, Pastdargom, capacity=150))
    system.add_edge(Edge(Pastdargom, HW_Damkodzha, capacity=150))
    system.add_edge(Edge(HW_EskiAnkhor, Nurobod, capacity=60))
    system.add_edge(Edge(Nurobod, sink_eskiankhor, capacity=60))

    # HW_AkKaraDarya
    system.add_edge(Edge(HW_AkKaraDarya, Oqdaryo, capacity=230))
    system.add_edge(Edge(Oqdaryo, RES_AkDarya, capacity=230))
    system.add_edge(Edge(Payariq, Ishtixon, capacity=105))
    system.add_edge(Edge(Ishtixon, RES_AkDarya, capacity=105))
    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_AkKaraDarya, HW_Damkodzha, capacity=550))

    # Damkodzha
    system.add_edge(Edge(HW_Damkodzha, RES_Kattakurgan, capacity=100))
    system.add_edge(Edge(RES_Kattakurgan, HW_Narpay, capacity=125))
    system.add_edge(Edge(HW_Damkodzha, HW_Narpay, capacity=80))
    system.add_edge(Edge(HW_Damkodzha, HW_Confluence, capacity=350))
    system.add_edge(Edge(HW_Damkodzha, Kattaqorgon, capacity=90))
    system.add_edge(Edge(Kattaqorgon, Xatirchi, capacity=90))
    system.add_edge(Edge(Xatirchi, HW_Karmana, capacity=90))

    # HW_Narpay
    system.add_edge(Edge(HW_Narpay, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_Narpay, Narpay, capacity=80))
    system.add_edge(Edge(Narpay, Paxtachi, capacity=80))
    system.add_edge(Edge(Paxtachi, Karmana, capacity=80))
    system.add_edge(Edge(Karmana, sink_downstream, capacity=80))

    # HW_Confluence
    system.add_edge(Edge(HW_Confluence, HW_Karmana, capacity=400))

    # HW_Karmana
    system.add_edge(Edge(HW_Karmana, Navbahor, capacity=45))
    system.add_edge(Edge(Navbahor, sink_downstream, capacity=45))
    system.add_edge(Edge(HW_Karmana, sink_downstream, capacity=400))
    system.add_edge(Edge(HW_Karmana, Powerplant, capacity=65))
    system.add_edge(Edge(Powerplant, sink_downstream, capacity=65))

    # HW-Ravadhoza distribution
    HW_Ravadhoza.set_distribution_parameters({
         'HW-AkKaraDarya': [0.7]*12,        
         'Toyloq': 0.05,           
         'Urgut': 0.1,
         'Bulungur': 0.05,          
         'Jomboy': 0.05,            
         'Sink-Jizzakh': 0.05               
    })
 
    # HW-EskiAnkhor distribution
    HW_EskiAnkhor.set_distribution_parameters({
         'Pastdargom': 0.610,        # 125/205
         'Nurobod': 0.390            # 80/205
    })
 
     # HW-AkKaraDarya distribution
    HW_AkKaraDarya.set_distribution_parameters({
         'Oqdaryo': 0.353,           # 300/850
         'HW-Damkhoza': 0.647       # 550/850
    })
 
     # HW-Damkhoza distribution
    HW_Damkodzha.set_distribution_parameters({
         'RES-Kattakurgan': 0.171,   # 100/585
         'HW-Narpay': 0.137,         # 80/585
         'HW-Confluence': 0.598,      # 350/585
         'Kattaqorgon': 0.094        # 55/585
    })
 
     # HW-Narpay distribution
    HW_Narpay.set_distribution_parameters({
         'HW-Confluence': 0.510,      # 125/245
         'Narpay': 0.49,            # 80/245
    })
 
     # HW-Confluence distribution
    HW_Confluence.set_distribution_parameters({
         'HW-Karmana': 1.0           # All flow goes to Karmana
    })
 
     # HW-Karmana distribution
    HW_Karmana.set_distribution_parameters({
         'Navbahor': 0.078,          # 45/580
         'Sink-Navoi': 0.862,    # 500/580
         'Navoiy-Powerplant': 0.060         # 35/580
    })

    return system

def create_ZRB_system_scenarios(start_year, start_month, num_time_steps, scenario='', period='', agr_scenario='', efficiency=''):

    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)

    # Create nodes
    supply = SupplyNode("MountainSource", easting=381835,northing=4374682, csv_file=f"data/scenarios/inflow/inflow_ravatkhoza_{scenario}_2012-2099.csv", start_year= start_year, start_month=start_month, num_time_steps=num_time_steps)
    # HydroWorks Nodes
    HW_Ravadhoza = HydroWorks("HW-Ravatkhoza", easting=363094.43,northing=4377810.64)
    HW_AkKaraDarya = HydroWorks("HW-AkKaraDarya", easting=333156.64,northing=4395650.43)
    HW_Damkodzha = HydroWorks("HW-Damkhoza", easting=284720.68, northing=4417759.40)
    HW_Narpay = HydroWorks("HW-Narpay", easting=270403.55,northing=4424501.92)
    HW_Confluence=HydroWorks("HW-Confluence", easting=239889.6,northing=4433214.0)
    HW_Karmana=HydroWorks("HW-Karmana", easting=209334.3,northing=4448118.7)
    HW_EskiAnkhor=HydroWorks("HW-EskiAnkhor", easting=315015,northing=4390976)
    
    # Creates demand nodes from csv file DemandNode_Info.csv (columns: name,easting,northing,field_efficiency,conveyance_efficiency,weight)
    Demand_info = pd.read_csv(f'./data/scenarios/config/demand_nodes_config_{efficiency}.csv', sep=',')
    demand_nodes = []
    for index, row in Demand_info.iterrows():
        demand_node = DemandNode(
            row['name'],
            easting=row['easting'],
            northing=row['northing'],
            csv_file=f"./data/scenarios/demand/demand_{scenario}_{agr_scenario}_{period}.csv",
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            field_efficiency=row['field_efficiency'],
            conveyance_efficiency=row['conveyance_efficiency'],
            weight=row['weight'],
        )
        demand_nodes.append(demand_node)
    
    # Unpack demand nodes into individual variables based on Demand_info['name'] row
    for index, row in Demand_info.iterrows():
        globals()[row['name']] = demand_nodes[index]
        #print(f"Created demand node: {row['name']}")

    # Demand Thermal powerplant Navoi (25 m³/s)
    Powerplant = DemandNode("Navoiy-Powerplant", demand_rates=25, non_consumptive_rate=17, easting=186146.3,northing=4454459.3, weight=1000)

    # Reservoir
    release_params_kattakurgan = {
        'h1': 504.0,
        'h2': 511.0,
        'w': 5.0,
        'm1': 1.5,
        'm2': 1.5,
    }
    release_params_akdarya = {
        'h1': 489.0,
        'h2': 495.0,
        'w': 30.0,
        'm1': 1.51,
        'm2': 1.54,
    }
    RES_Kattakurgan =StorageNode("RES-Kattakurgan",hv_file='./data/baseline/reservoir/reservoir_kattakurgan_hv.csv',easting=265377.2,northing= 4414217.5, initial_storage=3e8,
                                 evaporation_file=f'./data/scenarios/reservoir/reservoir_evaporation_{scenario}_2015-2100.csv', start_year=start_year, start_month=start_month, 
                                 num_time_steps=num_time_steps, release_params=release_params_kattakurgan, dead_storage=32e5)
    RES_AkDarya = StorageNode("RES-Akdarya", hv_file='./data/baseline/reservoir/reservoir_akdarya_hv.csv' ,easting= 274383.7,northing=4432954.7, initial_storage=4e7, 
                              evaporation_file=f'./data/scenarios/reservoir/reservoir_evaporation_{scenario}_2015-2100.csv', start_year=start_year, start_month=start_month, 
                              num_time_steps=num_time_steps, release_params=release_params_akdarya, dead_storage=14e5)
    
    # Sink Nodes
    sink_tuyatortor = SinkNode("Sink-Jizzakh", min_flow_csv_file='./data/scenarios/min_flow/min_flow_jizzakh_2015-2100.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("Sink-Kashkadarya",min_flow_csv_file='./data/scenarios/min_flow/min_flow_kashkadarya_2015-2100.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=272551,northing=4361872)
    sink_downstream = SinkNode("Sink-Navoi", min_flow_csv_file='./data/scenarios/min_flow/min_flow_navoi_2015-2100.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=153771,northing=4454402)

    # Add nodes to the system
    supply_node = [supply]  # List of supply nodes
    reservoir = [RES_Kattakurgan, RES_AkDarya]  # List of reservoir nodes
    hydroworks = [HW_EskiAnkhor, HW_Ravadhoza, HW_AkKaraDarya, HW_Damkodzha, HW_Narpay, HW_Confluence, HW_Karmana]  # List of agricultural demand nodes
    demand_node = [Bulungur, Ishtixon, Jomboy, Karmana, Kattaqorgon, Narpay, Navbahor, Nurobod, Oqdaryo, Pastdargom, Paxtachi, Payariq, Samarqand, Toyloq, Urgut, Xatirchi, Powerplant]  # List of demand nodes
    sink_node = [sink_tuyatortor, sink_eskiankhor, sink_downstream]  # List of sink nodes

    # Iterate through each category and add nodes to the system
    for node in supply_node + demand_node + reservoir + hydroworks + sink_node:
        system.add_node(node)

    # Add Edges to the system
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=1230))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=885))

    # Supply for Bulungur, Jomboy and Payriq (and Jizzakh-Region)
    system.add_edge(Edge(HW_Ravadhoza, Bulungur, capacity=45))
    system.add_edge(Edge(HW_Ravadhoza, Jomboy, capacity=60))
    system.add_edge(Edge(Bulungur, Jomboy, capacity=45))
    system.add_edge(Edge(Jomboy, Payariq, capacity=105))
    system.add_edge(Edge(HW_Ravadhoza, sink_tuyatortor, capacity=35))


    # Supply for Toyloq, Urgut, Samarqand
    system.add_edge(Edge(HW_Ravadhoza, Toyloq, capacity=80))
    system.add_edge(Edge(Toyloq, Samarqand, capacity=80))
    system.add_edge(Edge(HW_Ravadhoza, Urgut, capacity=125))
    system.add_edge(Edge(Urgut, Samarqand, capacity=125))
    system.add_edge(Edge(Samarqand, HW_EskiAnkhor, capacity=205))
    system.add_edge(Edge(HW_EskiAnkhor, Pastdargom, capacity=150))
    system.add_edge(Edge(Pastdargom, HW_Damkodzha, capacity=150))
    system.add_edge(Edge(HW_EskiAnkhor, Nurobod, capacity=60))
    system.add_edge(Edge(Nurobod, sink_eskiankhor, capacity=60))

    # HW_AkKaraDarya
    system.add_edge(Edge(HW_AkKaraDarya, Oqdaryo, capacity=230))
    system.add_edge(Edge(Oqdaryo, RES_AkDarya, capacity=230))
    system.add_edge(Edge(Payariq, Ishtixon, capacity=105))
    system.add_edge(Edge(Ishtixon, RES_AkDarya, capacity=105))
    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_AkKaraDarya, HW_Damkodzha, capacity=550))

    # Damkodzha
    system.add_edge(Edge(HW_Damkodzha, RES_Kattakurgan, capacity=100))
    system.add_edge(Edge(RES_Kattakurgan, HW_Narpay, capacity=125))
    system.add_edge(Edge(HW_Damkodzha, HW_Narpay, capacity=80))
    system.add_edge(Edge(HW_Damkodzha, HW_Confluence, capacity=350))
    system.add_edge(Edge(HW_Damkodzha, Kattaqorgon, capacity=90))
    system.add_edge(Edge(Kattaqorgon, Xatirchi, capacity=90))
    system.add_edge(Edge(Xatirchi, HW_Karmana, capacity=90))

    # HW_Narpay
    system.add_edge(Edge(HW_Narpay, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_Narpay, Narpay, capacity=80))
    system.add_edge(Edge(Narpay, Paxtachi, capacity=80))
    system.add_edge(Edge(Paxtachi, Karmana, capacity=80))
    system.add_edge(Edge(Karmana, sink_downstream, capacity=80))

    # HW_Confluence
    system.add_edge(Edge(HW_Confluence, HW_Karmana, capacity=400))

    # HW_Karmana
    system.add_edge(Edge(HW_Karmana, Navbahor, capacity=45))
    system.add_edge(Edge(Navbahor, sink_downstream, capacity=45))
    system.add_edge(Edge(HW_Karmana, sink_downstream, capacity=400))
    system.add_edge(Edge(HW_Karmana, Powerplant, capacity=65))
    system.add_edge(Edge(Powerplant, sink_downstream, capacity=65))

    return system

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
        elif isinstance(obj, np.ndarray):  # Handle numpy arrays
            return [float(x) for x in obj]
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

def run_system_with_optimized_parameters(system_creator, optimization_results, start_year, start_month, 
                                         num_time_steps, name='ZRB_system', scenario='', period='', agr_scenario='', efficiency=''):
    
    # Create new system
    system = system_creator(start_year, start_month, num_time_steps, scenario, period, agr_scenario, efficiency)
    
    # Load optimized parameters
    system = load_optimized_parameters(system, optimization_results)
    
    # Run simulation
    system.simulate(num_time_steps)

    vis=WaterSystemVisualizer(system, name)
    vis.plot_minimum_flow_compliance()
    vis.plot_storage_dynamics()
    vis.plot_reservoir_dynamics()
    vis.plot_flow_compliance_heatmap()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.plot_objective_function_breakdown()
    print("Visualizations complete")
    
    return system

def run_optimization(system_creator, start_year=2017, start_month=1, num_time_steps=12, scenario = '', period = '', agr_scenario= ' ', efficiency = ' ',
                     ngen=100, pop_size=2000, cxpb=0.5, mutpb=0.2):
    
    ZRB_system = system_creator(start_year, start_month, num_time_steps, scenario, period, agr_scenario, efficiency)

    
    optimizer = MultiGeneticOptimizer(
        base_system=ZRB_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        ngen=ngen,
        population_size=pop_size,
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
    print(f"Corss-over probability: {results['crossover_probability']}")
    print(f"Mutation probability: {results['mutation_probability']}")
    print(f"Final objective value: {results['objective_value']:,.0f} m³")
    
    print("\nOptimal Reservoir Parameters:")
    for res_id, params in results['optimal_reservoir_parameters'].items():
        print(f"\n{res_id}:")
        for param, values in params.items():
            print(f"{param}: [", end="")
            print(", ".join(f"{v:.3f}" for v in values), end="")
            print("]")
        
    print("\nOptimal Hydroworks Parameters:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for hw_id, params in results['optimal_hydroworks_parameters'].items():
        print(f"\n{hw_id}:")
        for target, values in params.items():
            print(f"{target}: [", end="")
            print(", ".join(f"{v:.3f}" for v in values), end="")
            print("]")

    optimizer.plot_convergence()

    return results

def run_tests(start_year=2017, start_month=1, num_time_steps=12):
 
     ZRB_system = create_ZRB_system_baseline(start_year, start_month, num_time_steps, scenario='', period='', agr_scenario='', efficiency='')
 
     print("ZRB system simulation:")
     ZRB_system._check_network()
     ZRB_system.simulate(num_time_steps)
     print("Simulation complete")
 
     print('ZRB system visualization:')
     vis_ZRB=WaterSystemVisualizer(ZRB_system, 'ZRB')
     vis_ZRB.plot_minimum_flow_compliance()
     vis_ZRB.plot_storage_dynamics()
     vis_ZRB.plot_reservoir_dynamics()
     vis_ZRB.plot_flow_compliance_heatmap()
     vis_ZRB.plot_spills()
     vis_ZRB.plot_reservoir_volumes()
     vis_ZRB.plot_system_demands_vs_inflow()
     vis_ZRB.plot_objective_function_breakdown()
     print("Visualizations complete")

# Run the sample tests
if __name__ == "__main__":

    #################
    ### Run Tests ###
    #################

    #run_tests(start_year=2017, start_month=1, num_time_steps=6*12)


    ###########################
    ### Run Baseline Period ###
    ###########################
    #'''
    start_year = 2017
    start_month = 1
    num_time_steps = 12*6

    loaded_results = load_parameters_from_file(f"./model_output/optimisation/baseline_param_test.json")
     
    system = run_system_with_optimized_parameters(
        create_ZRB_system_baseline,
        loaded_results,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps, 
        name= 'Baseline Period',
        scenario = '',
        period = '',
        agr_scenario = '', 
        efficiency = ''
    )
    
    #'''
    ###########################
    ### Run Future Scenario ###
    ###########################
    '''
    start_year = 2041
    start_month = 1
    num_time_steps = 12*29

    loaded_results = load_parameters_from_file(f"data/optimization/optimised_parameter/param_ssp126_mid_century_diversification.json")
     
    system = run_system_with_optimized_parameters(
        create_ZRB_system_scenarios,
        loaded_results,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps, 
        name= 'Diversification SSP126',
        scenario = 'ssp126',
        period = '2041-2070',
        agr_scenario = 'diversification', 
        efficiency = 'improved_efficiency' # or 'noeff'
    )
    '''
    ############################################
    ### Run Optimization for Baseline Period ###
    ############################################
    '''
    start_year = 2017
    start_month = 1
    num_time_steps = 12*6

    # GA settings
    ngen = 10
    pop_size = 20
    cxpb = 0.65
    mutpb = 0.32

    results = run_optimization(
        create_ZRB_system_baseline,
        start_year, 
        start_month, 
        num_time_steps,
        scenario = '', 
        period = '', 
        agr_scenario= ' ', 
        efficiency = ' ', 
        ngen=ngen, 
        pop_size=pop_size, 
        cxpb=cxpb, 
        mutpb= mutpb
    )        

    save_optimized_parameters(results, f"./model_output/optimisation/baseline_param_test.json")
    '''
    ############################################
    ### Run Optimization for Future Scenario ###
    ############################################
    '''
    start_year = 2041
    start_month = 1
    num_time_steps = 12*30

    # GA settings
    ngen = 10
    pop_size = 20
    cxpb = 0.65
    mutpb = 0.32

    results = run_optimization(
        create_ZRB_system_scenarios,
        start_year, 
        start_month, 
        num_time_steps,
        scenario = 'ssp126',
        period = '2041-2070',
        agr_scenario = 'diversification', 
        efficiency = 'improved_efficiency', # or 'noeff' 
        ngen=ngen, 
        pop_size=pop_size, 
        cxpb=cxpb, 
        mutpb= mutpb
    )        

    save_optimized_parameters(results, f"./model_output/optimisation/ssp126_div_mid_impeff_param_test.json")
    '''


    ##################################
    ### Options for Code Profiling ###
    ##################################
    '''
    # Start profiling
    #profiler = cProfile.Profile()
    #profiler.enable()

    # Stop profiling
    #profiler.disable()

    # Save profiling stats
    #profile_output = 'cprofile_stats.prof'
    #profiler.dump_stats(profile_output)

    # Display profiling stats
    #stream = io.StringIO()
    #stats = pstats.Stats(profiler, stream=stream)
    #stats.sort_stats('cumulative')  # Sort by cumulative time
    #stats.print_stats(40)  # Print top 20 functions
    #print(stream.getvalue())
    '''
    