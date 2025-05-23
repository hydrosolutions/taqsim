from typing import Callable, Dict, List, Optional, Union
import pandas as pd
import json
from water_system import (WaterSystem, SupplyNode, StorageNode, DemandNode, 
                          SinkNode, HydroWorks,RunoffNode, Edge)


def create_ZRB_system(start_year: int,start_month: int,num_time_steps: int,system_type: str = "baseline",
    scenario: str = '',period: str = '',agr_scenario: str = '',efficiency: str = '') -> WaterSystem:
    """
    Flexible function to create a ZRB water system model for either baseline or scenario simulations.
    
    Args:
        start_year (int): Start year for simulation
        start_month (int): Start month for simulation (1-12)
        num_time_steps (int): Number of time steps to simulate
        system_type (str): "baseline" or "scenario"
        scenario (str): Climate scenario (e.g., 'ssp126') - only used for scenario simulations
        period (str): Time period (e.g., '2041-2070') - only used for scenario simulations
        agr_scenario (str): Agricultural scenario - only used for scenario simulations
        efficiency (str): Efficiency scenario (e.g., 'improved_efficiency') - only used for scenario simulations
        
    Returns:
        WaterSystem: Configured water system ready for simulation
    """
    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)
    
    # Set base path according to system type
    if system_type == 'simplified_ZRB':
        base_path = './data/simplified_ZRB'
    else:
        base_path = f"./data/{'baseline' if system_type == 'baseline' else 'scenarios'}"
    
    # Create Source Node with appropriate path
    if system_type == "scenario":
        inflow_path = f"{base_path}/inflow/inflow_ravatkhoza_{scenario}_2012-2099.csv"
    else:
        inflow_path = f"{base_path}/inflow/inflow_ravatkhoza.csv"
    
    Source = SupplyNode("Source", 
                       easting=381835, 
                       northing=4374682, 
                       csv_file=inflow_path, 
                       start_year=start_year, 
                       start_month=start_month, 
                       num_time_steps=num_time_steps)
    system.add_node(Source)
    
    # Load Hydroworks Node Configuration
    hydrowork_info = pd.read_csv(f'{base_path}/config/hydrowork_nodes_config.csv', sep=',')
    for index, row in hydrowork_info.iterrows():
        hw_node = HydroWorks(
            row['name'],
            easting=row['easting'],
            northing=row['northing']
        )
        globals()[row['name']] = hw_node
        system.add_node(hw_node)

    # Determine precipitation data file path (so far only for simplified ZRB possible)
    if system_type == "simplified_ZRB":
        precip_data_path = f"{base_path}/precipitation/precipitation_2017-2022.csv"


        # Load Runoff Node Configuration
        runoff_info = pd.read_csv(f'{base_path}/config/runoff_nodes_config.csv', sep=',')
        for index, row in runoff_info.iterrows():
            runoff_node = RunoffNode(
                row['name'],
                area=row['area'],
                runoff_coefficient=row['runoff_coefficient'],
                easting=row['easting'],
                northing=row['northing'],
                rainfall_csv=precip_data_path,
                start_year=start_year,
                start_month=start_month,
                num_time_steps=num_time_steps
            )
            globals()[row['name']] = runoff_node
            system.add_node(runoff_node)
  
    
    # Load Demand Node Configuration
    demand_config_path = f'{base_path}/config/demand_nodes_config'
    if system_type == "scenario" and efficiency:
        demand_config_path += f"_{efficiency}"
    demand_config_path += ".csv"
    
    demand_info = pd.read_csv(demand_config_path, sep=',')
    
    # Determine demand data file path
    if system_type == "scenario":
        demand_data_path = f"{base_path}/demand/demand_{scenario}_{agr_scenario}_{period}.csv"
    else:
        demand_data_path = f"{base_path}/demand/demand_all_districts_2017-2022_monthly.csv"
        
    
    # Create demand nodes
    for index, row in demand_info.iterrows():
        demand_node = DemandNode(
            row['name'],
            easting=row['easting'],
            northing=row['northing'],
            csv_file=demand_data_path,
            start_year=start_year,
            start_month=start_month,
            num_time_steps=num_time_steps,
            field_efficiency=row['field_efficiency'],
            conveyance_efficiency=row['conveyance_efficiency'],
        )
        globals()[row['name']] = demand_node
        system.add_node(demand_node)
    
    # Add Industrial Demand Node
    Powerplant = DemandNode("Powerplant", 
                           easting=186146.3,
                           northing=4454459.3, 
                           constant_demand_rate=25, 
                           non_consumptive_rate=17, 
                           num_time_steps=num_time_steps,
                           priority=1)
    system.add_node(Powerplant)
    
    # Determine evaporation file path
    if system_type == "scenario":
        evap_path = f"{base_path}/reservoir/reservoir_evaporation_{scenario}_2015-2100.csv"
    else :
        evap_path = f"{base_path}/reservoir/reservoir_evaporation.csv"
    
    # Add Reservoir Nodes
    RES_Kattakurgan = StorageNode("RES_Kattakurgan",
                                 easting=265377.2,
                                 northing=4414217.5, 
                                 hv_file=f"./data/baseline/reservoir/reservoir_kattakurgan_hv.csv",
                                 evaporation_file=evap_path,
                                 start_year=start_year,
                                 start_month=start_month,
                                 num_time_steps=num_time_steps,
                                 initial_storage=32e5,#3e8,
                                 dead_storage=32e5,
                                 buffer_coef=0.2)
    
    RES_Akdarya = StorageNode("RES_Akdarya",
                             easting=274383.7,
                             northing=4432954.7,
                             hv_file=f"./data/baseline/reservoir/reservoir_akdarya_hv.csv",
                             evaporation_file=evap_path, 
                             start_year=start_year,
                             start_month=start_month,
                             num_time_steps=num_time_steps,
                             initial_storage=14e5,#4e7,
                             dead_storage=14e5,
                             buffer_coef=0.2)
    
    # Add reservoirs to system
    system.add_node(RES_Kattakurgan)
    system.add_node(RES_Akdarya)
    
    # Determine min flow paths
    if system_type == "scenario":
        jizzakh_path = f"{base_path}/min_flow/min_flow_jizzakh_2015-2100.csv"
        kashkadarya_path = f"{base_path}/min_flow/min_flow_kashkadarya_2015-2100.csv"
        navoi_path = f"{base_path}/min_flow/min_flow_navoi_2015-2100.csv"
    else:
        jizzakh_path = f"{base_path}/min_flow/min_flow_jizzakh.csv"
        kashkadarya_path = f"{base_path}/min_flow/min_flow_kashkadarya.csv"
        navoi_path = f"{base_path}/min_flow/min_flow_navoi.csv"
    
    # Add Sink Nodes
    Sink_Jizzakh = SinkNode("Sink_Jizzakh", 
                           csv_file=jizzakh_path,
                           start_year=start_year, 
                           start_month=start_month, 
                           num_time_steps=num_time_steps,
                           weight=1, 
                           easting=376882.3, 
                           northing=4411307.9)
    
    Sink_Kashkadarya = SinkNode("Sink_Kashkadarya", 
                               csv_file=kashkadarya_path,
                               start_year=start_year, 
                               start_month=start_month, 
                               num_time_steps=num_time_steps,
                               weight=1, 
                               easting=272551, 
                               northing=4361872)
    
    Sink_Navoi = SinkNode("Sink_Navoi", 
                         csv_file=navoi_path,
                         start_year=start_year, 
                         start_month=start_month, 
                         num_time_steps=num_time_steps,
                         weight=1, 
                         easting=153771, 
                         northing=4454402)
    
    # Add sink nodes to system
    system.add_node(Sink_Jizzakh)
    system.add_node(Sink_Kashkadarya) 
    system.add_node(Sink_Navoi)
    
    # Load Edge Configuration
    with open(f'{base_path}/config/edges_config.json', 'r') as file:
        edge_list = json.load(file)
    
    for node_id in edge_list:
        node = system.graph.nodes[node_id]['node']
        for edge in edge_list[node_id]:
            target_node = system.graph.nodes[edge['target']]['node']
            system.add_edge(Edge(node, target_node, capacity=edge['capacity'], ecological_flow=edge['ecological_flow']))
    
    # Finalize and validate the system
    system._check_network()
    return system
