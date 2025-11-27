from typing import Callable, Dict, List, Optional, Union
import pandas as pd
import json
from taqsim import (WaterSystem, SupplyNode, StorageNode, DemandNode,
                    SinkNode, HydroWorks, RunoffNode, Edge)


def create_ZRB_system(start_year: int, start_month: int, num_time_steps: int) -> WaterSystem:
    """
    Create the ZRB water system model.

    Args:
        start_year (int): Simulation start year.
        start_month (int): Simulation start month.
        num_time_steps (int): Number of monthly time steps to simulate.

    Returns:
        WaterSystem: Configured ZRB water system object.
    """
    # Set time step duration (seconds in an average month)
    dt = 30.44 * 24 * 3600
    # Initialize the water system
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)
    # Base path for data files 
    base_path = './data/ZRB_baseline'

    # --- Add Supply Node (main inflow) ---
    inflow_path = f"{base_path}/inflow/inflow_ravatkhoza.csv"
    Source = SupplyNode(
        "Source",
        easting=381835,
        northing=4374682,
        csv_file=inflow_path,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps
    )
    system.add_node(Source)

    # --- Add HydroWork Nodes (diversions/confluences) ---
    hydrowork_info = pd.read_csv(f'{base_path}/config/hydrowork_nodes_config.csv', sep=',')
    for index, row in hydrowork_info.iterrows():
        hw_node = HydroWorks(
            row['name'],
            easting=row['easting'],
            northing=row['northing']
        )
        globals()[row['name']] = hw_node  # Make node accessible by name (optional)
        system.add_node(hw_node)

    # --- Add Runoff Nodes (rainfall-runoff generation) ---
    precip_data_path = f"{base_path}/precipitation/precipitation_2017-2022.csv"
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

    # --- Add Demand Nodes (irrigation/urban/industrial) ---
    demand_config_path = f'{base_path}/config/demand_nodes_config.csv'
    demand_info = pd.read_csv(demand_config_path, sep=',')
    demand_data_path = f"{base_path}/demand/demand_all_districts_2017-2022_monthly.csv"
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
            priority=row['priority']
        )
        globals()[row['name']] = demand_node
        system.add_node(demand_node)

    # --- Add Powerplant Node (special demand, non-consumptive) ---
    Powerplant = DemandNode(
        "Powerplant",
        easting=186146.3,
        northing=4454459.3,
        constant_demand_rate=25,
        non_consumptive_rate=17,
        num_time_steps=num_time_steps,
        priority=1
    )
    system.add_node(Powerplant)

    # --- Add Storage Nodes (reservoirs) ---
    evap_path = f"{base_path}/reservoir/reservoir_evaporation.csv"
    RES_Kattakurgan = StorageNode(
        "RES_Kattakurgan",
        easting=265377.2,
        northing=4414217.5,
        hv_file=f"{base_path}/reservoir/reservoir_kattakurgan_hv.csv",
        evaporation_file=evap_path,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        initial_storage=32e5,
        dead_storage=32e5,
        buffer_coef=0.2
    )
    RES_Akdarya = StorageNode(
        "RES_Akdarya",
        easting=274383.7,
        northing=4432954.7,
        hv_file=f"{base_path}/reservoir/reservoir_akdarya_hv.csv",
        evaporation_file=evap_path,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        initial_storage=14e5,
        dead_storage=14e5,
        buffer_coef=0.2
    )
    system.add_node(RES_Kattakurgan)
    system.add_node(RES_Akdarya)

    # --- Add Sink Nodes (system outlets/environmental flows) ---
    jizzakh_path = f"{base_path}/min_flow/min_flow_jizzakh.csv"
    kashkadarya_path = f"{base_path}/min_flow/min_flow_kashkadarya.csv"
    navoi_path = f"{base_path}/min_flow/min_flow_navoi.csv"
    Sink_Jizzakh = SinkNode(
        "Sink_Jizzakh",
        csv_file=jizzakh_path,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        easting=376882.3,
        northing=4411307.9
    )
    Sink_Kashkadarya = SinkNode(
        "Sink_Kashkadarya",
        csv_file=kashkadarya_path,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        easting=272551,
        northing=4361872
    )
    Sink_Navoi = SinkNode(
        "Sink_Navoi",
        csv_file=navoi_path,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        easting=153771,
        northing=4454402
    )
    system.add_node(Sink_Jizzakh)
    system.add_node(Sink_Kashkadarya)
    system.add_node(Sink_Navoi)

    # --- Add Edges (connect all nodes according to config) ---
    with open(f'{base_path}/config/edges_config.json', 'r') as file:
        edge_list = json.load(file)
    for node_id in edge_list:
        node = system.graph.nodes[node_id]['node']
        for edge in edge_list[node_id]:
            target_node = system.graph.nodes[edge['target']]['node']
            system.add_edge(
                Edge(
                    node,
                    target_node,
                    capacity=edge['capacity'],
                    ecological_flow=edge['ecological_flow']
                )
            )

    # --- Final network validation ---
    system._check_network()
    return system

## TO DO: The input data for the future scenarios considering the new irrigation districts,
# still needs to be prepared, and the system creator function needs to be updated accordingly.
def create_future_scenario_ZRB_system(
    start_year: int,
    start_month: int,
    num_time_steps: int,
    scenario: str,
    period: str,
    agr_scenario: str,
    efficiency: str = ''
) -> WaterSystem:
    """
    Create a ZRB water system model for scenario simulations.
    """
    dt = 30.44 * 24 * 3600  # Average month in seconds
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)
    base_path = "./data/scenarios"

    inflow_path = f"{base_path}/inflow/inflow_ravatkhoza_{scenario}_2012-2099.csv"
    Source = SupplyNode("Source",
                        easting=381835,
                        northing=4374682,
                        csv_file=inflow_path,
                        start_year=start_year,
                        start_month=start_month,
                        num_time_steps=num_time_steps)
    system.add_node(Source)

    hydrowork_info = pd.read_csv(f'{base_path}/config/hydrowork_nodes_config.csv', sep=',')
    for index, row in hydrowork_info.iterrows():
        hw_node = HydroWorks(
            row['name'],
            easting=row['easting'],
            northing=row['northing']
        )
        globals()[row['name']] = hw_node
        system.add_node(hw_node)

    demand_config_path = f'{base_path}/config/demand_nodes_config'
    if efficiency:
        demand_config_path += f"_{efficiency}"
    demand_config_path += ".csv"
    demand_info = pd.read_csv(demand_config_path, sep=',')
    demand_data_path = f"{base_path}/demand/demand_{scenario}_{agr_scenario}_{period}.csv"

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
            priority=row['priority']
        )
        globals()[row['name']] = demand_node
        system.add_node(demand_node)

    Powerplant = DemandNode("Powerplant",
                            easting=186146.3,
                            northing=4454459.3,
                            constant_demand_rate=25,
                            non_consumptive_rate=17,
                            num_time_steps=num_time_steps,
                            priority=1)
    system.add_node(Powerplant)

    evap_path = f"{base_path}/reservoir/reservoir_evaporation_{scenario}_2015-2100.csv"
    RES_Kattakurgan = StorageNode("RES_Kattakurgan",
                                  easting=265377.2,
                                  northing=4414217.5,
                                  hv_file=f"./data/baseline/reservoir/reservoir_kattakurgan_hv.csv",
                                  evaporation_file=evap_path,
                                  start_year=start_year,
                                  start_month=start_month,
                                  num_time_steps=num_time_steps,
                                  initial_storage=32e5,
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
                              initial_storage=14e5,
                              dead_storage=14e5,
                              buffer_coef=0.2)
    system.add_node(RES_Kattakurgan)
    system.add_node(RES_Akdarya)

    jizzakh_path = f"{base_path}/min_flow/min_flow_jizzakh_2015-2100.csv"
    kashkadarya_path = f"{base_path}/min_flow/min_flow_kashkadarya_2015-2100.csv"
    navoi_path = f"{base_path}/min_flow/min_flow_navoi_2015-2100.csv"

    Sink_Jizzakh = SinkNode("Sink_Jizzakh",
                            csv_file=jizzakh_path,
                            start_year=start_year,
                            start_month=start_month,
                            num_time_steps=num_time_steps,
                            easting=376882.3,
                            northing=4411307.9)
    Sink_Kashkadarya = SinkNode("Sink_Kashkadarya",
                                csv_file=kashkadarya_path,
                                start_year=start_year,
                                start_month=start_month,
                                num_time_steps=num_time_steps,
                                easting=272551,
                                northing=4361872)
    Sink_Navoi = SinkNode("Sink_Navoi",
                          csv_file=navoi_path,
                          start_year=start_year,
                          start_month=start_month,
                          num_time_steps=num_time_steps,
                          easting=153771,
                          northing=4454402)
    system.add_node(Sink_Jizzakh)
    system.add_node(Sink_Kashkadarya)
    system.add_node(Sink_Navoi)

    with open(f'{base_path}/config/edges_config.json', 'r') as file:
        edge_list = json.load(file)
    for node_id in edge_list:
        node = system.graph.nodes[node_id]['node']
        for edge in edge_list[node_id]:
            target_node = system.graph.nodes[edge['target']]['node']
            system.add_edge(Edge(node, target_node, capacity=edge['capacity'], ecological_flow=edge['ecological_flow']))

    system._check_network()
    return system

# Note: The tuman system is created with the same structure as the ZRB system,
# but it uses different data files and configurations specific to the administrative districts/tumans.
def create_tuman_ZRB_system(start_year: int, start_month: int, num_time_steps: int) -> WaterSystem:
    """
    Create a ZRB water system model for baseline simulations.
    """
    dt = 30.44 * 24 * 3600  # Average month in seconds
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)
    base_path = "./data/baseline"

    inflow_path = f"{base_path}/inflow/inflow_ravatkhoza.csv"
    Source = SupplyNode("Source",
                        easting=381835,
                        northing=4374682,
                        csv_file=inflow_path,
                        start_year=start_year,
                        start_month=start_month,
                        num_time_steps=num_time_steps)
    system.add_node(Source)

    hydrowork_info = pd.read_csv(f'{base_path}/config/hydrowork_nodes_config.csv', sep=',')
    for index, row in hydrowork_info.iterrows():
        hw_node = HydroWorks(
            row['name'],
            easting=row['easting'],
            northing=row['northing']
        )
        globals()[row['name']] = hw_node
        system.add_node(hw_node)

    demand_config_path = f'{base_path}/config/demand_nodes_config.csv'
    demand_info = pd.read_csv(demand_config_path, sep=',')
    demand_data_path = f"{base_path}/demand/demand_all_districts_2017-2022_monthly.csv"

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
            priority=row['priority']
        )
        globals()[row['name']] = demand_node
        system.add_node(demand_node)

    Powerplant = DemandNode("Powerplant",
                            easting=186146.3,
                            northing=4454459.3,
                            constant_demand_rate=25,
                            non_consumptive_rate=17,
                            num_time_steps=num_time_steps,
                            priority=1)
    system.add_node(Powerplant)

    evap_path = f"{base_path}/reservoir/reservoir_evaporation.csv"
    RES_Kattakurgan = StorageNode("RES_Kattakurgan",
                                  easting=265377.2,
                                  northing=4414217.5,
                                  hv_file=f"./data/baseline/reservoir/reservoir_kattakurgan_hv.csv",
                                  evaporation_file=evap_path,
                                  start_year=start_year,
                                  start_month=start_month,
                                  num_time_steps=num_time_steps,
                                  initial_storage=32e5,
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
                              initial_storage=14e5,
                              dead_storage=14e5,
                              buffer_coef=0.2)
    system.add_node(RES_Kattakurgan)
    system.add_node(RES_Akdarya)

    jizzakh_path = f"{base_path}/min_flow/min_flow_jizzakh.csv"
    kashkadarya_path = f"{base_path}/min_flow/min_flow_kashkadarya.csv"
    navoi_path = f"{base_path}/min_flow/min_flow_navoi.csv"

    Sink_Jizzakh = SinkNode("Sink_Jizzakh",
                            csv_file=jizzakh_path,
                            start_year=start_year,
                            start_month=start_month,
                            num_time_steps=num_time_steps,
                            easting=376882.3,
                            northing=4411307.9)
    Sink_Kashkadarya = SinkNode("Sink_Kashkadarya",
                                csv_file=kashkadarya_path,
                                start_year=start_year,
                                start_month=start_month,
                                num_time_steps=num_time_steps,
                                easting=272551,
                                northing=4361872)
    Sink_Navoi = SinkNode("Sink_Navoi",
                          csv_file=navoi_path,
                          start_year=start_year,
                          start_month=start_month,
                          num_time_steps=num_time_steps,
                          easting=153771,
                          northing=4454402)
    system.add_node(Sink_Jizzakh)
    system.add_node(Sink_Kashkadarya)
    system.add_node(Sink_Navoi)

    with open(f'{base_path}/config/edges_config.json', 'r') as file:
        edge_list = json.load(file)
    for node_id in edge_list:
        node = system.graph.nodes[node_id]['node']
        for edge in edge_list[node_id]:
            target_node = system.graph.nodes[edge['target']]['node']
            system.add_edge(Edge(node, target_node, capacity=edge['capacity'], ecological_flow=edge['ecological_flow']))

    system._check_network()
    return system



