import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import os
import json
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer, MultiGeneticOptimizer
import ctypes

def prevent_sleep():
    """Prevents the system from going to sleep."""
    try:
        # Windows only: Prevent system from sleeping
        print("Preventing system from sleeping...")
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    except Exception as e:
        print(f"Sleep prevention failed: {e}")

def allow_sleep():
    """Allows the system to sleep again."""
    try:
        # Windows only: Reset system sleep settings
        print("Resetting sleep settings...")
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception as e:
        print(f"Failed to reset sleep settings: {e}")

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
    HW_Damkodzha = HydroWorks("HW-Damkodzha", easting=284720.68, northing=4417759.40)
    HW_Narpay = HydroWorks("HW-Narpay", easting=270403.55,northing=4424501.92)
    HW_Confluence=HydroWorks("HW-Confluence", easting=239889.6,northing=4433214.0)
    HW_Karmana=HydroWorks("HW-Karmana", easting=209334.3,northing=4448118.7)
    HW_EskiAnkhor=HydroWorks("HW-EskiAnkhor", easting=315015,northing=4390976)
    HW_PC22=HydroWorks("HW-PC22", easting=362679.7,northing=4379566.2)
    
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
            conveyance_efficiency=0.65,
            weight=1.0
        )
        demand_nodes.append(demand_node)
    
    # Unpack demand nodes into individual variables based on Demand_info['name'] row
    for index, row in Demand_info.iterrows():
        globals()[row['name']] = demand_nodes[index]
        #print(f"Created demand node: {row['name']}")

    # Demand Thermal powerplant Navoi (25 m³/s)
    Powerplant = DemandNode("Navoi-Powerplant", demand_rates=25,easting=186146.3,northing=4451659.3, weight=1000)
    Jizzakh = DemandNode("Jizzakh", easting=376882.3,northing=4401307.9, csv_file='./data/Sink_Eski Tuyatortor_monthly_2000_2022.csv', 
                         start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, weight=1)
    Kashkadarya = DemandNode("Kashkadarya", easting=272551,northing=4371872, csv_file='./data/Sink_Eski Ankhor_monthly_2000_2022.csv', 
                             start_year=start_year, start_month=start_month, num_time_steps=num_time_steps, weight=1)

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
                                 evaporation_file='./data/reservoir_et_2010_2019_predicted.csv', start_year=start_year, start_month=start_month, 
                                 num_time_steps=num_time_steps, release_params=release_params_kattakurgan)
    RES_AkDarya = StorageNode("RES-Akdarya", hva_file='./data/Akdarya_H_V_A.csv' ,easting= 274383.7,northing=4432954.7, initial_storage=6e7, 
                              evaporation_file='./data/Reservoir_ET_2010_2023.csv', start_year=start_year, start_month=start_month, 
                              num_time_steps=num_time_steps, release_params=release_params_akdarya)
    
    # Sink Nodes
    sink_tuyatortor = SinkNode("Sink-Jizzakh", easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("Sink-Kashkadarya", easting=272551,northing=4361872)
    sink_downstream = SinkNode("Sink-Navoi", easting=153771,northing=4454402)

    # Add nodes to the system
    supply_node = [supply]  # List of supply nodes
    reservoir = [RES_Kattakurgan, RES_AkDarya]  # List of reservoir nodes
    hydroworks = [HW_PC22,HW_EskiAnkhor, HW_Ravadhoza, HW_AkKaraDarya, HW_Damkodzha, HW_Narpay, HW_Confluence, HW_Karmana]  # List of agricultural demand nodes
    demand_node = [Bulungur, Ishtixon, Jomboy, Karmana, Kattaqorgon, Narpay, Navbahor, Nurobod, Oqdaryo, Pastdargom, Paxtachi, Payariq, Samarqand, Toyloq, Urgut, Xatirchi, Powerplant, Jizzakh, Kashkadarya]  # List of demand nodes
    sink_node = [sink_tuyatortor, sink_eskiankhor, sink_downstream]  # List of sink nodes

    # Iterate through each category and add nodes to the system
    for node in supply_node + demand_node + reservoir + hydroworks + sink_node:
        system.add_node(node)

    # Add Edges to the system
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=1350))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=885))

    # Supply for Bulungur, Jomboy and Payriq (and Jizzakh-Region)
    system.add_edge(Edge(HW_Ravadhoza, HW_PC22, capacity=125))
    system.add_edge(Edge(HW_PC22, Bulungur, capacity=65))
    system.add_edge(Edge(HW_PC22, Jomboy, capacity=50))
    system.add_edge(Edge(Bulungur, Jomboy, capacity=65))
    system.add_edge(Edge(Jomboy, Payariq, capacity=115))
    system.add_edge(Edge(HW_PC22, Jizzakh, capacity=65))
    system.add_edge(Edge(Jizzakh, sink_tuyatortor, capacity=65))

    # Supply for Toyloq, Urgut, Samarqand
    system.add_edge(Edge(HW_Ravadhoza, Toyloq, capacity=80))
    system.add_edge(Edge(Toyloq, Samarqand, capacity=80))
    system.add_edge(Edge(HW_Ravadhoza, Urgut, capacity=125))
    system.add_edge(Edge(Urgut, Samarqand, capacity=125))
    system.add_edge(Edge(Samarqand, HW_EskiAnkhor, capacity=205))
    system.add_edge(Edge(HW_EskiAnkhor, Pastdargom, capacity=125))
    system.add_edge(Edge(Pastdargom, HW_Damkodzha, capacity=125))
    system.add_edge(Edge(HW_EskiAnkhor, Nurobod, capacity=80))
    system.add_edge(Edge(Nurobod, Kashkadarya, capacity=80))
    system.add_edge(Edge(Kashkadarya, sink_eskiankhor, capacity=80))

    # HW_AkKaraDarya
    system.add_edge(Edge(HW_AkKaraDarya, Oqdaryo, capacity=300))
    system.add_edge(Edge(Oqdaryo, RES_AkDarya, capacity=300))
    system.add_edge(Edge(Payariq, Ishtixon, capacity=115))
    system.add_edge(Edge(Ishtixon, RES_AkDarya, capacity=115))
    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=55))
    system.add_edge(Edge(HW_AkKaraDarya, HW_Damkodzha, capacity=550))

    # Damkodzha
    system.add_edge(Edge(HW_Damkodzha, RES_Kattakurgan, capacity=100))
    system.add_edge(Edge(RES_Kattakurgan, HW_Narpay, capacity=65))
    system.add_edge(Edge(HW_Damkodzha, HW_Narpay, capacity=80))
    system.add_edge(Edge(HW_Damkodzha, HW_Confluence, capacity=350))
    system.add_edge(Edge(HW_Damkodzha, Kattaqorgon, capacity=55))
    system.add_edge(Edge(Kattaqorgon, Xatirchi, capacity=95))
    system.add_edge(Edge(Xatirchi, HW_Karmana, capacity=95))

    # HW_Narpay
    system.add_edge(Edge(HW_Narpay, HW_Confluence, capacity=125))
    system.add_edge(Edge(HW_Narpay, Narpay, capacity=80))
    system.add_edge(Edge(Narpay, Paxtachi, capacity=80))
    system.add_edge(Edge(Paxtachi, Karmana, capacity=80))
    system.add_edge(Edge(Karmana, sink_downstream, capacity=80))
    system.add_edge(Edge(HW_Narpay, Kattaqorgon, capacity=40))

    # HW_Confluence
    system.add_edge(Edge(HW_Confluence, HW_Karmana, capacity=530))

    # HW_Karmana
    system.add_edge(Edge(HW_Karmana, Navbahor, capacity=45))
    system.add_edge(Edge(Navbahor, sink_downstream, capacity=45))
    system.add_edge(Edge(HW_Karmana, sink_downstream, capacity=545))
    system.add_edge(Edge(HW_Karmana, Powerplant, capacity=35))
    system.add_edge(Edge(Powerplant, sink_downstream, capacity=35))


    # HW-Ravadhoza distribution
    HW_Ravadhoza.set_distribution_parameters({
        'HW-AkKaraDarya': 0.728,   
        'HW-PC22': 0.103,           
        'Toyloq': 0.066,           
        'Urgut': 0.103              
    })

    # HW-PC22 distribution
    HW_PC22.set_distribution_parameters({
        'Bulungur': 0.361,          # 65/180
        'Jomboy': 0.278,            # 50/180
        'Jizzakh': 0.361            # 65/180
    })

    # HW-EskiAnkhor distribution
    HW_EskiAnkhor.set_distribution_parameters({
        'Pastdargom': 0.610,        # 125/205
        'Nurobod': 0.390            # 80/205
    })

    # HW-AkKaraDarya distribution
    HW_AkKaraDarya.set_distribution_parameters({
        'Oqdaryo': 0.353,           # 300/850
        'HW-Damkodzha': 0.647       # 550/850
    })

    # HW-Damkodzha distribution
    HW_Damkodzha.set_distribution_parameters({
        'RES-Kattakurgan': 0.171,   # 100/585
        'HW-Narpay': 0.137,         # 80/585
        'HW-Confluence': 0.598,      # 350/585
        'Kattaqorgon': 0.094        # 55/585
    })

    # HW-Narpay distribution
    HW_Narpay.set_distribution_parameters({
        'HW-Confluence': 0.510,      # 125/245
        'Narpay': 0.327,            # 80/245
        'Kattaqorgon': 0.163        # 40/245
    })

    # HW-Confluence distribution
    HW_Confluence.set_distribution_parameters({
        'HW-Karmana': 1.0           # All flow goes to Karmana
    })

    # HW-Karmana distribution
    HW_Karmana.set_distribution_parameters({
        'Navbahor': 0.078,          # 45/580
        'Sink-Navoi': 0.862,    # 500/580
        'Navoi-Powerplant': 0.060         # 35/580
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

def run_system_with_optimized_parameters(system_creator, optimization_results, 
                                       start_year, start_month, num_time_steps):
    """
    Create and run a water system using optimized parameters.
    
    Args:
        system_creator (function): Function that creates the water system
        optimization_results (dict): Results from optimization or loaded from file
        start_year (int): Start year for simulation
        start_month (int): Start month (1-12)
        num_time_steps (int): Number of time steps to simulate
        
    Returns:
        WaterSystem: Simulated system with optimized parameters
    """
    # Create new system
    system = system_creator(start_year, start_month, num_time_steps)
    
    # Load optimized parameters
    system = load_optimized_parameters(system, optimization_results)
    
    # Run simulation
    system.simulate(num_time_steps)

    vis=WaterSystemVisualizer(system, 'ZRB_system')
    vis.plot_sink_outflows()
    vis.plot_demand_deficit_heatmap()
    vis.plot_reservoir_volumes()
    vis.print_water_balance_summary()
    vis.plot_hydroworks_flows('HW-Karmana')
    vis.plot_system_demands_vs_inflow()
    vis.plot_network_layout_2()

    # Get the storage node from the system's graph
    storage_node = system.graph.nodes['RES-Akdarya']['node']
    vis.plot_release_function(storage_node)
    storage_node = system.graph.nodes['RES-Kattakurgan']['node']
    vis.plot_release_function(storage_node)
    
    html_file = vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    
    return system

def run_optimization(start_year=2017, start_month=1, num_time_steps=12, ngen=100, pop_size=2000):
    optimizer = MultiGeneticOptimizer(
        create_seasonal_ZRB_system,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps,
        population_size=pop_size
    )

    results = optimizer.optimize(ngen=ngen)

    print("\nOptimization Results:")
    print("-" * 50)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Final objective value: {results['objective_value']:,.0f} m³")
    
    print("\nOptimal Reservoir Parameters:")
    for res_id, params in results['optimal_reservoir_parameters'].items():
        print(f"\n{res_id}:")
        for param, values in params.items():
            print(f"{param}: ", end="")
            print([f"{v:.3f}" for v in values])
        
    print("\nOptimal Hydroworks Parameters:")
    for hw_id, params in results['optimal_hydroworks_parameters'].items():
        print(f"\n{hw_id}:")
        for target, values in params.items():
            print(f"{target}: ", end="")
            print([f"{v:.3f}" for v in values])

    optimizer.plot_convergence()

      
    # Save optimization results
    save_optimized_parameters(results, f"optimized_parameters_ZRB_ngen{ngen}_pop{pop_size}.json")

    # Run system with optimized parameters
    optimized_system = run_system_with_optimized_parameters(
        create_seasonal_ZRB_system,  # Your system creator function
        results,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps
    )

    # Visualize the optimized system
    print("Optimized system visualization:")
    vis=WaterSystemVisualizer(optimized_system, 'optimized_ZRB_system')
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    vis.plot_reservoir_dynamics()

    # Get the storage node from the system's graph
    storage_node = optimized_system.graph.nodes['RES-Akdarya']['node']
    vis.plot_release_function(storage_node)
    storage_node = optimized_system.graph.nodes['RES-Kattakurgan']['node']
    vis.plot_release_function(storage_node)

    html_file=vis.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')

    return results

def run_sample_tests(start_year=2017, start_month=1, num_time_steps=12):

    print("\n" + "="*50 + "\n")

    # Definition of simulation period
    start_year = start_year
    start_month = start_month
    num_time_steps = num_time_steps
    ZRB_system = create_seasonal_ZRB_system(start_year, start_month, num_time_steps)

    print("ZRB system simulation:")
    ZRB_system._check_network()
    ZRB_system.simulate(num_time_steps)
    print("Simulation complete")
    
    print('ZRB system visualization:')
    save_water_balance_to_csv(ZRB_system, "balance_table_ZRB_system.csv")
    vis_ZRB=WaterSystemVisualizer(ZRB_system, 'ZRB')
    vis_ZRB.print_water_balance_summary()
    vis_ZRB.plot_reservoir_dynamics()
    vis_ZRB.plot_network_layout()
    vis_ZRB.plot_network_layout_2()

    # Get the storage node from the system's graph
    storage_node = ZRB_system.graph.nodes['RES-Akdarya']['node']
    vis_ZRB.plot_release_function(storage_node)
    storage_node = ZRB_system.graph.nodes['RES-Kattakurgan']['node']
    vis_ZRB.plot_release_function(storage_node)
    
    vis_ZRB.plot_demand_deficit_heatmap()
    vis_ZRB.plot_storage_dynamics()
    vis_ZRB.plot_network_layout()
    vis_ZRB.plot_edge_flow_summary()
    vis_ZRB.plot_edge_flows()

    html_file = vis_ZRB.create_interactive_network_visualization()
    print(f"Interactive visualization saved to: {html_file}")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')

    print("Visualizations complete")

# Run the sample tests
if __name__ == "__main__":
    prevent_sleep()
    
    start_year = 2017
    start_month = 1
    num_time_steps = 12*3
    
    run_sample_tests(start_year, start_month, num_time_steps)
    #run_optimization(start_year, start_month, num_time_steps, ngen=10, pop_size=10)
    # Load parameters from file

    """
    loaded_results = load_parameters_from_file("optimized_parameters_ZRB_ngen10_pop10.json")

    # Create and run system with loaded parameters
    system = run_system_with_optimized_parameters(
        create_seasonal_ZRB_system,
        loaded_results,
        start_year=start_year,
        start_month=start_month,
        num_time_steps=num_time_steps
    )
    """
    
    allow_sleep()

  