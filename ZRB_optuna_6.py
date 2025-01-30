import pandas as pd
import numpy as np
import webbrowser
import os
import json
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, Edge, WaterSystemVisualizer, MultiGeneticOptimizer
import optuna
import plotly
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_intermediate_values, plot_timeline, plot_slice, plot_edf

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
            conveyance_efficiency=0.645,
            weight=1.0
        )
        demand_nodes.append(demand_node)
    
    # Unpack demand nodes into individual variables based on Demand_info['name'] row
    for index, row in Demand_info.iterrows():
        globals()[row['name']] = demand_nodes[index]
        #print(f"Created demand node: {row['name']}")

    # Demand Thermal powerplant Navoi (25 m³/s)
    Powerplant = DemandNode("Navoi-Powerplant", demand_rates=25, non_consumptive_rate=17, easting=186146.3,northing=4454459.3, weight=1000)

    # Reservoir
    release_params_kattakurgan = {
        'h1': 504.0,
        'h2': 511.0,
        'w': 5.0,
        'm1': 1.5,
        'm2': 1.5,
    }
    release_params_akdarya = {
        'h1': 490.0,
        'h2': 496.0,
        'w': 10.0,
        'm1': 1.5,
        'm2': 1.5,
    }
    RES_Kattakurgan =StorageNode("RES-Kattakurgan",hva_file='./data/Kattakurgan_H_V_A.csv',easting=265377.2,northing= 4414217.5, initial_storage=5e8,
                                 evaporation_file='./data/extended_predicted_reservoir_et_2010_2022.csv', start_year=start_year, start_month=start_month, 
                                 num_time_steps=num_time_steps, release_params=release_params_kattakurgan, dead_storage=32e5)
    RES_AkDarya = StorageNode("RES-Akdarya", hva_file='./data/Akdarya_H_V_A.csv' ,easting= 274383.7,northing=4432954.7, initial_storage=8e7, 
                              evaporation_file='./data/extended_predicted_reservoir_et_2010_2022.csv', start_year=start_year, start_month=start_month, 
                              num_time_steps=num_time_steps, release_params=release_params_akdarya, dead_storage=14e5)
    
    # Sink Nodes
    sink_tuyatortor = SinkNode("Sink-Jizzakh", min_flow_csv_file='./data/Jizzakh_min_flow_monthly_2000_2022.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=376882.3,northing=4411307.9)
    sink_eskiankhor = SinkNode("Sink-Kashkadarya",min_flow_csv_file='./data/Kashkadarya_min_flow_monthly_2000_2022.csv', start_year=start_year, 
                               start_month=start_month, num_time_steps=num_time_steps, weight=10, easting=272551,northing=4361872)
    sink_downstream = SinkNode("Sink-Navoi", min_flow_csv_file='./data/Navoi_min_flow_monthly_plus_norm_1968_2022.csv', start_year=start_year, 
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
    system.add_edge(Edge(supply, HW_Ravadhoza, capacity=1350))
    system.add_edge(Edge(HW_Ravadhoza, HW_AkKaraDarya, capacity=885))

    # Supply for Bulungur, Jomboy and Payriq (and Jizzakh-Region)
    system.add_edge(Edge(HW_Ravadhoza, Bulungur, capacity=40))
    system.add_edge(Edge(HW_Ravadhoza, Jomboy, capacity=50))
    system.add_edge(Edge(Bulungur, Jomboy, capacity=40))
    system.add_edge(Edge(Jomboy, Payariq, capacity=90))
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
    system.add_edge(Edge(Payariq, Ishtixon, capacity=90))
    system.add_edge(Edge(Ishtixon, RES_AkDarya, capacity=90))
    system.add_edge(Edge(RES_AkDarya, HW_Confluence, capacity=80))
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
    system.add_edge(Edge(HW_Karmana, Powerplant, capacity=35))
    system.add_edge(Edge(Powerplant, sink_downstream, capacity=35))

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

def run_ipynb_optimization(start_year=2017, start_month=1, num_time_steps=12, ngen=100, pop_size=2000, cxpb=0.5, mutpb=0.2):
    ZRB_system = create_seasonal_ZRB_system(start_year, start_month, num_time_steps)

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
    print(f"Message: {results['message']}")
    print(f"Population size: {results['population_size']}")
    print(f"Generations: {results['generations']}")
    print(f"Corss-over probability: {results['crossover_probability']}")
    print(f"Mutation probability: {results['mutation_probability']}")
    print(f"Final objective value: {results['objective_value']:,.0f} m³")

    save_optimized_parameters(results, f"./GA_experiments/optimized_parameters_ZRB_ngen{ngen}_pop{pop_size}_cxpb{cxpb}_mutpb{mutpb}.json")
    optimizer.plot_convergence()
    return results

# Run the sample tests
if __name__ == "__main__":
    start_year = 2017
    start_month = 1
    num_time_steps = 12*6
    
    def objective(trial):
        population = trial.suggest_int('pop_size', 1000, 2000 )
        generations = trial.suggest_int('generations', 50, 200)
        cxpb = trial.suggest_float('cxpb', 0, 1)
        mutpb = trial.suggest_float('mutpb', 0, 1)

        optimization_results = run_ipynb_optimization(start_year, start_month, num_time_steps, generations, population, cxpb, mutpb)
        return optimization_results['objective_value']

    # Create a study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Print the best parameters and fitness value
    print("Best Parameters:", study.best_params)
    print("Best Objective Value:", study.best_value)

    # save the study
    study_name = "ZRB_study"
    study_file = f"{study_name}.pkl"
    study.trials_dataframe().to_csv(f"{study_name}.csv")
    study.trials_dataframe().to_pickle(study_file)

    #save the plots
    plot_optimization_history(study).write_html(f"GA_experiments/{study_name}_history.html")
    plot_param_importances(study).write_html(f"GA_experiments/{study_name}_importances.html")
    plot_contour(study).write_html(f"GA_experiments/{study_name}_contour.html")
    plot_timeline(study).write_html(f"GA_experiments/{study_name}_timeline.html")
    plot_slice(study).write_html(f"GA_experiments/{study_name}_slice.html")
    plot_edf(study).write_html(f"GA_experiments/{study_name}_edf.html")
