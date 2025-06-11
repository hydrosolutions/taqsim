from water_system import  (WaterSystemVisualizer, ParetoVisualizer)
from datetime import datetime
from system_creator_ZRB import create_simplified_ZRB_system
from water_system.io_utils import load_optimized_parameters, load_parameters_from_file
import json

if __name__ == "__main__":

    start = datetime.now()

    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 6  # 6 years of monthly data

    # Create new system
    system = create_simplified_ZRB_system(start_year, start_month, num_time_steps)

    sol=load_parameters_from_file(f"./data/simplified_ZRB/parameter/parameter_3obj_100gen_3000pop.json")

    dashboard = ParetoVisualizer(sol, objective_names=[f'Irrig. Supply Deficit [km³/a]', f'Industry Supply Deficit [km³/a]', f'Ecological Flow Deficit [km³/a]'])
    dashboard.generate_full_report()
    
    # Load optimized parameters
    # Example of running the simulation with optimized parameters for a simplified ZRB system
    loaded_results = load_parameters_from_file(f"./data/simplified_ZRB/parameter/parameter_3obj_100gen_3000pop.json")
    
    sol_id = 70  # Example solution ID to load
    system = load_optimized_parameters(system, pareto_solutions=loaded_results, solution_id=sol_id)
    print("Optimized parameters loaded successfully")
    # Run simulation
    system.simulate(num_time_steps)
    print("Simulation complete")

    vis=WaterSystemVisualizer(system, name=f'Solution_{sol_id}')
    vis.plot_network_overview()
    vis.plot_minimum_flow_compliance()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    vis.plot_network_layout()
    print("Visualizations complete")
    
    end = datetime.now()
    print(f"Execution time: {end - start}")


