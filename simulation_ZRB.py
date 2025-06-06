from water_system import  WaterSystemVisualizer
from datetime import datetime
from system_creator_ZRB import create_simplified_ZRB_system
from water_system.io_utils import load_optimized_parameters, load_parameters_from_file


if __name__ == "__main__":

    start = datetime.now()

    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 6  # 6 years of monthly data

    # Create new system
    system = create_simplified_ZRB_system(start_year, start_month, num_time_steps)

    # Example of running the simulation with optimized parameters for a simplified ZRB system
    loaded_results = load_parameters_from_file(f"./data/simplified_ZRB/parameter/test99.json", solution_id=5)

    
    # Load optimized parameters
    system = load_optimized_parameters(system, loaded_results)
    print("Optimized parameters loaded successfully")
    # Run simulation
    system.simulate(num_time_steps)
    print("Simulation complete")

    vis=WaterSystemVisualizer(system, name=f'ZRB_simulation')
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
    

    