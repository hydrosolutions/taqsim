"""
╔═════════════════════════════════════════════════════════════════════════╗
║           ████████╗ ██████╗  ██████╗ ███████╗██╗███╗   ███╗             ║
║           ╚══██╔══╝██╔═══██╗██╔═══██╗██╔════╝██║████╗ ████║             ║
║              ██║   ██║   ██║██║   ██║███████╗██║██╔████╔██║             ║
║              ██║   ██║   ██║██║▄▄ ██║╚════██║██║██║╚██╔╝██║             ║
║              ██║   ╚██████╔╝╚██████╔╝███████║██║██║ ╚═╝ ██║             ║
║              ╚═╝    ╚═════╝  ╚══▀▀═╝ ╚══════╝╚═╝╚═╝     ╚═╝             ║
║                                                                         ║
║           Advanced Water Resource System Modeling Framework             ║
║     ┌─────────────────────────────────────────────────────────────┐     ║
║     │ • Network-based simulation with specialized node types      │     ║
║     │ • Multi-objective optimization of water resource systems    │     ║
║     │ • Pareto-optimal solutions for sustainable water management │     ║
║     └─────────────────────────────────────────────────────────────┘     ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
"""

from taqsim import ParetoVisualizer, WaterSystemVisualizer
from taqsim.io_utils import load_optimized_parameters, load_parameters_from_file
from ZRB_system_creator import create_ZRB_system

if __name__ == "__main__":

    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 6  # 6 years of monthly data

    # Create new system
    system = create_ZRB_system(start_year, start_month, num_time_steps)

    # Load water allocation parameters from file
    optimized_parameters=load_parameters_from_file("./data/ZRB_baseline/parameter/parameter_3obj_100gen_3000pop.json")

    # Create a ParetoVisualizer to visualize the optimization solutions
    # the objective_names are only used for visualization purposes
    dashboard = ParetoVisualizer(optimized_parameters, objective_names=['Irrig. Supply Deficit [km³/a]', 'Industry Supply Deficit [km³/a]', 'Ecological Flow Deficit [km³/a]'])
    dashboard.generate_full_report()

    # Load optimized parameters into the system,
    # with the specified solution ID a solution from the Pareto front can be selected
    sol_id = 70  # Example solution ID to load
    system = load_optimized_parameters(system, pareto_solutions=optimized_parameters, solution_id=sol_id)

    # Run simulation: will run a simulation of the water system with the loaded parameters
    system.simulate(num_time_steps)

    # Createing some visualizations of the water system in the model_output/figures folder
    # TO DO: more visualizations or adaptations of the existing ones in the water_system module (visualization.py)
    vis=WaterSystemVisualizer(system, name=f'Solution_{sol_id}')
    vis.plot_network_overview()
    vis.plot_minimum_flow_compliance()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()
    vis.plot_network_layout()



