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
from water_system import (DeapOptimizer, ParetoVisualizer)
from water_system.io_utils import save_optimized_parameters, load_parameters_from_file
from datetime import datetime
from ZRB_system_creator import create_ZRB_system

if __name__ == "__main__":
    
    # Logging start time to measure execution time
    start = datetime.now()
    
    # Define parameters for the optimization
    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 6  # 6 years of monthly data
    # Optimization parameters
    number_of_generations = 10
    population_size = 300
    crossover_probability = 0.65
    mutation_probability = 0.32

    # Define objective weights for the optimization
    # The number of objectives is determined by the number of keys ('objective_1', 'objective_2', etc.) in the dictionary.
    # Each value is a list of five weights, corresponding to the five available objective components:
    #   [Regular Demand Deficit, Priority Demand Deficit, Spillage, Minimum Flow Deficit at Sinks, Unmet Ecological Flow at Edges]
    # For each objective, set the weights to 1 for the components you want to include, and 0 for those you want to ignore.
    # Example below:
    objective_weights = {
        'objective_1': [1, 0, 0, 0, 0],  # Only regular demand deficit is minimized in this objective
        'objective_2': [0, 1, 1, 0, 0],  # Priority demand deficit and spillage are minimized together
        'objective_3': [0, 0, 0, 1, 0],  # Only minimum flow deficit at sinks is minimized
    }

    # Create the base water system
    water_system = create_ZRB_system(start_year, start_month, num_time_steps)

    # Initialize the optimizer
    optimizer = DeapOptimizer(
        base_system=water_system,
        num_time_steps=num_time_steps,
        ngen=number_of_generations,
        population_size=population_size,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        objective_weights=objective_weights
    )

    # Run the optimization
    results = optimizer.optimize()

    # Plot convergence and Pareto front
    optimizer.plot_convergence()
    optimizer.plot_total_objective_convergence()

    # Print optimization results
    print("\nOptimization Results:")
    print("-" * 60)
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    # Print the number of solutions in the Pareto front
    print(f"\nNumber of non-dominated solutions: {len(results['pareto_front'])}")

    # Save the optimized parameters
    save_optimized_parameters(results, f"./model_output/optimization/parameter/parameter_{len(objective_weights)}obj_{number_of_generations}gen_{population_size}pop.json")
    sol =load_parameters_from_file(f"./model_output/optimization/parameter/parameter_{len(objective_weights)}obj_{number_of_generations}gen_{population_size}pop.json")
    # Generate Pareto Front visualizations
    dashboard = ParetoVisualizer(sol)
    dashboard.generate_full_report()

    # Log the end time and calculate execution duration
    end = datetime.now()
    print(f"Execution time: {end - start}")
