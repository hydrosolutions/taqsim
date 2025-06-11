import os
from water_system import (DeapOptimizer, ParetoVisualizer)
from water_system.io_utils import save_optimized_parameters, load_parameters_from_file
from datetime import datetime
from system_creator_ZRB import create_simplified_ZRB_system

if __name__ == "__main__":

    start = datetime.now()
    
    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 6  # 6 years of monthly data

    number_of_generations = 10
    population_size = 3000
    crossover_probability = 0.65
    mutation_probability = 0.3

    objective_weights ={
            'objective_1': [1,0,0,0,0],
            'objective_2': [0,1,1,0,0],
            'objective_3': [0,0,0,1,0],
        }

    # Create the base water system
    water_system = create_simplified_ZRB_system(start_year, start_month, num_time_steps)

    # Initialize the single-objective optimizer
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

    save_optimized_parameters(results, f"./model_output/optimization/parameter/parameter_{len(objective_weights)}obj_{number_of_generations}gen_{population_size}pop.json")
    sol =load_parameters_from_file(f"./model_output/optimization/parameter/parameter_{len(objective_weights)}obj_{number_of_generations}gen_{population_size}pop.json")

    dashboard = ParetoVisualizer(sol)
    dashboard.generate_full_report()

    end = datetime.now()
    print(f"Execution time: {end - start}")
