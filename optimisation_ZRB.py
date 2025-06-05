import os
from water_system import (DeapOptimizer, ParetoVisualizer)
from water_system.io_utils import save_optimized_parameters
from datetime import datetime
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_timeline, plot_slice, plot_edf
from system_creator_ZRB import create_simplified_ZRB_system

if __name__ == "__main__":

    start = datetime.now()
    
    start_year = 2017
    start_month = 1
    num_time_steps = 12 * 6  # 6 years of monthly data

    number_of_generations = 10
    population_size = 30
    crossover_probability = 0.65
    mutation_probability = 0.3

    objective_weights ={
            'objective_1': [1,0,0,0,0],
            'objective_2': [0,1,1,0,0],
            'objective_3': [0,0,0,1,1]
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
    print(f"Final objective values:")
    for i, value in enumerate(results['objective_values'], 1):
        print(f"  - Objective {i}:    {value:,.3f} km³/a")
    
    # Print the number of solutions in the Pareto front
    print(f"\nNumber of non-dominated solutions: {len(results['pareto_front'])}")

    save_optimized_parameters(results, f"./model_output/optimization/parameter/parameter_{len(objective_weights)}obj_{number_of_generations}gen_{population_size}pop.json")
    
    
    dashboard = ParetoVisualizer(results['pareto_front'])
    dashboard.generate_full_report()


    # Option for an Optuna study in order to find best GA parameters (cxpb, mutpb, ngen, pop_size)
    optunastudy = False
    '''if optunastudy:
        # Making an Optuna study
        def objective(trial):
            # Define the hyperparameters to optimize
            ngen = trial.suggest_int("ngen", 1, 10)
            pop_size = trial.suggest_int("pop_size", 5, 30)
            cxpb = trial.suggest_float("cxpb", 0.5, 1.0)
            mutpb = trial.suggest_float("mutpb", 0.1, 0.5)

            # Run the optimization
            results = run_optimization(
                create_simplified_ZRB_system,
                start_year=2017, 
                start_month=1, 
                num_time_steps=12*6,
                ngen=ngen, 
                pop_size=pop_size, 
                cxpb=cxpb, 
                mutpb=mutpb,
                number_of_objectives=1, 
                objective_weights=objective_weights
            )

            return results['objective_value']

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)

        # Print the best parameters and fitness value
        print("Best Parameters:", study.best_params)
        print("Best Objective Value:", study.best_value)

        # save the study
        study_name = "ZRB_study"
        study_file = f"{study_name}.pkl"
        study.trials_dataframe().to_csv(f"{study_name}.csv")
        study.trials_dataframe().to_pickle(study_file)

        # create the study directory
        if not os.path.exists("model_output/optuna"):
            os.makedirs("model_output/optuna")
        #save the plots
        plot_optimization_history(study).write_html(f"model_output/optuna/{study_name}_history.html")
        plot_param_importances(study).write_html(f"model_output/optuna/{study_name}_importances.html")
        plot_contour(study).write_html(f"model_output/optuna/{study_name}_contour.html")
        plot_timeline(study).write_html(f"model_output/optuna/{study_name}_timeline.html")
        plot_slice(study).write_html(f"model_output/optuna/{study_name}_slice.html")
        plot_edf(study).write_html(f"model_output/optuna/{study_name}_edf.html")
'''

    end = datetime.now()
    print(f"Execution time: {end - start}")
