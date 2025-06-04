from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, Edge
from water_system import DeapOptimizer, WaterSystemVisualizer
from water_system.io_utils import load_optimized_parameters, load_parameters_from_file, save_optimized_parameters

if __name__ == "__main__":
    # Created a dummy water system with various nodes and edges

    dt = 30.44 * 24 * 3600  # Average month in seconds
    my_water_system = WaterSystem(dt=dt, start_year=2020, start_month=1)

    supply1 = SupplyNode(
        id="Source1",
        easting=100,
        northing=600,
        constant_supply_rate=100,  # m³/time step
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(supply1)

    supply2 = SupplyNode(
        id="Source2",
        easting=100,
        northing=200,
        csv_file='./data/dummy_data/supply_timeseries.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(supply2)

    runoff = RunoffNode(
        id="SurfaceRunoff",
        easting=300,
        northing=500,
        area=50,  # km²
        runoff_coefficient=0.3,
        rainfall_csv="./data/dummy_data/rainfall_timeseries.csv", # in mm
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(runoff)

    # Constant minimum flow
    sink1 = SinkNode(
        id="RiverMouth",
        easting=500,
        northing=200,
        constant_min_flow=10,
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(sink1)

    # Time-varying minimum flow from CSV
    sink2 = SinkNode(
        id="EnvFlow",
        easting=400,
        northing=400,
        csv_file='./data/dummy_data/sink_min_flow.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(sink2)

    hydrowork = HydroWorks(
        id="HydroWorks1",
        easting=300,
        northing=300
    )
    my_water_system.add_node(hydrowork)

    demand1 = DemandNode(
        id="agriculture",
        easting=300,
        northing=400,
        csv_file='./data/dummy_data/demand_timeseries.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12,
        field_efficiency=0.8,
        conveyance_efficiency=0.7,
        priority=2
    )
    my_water_system.add_node(demand1)

    demand2 = DemandNode(
        id="Industry",
        easting=400,
        northing=200,
        constant_demand_rate=60,   # m³/s
        non_consumptive_rate=10,   # m³/s (returns to system)
        start_year=2020,
        start_month=1,
        num_time_steps=12,
        priority=1
    )
    my_water_system.add_node(demand2)

    storage = StorageNode(
        id="Reservoir",
        easting=200,
        northing=300,
        hv_file='./data/dummy_data/reservoir_hv.csv',  # Height-volume relationship
        evaporation_file='./data/dummy_data/reservoir_ev_timeseries.csv', # Monthly reservoir evaporation in mm
        start_year=2020,
        start_month=1,
        num_time_steps=12,
        initial_storage=5e6,              # m³
        dead_storage=1e5,                 # m³
        buffer_coef=0.5
    )
    my_water_system.add_node(storage)

    # Add edges between nodes
    my_water_system.add_edge(Edge(supply1, storage, 100))
    my_water_system.add_edge(Edge(supply2, storage, 80))
    my_water_system.add_edge(Edge(storage, hydrowork, 140))
    my_water_system.add_edge(Edge(hydrowork, demand1, 80))
    my_water_system.add_edge(Edge(hydrowork, demand2, 80))
    my_water_system.add_edge(Edge(runoff, demand1, 60))
    my_water_system.add_edge(Edge(demand1, sink2, 100))
    my_water_system.add_edge(Edge(demand2, sink1, 100))

    my_water_system._check_network()

    ## Setting up an Optimization problem


    two_objectives = {'objective_1':[1,2,3,0,0.0]} 

    MyProblem = DeapOptimizer(
                    base_system=my_water_system,
                    num_time_steps=12,  # 12 month are optimized
                    objective_weights=two_objectives,
                    ngen=100,        # Optimizing over 50 generations
                    population_size=500, # A Population consists of 100 individuals
                    cxpb=0.6,       # 0.6 probability for crossover
                    mutpb=0.2,      # 0.2 probability for mutation 
    )

    # Run the optimization
    results = MyProblem.optimize()
    save_optimized_parameters(
        results, 
        filename="./data/dummy_data/optimization_results.json"
    )

    # Plot convergence and Pareto front
    MyProblem.plot_convergence()
    MyProblem.plot_total_objective_convergence()

    results = load_parameters_from_file("./data/dummy_data/optimization_results.json")
    my_water_system = load_optimized_parameters(my_water_system, results)
    my_water_system.simulate(time_steps=12)

    vis = WaterSystemVisualizer(my_water_system, name='Dummy_Water_System_Visualization')
    vis.plot_network_overview()
    vis.plot_minimum_flow_compliance()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()