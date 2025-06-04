from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, Edge
from water_system import DeapOptimizer

if __name__ == "__main__":
    # Created a dummy water system with various nodes and edges

    dt = 30.44 * 24 * 3600  # Average month in seconds
    my_water_system = WaterSystem(dt=dt, start_year=2020, start_month=1)

    supply1 = SupplyNode(
        id="Source1",
        easting=100,
        northing=200,
        constant_supply_rate=100,  # m³/time step
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(supply1)

    supply2 = SupplyNode(
        id="Source2",
        easting=100,
        northing=300,
        csv_file='./data/dummy_data/supply_timeseries.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(supply2)

    runoff = RunoffNode(
        id="SurfaceRunoff",
        easting=140,
        northing=240,
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
        easting=130,
        northing=230,
        constant_min_flow=10,
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(sink1)

    # Time-varying minimum flow from CSV
    sink2 = SinkNode(
        id="EnvFlow",
        easting=135,
        northing=235,
        csv_file='./data/dummy_data/sink_min_flow.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(sink2)

    hydrowork = HydroWorks(
        id="HydroWorks1",
        easting=115,
        northing=215
    )
    my_water_system.add_node(hydrowork)

    demand1 = DemandNode(
        id="agriculture",
        easting=220,
        northing=320,
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
        easting=120,
        northing=220,
        constant_demand_rate=40,   # m³/s
        non_consumptive_rate=4,   # m³/s (returns to system)
        start_year=2020,
        start_month=1,
        num_time_steps=12,
        priority=1
    )
    my_water_system.add_node(demand2)

    storage = StorageNode(
        id="Reservoir",
        easting=110,
        northing=210,
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
    my_water_system.add_edge(Edge(storage, hydrowork, 100))
    my_water_system.add_edge(Edge(hydrowork, demand1, 50))
    my_water_system.add_edge(Edge(hydrowork, demand2, 30))
    my_water_system.add_edge(Edge(runoff, demand1, 60))
    my_water_system.add_edge(Edge(demand1, sink1, 80))
    my_water_system.add_edge(Edge(demand2, sink2, 20))

    my_water_system._check_network()

    ## Setting up an Optimization problem


    two_objectives = {'objective_1':[1,0,0,0,0.0], 
                    'objective_2':[0,1,0,0,0],
                    'objective_3':[0,0,0,1,0]} 

    MyProblem = DeapOptimizer(
                    base_system=my_water_system,
                    num_time_steps=12,  # 12 month are optimized
                    objective_weights=two_objectives,
                    ngen=5,        # Optimizing over 50 generations
                    population_size=10, # A Population consists of 100 individuals
                    cxpb=0.6,       # 0.6 probability for crossover
                    mutpb=0.2,      # 0.2 probability for mutation 
    )

    # Run the optimization
    results = MyProblem.optimize()

    # Plot convergence and Pareto front
    MyProblem.plot_convergence()
    MyProblem.plot_total_objective_convergence()