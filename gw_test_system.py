from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, Edge
from water_system import DeapOptimizer, WaterSystemVisualizer
from water_system.io_utils import load_optimized_parameters, load_parameters_from_file, save_optimized_parameters
from water_system.gw_nodes_edges import AquiferNode, GroundwaterEdge

if __name__ == "__main__":
    # Created a dummy water system with various nodes and edges

    dt = 30.44 * 24 * 3600  # Average month in seconds
    my_water_system = WaterSystem(dt=dt, start_year=2020, start_month=1)

    supply = SupplyNode(
        id="Source2",
        easting=100,
        northing=300,
        csv_file='./data/dummy_data/supply_timeseries.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(supply)

    runoff1 = RunoffNode(
        id="RF1",
        easting=400,
        northing=500,
        area=50,  # km²
        runoff_coefficient=0.3,
        rainfall_csv="./data/dummy_data/rainfall_timeseries.csv", # in mm
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(runoff1)

    runoff2 = RunoffNode(
        id="RF2",
        easting=400,
        northing=100,
        area=100,  # km²
        runoff_coefficient=0.3,
        rainfall_csv="./data/dummy_data/rainfall_timeseries.csv", # in mm
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(runoff2)

    # Constant minimum flow
    sink = SinkNode(
        id="RiverMouth",
        easting=500,
        northing=200,
        constant_min_flow=10,
        start_year=2020,
        start_month=1,
        num_time_steps=12
    )
    my_water_system.add_node(sink)


    hydrowork = HydroWorks(
        id="HydroWorks1",
        easting=300,
        northing=300
    )
    my_water_system.add_node(hydrowork)

    demand1 = DemandNode(
        id="agriculture1",
        easting=400,
        northing=400,
        csv_file='./data/dummy_data/demand_timeseries.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12,
        field_efficiency=0.8,
        conveyance_efficiency=0.5,
        priority=2
    )
    my_water_system.add_node(demand1)

        # Create aquifer
    aquifer = AquiferNode(
        id="Aquifer1",
        easting=300,
        northing=400,
        area=50,  # 50 km²
        max_thickness=50.0,  # 50 m
        porosity=0.2,  # 20%
        initial_head=25.0  # 25 m
    )
    my_water_system.add_node(aquifer)

    demand2 = DemandNode(
        id="agriculture2",
        easting=400,
        northing=200,
        csv_file='./data/dummy_data/demand_timeseries.csv',
        start_year=2020,
        start_month=1,
        num_time_steps=12,
        field_efficiency=0.9,
        conveyance_efficiency=0.8,
        priority=2
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
    my_water_system.add_edge(Edge(supply, storage, 100))
    my_water_system.add_edge(Edge(storage, hydrowork, 140))
    my_water_system.add_edge(Edge(hydrowork, demand1, 80))
    my_water_system.add_edge(Edge(hydrowork, demand2, 80))
    my_water_system.add_edge(Edge(runoff1, demand1, 60))
    my_water_system.add_edge(Edge(runoff2, demand2, 60))
    my_water_system.add_edge(Edge(demand1, sink, 100))
    my_water_system.add_edge(Edge(demand2, sink, 100))

    recharge_edge = GroundwaterEdge(
        source=supply,
        target=aquifer,
        edge_type="recharge",
        recharge_fraction=0.2  # 20% of source discharge
    )
    my_water_system.add_edge(recharge_edge)

    horizontal_edge = GroundwaterEdge(
        source=aquifer,
        target=sink,
        edge_type="horizontal",
        conductivity=1e-4,  # 1e-4 m/s
        area=1000.0,  # 1000 m² cross-sectional area
        length=2000.0  # 2 km distance
    )
    my_water_system.add_edge(horizontal_edge)

    my_water_system._check_network()

    ## Setting up an Optimization problem


    objectives = {'objective_1':[1,1,1,0,0]} 

    MyProblem = DeapOptimizer(
                    base_system=my_water_system,
                    num_time_steps=12,  # 12 month are optimized
                    objective_weights=objectives,
                    ngen=10,        # Optimizing over 50 generations
                    population_size=50, # A Population consists of 100 individuals
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
    my_water_system = load_optimized_parameters(my_water_system, results, solution_id=0)
    my_water_system.simulate(time_steps=12)

    vis = WaterSystemVisualizer(my_water_system, name='GW_Test_System_Visualization')
    vis.plot_network_overview()
    vis.plot_minimum_flow_compliance()
    vis.plot_spills()
    vis.plot_reservoir_volumes()
    vis.plot_system_demands_vs_inflow()
    vis.plot_demand_deficit_heatmap()
    vis.print_water_balance_summary()