from typing import Callable, Dict, List, Optional, Union
import pandas as pd
import json
from water_system import WaterSystem, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks,RunoffNode, Edge, WaterSystemVisualizer, SingleObjectiveOptimizer, MultiObjectiveOptimizer, ParetoFrontDashboard


def create_system(start_year: int,start_month: int,num_time_steps: int, system_type:str = '', scenario:str='', period:str='', agr_scenario:str='', efficiency:str='' ) -> WaterSystem:
    """
    Flexible function to create a ZRB water system model for either baseline or scenario simulations.
    
    Args:
        start_year (int): Start year for simulation
        start_month (int): Start month for simulation (1-12)
        num_time_steps (int): Number of time steps to simulate
        
    Returns:
        WaterSystem: Configured water system ready for simulation
    """
    # Set up the system with monthly time steps
    dt = 30.44 * 24 * 3600  # Average month in seconds
    system = WaterSystem(dt=dt, start_year=start_year, start_month=start_month)
    
    # Add Supply Node
    source = SupplyNode("Source", 
                       easting=0, 
                       northing=5, 
                       constant_supply_rate=80,
                       start_year=start_year, 
                       start_month=start_month, 
                       num_time_steps=num_time_steps)
    system.add_node(source)
    # Add Storage Node
    reservoir = StorageNode("Reservoir",
                            easting=5, 
                            northing=5, 
                            hv_file='data/test_system/reservoir_hv.csv',
                            initial_storage=1000000,
                            dead_storage=100000, 
                            start_year=start_year,
                            start_month=start_month,
                            num_time_steps=num_time_steps)
    system.add_node(reservoir)
    # Add Edge from Source to Reservoir
    system.add_edge(Edge(source, reservoir, capacity=100))

    # Add HydroWorks Node
    hydrowork1 = HydroWorks("Hydrowork1",
                           easting=10, 
                           northing=5)
    system.add_node(hydrowork1)
    # Add Edge from Source to HydroWorks
    system.add_edge(Edge(reservoir, hydrowork1, capacity=80))

    # Add Demand Node
    demand1= DemandNode("demand1", 
                            easting=20, 
                            northing=0, 
                            constant_demand_rate=40, 
                            non_consumptive_rate=0, 
                            num_time_steps=num_time_steps,
                            weight=1)
    demand2= DemandNode("demand2", 
                            easting=20, 
                            northing=10, 
                            constant_demand_rate=35, 
                            non_consumptive_rate=5, 
                            num_time_steps=num_time_steps,
                            weight=1)
    system.add_node(demand1)
    system.add_node(demand2)
    # Add Edge from HydroWorks to Demand Node
    system.add_edge(Edge(hydrowork1, demand1, capacity=100))
    system.add_edge(Edge(hydrowork1, demand2, capacity=100))
    
    # Add Sink Node
    sink = SinkNode("Sink",
                     easting=30, 
                     northing=5, 
                     constant_min_flow=10,
                     weight=1, 
                     num_time_steps=num_time_steps)
    system.add_node(sink)
    # Add Edge from Demand Node to Sink Node
    system.add_edge(Edge(demand1, sink, capacity=100))
    system.add_edge(Edge(demand2, sink, capacity=100))
    

    system._check_network()
    return system
