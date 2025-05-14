"""
This module provides an example usage of the OptimizedWaterSystem class.
It demonstrates how to set up a water system with various node types and
optimize water allocation using linear programming.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from water_system import WaterSystem, Edge, SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode, OptimizedWaterSystem


def create_example_system():
    """
    Create an example water system with various node types.
    
    Returns:
        OptimizedWaterSystem: A configured water system ready for optimization.
    """
    # Initialize the system with monthly time steps
    system = OptimizedWaterSystem(dt=2629800, start_year=2020, start_month=1)
    
    # Create nodes
    # 1. Supply node (a river)
    river = SupplyNode(
        id="River",
        constant_supply_rate=10.0,  # 10 m³/s constant flow
        easting=0,
        northing=0, 
        start_year=2020,
        start_month=1,
        num_time_steps=12,  # 12 monthly time steps
    )
    
    # 2. Runoff node (surface runoff from rainfall)
    runoff = RunoffNode(
        id="Runoff",
        area=100.0,  # 100 km² catchment
        runoff_coefficient=0.3,  # 30% of rainfall becomes runoff
        easting=20,
        northing=0,
        # In a real case, you would provide a CSV with rainfall data
        rainfall_csv=None,
        start_year=2020,
        start_month=1,
        num_time_steps=12  # 12 monthly time steps
    )
    # Set some dummy rainfall data for demonstration
    runoff.rainfall_data = [50, 40, 30, 20, 15, 10, 5, 10, 20, 30, 40, 50]  # mm/month
    
    # 3. Storage node (a reservoir)
    reservoir = StorageNode(
        id="Reservoir",
        easting=50,
        northing=0,
        initial_storage=5000000,  # 5 million m³ initial storage
        dead_storage=1000000,  # 1 million m³ dead storage
        buffer_coef=0.5,  # Buffer coefficient for low storage operation
        hv_file='./data/baseline/reservoir/reservoir_kattakurgan_hv.csv',  # In a real case, you would provide a CSV with height-volume data
        num_time_steps=12, 
        evaporation_file='./data/baseline/reservoir/reservoir_evaporation.csv'
    )

    # Dummy evaporation rates in mm/month
    reservoir.evaporation_rates = [100, 110, 120, 130, 140, 150, 160, 150, 140, 130, 120, 110]
    
    # Set release parameters for the reservoir
    reservoir.set_release_params({
        'Vr': [1000000] * 12,  # Target release: 1 million m³/month
        'V1': [2000000] * 12,  # Buffer zone starts at 2 million m³
        'V2': [8000000] * 12   # Conservation zone starts at 8 million m³
    })
    
    # 4. Distribution node (water allocation)
    distribution = HydroWorks(
        id="Distribution",
        easting=100,
        northing=0
    )
    
    # 5. Agricultural demand node
    agriculture = DemandNode(
        id="Agriculture",
        easting=150,
        northing=20,
        constant_demand_rate=3.0,  # 3 m³/s
        non_consumptive_rate=0.5,  # 0.5 m³/s returns to system
        field_efficiency=0.75,  # 75% field efficiency
        conveyance_efficiency=0.8,  # 80% conveyance efficiency
        weight=2,  # Higher weight (more important)
        num_time_steps=12
    )
    
    # 6. Urban demand node
    urban = DemandNode(
        id="Urban",
        easting=150,
        northing=0,
        constant_demand_rate=2.0,  # 2 m³/s
        non_consumptive_rate=1.2,  # 1.2 m³/s returns to system (e.g., treated wastewater)
        field_efficiency=0.9,  # 90% end-use efficiency
        conveyance_efficiency=0.95,  # 95% conveyance efficiency
        weight=3,  # Highest weight (most important)
        num_time_steps=12
    )
    
    # 7. Environmental sink node (downstream flow requirement)
    environment = SinkNode(
        id="Environment",
        constant_min_flow=1.5,  # 1.5 m³/s minimum flow requirement
        easting=200,
        northing=0,
        weight=1,  # Lower weight
        num_time_steps=12
    )
    
    # Add nodes to the system
    system.add_node(river)
    system.add_node(runoff)
    system.add_node(reservoir)
    system.add_node(distribution)
    system.add_node(agriculture)
    system.add_node(urban)
    system.add_node(environment)
    
    # Create and add edges
    # River to Reservoir
    edge1 = Edge(
        source=river,
        target=reservoir,
        capacity=15.0,  # 15 m³/s capacity
        length=50.0,  # 50 km length
        loss_factor=0  # 2% loss per km
    )
    
    # Runoff to Reservoir
    edge2 = Edge(
        source=runoff,
        target=reservoir,
        capacity=10.0,  # 10 m³/s capacity
        length=30.0,  # 30 km length
        loss_factor=0  # 1% loss per km
    )
    
    # Reservoir to Distribution
    edge3 = Edge(
        source=reservoir,
        target=distribution,
        capacity=12.0,  # 12 m³/s capacity
        length=50.0,  # 50 km length
        loss_factor=0  # 1.5% loss per km
    )
    
    # Distribution to Agriculture
    edge4 = Edge(
        source=distribution,
        target=agriculture,
        capacity=5.0,  # 5 m³/s capacity
        length=50.0,  # 50 km length
        loss_factor=0  # 2% loss per km
    )
    
    # Distribution to Urban
    edge5 = Edge(
        source=distribution,
        target=urban,
        capacity=4.0,  # 4 m³/s capacity
        length=50.0,  # 50 km length
        loss_factor=0  # 1% loss per km
    )
    
    # Distribution to Environment
    edge6 = Edge(
        source=distribution,
        target=environment,
        capacity=8.0,  # 8 m³/s capacity
        length=100.0,  # 100 km length
        loss_factor=0  # 3% loss per km
    )
    
    # Agriculture to Environment (return flow)
    edge7 = Edge(
        source=agriculture,
        target=environment,
        capacity=2.0,  # 2 m³/s capacity
        length=70.0,  # 70 km length
        loss_factor=0.0  # 2.5% loss per km
    )
    
    # Urban to Environment (return flow)
    edge8 = Edge(
        source=urban,
        target=environment,
        capacity=3.0,  # 3 m³/s capacity
        length=50.0,  # 50 km length
        loss_factor=0.0  # 1% loss per km
    )
    
    # Add edges to the system
    system.add_edge(edge1)
    system.add_edge(edge2)
    system.add_edge(edge3)
    system.add_edge(edge4)
    system.add_edge(edge5)
    system.add_edge(edge6)
    system.add_edge(edge7)
    system.add_edge(edge8)
    
    # Set distribution parameters for the HydroWorks node
    # Distribution ratios vary by month to represent seasonal priorities
    # These distributions sum to 1 for each month and represent the target 
    # allocation of water from the Distribution node to its targets
    dry_season = [0.6, 0.3, 0.1]    # Agriculture, Urban, Environment (dry months)
    wet_season = [0.4, 0.3, 0.3]    # Agriculture, Urban, Environment (wet months)
    
    # Create monthly distribution parameters (simplified: 6 months dry, 6 months wet)
    agriculture_dist = dry_season[0] * np.ones(12)
    agriculture_dist[6:] = wet_season[0]
    
    urban_dist = dry_season[1] * np.ones(12)
    urban_dist[6:] = wet_season[1]
    
    environment_dist = dry_season[2] * np.ones(12)
    environment_dist[6:] = wet_season[2]
    
    distribution.set_distribution_parameters({
        "Agriculture": agriculture_dist,
        "Urban": urban_dist,
        "Environment": environment_dist
    })
    
    return system

def run_optimization_example():
    """
    Run an optimization example on a sample water system.
    """
    # Create example system
    system = create_example_system()
    
    # Number of time steps to simulate (12 months)
    time_steps = 12
    
    # Define weights for optimization objectives
    objective_weights = {
        'demand': 1.0,     # Weight for demand shortfalls
        'min_flow': 0.5,   # Weight for minimum flow violations
        'storage': 0.2,    # Weight for final storage volumes
        'losses': 0.1      # Weight for system losses
    }
    
    # Run optimization
    print("\nOptimizing water system...")
    optimization_results = system.optimize(time_steps, objective_weights)
    
    # Get optimization summary
    summary = system.get_optimization_summary()
    print("\nOptimization Summary:")
    print(summary)
    
    # Run comparison with standard simulation
    print("\nComparing optimization with standard simulation...")
    comparison = system.compare_with_simulation(time_steps)
    print("\nComparison Results:")
    print(comparison)
    
    # Get water balance
    water_balance = system.get_water_balance()
    print("\nWater Balance:")
    print(water_balance.head())
    
    # Plot results
    print("\nGenerating plots...")
    plot_optimization_results(system, time_steps)
    
    print("\nOptimization example completed successfully.")
    
def plot_optimization_results(system, time_steps):
    """
    Plot optimization results for visualization.
    
    Args:
        system (OptimizedWaterSystem): The optimized water system.
        time_steps (int): Number of time steps simulated.
    """
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Time labels for x-axis
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Plot 1: Reservoir Storage
    plt.figure(figsize=(10, 6))
    
    for node_id, node_data in system.graph.nodes(data=True):
        node = node_data['node']
        if isinstance(node, StorageNode):
            plt.plot(range(len(node.storage)), node.storage, 'b-', linewidth=2, label='Storage Volume')
            plt.axhline(y=node.capacity, color='r', linestyle='--', label='Capacity')
            plt.axhline(y=node.dead_storage, color='g', linestyle='--', label='Dead Storage')
            
            # Add release policy zones if available
            if hasattr(node, 'release_params') and 'V1' in node.release_params and 'V2' in node.release_params:
                v1_values = node.release_params['V1']
                v2_values = node.release_params['V2']
                
                if len(v1_values) == 12:
                    plt.plot(range(12), v1_values, 'k:', label='V1 (Buffer Zone)')
                    plt.plot(range(12), v2_values, 'k-.', label='V2 (Conservation Zone)')
            
    plt.title('Reservoir Storage Volume Over Time')
    plt.xlabel('Month')
    plt.ylabel('Volume (m³)')
    plt.xticks(range(min(12, time_steps)), months[:min(12, time_steps)])
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'reservoir_storage.png')
    
    # Plot 2: Demand Satisfaction
    plt.figure(figsize=(10, 6))
    
    for node_id, node_data in system.graph.nodes(data=True):
        node = node_data['node']
        if isinstance(node, DemandNode):
            # Calculate total satisfied demand
            satisfied_total = [
                (node.satisfied_consumptive_demand[t] if t < len(node.satisfied_consumptive_demand) else 0) +
                (node.satisfied_non_consumptive_demand[t] if t < len(node.satisfied_non_consumptive_demand) else 0)
                for t in range(time_steps)
            ]
            
            plt.plot(range(time_steps), node.demand_rates[:time_steps], 'r-', label=f'{node_id} Demand')
            plt.plot(range(time_steps), satisfied_total, 'g-', label=f'{node_id} Satisfied')
    
    plt.title('Demand Satisfaction')
    plt.xlabel('Month')
    plt.ylabel('Flow Rate (m³/s)')
    plt.xticks(range(min(12, time_steps)), months[:min(12, time_steps)])
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'demand_satisfaction.png')
    
    # Plot 3: Minimum Flow Requirements
    plt.figure(figsize=(10, 6))
    
    for node_id, node_data in system.graph.nodes(data=True):
        node = node_data['node']
        if isinstance(node, SinkNode):
            min_flows = node.min_flows[:time_steps]
            actual_flows = node.flow_history[:time_steps]
            
            plt.plot(range(time_steps), min_flows, 'r--', label=f'{node_id} Min Requirement')
            plt.plot(range(time_steps), actual_flows, 'b-', label=f'{node_id} Actual Flow')
    
    plt.title('Minimum Flow Requirements')
    plt.xlabel('Month')
    plt.ylabel('Flow Rate (m³/s)')
    plt.xticks(range(min(12, time_steps)), months[:min(12, time_steps)])
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'min_flow_requirements.png')
    
    # Plot 4: Supply Sources
    plt.figure(figsize=(10, 6))
    
    for node_id, node_data in system.graph.nodes(data=True):
        node = node_data['node']
        if isinstance(node, SupplyNode):
            plt.plot(range(time_steps), node.supply_rates[:time_steps], 'b-', label=f'{node_id} Supply')
        elif isinstance(node, RunoffNode):
            plt.plot(range(time_steps), node.runoff_history[:time_steps], 'g-', label=f'{node_id} Runoff')
    
    plt.title('Water Supply Sources')
    plt.xlabel('Month')
    plt.ylabel('Flow Rate (m³/s)')
    plt.xticks(range(min(12, time_steps)), months[:min(12, time_steps)])
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'supply_sources.png')
    
    # Plot 5: System Losses
    plt.figure(figsize=(10, 6))
    
    water_balance = system.get_water_balance()
    if not water_balance.empty:
        plt.plot(water_balance['time_step'], water_balance['edge losses'], 'r-', label='Edge Losses')
        plt.plot(water_balance['time_step'], water_balance['reservoir ET losses'], 'b-', label='Reservoir ET Losses')
        plt.plot(water_balance['time_step'], water_balance['reservoir spills'], 'g-', label='Reservoir Spills')
        plt.plot(water_balance['time_step'], water_balance['hydroworks spills'], 'y-', label='HydroWorks Spills')
        
        # Plot total losses
        total_losses = (
            water_balance['edge losses'] + 
            water_balance['reservoir ET losses'] + 
            water_balance['reservoir spills'] + 
            water_balance['hydroworks spills']
        )
        plt.plot(water_balance['time_step'], total_losses, 'k--', label='Total Losses')
    
    plt.title('System Losses')
    plt.xlabel('Month')
    plt.ylabel('Volume (m³)')
    plt.xticks(range(min(12, time_steps)), months[:min(12, time_steps)])
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'system_losses.png')
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    run_optimization_example()