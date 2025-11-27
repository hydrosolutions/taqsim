"""
Objective functions for water system optimization.

This module provides reusable objective functions for multi-objective optimization
of water resource systems. The objectives are designed to be compatible with both
DEAP and pymoo optimization frameworks and can be flexibly combined as needed.

All objectives are returned as annualized values in cubic kilometers (km³/year).

Functions:
    - regular_demand_deficit: Total annual regular demand deficit.
    - priority_demand_deficit: Total annual priority demand deficit.
    - sink_node_min_flow_deficit: Total annual minimum flow deficit at sink nodes.
    - total_spillage: Total annual spillage from hydroworks and reservoirs.
    - total_unmet_ecological_flow: Total annual unmet ecological flow for all edges with ecological flow requirements.

Arguments:
    system: The water system object containing the network graph and simulation results.
    *_ids: Lists of node IDs relevant to each objective.
    dt: Time step duration (seconds).
    num_years: Number of years in the simulation period.

Returns:
    Each function returns a float representing the annualized objective value in km³/year.
"""

import numpy as np


def regular_demand_deficit(system, regular_demand_ids, dt, num_years):
    """
    Calculate the total annual regular demand deficit.

    Args:
        system: The water system object.
        regular_demand_ids: List of node IDs for regular demand nodes.
        dt: Time step duration (seconds).
        num_years: Number of years in the simulation.

    Returns:
        float: Annualized regular demand deficit (km³/year).
    """
    total = 0
    for node_id in regular_demand_ids:
        demand_node = system.graph.nodes[node_id]["node"]
        deficit = np.array(demand_node.unmet_demand) * dt
        total += np.sum(deficit)
    return total / num_years / 1e9


def priority_demand_deficit(system, priority_demand_ids, dt, num_years):
    """
    Calculate the total annual priority demand deficit.

    Args:
        system: The water system object.
        priority_demand_ids: List of node IDs for priority demand nodes.
        dt: Time step duration (seconds).
        num_years: Number of years in the simulation.

    Returns:
        float: Annualized priority demand deficit (km³/year).
    """
    total = 0
    for node_id in priority_demand_ids:
        demand_node = system.graph.nodes[node_id]["node"]
        deficit = np.array(demand_node.unmet_demand) * dt
        total += np.sum(deficit)
    return total / num_years / 1e9


def sink_node_min_flow_deficit(system, sink_ids, dt, num_years):
    """
    Calculate the total annual minimum flow deficit at sink nodes.

    Args:
        system: The water system object.
        sink_ids: List of node IDs for sink nodes.
        dt: Time step duration (seconds).
        num_years: Number of years in the simulation.

    Returns:
        float: Annualized minimum flow deficit (km³/year).
    """
    total = 0
    for node_id in sink_ids:
        sink_node = system.graph.nodes[node_id]["node"]
        deficit = np.array(sink_node.flow_deficits) * dt
        total += np.sum(deficit)
    return total / num_years / 1e9


def total_spillage(system, hydroworks_ids, reservoir_ids, num_years):
    """
    Calculate the total annual spillage from hydroworks and reservoirs.

    Args:
        system: The water system object.
        hydroworks_ids: List of node IDs for hydropower works.
        reservoir_ids: List of node IDs for reservoirs.
        num_years: Number of years in the simulation.

    Returns:
        float: Annualized total spillage (km³/year).
    """
    total = 0
    for node_id in hydroworks_ids:
        total += np.sum(system.graph.nodes[node_id]["node"].spill_register)
    for node_id in reservoir_ids:
        total += np.sum(system.graph.nodes[node_id]["node"].spillway_register)
    return total / num_years / 1e9


def total_unmet_ecological_flow(system, dt, num_years):
    """
    Calculate the total annual unmet ecological flow for all edges with ecological_flow > 0.

    Args:
        system: The water system object.
        dt: Time step duration (seconds).
        num_years: Number of years in the simulation.

    Returns:
        float: Annualized total unmet ecological flow (km³/year).
    """
    total_unmet = 0
    for _, _, edge_data in system.graph.edges(data=True):
        edge = edge_data["edge"]
        if getattr(edge, "ecological_flow", 0) > 0:
            # Sum unmet ecological flow for this edge
            total_unmet += sum(edge.unmet_ecological_flow) * dt
    return total_unmet / num_years / 1e9
