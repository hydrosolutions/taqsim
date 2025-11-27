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

This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation.

The WaterSystem class allows users to add nodes and edges to the system and run simulations
with specialized node types including SupplyNode, StorageNode, DemandNode, SinkNode,
HydroWorks, and RunoffNode.
"""

from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

from .edge import Edge
from .nodes import DemandNode, HydroWorks, RunoffNode, SinkNode, StorageNode, SupplyNode
from .validation import (
    validate_month,
    validate_positive_float,
    validate_year,
)

# Define a type for any valid node type
NodeType = Union[SupplyNode, StorageNode, HydroWorks, DemandNode, SinkNode, RunoffNode]


class WaterSystem:
    """
    Represents a water system as a directed graph and provides methods for simulation and visualization.

    Attributes:
        graph (nx.DiGraph): A NetworkX directed graph representing the water system.
        time_steps (int): The number of time steps in the most recent simulation.
        dt (float): The length of each time step in seconds.
        start_year (int): The start year of the simulation period.
        start_month (int): The start month of the simulation period.
        has_been_checked (bool): Flag indicating if the network configuration has been validated.
    """

    def __init__(self, dt: float = 2629800, start_year: int = 2017, start_month: int = 1) -> None:
        """
        Initialize a new WaterSystem instance.

        Args:
            dt (float): The length of each time step in seconds. Defaults to one month (2629800 seconds).
            start_year (int): The starting year for the simulation.
            start_month (int): The starting month (1-12) for the simulation.
        """
        WaterSystem.reset_node_registries()  # Reset node registries for new instance

        # Validate dt
        validate_positive_float(dt, "dt")
        # Validate start year and month
        validate_year(start_year)
        validate_month(start_month)

        self.graph = nx.DiGraph()
        self.time_steps = 0
        self.dt = dt
        self.start_year = start_year
        self.start_month = start_month
        self.has_been_checked = False  # Flag to indicate if the network has been checked

    @staticmethod
    def reset_node_registries():
        """Reset all node class registries"""
        StorageNode.all_ids.clear()
        HydroWorks.all_ids.clear()
        DemandNode.all_ids.clear()
        SinkNode.all_ids.clear()

    def add_node(self, node: NodeType) -> None:
        """
        Add a node to the water system.

        Args:
            node: The node to be added to the system (must be one of the specialized node types).

        This method adds the node to the graph and stores its specific type as an attribute.
        """
        # Get the actual node type name
        node_type = type(node).__name__
        self.graph.add_node(node.id, node=node, node_type=node_type)

    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge to the water system.

        Args:
            edge (Edge): The edge to be added to the system.

        This method adds the edge to the graph, connecting its source and target nodes.
        """
        self.graph.add_edge(edge.source.id, edge.target.id, edge=edge)

    def _check_network(self) -> None:
        """
        Comprehensive check of network configuration for potential issues.
        Performs multiple validations on network structure, node configuration,
        edge properties, and data consistency.

        Raises:
            ValueError: If critical network configuration issues are found
        """
        self._check_network_structure()
        self._check_node_configuration()
        print("Network checking was successful.")
        self.has_been_checked = True  # Set the flag to indicate the network has been checked

    def _check_network_structure(self) -> None:
        """Check overall network structure and connectivity."""
        # Check for empty network
        if len(self.graph) == 0:
            raise ValueError("Network is empty. Add nodes and edges before simulation.")

        # Check for isolated nodes
        if not nx.is_weakly_connected(self.graph):
            isolated = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) == 0]
            raise ValueError(f"Network contains isolated nodes: {isolated}")

        # Check for cycles
        try:
            nx.find_cycle(self.graph)
            raise ValueError("Network contains cycles. Water system must be acyclic.")
        except nx.NetworkXNoCycle:
            pass  # This is what we want - no cycles

        # Check for existence of supply and demand/sink nodes
        node_types = set()
        for _, data in self.graph.nodes(data=True):
            node = data["node"]
            node_types.add(type(node))

        if SupplyNode not in node_types and RunoffNode not in node_types:
            raise ValueError("Network must contain at least one SupplyNode or RunoffNode")
        if SinkNode not in node_types:
            raise ValueError("Network must contain at least one SinkNode")

        # Check if all paths lead to demand or sink
        terminal_nodes = {n for n, data in self.graph.nodes(data=True) if isinstance(data["node"], (SinkNode))}
        for node in self.graph.nodes():
            if node not in terminal_nodes:
                paths_exist = any(nx.has_path(self.graph, node, term) for term in terminal_nodes)
                if not paths_exist:
                    raise ValueError(f"Node {node} has no path to any SinkNode")

    def _check_node_configuration(self) -> None:
        """Check individual node configurations and connections."""
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data["node"]

            # Check node types based on specific requirements
            if isinstance(node, (SupplyNode, RunoffNode)):
                in_degree = self.graph.in_degree(node_id)
                if in_degree > 0:
                    raise ValueError(f"Node {node_id} should not have any inflows")
                if not hasattr(node, "outflow_edge") or node.outflow_edge is None:
                    raise ValueError(f"Node {node_id} must have exactly one outflow")

            # Check SinkNode configuration
            elif isinstance(node, SinkNode):
                out_degree = self.graph.out_degree(node_id)
                if out_degree > 0:
                    raise ValueError(f"SinkNode {node_id} should not have any outflows")
                if self.graph.in_degree(node_id) == 0:
                    raise ValueError(f"SinkNode {node_id} must have at least one inflow")

            # Check StorageNode and DemandNode configuration
            elif isinstance(node, (StorageNode, DemandNode)):
                in_degree = self.graph.in_degree(node_id)
                if in_degree == 0:
                    raise ValueError(f"Node {node_id} must have at least one inflow")
                if not hasattr(node, "outflow_edge") or node.outflow_edge is None:
                    raise ValueError(f"Node {node_id} must have exactly one outflow")

            # Check HydroWorks configuration
            elif isinstance(node, HydroWorks):
                in_degree = self.graph.in_degree(node_id)
                out_degree = self.graph.out_degree(node_id)
                if in_degree == 0:
                    raise ValueError(f"HydroWorks {node_id} must have at least one inflow")
                if out_degree == 0:
                    raise ValueError(f"HydroWorks {node_id} must have at least one outflow")

            # Check capacity mismatches for DemandNode
            if isinstance(node, DemandNode):
                # Get all incoming and outgoing edges
                inflow_edges = [data["edge"] for _, _, data in self.graph.in_edges(node_id, data=True)]

                total_inflow_capacity = sum(edge.capacity for edge in inflow_edges)
                outflow_capacity = node.outflow_edge.capacity if node.outflow_edge else 0

                if outflow_capacity < total_inflow_capacity:
                    print(
                        f"Warning: Node {node_id} has lower outflow capacity ({outflow_capacity} m³/s) "
                        f"than inflow capacity ({total_inflow_capacity} m³/s). "
                        f"This can lead to undefined water losses and errors in the water balance."
                    )

    def simulate(self, time_steps: int) -> None:
        """
        Run the water system simulation for a specified number of time steps.

        Args:
            time_steps (int): The number of time steps to simulate.

        This method updates all nodes and edges in the system for each time step,
        following a topological sort to ensure proper flow propagation.
        """
        if not self.has_been_checked:
            self._check_network()  # Check network configuration before simulation
        self.time_steps = time_steps

        # Perform a topological sort to determine the correct order for node updates
        sorted_nodes = list(nx.topological_sort(self.graph))

        for t in range(time_steps):
            # Update nodes in topologically sorted order
            for node_id in sorted_nodes:
                node_data = self.graph.nodes[node_id]
                node_data["node"].update(t, self.dt)

    def _collect_node_data_for_timestep(self, time_steps: np.ndarray) -> dict[str, np.ndarray]:
        """
        Collect data from all nodes for all timesteps.

        Args:
            time_steps: Array of timestep indices

        Returns:
            Dictionary containing arrays of node data
        """
        storage_start = np.zeros(self.time_steps)
        storage_end = np.zeros(self.time_steps)
        reservoir_et_losses = np.zeros(self.time_steps)
        reservoir_spills = np.zeros(self.time_steps)
        hydroworks_spills = np.zeros(self.time_steps)
        source = np.zeros(self.time_steps)
        surfacerunoff = np.zeros(self.time_steps)
        sink = np.zeros(self.time_steps)
        sink_min_flow = np.zeros(self.time_steps)
        sink_min_flow_deficit = np.zeros(self.time_steps)
        demands = np.zeros(self.time_steps)
        demands_non_consumptive = np.zeros(self.time_steps)
        supplied_consumptive_demand = np.zeros(self.time_steps)
        supplied_non_consumptive_demand = np.zeros(self.time_steps)
        unmet_demand = np.zeros(self.time_steps)

        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data["node"]

            if isinstance(node, SupplyNode):
                supply_rates = np.array([node.supply_rates[t] for t in time_steps])
                source += supply_rates * self.dt

            elif isinstance(node, RunoffNode):
                runoff_rates = np.array([node.runoff_history[t] for t in time_steps])
                surfacerunoff += runoff_rates * self.dt

            elif isinstance(node, DemandNode):
                demand_rates = np.array([node.demand_rates[t] for t in time_steps])
                non_consumptive_rate = np.array([node.non_consumptive_rate] * self.time_steps)
                satisfied_consumptive_rates = np.array(
                    [
                        node.satisfied_consumptive_demand[t] if t < len(node.satisfied_consumptive_demand) else 0
                        for t in time_steps
                    ]
                )
                satisfied_non_consumptive_rates = np.array(
                    [
                        node.satisfied_non_consumptive_demand[t]
                        if t < len(node.satisfied_non_consumptive_demand)
                        else 0
                        for t in time_steps
                    ]
                )

                demands += demand_rates * self.dt
                demands_non_consumptive += non_consumptive_rate * self.dt
                supplied_consumptive_demand += satisfied_consumptive_rates * self.dt
                supplied_non_consumptive_demand += satisfied_non_consumptive_rates * self.dt
                unmet_demand += (demand_rates - satisfied_consumptive_rates - satisfied_non_consumptive_rates) * self.dt

            elif isinstance(node, StorageNode):
                storage = np.array(node.storage[: self.time_steps + 1])
                storage_start += storage[:-1]
                storage_end += storage[1:]
                reservoir_et_losses += np.array(node.evaporation_losses[: self.time_steps])
                reservoir_spills += np.array(node.spillway_register[: self.time_steps])

            elif isinstance(node, HydroWorks):
                hydroworks_spills += np.array(node.spill_register[: self.time_steps])

            elif isinstance(node, SinkNode):
                for source_id in node.inflow_edges:
                    edge = node.inflow_edges[source_id]
                    outflow_rates = np.array([edge.flow_after_losses[t] for t in time_steps])
                    sink += outflow_rates * self.dt

                sink_min_fl = np.array([node.min_flows[t] for t in time_steps])
                sink_min_flow += sink_min_fl * self.dt
                sink_deficit = np.array([node.flow_deficits[t] for t in time_steps])
                sink_min_flow_deficit += sink_deficit * self.dt

        storage_change = storage_end - storage_start

        return {
            "storage_start": storage_start,
            "storage_end": storage_end,
            "storage_change": storage_change,
            "reservoir_et_losses": reservoir_et_losses,
            "reservoir_spills": reservoir_spills,
            "hydroworks_spills": hydroworks_spills,
            "source": source,
            "surfacerunoff": surfacerunoff,
            "sink": sink,
            "sink_min_flow": sink_min_flow,
            "sink_min_flow_deficit": sink_min_flow_deficit,
            "demands": demands,
            "demands_non_consumptive": demands_non_consumptive,
            "supplied_consumptive_demand": supplied_consumptive_demand,
            "supplied_non_consumptive_demand": supplied_non_consumptive_demand,
            "unmet_demand": unmet_demand,
        }

    def _collect_edge_losses(self, time_steps: np.ndarray) -> np.ndarray:
        """
        Collect total edge losses across the network for all timesteps.

        Args:
            time_steps: Array of timestep indices

        Returns:
            Array of total edge losses in m³
        """
        edge_losses = np.zeros(self.time_steps)
        for _, _, edge_data in self.graph.edges(data=True):
            edge = edge_data["edge"]
            edge_losses += np.array(edge.losses[: self.time_steps]) * self.dt
        return edge_losses

    def _calculate_balance_error(self, node_data: dict[str, np.ndarray], edge_losses: np.ndarray) -> np.ndarray:
        """
        Calculate water balance error for all timesteps.

        Args:
            node_data: Dictionary containing node data arrays
            edge_losses: Array of edge losses

        Returns:
            Array of balance errors in m³
        """
        return (
            node_data["source"]
            + node_data["surfacerunoff"]
            - node_data["supplied_consumptive_demand"]
            - node_data["sink"]
            - edge_losses
            - node_data["reservoir_spills"]
            - node_data["reservoir_et_losses"]
            - node_data["hydroworks_spills"]
            - node_data["storage_change"]
        )

    def _create_balance_dataframe(
        self,
        time_steps: np.ndarray,
        node_data: dict[str, np.ndarray],
        edge_losses: np.ndarray,
        balance_error: np.ndarray,
    ) -> pd.DataFrame:
        """
        Create and format the water balance DataFrame.

        Args:
            time_steps: Array of timestep indices
            node_data: Dictionary containing node data arrays
            edge_losses: Array of edge losses
            balance_error: Array of balance errors

        Returns:
            Formatted DataFrame with water balance data
        """
        df = pd.DataFrame(
            {
                "time_step": time_steps,
                "storage_start": node_data["storage_start"],
                "storage_end": node_data["storage_end"],
                "storage_change": node_data["storage_change"],
                "reservoir ET losses": node_data["reservoir_et_losses"],
                "reservoir spills": node_data["reservoir_spills"],
                "hydroworks spills": node_data["hydroworks_spills"],
                "source": node_data["source"],
                "surfacerunoff": node_data["surfacerunoff"],
                "sink": node_data["sink"],
                "sink min flow requirement": node_data["sink_min_flow"],
                "sink min flow deficit": node_data["sink_min_flow_deficit"],
                "edge losses": edge_losses,
                "demands": node_data["demands"],
                "demands non consumptive": node_data["demands_non_consumptive"],
                "supplied consumptive demand": node_data["supplied_consumptive_demand"],
                "supplied non consumptive demand": node_data["supplied_non_consumptive_demand"],
                "unmet demand": node_data["unmet_demand"],
                "balance_error": balance_error,
            }
        )

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].round(3)

        return df

    def get_water_balance(self) -> pd.DataFrame:
        """
        Calculate system-wide water balance for each time step using volumes in m³.

        Returns:
            pandas.DataFrame: A DataFrame containing water balance volumes for each timestep
        """
        if self.time_steps == 0:
            return pd.DataFrame()

        time_steps = np.arange(self.time_steps)

        node_data = self._collect_node_data_for_timestep(time_steps)
        edge_losses = self._collect_edge_losses(time_steps)
        balance_error = self._calculate_balance_error(node_data, edge_losses)

        return self._create_balance_dataframe(time_steps, node_data, edge_losses, balance_error)
