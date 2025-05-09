"""
This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation.

The WaterSystem class allows users to add nodes and edges to the system and run simulations
"""
from typing import Dict, List, Union, Optional
import networkx as nx
import pandas as pd
from .structure import SupplyNode, StorageNode, HydroWorks, DemandNode, SinkNode, RunoffNode
from .edge import Edge
import numpy as np

class WaterSystem:
    """
    Represents a water system as a directed graph and provides methods for simulation and visualization.

    Attributes:
        graph (nx.DiGraph): A NetworkX directed graph representing the water system.
        time_steps (int): The number of time steps in the most recent simulation.
    """

    def __init__(self, dt: float = 2629800, start_year: int = 2017, start_month: int = 1) -> None:
        """
        Initialize a new WaterSystem instance.

        Args:
            dt (float): The length of each time step in seconds. Defaults to one month.
        """
        self.graph = nx.DiGraph()
        self.time_steps = 0
        self.dt = dt
        self.start_year = start_year
        self.start_month = start_month

        self.has_been_checked = False  # Flag to indicate if the network has been checked
        
    def add_node(self, node: Union[SupplyNode, StorageNode, HydroWorks, DemandNode, SinkNode, RunoffNode]) -> None:
        """
        Add a node to the water system.

        Args:
            node (Node): The node to be added to the system.

        This method adds the node to the graph and stores its type as an attribute.
        """
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
        self._check_edge_properties()
        self._check_data_consistency()
        print("Network checking was successfull.")
        self.has_been_checked = True  # Set the flag to indicate the network has been checked

    def _check_network_structure(self)-> None:
        """Check overall network structure and connectivity."""
        # Check for empty network
        if len(self.graph) == 0:
            raise ValueError("Network is empty. Add nodes and edges before simulation.")

        # Check for isolated nodes
        if not nx.is_weakly_connected(self.graph):
            isolated = [n for n in self.graph.nodes() 
                       if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) == 0]
            raise ValueError(f"Network contains isolated nodes: {isolated}")

        # Check for cycles
        try:
            nx.find_cycle(self.graph)
            raise ValueError("Network contains cycles. Water system must be acyclic.")
        except nx.NetworkXNoCycle:
            pass  # This is what we want - no cycles

        # Check for existence of supply and demand/sink nodes
        node_types = {type(data['node']) for _, data in self.graph.nodes(data=True)}
        if SupplyNode not in node_types:
            raise ValueError("Network must contain at least one SupplyNode")
        if not any(t in node_types for t in [DemandNode, SinkNode]):
            raise ValueError("Network must contain at least one DemandNode or SinkNode")

        # Check if all paths lead to demand or sink
        terminal_nodes = {n for n, data in self.graph.nodes(data=True) 
                         if isinstance(data['node'], (DemandNode, SinkNode))}
        for node in self.graph.nodes():
            if node not in terminal_nodes:
                paths_exist = any(nx.has_path(self.graph, node, term) 
                                for term in terminal_nodes)
                if not paths_exist:
                    raise ValueError(f"Node {node} has no path to any DemandNode or SinkNode")

    def _check_node_configuration(self)-> None:
        """Check individual node configurations and connections."""
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            in_degree = len(node.inflow_edges)
            out_degree = len(node.outflow_edges)

            # Check SupplyNode configuration
            if isinstance(node, (SupplyNode, RunoffNode)):
                if in_degree > 0:
                    raise ValueError(f"Node {node_id} should not have any inflows")
                if out_degree != 1:
                    raise ValueError(f"Node {node_id} must have exactly one outflow")

            # Check SinkNode configuration
            elif isinstance(node, SinkNode):
                if out_degree > 0:
                    raise ValueError(f"SinkNode {node_id} should not have any outflows")
                if in_degree == 0:
                    raise ValueError(f"SinkNode {node_id} must have one inflow")

            # Check StorageNode configuration
            elif isinstance(node, (StorageNode, DemandNode)):
                if in_degree == 0:
                    raise ValueError(f"Node {node_id} must have at least one inflow")
                if out_degree != 1:
                    raise ValueError(f"Node {node_id} must have exactly one outflow")
                
            # Check HydroWorks configuration
            elif isinstance(node, HydroWorks):
                if in_degree == 0:
                    raise ValueError(f"HydroWorks {node_id} must have at least one inflow")
                if out_degree == 0:
                    raise ValueError(f"HydroWorks {node_id} must have at least one outflow")

            # Check single outflow constraint for other nodes
            elif not isinstance(node, (HydroWorks, SinkNode)):
                if out_degree != 1:
                    raise ValueError(
                        f"Node {node_id} has {out_degree} outflows. "
                        f"All nodes except HydroWorks and SinkNode must have exactly one outflow."
                    )

            # Check capacity mismatches
            if isinstance(node, (DemandNode)):
                total_inflow_capacity = sum(edge.capacity for edge in node.inflow_edges.values())
                total_outflow_capacity = sum(edge.capacity for edge in node.outflow_edges.values())
                
                if total_outflow_capacity < total_inflow_capacity:
                    print(
                        f"Warning: Node {node_id} has lower outflow capacity ({total_outflow_capacity} m³/s) "
                        f"than inflow capacity ({total_inflow_capacity} m³/s). "
                        f"This can lead to undefined water losses and errors in the water balance."
                    )

    def _check_edge_properties(self)-> None:
        """Check edge properties and connections."""
        for source, target, edge_data in self.graph.edges(data=True):
            edge = edge_data['edge']
    

            # Check edge capacity
            if edge.capacity <= 0:
                raise ValueError(f"Edge from {source} to {target} has invalid capacity: {edge.capacity}")

            # Check loss factors
            if edge.loss_factor < 0 or edge.loss_factor > 1:
                raise ValueError(
                    f"Edge from {source} to {target} has invalid loss factor: {edge.loss_factor}"
                )
            if edge.loss_factor > 0.5:
                print(f"Warning: Edge from {source} to {target} has unusually high loss factor: {edge.loss_factor}")

    def _check_data_consistency(self)-> None:
        """Check consistency of node data and time series."""
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']

            # Check SupplyNode data
            if isinstance(node, SupplyNode):
                if not node.supply_rates:
                    raise ValueError(f"SupplyNode {node_id} has no supply rates defined")

            # Check DemandNode data
            elif isinstance(node, DemandNode):
                if not node.demand_rates:
                    raise ValueError(f"DemandNode {node_id} has no demand rates defined")

            # Check StorageNode data
            elif isinstance(node, StorageNode):
                # Check hv relationships
                if not node.hv_data:
                    raise ValueError(f"StorageNode {node_id} missing height-volume-area relationship")
                
                # Check evaporation data if node has evaporation rates
                if node.evaporation_rates is not None:
                    if not node.evaporation_rates:
                        raise ValueError(f"StorageNode {node_id} has empty evaporation rates")

                # Check initial storage
                if node.storage and node.storage[0] > node.capacity:
                    raise ValueError(
                        f"StorageNode {node_id} initial storage ({node.storage[0]}) "
                        f"exceeds capacity ({node.capacity})"
                    )
      
    def simulate(self, time_steps: int) -> None:
        """
        Run the water system simulation for a specified number of time steps.

        Args:
            time_steps (int): The number of time steps to simulate.

        This method updates all nodes and edges in the system for each time step.
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
                node_data['node'].update(t, self.dt)

    def get_water_balance(self):
        """
        Calculate system-wide water balance for each time step using volumes in m³.
        
        Returns:
            pandas.DataFrame: A DataFrame containing water balance volumes for each timestep
        """
        if self.time_steps == 0:
            return pd.DataFrame()
        
        # Initialize arrays to store data
        time_steps = np.arange(self.time_steps)
        storage_start = np.zeros(self.time_steps)
        storage_end = np.zeros(self.time_steps)
        storage_change = np.zeros(self.time_steps)
        reservoir_et_losses = np.zeros(self.time_steps)
        reservoir_spills = np.zeros(self.time_steps)
        hydroworks_spills = np.zeros(self.time_steps)
        source = np.zeros(self.time_steps)
        surfacerunoff = np.zeros(self.time_steps)  # Added for runoff
        sink = np.zeros(self.time_steps)
        edge_losses = np.zeros(self.time_steps)
        demands = np.zeros(self.time_steps)
        supplied_demand = np.zeros(self.time_steps)
        unmet_demand = np.zeros(self.time_steps)
        
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            
            if isinstance(node, SupplyNode):
                supply_rates = np.array([node.get_supply_rate(t) for t in time_steps])
                source += supply_rates * self.dt
            
            # Add runoff contribution 
            elif isinstance(node, RunoffNode):
                runoff_rates = np.array([node.get_runoff(t) for t in time_steps])
                surfacerunoff += runoff_rates * self.dt
                
            elif isinstance(node, DemandNode):
                demand_rates = np.array([node.get_demand_rate(t) for t in time_steps])
                satisfied_rates = np.array([node.satisfied_consumptive_demand[t] if t < len(node.satisfied_consumptive_demand) else 0 for t in time_steps])
                
                demands += demand_rates * self.dt
                supplied_demand += satisfied_rates * self.dt
                unmet_demand += (demand_rates - satisfied_rates) * self.dt
                
            elif isinstance(node, StorageNode):
                storage = np.array(node.storage[:self.time_steps+1])
                storage_start += storage[:-1]
                storage_end += storage[1:]
                storage_change = storage_end - storage_start
                reservoir_et_losses += np.array(node.evaporation_losses[:self.time_steps])
                reservoir_spills += np.array(node.spillway_register[:self.time_steps])
                
            elif isinstance(node, HydroWorks):
                hydroworks_spills += np.array(node.spill_register[:self.time_steps])
                
            elif isinstance(node, SinkNode):
                for edge in node.inflow_edges.values():
                    outflow_rates = np.array([edge.get_edge_flow_after_losses(t) for t in time_steps])
                    sink += outflow_rates * self.dt
        
        for _, _, edge_data in self.graph.edges(data=True):
            edge = edge_data['edge']
            edge_losses += np.array(edge.losses[:self.time_steps]) * self.dt
        
        balance_error = (
            source 
            + surfacerunoff  # Add runoff to inputs
            - supplied_demand
            - sink
            - edge_losses
            - reservoir_spills
            - reservoir_et_losses
            - hydroworks_spills
            - storage_change
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'time_step': time_steps,
            'storage_start': storage_start,
            'storage_end': storage_end,
            'storage_change': storage_change,
            'reservoir ET losses': reservoir_et_losses,
            'reservoir spills': reservoir_spills,
            'hydroworks spills': hydroworks_spills,
            'source': source,
            'surfacerunoff': surfacerunoff,  # Added runoff column
            'sink': sink,
            'edge losses': edge_losses,
            'demands': demands,
            'supplied demand': supplied_demand,
            'unmet demand': unmet_demand,
            'balance_error': balance_error
        })
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        
        return df