"""
This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation.

The WaterSystem class allows users to add nodes and edges to the system and run simulations
"""

import networkx as nx
import pandas as pd
from .structure import SupplyNode, StorageNode, HydroWorks, DemandNode, SinkNode
from .edge import Edge
import numpy as np

class WaterSystem:
    """
    Represents a water system as a directed graph and provides methods for simulation and visualization.

    Attributes:
        graph (nx.DiGraph): A NetworkX directed graph representing the water system.
        time_steps (int): The number of time steps in the most recent simulation.
    """

    def __init__(self, dt=2629800, start_year=2017, start_month=1):  # Default to average month in seconds (365.25 days / 12 months * 24 hours * 3600 seconds)
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
        
    def add_node(self, node):
        """
        Add a node to the water system.

        Args:
            node (Node): The node to be added to the system.

        This method adds the node to the graph and stores its type as an attribute.
        """
        node_type = type(node).__name__
        if node_type == 'SupplyNode':
            self.graph.add_node(node.id, node=node, node_type=node_type, supply_rates=node.supply_rates)
        else:
            self.graph.add_node(node.id, node=node, node_type=node_type)

    def add_edge(self, edge):
        """
        Add an edge to the water system.

        Args:
            edge (Edge): The edge to be added to the system.

        This method adds the edge to the graph, connecting its source and target nodes.
        """
        self.graph.add_edge(edge.source.id, edge.target.id, edge=edge)

    def _check_network(self):
        """
        Comprehensive check of network configuration for potential issues.
        Performs multiple validations on network structure, node configuration,
        edge properties, and data consistency.
        
        Raises:
            ValueError: If critical network configuration issues are found
        """
        #print("Checking network configuration...")
        self._check_network_structure()
        self._check_node_configuration()
        self._check_edge_properties()
        self._check_data_consistency()
        #print("Network configuration check complete. No issues found.")

    def _check_network_structure(self):
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

    def _check_node_configuration(self):
        """Check individual node configurations and connections."""
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            in_degree = len(node.inflow_edges)
            out_degree = len(node.outflow_edges)

            # Check SupplyNode configuration
            if isinstance(node, SupplyNode):
                if in_degree > 0:
                    raise ValueError(f"SupplyNode {node_id} should not have any inflows")
                if out_degree == 0:
                    raise ValueError(f"SupplyNode {node_id} must have one outflow")

            # Check SinkNode configuration
            elif isinstance(node, SinkNode):
                if out_degree > 0:
                    raise ValueError(f"SinkNode {node_id} should not have any outflows")
                if in_degree == 0:
                    raise ValueError(f"SinkNode {node_id} must have one inflow")

            # Check StorageNode configuration
            elif isinstance(node, StorageNode):
                if in_degree == 0:
                    raise ValueError(f"StorageNode {node_id} must have at least one inflow")
                if out_degree == 0:
                    raise ValueError(f"StorageNode {node_id} must have one outflow")
                
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
            if not isinstance(node, (StorageNode, SupplyNode, SinkNode)):
                total_inflow_capacity = sum(edge.capacity for edge in node.inflow_edges.values())
                total_outflow_capacity = sum(edge.capacity for edge in node.outflow_edges.values())
                
                if total_outflow_capacity < total_inflow_capacity:
                    print(
                        f"Warning: Node {node_id} has lower outflow capacity ({total_outflow_capacity} m³/s) "
                        f"than inflow capacity ({total_inflow_capacity} m³/s). "
                        f"This can lead to undefined water losses and errors in the water balance."
                    )

    def _check_edge_properties(self):
        """Check edge properties and connections."""
        for u, v, edge_data in self.graph.edges(data=True):
            edge = edge_data['edge']
            source_node = self.graph.nodes[u]['node']
            target_node = self.graph.nodes[v]['node']

            # Check edge capacity
            if edge.capacity <= 0:
                raise ValueError(f"Edge from {u} to {v} has invalid capacity: {edge.capacity}")

            # Check loss factors
            if edge.loss_factor < 0 or edge.loss_factor > 1:
                raise ValueError(
                    f"Edge from {u} to {v} has invalid loss factor: {edge.loss_factor}"
                )
            if edge.loss_factor > 0.5:
                print(f"Warning: Edge from {u} to {v} has unusually high loss factor: {edge.loss_factor}")

    def _check_data_consistency(self):
        """Check consistency of node data and time series."""
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']

            # Check SupplyNode data
            if isinstance(node, SupplyNode):
                if not node.supply_rates and node.default_supply_rate <= 0:
                    print(
                        f"Warning: SupplyNode {node_id} has no supply rates defined "
                        f"and default rate is {node.default_supply_rate}"
                    )

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
                    if len(node.evaporation_rates) == 0:
                        print(f"Warning: StorageNode {node_id} has empty evaporation rates")

                # Check initial storage
                if node.storage and node.storage[0] > node.capacity:
                    raise ValueError(
                        f"StorageNode {node_id} initial storage ({node.storage[0]}) "
                        f"exceeds capacity ({node.capacity})"
                    )
      
    def simulate(self, time_steps):
        """
        Run the water system simulation for a specified number of time steps.

        Args:
            time_steps (int): The number of time steps to simulate.

        This method updates all nodes and edges in the system for each time step.
        """
        self.time_steps = time_steps
        # Perform a topological sort to determine the correct order for node updates
        sorted_nodes = list(nx.topological_sort(self.graph))

        for t in range(time_steps):
            # Update nodes in topologically sorted order
            for node_id in sorted_nodes:
                node_data = self.graph.nodes[node_id]
                node_data['node'].update(t, self.dt)
                
                # Update edges after all nodes have been updated
                for _, _, edge_data in self.graph.edges(data=True):
                    if not isinstance(edge_data['edge'].source, (SupplyNode, StorageNode, HydroWorks, DemandNode)):
                        edge_data['edge'].update(t)
             
    def get_water_balance(self):
        """
        Calculate system-wide water balance for each time step using volumes in m³.
        Initial storage is only considered at the first timestep.
        Storage change is calculated as current minus previous storage, with zero at first timestep.
        
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
