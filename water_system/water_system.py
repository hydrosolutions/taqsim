"""
This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation.

The WaterSystem class allows users to add nodes and edges to the system and run simulations
"""

import networkx as nx
import pandas as pd
from .structure import SupplyNode, StorageNode, HydroWorks, DemandNode, SinkNode
from .edge import Edge
from .visualization import WaterSystemVisualizer

class WaterSystem:
    """
    Represents a water system as a directed graph and provides methods for simulation and visualization.

    Attributes:
        graph (nx.DiGraph): A NetworkX directed graph representing the water system.
        time_steps (int): The number of time steps in the most recent simulation.
    """

    def __init__(self, dt=2629800):  # Default to average month in seconds (365.25 days / 12 months * 24 hours * 3600 seconds)
        """
        Initialize a new WaterSystem instance.

        Args:
            dt (float): The length of each time step in seconds. Defaults to one month.
        """
        self.graph = nx.DiGraph()
        self.time_steps = 0
        self.dt = dt

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
        
        # Initialize lists to store data
        balance_data = []
        
        for t in range(self.time_steps):
        
            # Initialize volumes for current timestep
            volumes = {
                'time_step': t,
                'storage_start': 0.0, # Storage at start of timestep
                'storage_end': 0.0,  # Storage at end of timestep
                'storage_change': 0.0,  # Change in storage (current - previous)
                'reservoir spills': 0.0,    # Reservoir spills
                'reservoir ET losses': 0.0,    # Reservoir evaporation losses
                'source': 0.0,    # Supply node contributions
                'sink': 0.0,   # Sink node outflows
                'edge losses': 0.0,    # Edge losses
                'demands': 0.0,   # Total demand
                'supplied demand': 0.0,  # Satisfied demand
                
                
                
            }
            
            # Calculate volumes for each component
            for node_id, node_data in self.graph.nodes(data=True):
                node = node_data['node']
                
                if isinstance(node, SupplyNode):
                    # Convert supply rate to volume
                    supply_rate = node.get_supply_rate(t)
                    volumes['source'] += supply_rate * self.dt
                    
                elif isinstance(node, DemandNode):
                    # Convert demand rates to volumes
                    demand_rate = node.get_demand_rate(t)
                    satisfied_rate = node.satisfied_demand[t] if t < len(node.satisfied_demand) else 0
                    
                    volumes['demands'] += demand_rate * self.dt
                    volumes['supplied demand'] += satisfied_rate * self.dt
                    
                elif isinstance(node, StorageNode):
                    # Add current storage
                    if t < len(node.storage)-1:
                        volumes['storage_end'] += node.storage[t+1]
                
                        volumes['storage_start'] += node.storage[t]

                        # Calculate storage change (current - previous)
                        storage_change = node.storage[t+1] - node.storage[t]
                        volumes['storage_change'] += storage_change
                        volumes['reservoir ET losses'] += node.evaporation_losses[t]
                    else:  # Last timestep
                        # Set storage change to 0 for last timestep
                        volumes['storage_change'] += 0
                        volumes['storage_end'] += node.storage[t]
                        volumes['storage_start'] += node.storage[t]
                    
                    # Add spills
                    if t < len(node.spillway_register):
                        volumes['reservoir spills'] += node.spillway_register[t]
                        
                elif isinstance(node, SinkNode):
                    for edge in node.inflow_edges.values():
                        outflow_rate = edge.get_edge_outflow(t)
                        volumes['sink'] += outflow_rate * self.dt
            
            # Calculate total losses from all edges
            for _, _, edge_data in self.graph.edges(data=True):
                edge = edge_data['edge']
                if t < len(edge.losses):
                    volumes['edge losses'] += edge.losses[t] * self.dt
            
            # Calculate balance error
            # For each timestep: source = supplied + outflow + losses + spills + storage_change
            volumes['balance_error'] = (
                volumes['source'] 
                - volumes['supplied demand']
                - volumes['sink']
                - volumes['edge losses']
                - volumes['reservoir spills']
                - volumes['reservoir ET losses']
                - volumes['storage_change']
            )
            
            balance_data.append(volumes)
        
        # Create DataFrame and round values
        df = pd.DataFrame(balance_data)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        
        return df