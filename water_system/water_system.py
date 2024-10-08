"""
This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation and matplotlib for visualization.

The WaterSystem class allows users to add nodes and edges to the system, run simulations,
and visualize the results.
"""

import networkx as nx
import matplotlib.pyplot as plt
from .structure import SupplyNode, StorageNode, DemandNode, Sink

class WaterSystem:
    """
    Represents a water system as a directed graph and provides methods for simulation and visualization.

    Attributes:
        graph (nx.DiGraph): A NetworkX directed graph representing the water system.
        time_steps (int): The number of time steps in the most recent simulation.
    """

    def __init__(self):
        """
        Initialize a new WaterSystem instance.
        """
        self.graph = nx.DiGraph()
        self.time_steps = 0

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
        Run a simulation of the water system for a specified number of time steps.

        Args:
            time_steps (int): The number of time steps to simulate.

        This method updates each node and edge in the system for each time step.
        """
        self.time_steps = time_steps
        for t in range(time_steps):
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]['node'].update(t)
            for _, _, edge_data in self.graph.edges(data=True):
                edge_data['edge'].update(t)

    def visualize(self):
        """
        Visualize the water system using matplotlib.

        This method creates a custom layout of the system, with nodes color-coded by type
        and edges labeled with their final flow values.
        """
        plt.figure(figsize=(15, 10))
        
        # Create a custom layout
        pos = {
            'Supply1': (0, 0.5),
            'Reservoir': (0.3, 0.5),
            'Agriculture': (0.6, 0.75),
            'Domestic': (0.6, 0.25),
            'Sink': (0.9, 0.5)
        }
        
        # Define node colors and sizes
        node_colors = {
            'SupplyNode': 'skyblue',
            'StorageNode': 'lightgreen',
            'DemandNode': 'lightcoral',
            'SinkNode': 'lightgray'
        }
        node_sizes = {
            'SupplyNode': 3000,
            'StorageNode': 5000,
            'DemandNode': 3000,
            'SinkNode': 3000
        }
        
        # Draw nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data['node_type']
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[node_id],
                                node_color=node_colors.get(node_type, 'lightgray'),
                                node_size=node_sizes.get(node_type, 3000))
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        # Draw edge labels (flow values)
        edge_labels = {}
        for (u, v, data) in self.graph.edges(data=True):
            if data['edge'].flow:
                edge_labels[(u, v)] = f"{data['edge'].flow[-1]:.2f}"
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        # Add node information
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            x, y = pos[node_id]
            if isinstance(node, SupplyNode):
                plt.text(x, y-0.1, f"Supply: {node.supply_history[-1]:.2f}", ha='center', va='center')
            elif isinstance(node, StorageNode):
                plt.text(x, y-0.1, f"Storage: {node.storage[-1]:.2f}/{node.capacity}", ha='center', va='center')
            elif isinstance(node, DemandNode):
                plt.text(x, y-0.1, f"Demand: {node.demand_rate:.2f}", ha='center', va='center')
            elif isinstance(node, SinkNode):
                plt.text(x, y-0.1, f"Outflow: {node.outflow_history[-1]:.2f}", ha='center', va='center')
        
        plt.title(f"Water System - Time Step: {self.time_steps}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()