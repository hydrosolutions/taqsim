import networkx as nx
import matplotlib.pyplot as plt
from .structure import SupplyNode, StorageNode, DemandNode

class WaterSystem:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.time_steps = 0

    def add_node(self, node):
        node_type = type(node).__name__
        self.graph.add_node(node.id, node=node, node_type=node_type)

    def add_edge(self, edge):
        self.graph.add_edge(edge.source.id, edge.target.id, edge=edge)

    def simulate(self, time_steps):
        self.time_steps = time_steps
        for t in range(time_steps):
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]['node'].update(t)
            for _, _, edge_data in self.graph.edges(data=True):
                edge_data['edge'].update(t)

    def visualize(self):
        # Determine node layers for multipartite layout
        layers = {'SupplyNode': 0, 'StorageNode': 1, 'DemandNode': 2}
        colors = []
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data['node_type']
            if node_type not in layers:
                layers[node_type] = 1  # Default to middle layer for unknown types
            
            if node_type == 'SupplyNode':
                colors.append('skyblue')
            elif node_type == 'StorageNode':
                colors.append('lightgreen')
            elif node_type == 'DemandNode':
                colors.append('lightcoral')
            else:
                colors.append('lightgray')

        # Create multipartite layout
        pos = nx.multipartite_layout(self.graph, subset_key='node_type')

        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)

        # Draw edge labels (flow values)
        edge_labels = {}
        for (u, v, data) in self.graph.edges(data=True):
            if data['edge'].flow:
                edge_labels[(u, v)] = f"{data['edge'].flow[-1]:.2f}"
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f"Water System - Time Step: {self.time_steps}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()