"""
This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation and matplotlib for visualization.

The WaterSystem class allows users to add nodes and edges to the system, run simulations,
and visualize the results.
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from .structure import SupplyNode, StorageNode, DemandNode, SinkNode
from .edge import Edge

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
        self.time_steps = time_steps
        for t in range(time_steps):
            # Update nodes in order: SupplyNode, StorageNode, DemandNode, SinkNode
            for node_type in [SupplyNode, StorageNode, DemandNode, SinkNode]:
                for node_id, node_data in self.graph.nodes(data=True):
                    if isinstance(node_data['node'], node_type):
                        node_data['node'].update(t)
            
            # Update edges after all nodes have been updated
            for _, _, edge_data in self.graph.edges(data=True):
                if not isinstance(edge_data['edge'].source, (SupplyNode, StorageNode, DemandNode)):
                    edge_data['edge'].update(t)

    def visualize(self):
        """
        Visualize the water system using matplotlib.

        This method creates a multipartite layout of the system, with nodes color-coded by type
        and edges labeled with their final flow values. The SinkNode is positioned on the left side.
        """
        # Determine node layers for multipartite layout
        layers = {'SinkNode': 0, 'DemandNode': 1, 'StorageNode': 2, 'SupplyNode': 3}
        node_colors = {'SinkNode': 'lightgray', 'SupplyNode': 'skyblue', 'StorageNode': 'lightgreen', 'DemandNode': 'lightcoral'}
        
        # Assign positions and colors
        pos = {}
        colors = []
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data['node_type']
            if node_type not in layers:
                layers[node_type] = 0  # Default to middle layer for unknown types
            layer = layers[node_type]
            pos[node_id] = (layer, len([n for n in pos if pos[n][0] == layer]))
            colors.append(node_colors.get(node_type, 'lightgray'))

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
    
    def print_water_balances(self):
        """
        Print water balances for each node at each time step.
        """
        for time_step in range(self.time_steps):
            print(f"\nTime Step {time_step}:")
            for node_id, node_data in self.graph.nodes(data=True):
                node = node_data['node']
                inflow = sum(edge.get_flow(time_step) for edge in node.inflows.values())
                outflow = sum(edge.get_flow(time_step) for edge in node.outflows.values())
                
                if isinstance(node, DemandNode):
                    balance = inflow - outflow - node.satisfied_demand[time_step]
                    print(f"  {node_id}: Inflow = {inflow:.2f}, Outflow = {outflow:.2f}, "
                        f"Satisfied Demand = {node.satisfied_demand[time_step]:.2f}, "
                        f"Balance = {balance:.2f}")
                elif isinstance(node, StorageNode):
                    storage_change = node.storage[time_step + 1] - node.storage[time_step] if time_step + 1 < len(node.storage) else 0
                    balance = inflow - outflow - storage_change
                    print(f"  {node_id}: Inflow = {inflow:.2f}, Outflow = {outflow:.2f}, "
                        f"Storage Change = {storage_change:.2f}, Balance = {balance:.2f}")
                else:
                    balance = inflow - outflow
                    print(f"  {node_id}: Inflow = {inflow:.2f}, Outflow = {outflow:.2f}, "
                        f"Balance = {balance:.2f}")
                    
    def get_water_balance_table(self):
        """
        Generate a table with water balance data for all nodes across all time steps.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the water balance data.
                        Columns represent different aspects of each node's water balance.
                        Rows represent time steps.
        """
        data = []
        node_columns = []

        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            node_type = type(node).__name__

            # Common columns for all node types
            base_columns = [
                f"{node_id}_Inflow",
                f"{node_id}_Outflow",
            ]

            # Add type-specific columns
            if isinstance(node, SupplyNode):
                type_columns = [f"{node_id}_SupplyRate"]
            elif isinstance(node, StorageNode):
                type_columns = [f"{node_id}_Storage", f"{node_id}_StorageChange"]
            elif isinstance(node, DemandNode):
                type_columns = [f"{node_id}_Demand", f"{node_id}_SatisfiedDemand", f"{node_id}_Deficit"]
            elif isinstance(node, SinkNode):
                type_columns = []  # SinkNodes don't have additional columns
            else:
                type_columns = []  # For any other node types

            node_columns.extend(base_columns + type_columns)

        # Initialize the data list with the correct number of time steps
        data = [dict.fromkeys(node_columns, 0) for _ in range(self.time_steps)]

        # Populate the data
        for time_step in range(self.time_steps):
            for node_id, node_data in self.graph.nodes(data=True):
                node = node_data['node']
                inflow = sum(edge.get_flow(time_step) for edge in node.inflows.values())
                outflow = sum(edge.get_flow(time_step) for edge in node.outflows.values())

                data[time_step][f"{node_id}_Inflow"] = inflow
                data[time_step][f"{node_id}_Outflow"] = outflow

                if isinstance(node, SupplyNode):
                    data[time_step][f"{node_id}_SupplyRate"] = node.get_supply_rate(time_step)
                elif isinstance(node, StorageNode):
                    data[time_step][f"{node_id}_Storage"] = node.storage[time_step]
                    storage_change = node.storage[time_step] - node.storage[time_step - 1] if time_step > 0 else node.storage[0]
                    data[time_step][f"{node_id}_StorageChange"] = storage_change
                elif isinstance(node, DemandNode):
                    data[time_step][f"{node_id}_Demand"] = node.demand_rate
                    data[time_step][f"{node_id}_SatisfiedDemand"] = node.satisfied_demand[time_step]
                    data[time_step][f"{node_id}_Deficit"] = node.demand_rate - node.satisfied_demand[time_step]
                # No additional data needed for SinkNode

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
        
        # Add a 'TimeStep' column
        df.insert(0, 'TimeStep', range(self.time_steps))

        return df