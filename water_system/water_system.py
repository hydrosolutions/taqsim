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

    def visualize(self, filename='water_system_layout.png', display=True):
        """
        Create and save a network layout plot for the water system, showing:
        - Actual flows and capacities on edges
        - Demand satisfaction on demand nodes
        - Actual supply on supply nodes
        - Total inflow on sink nodes
        - Actual storage and capacity on storage nodes
        
        Args:
        filename (str): The name of the PNG file to save to.
        """
        pos = nx.spring_layout(self.graph, k=0.9, iterations=50)
        
        plt.figure(figsize=(10, 9))
        plt.title('Water System Network Layout and Flows', fontsize=20)
        
        node_colors = {
            SupplyNode: 'skyblue',
            StorageNode: 'lightgreen',
            DemandNode: 'salmon',
            SinkNode: 'lightgray'
        }
        
        node_size = 5000
        
        for node_type, color in node_colors.items():
            node_list = [node for node, data in self.graph.nodes(data=True) if isinstance(data['node'], node_type)]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=node_list, node_color=color, node_size=node_size, alpha=0.8)
        
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, arrowsize=75)
        
        # Update node labels
        labels = {}
        for node, data in self.graph.nodes(data=True):
            node_instance = data['node']
            if isinstance(node_instance, SupplyNode):
                actual_supply = node_instance.supply_history[-1] if node_instance.supply_history else 0
                labels[node] = f"{node}\nSupply Node\nActual: {actual_supply:.1f}"
            elif isinstance(node_instance, DemandNode):
                satisfied_demand = node_instance.satisfied_demand[-1] if node_instance.satisfied_demand else 0
                labels[node] = f"{node}\nDemand Node\n{satisfied_demand:.1f} ({node_instance.demand_rate})"
            elif isinstance(node_instance, SinkNode):
                total_inflow = sum(edge.flow[-1] if edge.flow else 0 for edge in node_instance.inflows.values())
                labels[node] = f"{node}\nSink Node\nTotal Inflow: {total_inflow:.1f}"
            elif isinstance(node_instance, StorageNode):
                actual_storage = node_instance.storage[-1] if node_instance.storage else 0
                labels[node] = f"{node}\nStorage Node\n{actual_storage:.1f} ({node_instance.capacity})"
            else:
                labels[node] = f"{node}\n{node_instance.__class__.__name__}"
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=9)
        
        # Update edge labels to show actual flow and capacity
        edge_labels = {}
        for u, v, d in self.graph.edges(data=True):
            edge = d['edge']
            actual_flow = edge.flow[-1] if edge.flow else 0
            capacity = edge.capacity
            edge_labels[(u, v)] = f'{actual_flow:.1f} ({capacity})'
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=14)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Network layout plot saved to {filename}")

        # Display the plot if requested
        if display:
            plt.show()
        else:
            plt.close()
    
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