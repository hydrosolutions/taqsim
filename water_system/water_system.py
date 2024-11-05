"""
This module defines the WaterSystem class, which is the main class for creating and managing
a water system simulation. It uses NetworkX for graph representation and matplotlib for visualization.

The WaterSystem class allows users to add nodes and edges to the system, run simulations,
and visualize the results.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from .structure import SupplyNode, StorageNode, HydroWorks, DemandNode, SinkNode
from .edge import Edge

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
        Generate a table with water balance data for all nodes across all time steps,
        including transmission losses.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the water balance data.
        """
        # Return empty DataFrame if no time steps have been simulated
        if self.time_steps == 0:
            return pd.DataFrame({'TimeStep': []})

        # Initialize lists for columns and data
        node_columns = []
        edge_columns = []
        data = []

        # Generate column names for nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data['node']
            
            # Common columns for all node types
            base_columns = [
                f"{node_id}_Inflow",
                f"{node_id}_Outflow",
            ]

            # Add type-specific columns
            if isinstance(node, SupplyNode):
                type_columns = [f"{node_id}_SupplyRate"]
            elif isinstance(node, StorageNode):
                type_columns = [
                    f"{node_id}_Storage",
                    f"{node_id}_StorageChange",
                    f"{node_id}_ExcessVolume"
                ]
            elif isinstance(node, DemandNode):
                type_columns = [
                    f"{node_id}_Demand",
                    f"{node_id}_SatisfiedDemand",
                    f"{node_id}_Deficit"
                ]
            else:
                type_columns = []

            node_columns.extend(base_columns + type_columns)

        # Generate column names for edges
        for u, v, d in self.graph.edges(data=True):
            edge_columns.extend([
                f"Edge_{u}_to_{v}_Inflow",
                f"Edge_{u}_to_{v}_Outflow",
                f"Edge_{u}_to_{v}_Losses",
                f"Edge_{u}_to_{v}_LossPercent"
            ])

        # Combine all columns
        all_columns = ['TimeStep'] + node_columns + edge_columns

        # Initialize data list with dictionaries
        data = []
        for time_step in range(self.time_steps):
            row_data = {'TimeStep': time_step}
            
            # Initialize all columns with 0
            for col in node_columns + edge_columns:
                row_data[col] = 0.0
                
            # Populate node data
            for node_id, node_data in self.graph.nodes(data=True):
                node = node_data['node']
                try:
                    inflow = sum(edge.get_flow(time_step) for edge in node.inflows.values())
                    outflow = sum(edge.get_flow(time_step) for edge in node.outflows.values())

                    row_data[f"{node_id}_Inflow"] = inflow
                    row_data[f"{node_id}_Outflow"] = outflow

                    if isinstance(node, SupplyNode):
                        row_data[f"{node_id}_SupplyRate"] = node.get_supply_rate(time_step)
                    elif isinstance(node, StorageNode):
                        row_data[f"{node_id}_Storage"] = node.storage[time_step]
                        storage_change = (node.storage[time_step] - node.storage[time_step - 1] 
                                        if time_step > 0 else node.storage[0])
                        row_data[f"{node_id}_StorageChange"] = storage_change
                        row_data[f"{node_id}_ExcessVolume"] = node.spillway_register[time_step]
                    elif isinstance(node, DemandNode):
                        row_data[f"{node_id}_Demand"] = node.get_demand_rate(time_step)
                        row_data[f"{node_id}_SatisfiedDemand"] = node.satisfied_demand[time_step]
                        row_data[f"{node_id}_Deficit"] = (node.get_demand_rate(time_step) - 
                                                        node.satisfied_demand[time_step])
                except Exception as e:
                    print(f"Warning: Error processing node {node_id} at time step {time_step}: {str(e)}")

            # Populate edge data
            for u, v, d in self.graph.edges(data=True):
                edge = d['edge']
                try:
                    if time_step < len(edge.inflow):
                        inflow = edge.inflow[time_step]
                        outflow = edge.flow[time_step]
                        losses = edge.losses[time_step]
                        loss_percent = (losses / inflow * 100) if inflow > 0 else 0
                        
                        row_data[f"Edge_{u}_to_{v}_Inflow"] = inflow
                        row_data[f"Edge_{u}_to_{v}_Outflow"] = outflow
                        row_data[f"Edge_{u}_to_{v}_Losses"] = losses
                        row_data[f"Edge_{u}_to_{v}_LossPercent"] = loss_percent
                except Exception as e:
                    print(f"Warning: Error processing edge {u}->{v} at time step {time_step}: {str(e)}")

            data.append(row_data)

        # Create DataFrame
        try:
            df = pd.DataFrame(data, columns=all_columns)
            return df
        except Exception as e:
            print(f"Error creating DataFrame: {str(e)}")
            # Return minimal DataFrame with just TimeStep column
            return pd.DataFrame({'TimeStep': range(self.time_steps)}) 

    def visualize(self, filename='water_system_layout.png', display=True):
        """
        Create and save a network layout plot for the water system, showing flows and losses.
        
        Args:
            filename (str): The name of the PNG file to save to. Defaults to 'water_system_layout.png'.
            display (bool): Whether to display the plot or not. Defaults to True.
        """
        # Setting node positions based on easting and northing
        pos = {}
        for node, data in self.graph.nodes(data=True):
            node_instance = data['node']
            pos[node] = (node_instance.easting, node_instance.northing)

        # Create figure and axis objects with a single subplot
        fig, ax = plt.subplots(figsize=(30, 25))
        plt.title('Water System Network Layout', fontsize=20)
        
        # Define node styling
        node_colors = {
            SupplyNode: 'skyblue',
            StorageNode: 'lightgreen',
            DemandNode: 'salmon',
            SinkNode: 'lightgray',
            HydroWorks: 'orange'
        }
        
        node_shapes = {
            SupplyNode: 's',    # square
            StorageNode: 's',   # square
            DemandNode: 'o',    # circle
            SinkNode: 's',      # square
            HydroWorks: 'o',    # circle
        }
        
        node_size = 5000
        
        # Draw nodes
        for node_type in node_colors.keys():
            node_list = [node for node, data in self.graph.nodes(data=True) 
                        if isinstance(data['node'], node_type)]
            nx.draw_networkx_nodes(self.graph, pos, 
                                nodelist=node_list, 
                                node_color=node_colors[node_type], 
                                node_shape=node_shapes[node_type],
                                node_size=node_size, 
                                alpha=0.8,
                                ax=ax)

        # Draw edges with standard color
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                            arrows=True, arrowsize=65, width=2, ax=ax)

        # Update node labels
        labels = {}
        for node, data in self.graph.nodes(data=True):
            node_instance = data['node']
            if isinstance(node_instance, SupplyNode):
                actual_supply = node_instance.supply_history[-1] if node_instance.supply_history else 0
                labels[node] = f"{node}\nSupply Node\nActual: {actual_supply:.1f}"
            elif isinstance(node_instance, DemandNode):
                satisfied_demand = node_instance.satisfied_demand[-1] if node_instance.satisfied_demand else 0
                current_demand = node_instance.get_demand_rate(len(node_instance.satisfied_demand) - 1)
                labels[node] = f"{node}\nDemand\n{satisfied_demand:.1f} ({current_demand:.1f})"
            elif isinstance(node_instance, SinkNode):
                total_inflow = sum(edge.flow[-1] if edge.flow else 0 
                                for edge in node_instance.inflows.values())
                labels[node] = f"{node}\nSink Node\nTotal Inflow: {total_inflow:.1f}"
            elif isinstance(node_instance, StorageNode):
                actual_storage = node_instance.storage[-1] if node_instance.storage else 0
                labels[node] = f"{node}\nStorage Node\n{actual_storage:.1f} ({node_instance.capacity})"
            elif isinstance(node_instance, HydroWorks):
                total_inflow = sum(edge.flow[-1] if edge.flow else 0 
                                for edge in node_instance.inflows.values())
                labels[node] = f"{node}\nHydroWorks\nInflow: {total_inflow:.1f}"
            else:
                labels[node] = f"{node}\n{node_instance.__class__.__name__}"
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=12, ax=ax)
        
        # Update edge labels to show flows and losses
        edge_labels = {}
        for u, v, d in self.graph.edges(data=True):
            edge = d['edge']
            if edge.flow and edge.inflow and edge.losses:
                inflow = edge.inflow[-1]
                outflow = edge.flow[-1]
                losses = edge.losses[-1]
                loss_percent = (losses / inflow * 100) if inflow > 0 else 0
                edge_labels[(u, v)] = (f'In: {inflow:.1f}m³/s\n'
                                    f'Out: {outflow:.1f}m³/s\n'
                                    f'Loss: {edge.loss_factor} [/km]\n'
                                    f'Cap: {edge.capacity}m³/s\n'
                                    f'L: {edge.length:.1f} km\n')
            else:
                edge_labels[(u, v)] = f'Cap: {edge.capacity}'
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, 
                                font_size=10, ax=ax)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Network layout plot saved to {filename}")

        # Display or close the plot
        if display:
            plt.show()
        else:
            plt.close(fig)