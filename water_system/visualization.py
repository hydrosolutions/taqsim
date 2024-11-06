"""
This module provides visualization tools for analyzing water system simulation results.
It includes time series visualizations for flows, storage levels, and demands.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import networkx as nx
from datetime import datetime
from .structure import SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks

class WaterSystemVisualizer:
    """
    A class for creating time series visualizations of water system simulation results.
    """
    
    def __init__(self, water_system, name=""):
        """
        Initialize the visualizer with a water system.
        
        Args:
            water_system (WaterSystem): The simulated water system to visualize.
        """
        self.system = water_system
        self.name = name
        self.df = water_system.get_water_balance_table()
        
        # Create images directory if it doesn't exist
        self.image_dir = os.path.join('.', 'figures')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _save_plot(self, plot_type, additional_info=""):
        """
        Helper method to save plots with consistent naming.
        
        Args:
            plot_type (str): Type of plot being saved
            additional_info (str): Additional information for filename
        """
        filename = f"{self.name}_{plot_type}"
        if additional_info:
            filename += f"_{additional_info}"
        filename += ".png"
        filepath = os.path.join(self.image_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath
        
    def plot_node_flows(self, node_ids):
        """
        Create a time series plot of inflows and outflows for specified nodes.
        
        Args:
            node_ids (list): List of node IDs to include in the visualization
        """
        plt.figure(figsize=(12, 6))
        
        for node_id in node_ids:
            inflow_col = f"{node_id}_Inflow"
            outflow_col = f"{node_id}_Outflow"
            
            if inflow_col in self.df.columns:
                plt.plot(self.df['TimeStep'], self.df[inflow_col], 
                        label=f"{node_id} Inflow", marker='o')
            if outflow_col in self.df.columns:
                plt.plot(self.df['TimeStep'], self.df[outflow_col], 
                        label=f"{node_id} Outflow", marker='s')
        
        plt.xlabel('Time Step')
        plt.ylabel('Flow Rate')
        plt.title('Node Flows Over Time')
        plt.grid(True)
        plt.legend()
        
        nodes_str = "_".join(node_ids)
        return self._save_plot("flows", nodes_str)
        
    def plot_storage_levels(self):
        """
        Create a time series plot of storage levels for all storage nodes.
        """
        storage_cols = [col for col in self.df.columns if col.endswith('_Storage')]
        if not storage_cols:
            print("No storage nodes found in the system.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for col in storage_cols:
            node_id = col.replace('_Storage', '')
            node_data = self.system.graph.nodes[node_id]['node']
            
            plt.plot(self.df['TimeStep'], self.df[col], 
                    label=f"{node_id} Storage", marker='o')
            plt.axhline(y=node_data.capacity, linestyle='--', 
                       label=f"{node_id} Capacity")
        
        plt.xlabel('Time Step')
        plt.ylabel('Storage Volume')
        plt.title('Storage Levels Over Time')
        plt.grid(True)
        plt.legend()
        
        return self._save_plot("storage_levels")
        
    def plot_demand_satisfaction(self):
        """
        Create a time series plot comparing demanded vs satisfied demand for all demand nodes.
        Uses distinct colors and patterns to make demands clearly distinguishable.
        """
        demand_cols = [col for col in self.df.columns if col.endswith('_Demand')]
        if not demand_cols:
            print("No demand nodes found in the system.")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Color scheme for different demand nodes
        # Using colorblind-friendly colors
        colors = [
            '#0077BB',  # Blue
            '#EE7733',  # Orange
            '#009988',  # Teal
            '#CC3311',  # Red
            '#33BBEE',  # Cyan
            '#EE3377',  # Magenta
            '#669900',  # Lime
            '#AA44BB',  # Purple
        ]
        
        # Line styles for demanded vs satisfied
        demanded_style = 'solid'
        satisfied_style = 'dashed'
        
        # Markers for better distinction
        demanded_marker = 'o'
        satisfied_marker = 's'
        
        for idx, col in enumerate(demand_cols):
            node_id = col.replace('_Demand', '')
            satisfied_col = f"{node_id}_SatisfiedDemand"
            color = colors[idx % len(colors)]  # Cycle through colors if more demands than colors
            
            # Plot demanded with solid line
            plt.plot(self.df['TimeStep'], 
                    self.df[col], 
                    label=f"{node_id} Demanded",
                    color=color,
                    linestyle=demanded_style,
                    marker=demanded_marker,
                    linewidth=1,
                    markersize=4,
                    markerfacecolor='white',
                    markeredgewidth=2)
            
            # Plot satisfied with dashed line
            plt.plot(self.df['TimeStep'], 
                    self.df[satisfied_col], 
                    label=f"{node_id} Satisfied",
                    color=color,
                    linestyle=satisfied_style,
                    marker=satisfied_marker,
                    linewidth=1,
                    markersize=4,
                    markerfacecolor='white',
                    markeredgewidth=2)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Flow Rate', fontsize=12)
        plt.title('Demand Satisfaction Over Time', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Enhance legend
        plt.legend(bbox_to_anchor=(1.05, 1), 
                loc='upper left',
                borderaxespad=0.,
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        return self._save_plot("demand_satisfaction")

    def plot_demand_deficit_heatmap(self):
        """
        Create enhanced heatmaps showing demand deficits across all demand nodes over time.
        """
        deficit_cols = [col for col in self.df.columns if col.endswith('_Deficit')]
        if not deficit_cols:
            print("No demand nodes found in the system.")
            return
            
        # Prepare data for heatmap
        deficit_data = self.df[deficit_cols].copy()
        deficit_data.columns = [col.replace('_Deficit', '') for col in deficit_cols]
        
        # Calculate percentage deficits
        percentage_data = deficit_data.copy()
        for node in deficit_data.columns:
            demand_col = f"{node}_Demand"
            total_demand = self.df[demand_col]
            percentage_data[node] = (deficit_data[node] / total_demand) * 100
        
        # Plot absolute deficits
        plt.figure(figsize=(12, 6))
        sns.heatmap(deficit_data.T, cmap='YlOrRd', 
                    xticklabels=self.df['TimeStep'],
                    yticklabels=deficit_data.columns,
                    annot=True, fmt='.1f')
        plt.xlabel('Time Step')
        plt.ylabel('Demand Node')
        plt.title('Absolute Water Deficits Over Time (flow units)')
        abs_filepath = self._save_plot("deficit_heatmap_absolute")
        
        # Plot percentage deficits
        plt.figure(figsize=(12, 6))
        sns.heatmap(percentage_data.T, cmap='YlOrRd', 
                    xticklabels=self.df['TimeStep'],
                    yticklabels=percentage_data.columns,
                    annot=True, fmt='.1f',
                    vmin=0, vmax=100)
        plt.xlabel('Time Step')
        plt.ylabel('Demand Node')
        plt.title('Percentage of Unmet Demand Over Time (%)')
        pct_filepath = self._save_plot("deficit_heatmap_percentage")
        
        return abs_filepath, pct_filepath
        
    def plot_supply_utilization(self):
        """
        Create a time series plot showing supply rates and actual outflows for supply nodes.
        """
        supply_cols = [col for col in self.df.columns if col.endswith('_SupplyRate')]
        if not supply_cols:
            print("No supply nodes found in the system.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for col in supply_cols:
            node_id = col.replace('_SupplyRate', '')
            outflow_col = f"{node_id}_Outflow"
            
            plt.plot(self.df['TimeStep'], self.df[col], 
                    label=f"{node_id} Available", marker='o')
            plt.plot(self.df['TimeStep'], self.df[outflow_col], 
                    label=f"{node_id} Used", marker='s')
        
        plt.xlabel('Time Step')
        plt.ylabel('Flow Rate')
        plt.title('Supply Utilization Over Time')
        plt.grid(True)
        plt.legend()
        
        return self._save_plot("supply_utilization")

    def plot_network_layout(self):
        """
        Create and save a network layout plot for the water system, showing flows and losses.
        Returns the path to the saved image file.
        """
        # Setting node positions based on easting and northing
        pos = {}
        for node, data in self.system.graph.nodes(data=True):
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
            node_list = [node for node, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], node_type)]
            nx.draw_networkx_nodes(self.system.graph, pos, 
                                nodelist=node_list, 
                                node_color=node_colors[node_type], 
                                node_shape=node_shapes[node_type],
                                node_size=node_size, 
                                alpha=0.8,
                                ax=ax)

        # Draw edges with standard color
        nx.draw_networkx_edges(self.system.graph, pos, edge_color='gray', 
                            arrows=True, arrowsize=65, width=2, ax=ax)

        # Update node labels
        labels = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            if isinstance(node_instance, SupplyNode):
                actual_supply = node_instance.supply_history[-1] if node_instance.supply_history else 0
                labels[node] = f"{node}\nSupply Node\nActual: {actual_supply:.1f}"
            elif isinstance(node_instance, DemandNode):
                satisfied_demand = node_instance.satisfied_demand[-1] if node_instance.satisfied_demand else 0
                current_demand = node_instance.get_demand_rate(len(node_instance.satisfied_demand) - 1)
                labels[node] = f"{node}\nDemand\n{satisfied_demand:.1f} ({current_demand:.1f})"
            elif isinstance(node_instance, SinkNode):
                total_inflow = sum(edge.outflow[-1] if edge.outflow else 0 
                                for edge in node_instance.inflow_edges.values())
                labels[node] = f"{node}\nSink Node\nTotal Inflow: {total_inflow:.1f}"
            elif isinstance(node_instance, StorageNode):
                actual_storage = node_instance.storage[-1] if node_instance.storage else 0
                labels[node] = f"{node}\nStorage Node\n{actual_storage:.1f} ({node_instance.capacity})"
            elif isinstance(node_instance, HydroWorks):
                total_inflow = sum(edge.outflow[-1] if edge.outflow else 0 
                                for edge in node_instance.inflow_edges.values())
                labels[node] = f"{node}\nHydroWorks\nInflow: {total_inflow:.1f}"
            else:
                labels[node] = f"{node}\n{node_instance.__class__.__name__}"
        
        nx.draw_networkx_labels(self.system.graph, pos, labels, font_size=12, ax=ax)
        
        # Update edge labels to show flows and losses
        edge_labels = {}
        for u, v, d in self.system.graph.edges(data=True):
            edge = d['edge']
            if hasattr(edge, 'inflow') and hasattr(edge, 'losses') and edge.outflow and edge.inflow and edge.losses:
                inflow = edge.inflow[-1]
                outflow = edge.outflow[-1]
                losses = edge.losses[-1]
                loss_percent = (losses / inflow * 100) if inflow > 0 else 0
                edge_labels[(u, v)] = (f'In: {inflow:.1f}m³/s\n'
                                    f'Out: {outflow:.1f}m³/s\n'
                                    f'Loss: {edge.loss_factor} [/km]\n'
                                    f'Cap: {edge.capacity}m³/s\n'
                                    f'L: {edge.length:.1f} km\n')
            else:
                edge_labels[(u, v)] = f'Cap: {edge.capacity}'
        
        nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, 
                                font_size=10, ax=ax)
        
        plt.axis('off')
        plt.tight_layout()
        
        return self._save_plot("network_layout")  

    def plot_storage_spills(self):
        """
        Create a time series plot showing storage node spills over time.
        """
        spill_cols = [col for col in self.df.columns if col.endswith('_ExcessVolume')]
        if not spill_cols:
            print("No storage nodes found in the system.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for col in spill_cols:
            node_id = col.replace('_ExcessVolume', '')
            plt.plot(self.df['TimeStep'], self.df[col], 
                    label=f"{node_id} Spills", marker='o')
        
        plt.xlabel('Time Step')
        plt.ylabel('Excess Volume')
        plt.title('Storage Node Spills Over Time')
        plt.grid(True)
        plt.legend()
        
        return self._save_plot("storage_spills")

