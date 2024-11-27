"""
This module provides visualization tools for analyzing water system simulation results.
It includes time series visualizations for flows, storage levels, and demands.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import json
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
        self.df = self.get_water_balance_table()
        
        # Create images directory if it doesn't exist
        self.image_dir = os.path.join('.', 'figures')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_water_balance_table(self):
            """
            Generate a table with water balance data for all nodes across all time steps,
            including transmission losses.

            Returns:
                pd.DataFrame: A pandas DataFrame containing the water balance data.
            """
            # Return empty DataFrame if no time steps have been simulated
            if self.system.time_steps == 0:
                return pd.DataFrame({'TimeStep': []})

            # Initialize lists for columns and data
            node_columns = []
            edge_columns = []
            data = []

            # Generate column names for nodes
            for node_id, node_data in self.system.graph.nodes(data=True):
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
            for u, v, d in self.system.graph.edges(data=True):
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
            for time_step in range(self.system.time_steps):
                row_data = {'TimeStep': time_step}
                
                # Initialize all columns with 0
                for col in node_columns + edge_columns:
                    row_data[col] = 0.0
                    
                # Populate node data
                for node_id, node_data in self.system.graph.nodes(data=True):
                    node = node_data['node']
                    try:
                        inflow = sum(edge.get_edge_outflow(time_step) for edge in node.inflow_edges.values())
                        outflow = sum(edge.get_edge_inflow(time_step) for edge in node.outflow_edges.values())

                        row_data[f"{node_id}_Inflow"] = inflow
                        row_data[f"{node_id}_Outflow"] = outflow

                        if isinstance(node, SupplyNode):
                            row_data[f"{node_id}_SupplyRate"] = node.get_supply_rate(time_step)
                        elif isinstance(node, StorageNode):
                            row_data[f"{node_id}_Storage"] = node.storage[time_step+1]
                            storage_change = (node.storage[time_step+1] - node.storage[time_step] 
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
                for u, v, d in self.system.graph.edges(data=True):
                    edge = d['edge']
                    try:
                        if time_step < len(edge.inflow):
                            inflow = edge.inflow[time_step]
                            outflow = edge.outflow[time_step]
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
                return pd.DataFrame({'TimeStep': range(self.system.time_steps)})

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
        
    def plot_node_inflows(self, node_ids):
        """
        Create a time series plot of inflows and outflows for specified nodes.
        
        Args:
            node_ids (list): List of node IDs to include in the visualization
        """
        plt.figure(figsize=(12, 6))
        
        for node_id in node_ids:
            inflow_col = f"{node_id}_Inflow"
            
            if inflow_col in self.df.columns:
                plt.plot(self.df['TimeStep'], self.df[inflow_col], 
                        label=f"{node_id} Inflow", marker='o')
        
        plt.xlabel('Time Step')
        plt.ylabel('Flow Rate [m³/s]')
        plt.title('Node Flows Over Time')
        plt.grid(True)
        plt.legend()
        
        nodes_str = "_".join(node_ids)
        return self._save_plot("flows", nodes_str)
        
    def plot_demand_satisfaction(self):
        """
        Create a time series plot comparing demanded vs satisfied demand for all demand nodes.
        Uses distinct colors and patterns to make demands clearly distinguishable.
        """
        demand_cols = [col for col in self.df.columns if col.endswith('_Demand')]
        if not demand_cols:
            print("No demand nodes found in the system.")
            return
            
        plt.figure(figsize=(12, 10))
        
        # Color scheme for different demand nodes
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
        
        demanded_style = 'solid'
        satisfied_style = 'dashed'
        demanded_marker = 'o'
        satisfied_marker = 's'
        
        for idx, col in enumerate(demand_cols):
            node_id = col.replace('_Demand', '')
            satisfied_col = f"{node_id}_SatisfiedDemand"
            color = colors[idx % len(colors)]
            
            plt.plot(self.df['TimeStep'], 
                    self.df[col], 
                    label=f"{node_id} Demanded",
                    color=color,
                    linestyle=demanded_style,
                    marker=demanded_marker,
                    linewidth=0.8,
                    markersize=3,
                    markerfacecolor='white',
                    markeredgewidth=1)
            
            plt.plot(self.df['TimeStep'], 
                    self.df[satisfied_col], 
                    label=f"{node_id} Satisfied",
                    color=color,
                    linestyle=satisfied_style,
                    marker=satisfied_marker,
                    linewidth=0.8,
                    markersize=3,
                    markerfacecolor='white',
                    markeredgewidth=1)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Flow Rate [m³/s]', fontsize=12)
        plt.title('Demand Satisfaction Over Time', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.legend(bbox_to_anchor=(1.05, 1), 
                loc='upper left',
                borderaxespad=0.,
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)
        
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
            
        deficit_data = self.df[deficit_cols].copy()
        deficit_data.columns = [col.replace('_Deficit', '') for col in deficit_cols]
        
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
                    annot=False,
                    cbar_kws={'label': 'Deficit [m³/s]'})
        plt.xlabel('Time Step')
        plt.ylabel('Demand Node')
        plt.title('Absolute Water Deficits Over Time')
        abs_filepath = self._save_plot("deficit_heatmap_absolute")
        
        # Plot percentage deficits
        plt.figure(figsize=(12, 6))
        sns.heatmap(percentage_data.T, cmap='YlOrRd', 
                    xticklabels=self.df['TimeStep'],
                    yticklabels=percentage_data.columns,
                    annot=False,
                    vmin=0, vmax=100,
                    cbar_kws={'label': 'Deficit [%]'})
        plt.xlabel('Time Step')
        plt.ylabel('Demand Node')
        plt.title('Percentage of Unmet Demand Over Time')
        pct_filepath = self._save_plot("deficit_heatmap_percentage")
        
        return abs_filepath, pct_filepath
        
    def plot_water_levels(self):
        """
        Create a time series plot showing water levels for all storage nodes that have hva data.
        Also shows the minimum and maximum waterlevel for reference.
        """
        storage_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], StorageNode) and data['node'].hva_data is not None]
        
        if not storage_nodes:
            print("No storage nodes with survey data found in the system.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for node_id, node in storage_nodes:
            # Get the water levels time series
            water_levels = node.water_level if hasattr(node, 'water_level') else []
            
            if water_levels:
                # Plot water level time series
                plt.plot(range(len(water_levels)), water_levels, 
                        label=f"{node_id} Water Level", 
                        marker='o',
                        linewidth=0.8,
                        markersize=3,
                        markerfacecolor='white',
                        markeredgewidth=1)
                
                # Plot minimum and maximum waterlevels as reference lines
                plt.axhline(y=node.hva_data['min_waterlevel'], 
                        linestyle='--', 
                        color='gray', 
                        alpha=0.5,
                        label=f"{node_id} Minimum Waterlevel")
                        
                plt.axhline(y=node.hva_data['max_waterlevel'], 
                        linestyle=':', 
                        color='red', 
                        alpha=0.5,
                        label=f"{node_id} Maximum Waterlevel")
        
        plt.xlabel('Time Step')
        plt.ylabel('Waterlevel [m]')
        plt.title('Reservoir Water Levels Over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Enhance legend
        plt.legend(bbox_to_anchor=(1.05, 1), 
                loc='upper left',
                borderaxespad=0.,
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)
        
        plt.tight_layout()
        
        return self._save_plot("water_levels")

    def plot_edge_flows(self):
        """
        Create time series plots showing inflows and outflows for all edges in the system.
        Includes comparison with edge capacities and flow reduction due to losses.
        
        Returns:
            str: Path to the saved plot file
        """
        # Get all edge-related flow columns from the DataFrame
        inflow_cols = [col for col in self.df.columns if col.startswith('Edge_') and col.endswith('_Inflow')]
        
        if not inflow_cols:
            print("No edge flow data found in the system.")
            return None
        
        # Calculate number of edges and set up subplots layout
        n_edges = len(inflow_cols)
        n_cols = min(2, n_edges)  # Maximum 2 columns
        n_rows = (n_edges + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5 * n_rows))
        
        # Create color scheme for consistent visualization
        flow_colors = {
            'inflow': '#2196F3',    # Blue for inflow
            'outflow': '#4CAF50',   # Green for outflow
            'capacity': '#FF9800'    # Orange for capacity
        }
        
        # Plot each edge's flows in its own subplot
        for idx, inflow_col in enumerate(inflow_cols):
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Extract edge name and corresponding columns
            edge_name = inflow_col.replace('Edge_', '').replace('_Inflow', '')
            edge_name = edge_name.replace('_to_', ' → ')
            outflow_col = inflow_col.replace('_Inflow', '_Outflow')
            
            # Get edge capacity from the water system graph
            source, target = edge_name.replace(' → ', '_to_').split('_to_')
            try:
                edge_data = self.system.graph[source][target]['edge']
                capacity = edge_data.capacity
            except:
                capacity = None
            
            # Plot flows
            ax.plot(self.df['TimeStep'], 
                    self.df[inflow_col],
                    label='Inflow',
                    color=flow_colors['inflow'],
                    marker='o',
                    linewidth=1.5,
                    markersize=4,
                    markerfacecolor='white')
            
            ax.plot(self.df['TimeStep'], 
                    self.df[outflow_col],
                    label='Outflow',
                    color=flow_colors['outflow'],
                    marker='s',
                    linewidth=1.5,
                    markersize=4,
                    markerfacecolor='white')
            
            # Plot capacity line if available
            if capacity is not None:
                ax.axhline(y=capacity, 
                        color=flow_colors['capacity'],
                        linestyle='--',
                        label=f'Capacity ({capacity} m³/s)')
            
            # Calculate and plot flow metrics
            mean_inflow = self.df[inflow_col].mean()
            mean_outflow = self.df[outflow_col].mean()
            mean_loss = mean_inflow - mean_outflow
            loss_percent = (mean_loss / mean_inflow * 100) if mean_inflow > 0 else 0
            
            # Add metrics text box
            metrics_text = (
                f'Mean Inflow: {mean_inflow:.2f} m³/s\n'
                f'Mean Outflow: {mean_outflow:.2f} m³/s\n'
                f'Mean Loss: {mean_loss:.2f} m³/s\n'
                f'Loss Percentage: {loss_percent:.1f}%'
            )
            
            ax.text(0.95, 0.95, metrics_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            # Customize subplot
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Flow Rate [m³/s]')
            ax.set_title(f'Flow Rates for Edge: {edge_name}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left')
            
            # Add shaded area between inflow and outflow to highlight losses
            ax.fill_between(self.df['TimeStep'],
                        self.df[inflow_col],
                        self.df[outflow_col],
                        alpha=0.2,
                        color='red',
                        label='Losses')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        return self._save_plot("edge_flows")

    def plot_edge_flow_summary(self):
        """
        Create a summary plot showing the distribution of flows and losses across all edges.
        Uses box plots to show flow variability and average loss percentages.
        
        Returns:
            str: Path to the saved plot file
        """
        # Get edge flow columns
        edge_cols = [col for col in self.df.columns if col.startswith('Edge_')]
        if not edge_cols:
            print("No edge flow data found in the system.")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Prepare data for box plots
        edge_names = []
        inflow_data = []
        outflow_data = []
        loss_pcts = []
        
        for inflow_col in [col for col in edge_cols if col.endswith('_Inflow')]:
            edge_name = inflow_col.replace('Edge_', '').replace('_Inflow', '')
            edge_name = edge_name.replace('_to_', ' → ')
            outflow_col = inflow_col.replace('_Inflow', '_Outflow')
            
            edge_names.append(edge_name)
            inflow_data.append(self.df[inflow_col])
            outflow_data.append(self.df[outflow_col])
            
            # Calculate loss percentages
            loss_pct = ((self.df[inflow_col] - self.df[outflow_col]) / 
                    self.df[inflow_col] * 100).mean()
            loss_pcts.append(loss_pct)
        
        # Plot flow distributions
        bp1 = ax1.boxplot([inflow_data[i] for i in range(len(edge_names))],
                        positions=range(0, len(edge_names)*2, 2),
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='blue'),
                        flierprops=dict(marker='o', markerfacecolor='lightblue'))
        
        bp2 = ax1.boxplot([outflow_data[i] for i in range(len(edge_names))],
                        positions=range(1, len(edge_names)*2, 2),
                        patch_artist=True,
                        boxprops=dict(facecolor='lightgreen', color='green'),
                        medianprops=dict(color='green'),
                        flierprops=dict(marker='o', markerfacecolor='lightgreen'))
        
        # Customize flow distribution plot
        ax1.set_xticks(range(0, len(edge_names)*2, 2))
        ax1.set_xticklabels(edge_names, rotation=45, ha='right')
        ax1.set_ylabel('Flow Rate [m³/s]')
        ax1.set_title('Flow Rate Distributions by Edge')
        ax1.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Inflow', 'Outflow'])
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Plot average loss percentages
        bars = ax2.bar(edge_names, loss_pcts)
        
        # Color bars based on loss percentage
        for bar, loss_pct in zip(bars, loss_pcts):
            if loss_pct <= 25:
                bar.set_color('lightgreen')
            elif loss_pct <= 50:
                bar.set_color('yellow')
            elif loss_pct <= 75:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Customize loss percentage plot
        ax2.set_xticklabels(edge_names, rotation=45, ha='right')
        ax2.set_ylabel('Average Loss Percentage [%]')
        ax2.set_title('Average Water Loss Percentage by Edge')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add reference lines for loss percentages
        for pct in [25, 50, 75]:
            ax2.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
            ax2.text(ax2.get_xlim()[1], pct, f'{pct}%', 
                    verticalalignment='bottom', horizontalalignment='right')
        
        plt.tight_layout()
        
        return self._save_plot("edge_flow_summary")

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
        
        nx.draw_networkx_labels(self.system.graph, pos, labels, font_size=14, ax=ax)
        """
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
                                    f'Cap: {edge.capacity}m³/s\n'
                                    f'L: {edge.length:.1f} km\n')
            else:
                edge_labels[(u, v)] = f'Cap: {edge.capacity}'
        
        nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, 
                                font_size=8, ax=ax)
        """
        plt.axis('off')
        plt.tight_layout()
        
        return self._save_plot("network_layout")  
 
    def plot_cumulative_volumes(self):
        """
        Create a plot showing the cumulative volumes of all water balance components,
        including storage change.
        
        Returns:
            str: Path to the saved plot file
        """
        if not hasattr(self, 'image_dir'):
            self.image_dir = os.path.join('.', 'figures')
            os.makedirs(self.image_dir, exist_ok=True)
        
        df = self.system.get_water_balance()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define components and their properties
        components = {
            'source': {
                'color': '#2196F3',  # Blue
                'label': 'Source'
            },
            'supplied demand': {
                'color': '#4CAF50',  # Green
                'label': 'Supplied Demand'
            },
            'sink': {
                'color': '#f44336',  # Red
                'label': 'Sink'
            },
            'edge losses': {
                'color': '#FF9800',  # Orange
                'label': 'Edge Losses'
            },
            'reservoir spills': {
                'color': '#9C27B0',  # Purple
                'label': 'Reservoir Spills'
            },
            'storage_change': {
                'color': '#795548',  # Brown
                'label': 'Storage Change'
            }
        }
        
        # Calculate and plot cumulative volumes for main components
        for comp, properties in components.items():
            if comp != 'storage_change':  # Handle storage_change separately
                cumulative = df[comp].cumsum()
                ax.plot(df['time_step'], cumulative,
                       label=properties['label'],
                       color=properties['color'],
                       linewidth=2,
                       marker='o',
                       markersize=4)
        
        # Handle storage change separately with dashed line
        cumulative_storage = df['storage_change'].cumsum()
        ax.plot(df['time_step'], cumulative_storage,
               label='Storage Change',
               color=components['storage_change']['color'],
               linewidth=2,
               linestyle='--',
               marker='s',
               markersize=4)
        
        # Customize plot
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Volume (m³)')
        ax.set_title('Cumulative Water Balance Components')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add text box with final volumes
        if len(df) > 0:
            text_str = "Final Volumes:\n"
            total_source = df['source'].sum()
            
            # Add all components including storage change
            for comp, properties in components.items():
                total = df[comp].sum()
                percentage = (total / total_source * 100) if total_source > 0 else 0
                
                # Format numbers with appropriate sign
                if comp == 'storage_change':
                    text_str += f"{properties['label']}: {total:+,.0f} m³ ({percentage:+.1f}%)\n"
                else:
                    text_str += f"{properties['label']}: {total:,.0f} m³ ({percentage:.1f}%)\n"
            
            # Add storage start and end values
            text_str += f"\nStorage Start: {df['storage_start'].iloc[0]:,.0f} m³\n"
            text_str += f"Storage End: {df['storage_end'].iloc[-1]:,.0f} m³"
            
            plt.text(1.05, 0.5, text_str,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='center')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(self.image_dir, f"{self.name}_cumulative_volumes.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path
    
    def print_water_balance_summary(self):
        """
        Print a comprehensive summary of the water balance results.
        Includes volumes, relative contributions, storage changes and balance error statistics.
        """
        df = self.system.get_water_balance()
        
        print(f"\nWater Balance Summary for {self.name}")
        print("=" * 50)
        
        if len(df) == 0:
            print("No data available - simulation may not have been run yet.")
            return
            
        def print_section(title):
            print(f"\n{title}")
            print("-" * len(title))
        
        # Time information
        print_section("Simulation Period")
        print(f"Number of timesteps: {len(df)}")
        print(f"Timestep duration: {self.system.dt:,.0f} seconds")
        
        # Source volumes
        print_section("Source Volumes")
        total_source = df['source'].sum()
        print(f"Total source volume: {total_source:,.0f} m³ (100%)")
        
        # Component volumes and percentages
        print_section("Sink Volumes")
        components = {
            'Supplied Demand': 'supplied demand',
            'Sink Outflow': 'sink',
            'Edge Losses': 'edge losses',
            'Res Spills': 'reservoir spills',
            'Res ET Losses': 'reservoir ET losses',
        }
        
        for label, comp in components.items():
            total = df[comp].sum()
            percentage = (total / total_source * 100) if total_source > 0 else 0
            print(f"{label:15s}: {total:15,.0f} m³ ({percentage:6.1f}%)")
        
        # Storage changes
        print_section("Storage")
        print(f"Initial storage:     {df['storage_start'].iloc[0]:15,.0f} m³")
        print(f"Final storage:       {df['storage_end'].iloc[-1]:15,.0f} m³")
        total_storage_change = df['storage_change'].sum()
        storage_percentage = (total_storage_change / total_source * 100) if total_source > 0 else 0
        print(f"Net storage change:  {total_storage_change:15,.0f} m³ ({storage_percentage:6.1f}%)")
        
        # Demand satisfaction
        print_section("Demand Satisfaction")
        total_demand = df['demands'].sum()
        total_satisfied = df['supplied demand'].sum()
        unmet_demand = df['unmet demand'].sum()
        satisfied_percentage = (total_satisfied / total_demand * 100) if total_demand > 0 else 0
        unmet_percentage = (unmet_demand / total_demand * 100) if total_demand > 0 else 0
        print(f"Total demand:        {total_demand:15,.0f} m³ ( 100.0%)")
        print(f"Satisfied demand:    {total_satisfied:15,.0f} m³ ({satisfied_percentage:6.1f}%)")
        print(f"Unmet demand:        {unmet_demand:15,.0f} m³ ({unmet_percentage:6.1f}%)")


        # Balance error statistics
        print_section("Balance Error Statistics")
        error_stats = df['balance_error'].describe()
        
        error_metrics = {
            'Mean': error_stats['mean'],
            'Std Dev': error_stats['std'],
            'Minimum': error_stats['min'],
            'Maximum': error_stats['max']
        }
        
        # Calculate relative errors
        total_flux = df['source'].sum()
        for label, value in error_metrics.items():
            print(f"{label:8s}: {value:15,.2f} m³")
        
        # Additional statistics
        print_section("Maximum Values")
        components = {
            'Source': 'source',
            'Supplied': 'supplied demand',
            'Sink': 'sink',
            'Losses': 'edge losses',
            'Spills': 'reservoir spills',
            'Res ET': 'reservoir ET losses'
        }
        
        for label, comp in components.items():
            max_value = df[comp].max()
            timestep = df.loc[df[comp].idxmax(), 'time_step']
            print(f"{label:10s}: {max_value:15,.0f} m³ at timestep {timestep:3.0f}")
        
        # Conservation check
        print_section("Conservation Check")
        total_in = total_source
        total_out = sum(df[comp].sum() for comp in ['supplied demand', 'sink', 'edge losses', 'reservoir spills', 'reservoir ET losses'])
        total_stored = total_storage_change
        
        print(f"Total in:          {total_in:15,.0f} m³")
        print(f"Total out:         {total_out:15,.0f} m³")
        print(f"Net storage:       {total_stored:15,.0f} m³")
        
        balance = total_in - total_out - total_stored
        print(f"Balance residual:  {balance:15,.0f} m³")
        
        print("\n" + "=" * 50)

    def plot_network_timesteps(self, timesteps=None, min_edge_width=1, max_edge_width=20):
        """
        Create network layout plots for specified timesteps, where edge widths represent flow rates.
        Nodes are shown with minimal labels and a legend indicates node types.
        
        Args:
            timesteps (list, optional): List of timesteps to visualize. If None, plots all timesteps.
            min_edge_width (float): Minimum edge width in the visualization. Defaults to 1.
            max_edge_width (float): Maximum edge width in the visualization. Defaults to 10.
            
        Returns:
            list: Paths to the saved plot files
        """
        # If no timesteps specified, plot all timesteps
        if timesteps is None:
            timesteps = range(self.system.time_steps)
        elif isinstance(timesteps, int):
            timesteps = [timesteps]
            
        # Get node positions based on easting and northing
        pos = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            pos[node] = (node_instance.easting, node_instance.northing)

        # Define node styling with descriptive names for legend
        node_styles = {
            SupplyNode: {'color': 'skyblue', 'shape': 's', 'name': 'Supply Node'},
            StorageNode: {'color': 'lightgreen', 'shape': 's', 'name': 'Storage Node'},
            DemandNode: {'color': 'salmon', 'shape': 'o', 'name': 'Demand Node'},
            SinkNode: {'color': 'lightgray', 'shape': 's', 'name': 'Sink Node'},
            HydroWorks: {'color': 'orange', 'shape': 'o', 'name': 'Hydroworks'}
        }
        
        node_size = 1000  # Reduced node size
        plot_paths = []
        
        # Find maximum flow rate across all timesteps for edge width scaling
        max_flow = 0
        for _, _, edge_data in self.system.graph.edges(data=True):
            edge = edge_data['edge']
            if edge.outflow:  # Check if edge has flow data
                max_flow = max(max_flow, max(edge.outflow))
        
        # Create a plot for each timestep
        for t in timesteps:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(15, 12))
            plt.title(f'Water System Network - Timestep {t}', pad=20)
            
            # Draw nodes for each type
            for node_type, style in node_styles.items():
                node_list = [node for node, data in self.system.graph.nodes(data=True) 
                            if isinstance(data['node'], node_type)]
                nx.draw_networkx_nodes(self.system.graph, pos, 
                                    nodelist=node_list,
                                    node_color=style['color'],
                                    node_shape=style['shape'],
                                    node_size=node_size,
                                    alpha=0.8,
                                    ax=ax)
            
            # Draw edges with widths based on flow rates
            edge_colors = []
            edge_widths = []
            edge_labels = {}
            
            for u, v, data in self.system.graph.edges(data=True):
                edge = data['edge']
                if t < len(edge.outflow):
                    flow = edge.outflow[t]
                    # Scale edge width between min_edge_width and max_edge_width
                    width = (((flow / max_flow) * (max_edge_width - min_edge_width)) 
                            + min_edge_width if max_flow > 0 else min_edge_width)
                    
                    # Color edges based on flow percentage of capacity
                    flow_pct = (flow / edge.capacity) if edge.capacity > 0 else 0
                    if flow_pct < 0.33:
                        color = 'lightblue'
                    elif flow_pct < 0.66:
                        color = 'blue'
                    else:
                        color = 'darkblue'
                        
                    edge_colors.append(color)
                    edge_widths.append(width)
                    edge_labels[(u, v)] = f'{flow:.1f} m³/s'
                else:
                    edge_colors.append('gray')
                    edge_widths.append(min_edge_width)
                    edge_labels[(u, v)] = '0'
            
            # Draw edges
            nx.draw_networkx_edges(self.system.graph, pos,
                                edge_color=edge_colors,
                                width=edge_widths,
                                arrows=True,
                                arrowsize=20,
                                ax=ax)
            
            # Draw simple node labels (just node IDs)
            labels = {node: node for node in self.system.graph.nodes()}
            nx.draw_networkx_labels(self.system.graph, pos, labels, font_size=8)
            
            # Draw edge labels
            edge_label_pos = nx.draw_networkx_edge_labels(
                self.system.graph, pos,
                edge_labels=edge_labels,
                font_size=7
            )
            
            # Create legend for node types and edge colors
            legend_elements = []
            
            # Add node type legend elements
            for style in node_styles.values():
                legend_elements.append(
                    plt.scatter([], [], 
                            c=style['color'],
                            marker=style['shape'],
                            s=100,
                            label=style['name'])
                )
            
            # Add edge color legend elements
            legend_elements.extend([
                plt.Line2D([0], [0], color='lightblue', lw=2, label='< 33% capacity'),
                plt.Line2D([0], [0], color='blue', lw=2, label='33-66% capacity'),
                plt.Line2D([0], [0], color='darkblue', lw=2, label='> 66% capacity')
            ])
            
            # Add legend with both node types and edge colors
            ax.legend(handles=legend_elements,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5),
                    title='Network Elements',
                    frameon=True,
                    fancybox=True,
                    shadow=True)
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save plot
            plot_path = self._save_plot(f"network_flow_t{t}")
            plot_paths.append(plot_path)
            plt.close()
        
        return plot_paths
    
    def create_interactive_network_visualization(self):
        """
        Create an interactive visualization of the water system network as a React component.
        Network shows flows changing over time with an interactive time slider.
        
        Returns:
            str: Path to the saved HTML file containing the interactive visualization
        """
        # Collect node data
        nodes = []
        for node_id, data in self.system.graph.nodes(data=True):
            node = data['node']
            node_type = type(node)
            
            # Determine node style based on type
            color = {
                SupplyNode: 'rgb(135, 206, 235)',  # skyblue
                StorageNode: 'rgb(144, 238, 144)',  # lightgreen
                DemandNode: 'rgb(250, 128, 114)',   # salmon
                SinkNode: 'rgb(211, 211, 211)',     # lightgray
                HydroWorks: 'rgb(255, 165, 0)'      # orange
            }.get(node_type, 'gray')
            
            shape = 'circle' if node_type in [DemandNode, HydroWorks] else 'square'
            
            nodes.append({
                'id': node_id,
                'easting': float(node.easting),
                'northing': float(node.northing),
                'color': color,
                'shape': shape
            })
        
        # Collect edge data
        edges = []
        max_flow = 0
        
        for u, v, data in self.system.graph.edges(data=True):
            edge = data['edge']
            flows = [float(f) for f in edge.outflow] if edge.outflow else []
            if flows:
                max_flow = max(max_flow, max(flows))
            
            edges.append({
                'source': u,
                'target': v,
                'capacity': float(edge.capacity),
                'flows': flows
            })
        
        # Create network data object
        network_data = {
            'nodes': nodes,
            'edges': edges,
            'maxFlow': max_flow,
            'timeSteps': self.system.time_steps
        }
        
        # Convert network data to JSON string
        network_data_json = json.dumps(network_data)
        
        # Load the HTML template from a separate file
        html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Water System Network</title>
        <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body>
        <div id="root"></div>
        <script type="text/babel">
            // Network data
            const networkData = NETWORK_DATA_PLACEHOLDER;
            
            const NetworkGraph = ({data, width = 1000, height = 800}) => {
                const [currentTimestep, setCurrentTimestep] = React.useState(0);
                const [isPlaying, setIsPlaying] = React.useState(false);
                const [playSpeed, setPlaySpeed] = React.useState(1000);

                const { nodes, edges, maxFlow, timeSteps } = data;

                if (!nodes || !edges || nodes.length === 0) {
                    return (
                        <div className="flex items-center justify-center w-full h-full min-h-[400px] bg-gray-50 rounded-lg border">
                            <p className="text-gray-500">No network data available</p>
                        </div>
                    );
                }

                // Calculate scales
                const eastings = nodes.map(n => n.easting);
                const northings = nodes.map(n => n.northing);
                const minEasting = Math.min(...eastings);
                const maxEasting = Math.max(...eastings);
                const minNorthing = Math.min(...northings);
                const maxNorthing = Math.max(...northings);

                const padding = 0.1;
                const eastingRange = maxEasting - minEasting;
                const northingRange = maxNorthing - minNorthing;
                const paddedMinEasting = minEasting - eastingRange * padding;
                const paddedMaxEasting = maxEasting + eastingRange * padding;
                const paddedMinNorthing = minNorthing - northingRange * padding;
                const paddedMaxNorthing = maxNorthing + northingRange * padding;

                const scaleX = x => ((x - paddedMinEasting) / (paddedMaxEasting - paddedMinEasting)) * width;
                const scaleY = y => height - ((y - paddedMinNorthing) / (paddedMaxNorthing - paddedMinNorthing)) * height;

                const getEdgeProperties = edge => {
                    const flow = edge.flows[currentTimestep] || 0;
                    const minWidth = 1;
                    const maxWidth = 50;
                    const width = flow > 0 ? ((flow / maxFlow) * (maxWidth - minWidth)) + minWidth : minWidth;

                    const flowPercent = edge.capacity > 0 ? flow / edge.capacity : 0;
                    let color;
                    if (flowPercent < 0.33) color = '#BFDBFE';
                    else if (flowPercent < 0.66) color = '#2563EB';
                    else color = '#1E3A8A';

                    return { width, color, flow };
                };

                React.useEffect(() => {
                    let interval;
                    if (isPlaying) {
                        interval = setInterval(() => {
                            setCurrentTimestep(prev => {
                                if (prev >= timeSteps - 1) {
                                    setIsPlaying(false);
                                    return 0;
                                }
                                return prev + 1;
                            });
                        }, playSpeed);
                    }
                    return () => clearInterval(interval);
                }, [isPlaying, playSpeed, timeSteps]);

                return (
                    <div className="flex flex-col gap-4 p-4 w-full bg-white rounded-lg shadow">
                        <div className="flex justify-between items-center">
                            <h2 className="text-xl font-bold">Water System Network - Timestep {currentTimestep}</h2>
                            <div className="flex gap-2">
                                <button 
                                    onClick={() => setCurrentTimestep(0)}
                                    className="p-2 rounded hover:bg-gray-100"
                                >
                                    Reset
                                </button>
                                <button 
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    className="p-2 rounded hover:bg-gray-100"
                                >
                                    {isPlaying ? 'Pause' : 'Play'}
                                </button>
                            </div>
                        </div>

                        <div className="flex gap-4">
                            <input
                                type="range"
                                min="0"
                                max={timeSteps - 1}
                                value={currentTimestep}
                                onChange={e => {
                                    setCurrentTimestep(Number(e.target.value));
                                    setIsPlaying(false);
                                }}
                                className="w-full"
                            />
                        </div>

                        <div className="flex gap-4">
                            <svg width={width} height={height} className="border rounded">
                                {/* Draw edges */}
                                {edges.map((edge, i) => {
                                    const source = nodes.find(n => n.id === edge.source);
                                    const target = nodes.find(n => n.id === edge.target);
                                    const { width: strokeWidth, color, flow } = getEdgeProperties(edge);
                                    
                                    const x1 = scaleX(source.easting);
                                    const y1 = scaleY(source.northing);
                                    const x2 = scaleX(target.easting);
                                    const y2 = scaleY(target.northing);

                                    // Calculate curved path
                                    const dx = x2 - x1;
                                    const dy = y2 - y1;
                                    const angle = Math.atan2(dy, dx);
                                    const length = Math.sqrt(dx * dx + dy * dy);
                                    
                                    const midX = (x1 + x2) / 2;
                                    const midY = (y1 + y2) / 2;
                                    const curveOffset = length * 0.2;
                                    const ctrlX = midX - curveOffset * Math.sin(angle);
                                    const ctrlY = midY + curveOffset * Math.cos(angle);

                                    return (
                                        <g key={i}>
                                            <path
                                                d={`M ${x1} ${y1} L ${x2} ${y2}`}
                                                stroke={color}
                                                strokeWidth={strokeWidth}
                                                fill="none"
                                                markerEnd="url(#arrowhead)"
                                            />
                                            <text
                                                x={midX}
                                                y={midY}
                                                dy={-10}
                                                textAnchor="middle"
                                                fill="black"
                                                fontSize="12"
                                            >
                                                {flow.toFixed(1)} m³/s
                                            </text>
                                        </g>
                                    );
                                })}

                                {/* Draw nodes */}
                                {nodes.map((node, i) => {
                                    const x = scaleX(node.easting);
                                    const y = scaleY(node.northing);
                                    const size = 20;

                                    return (
                                        <g key={i}>
                                            {node.shape === 'circle' ? (
                                                <circle
                                                    cx={x}
                                                    cy={y}
                                                    r={size / 2}
                                                    fill={node.color}
                                                    stroke="black"
                                                    strokeWidth="1"
                                                />
                                            ) : (
                                                <rect
                                                    x={x - size / 2}
                                                    y={y - size / 2}
                                                    width={size}
                                                    height={size}
                                                    fill={node.color}
                                                    stroke="black"
                                                    strokeWidth="1"
                                                />
                                            )}
                                            <text
                                                x={x}
                                                y={y + size}
                                                textAnchor="middle"
                                                fill="black"
                                                fontSize="12"
                                            >
                                                {node.id}
                                            </text>
                                        </g>
                                    );
                                })}
                            </svg>

                            <div className="border rounded p-4 bg-gray-50">
                                <h3 className="font-bold mb-2">Legend</h3>
                                <div className="space-y-2">
                                    <div>
                                        <h4 className="text-sm font-semibold">Node Types</h4>
                                        <div className="space-y-1">
                                            {[
                                                { type: 'Supply Node', color: 'rgb(135, 206, 235)', shape: 'square' },
                                                { type: 'Storage Node', color: 'rgb(144, 238, 144)', shape: 'square' },
                                                { type: 'Demand Node', color: 'rgb(250, 128, 114)', shape: 'circle' },
                                                { type: 'Sink Node', color: 'rgb(211, 211, 211)', shape: 'square' },
                                                { type: 'Hydroworks', color: 'rgb(255, 165, 0)', shape: 'circle' }
                                            ].map((item, i) => (
                                                <div key={i} className="flex items-center gap-2">
                                                    <div 
                                                        className="w-4 h-4 rounded"
                                                        style={{ backgroundColor: item.color }}
                                                    />
                                                    <span className="text-sm">{item.type}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-semibold">Flow Capacity</h4>
                                        <div className="space-y-1">
                                            {[
                                                { type: '< 33% capacity', color: '#BFDBFE' },
                                                { type: '33-66% capacity', color: '#2563EB' },
                                                { type: '> 66% capacity', color: '#1E3A8A' }
                                            ].map((item, i) => (
                                                <div key={i} className="flex items-center gap-2">
                                                    <div 
                                                        className="w-8 h-1 rounded"
                                                        style={{ backgroundColor: item.color }}
                                                    />
                                                    <span className="text-sm">{item.type}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            };

            // Render the app
            ReactDOM.render(
                <NetworkGraph data={networkData} />,
                document.getElementById('root')
            );
        </script>
    </body>
    </html>
    """
        
        # Replace the placeholder with actual data
        html_content = html_template.replace('NETWORK_DATA_PLACEHOLDER', network_data_json)
        
        # Save the HTML file
        filename = f"{self.name}_interactive_network.html" if self.name else "water_system_network.html"
        filepath = os.path.join(self.image_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def plot_storage_spills(self):
        """
        Create time series plots showing storage dynamics including volumes, elevations, spills, 
        and evaporation losses for each storage node over time in separate subplots.
        
        Returns:
            str: Path to the saved plot file
        """
        # Find storage nodes in the system
        storage_cols = [(col, col.replace('_Storage', '')) for col in self.df.columns 
                        if col.endswith('_Storage')]
        
        if not storage_cols:
            print("No storage nodes found in the system.")
            return None
            
        # Get unique node names
        unique_nodes = sorted(set(node for _, node in storage_cols))
        n_nodes = len(unique_nodes)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 6*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Color scheme
        colors = {
            'storage': '#2196F3',     # Blue for storage volume
            'spills': '#f44336',      # Red for spills
            'evaporation': '#FF9800', # Orange for evaporation losses
            'elevation': '#4CAF50'    # Green for water surface elevation
        }
        
        # Plot for each storage node
        for idx, node_name in enumerate(unique_nodes):
            ax = axes[idx]
            
            # Plot storage volume
            storage_col = f"{node_name}_Storage"
            storage = self.df[storage_col]
            ax.plot(self.df['TimeStep'], storage, 
                   color=colors['storage'],
                   label='Storage Volume',
                   linewidth=2)
            
            # Plot spills if available
            spill_col = f"{node_name}_ExcessVolume"
            if spill_col in self.df.columns:
                spills = self.df[spill_col]
                ax.plot(self.df['TimeStep'], spills,
                       color=colors['spills'],
                       label='Spillway Volume',
                       linestyle='--',
                       linewidth=1.5)
            
            # Find the storage node instance to get evaporation data
            node_instance = None
            for _, node_data in self.system.graph.nodes(data=True):
                if isinstance(node_data['node'], StorageNode) and node_data['node'].id == node_name:
                    node_instance = node_data['node']
                    break
            
            # Plot evaporation losses if available
            if node_instance and hasattr(node_instance, 'evaporation_losses') and node_instance.evaporation_losses:
                evap_losses = node_instance.evaporation_losses
                ax.plot(range(len(evap_losses)), evap_losses,
                       color=colors['evaporation'],
                       label='Evaporation Loss',
                       linestyle=':',
                       linewidth=1.5)
            
            # Create second y-axis for elevation
            ax2 = ax.twinx()
            if hasattr(node_instance, 'water_level') and node_instance.water_level:
                elevations = node_instance.water_level
                ax2.plot(range(len(elevations)), elevations,
                        color=colors['elevation'],
                        label='Water Surface Elevation',
                        linestyle='-.',
                        linewidth=1.5)
                ax2.set_ylabel('Elevation [m.a.s.l]', color=colors['elevation'])
                ax2.tick_params(axis='y', labelcolor=colors['elevation'])
            
            # Add reservoir capacity line
            if node_instance:
                capacity = node_instance.capacity
                ax.axhline(y=capacity, 
                          color='gray', 
                          linestyle='--', 
                          alpha=0.5,
                          label='Reservoir Capacity')
                ax.text(ax.get_xlim()[1], capacity, 
                       f'Capacity: {capacity:,.0f} m³',
                       verticalalignment='bottom',
                       horizontalalignment='right')
            
            # Add summary statistics in text box
            stats_text = (
                f"Statistics:\n"
                f"Mean Storage: {storage.mean():,.0f} m³\n"
                f"Max Storage: {storage.max():,.0f} m³\n"
                f"Min Storage: {storage.min():,.0f} m³\n"
            )
            
            if spill_col in self.df.columns:
                spill_volume = spills.sum()
                stats_text += f"Total Spill Volume: {spill_volume:,.0f} m³\n"
            
            if node_instance and node_instance.evaporation_losses:
                total_evap = sum(node_instance.evaporation_losses)
                mean_evap = np.mean(node_instance.evaporation_losses)
                stats_text += (f"Total Evaporation Loss: {total_evap:,.0f} m³\n"
                             f"Mean Monthly Evaporation: {mean_evap:,.1f} m³/s\n")
            
            if hasattr(node_instance, 'water_level') and node_instance.water_level:
                stats_text += (f"Mean Elevation: {np.mean(node_instance.water_level):.1f} m\n"
                             f"Max Elevation: {max(node_instance.water_level):.1f} m\n"
                             f"Min Elevation: {min(node_instance.water_level):.1f} m")
            
            # Add stats text box
            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            # Customize subplot
            ax.set_title(f'{node_name} Storage Dynamics')
            if idx == n_nodes - 1:  # Only add xlabel to bottom subplot
                ax.set_xlabel('Time Step')
            ax.set_ylabel('Volume [m³]')
            ax.grid(True, alpha=0.3)
            
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            if hasattr(node_instance, 'water_level') and node_instance.water_level:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                         bbox_to_anchor=(1.15, 1),
                         loc='upper left',
                         borderaxespad=0.,
                         frameon=True,
                         fancybox=True,
                         shadow=True)
            else:
                ax.legend(bbox_to_anchor=(1.15, 1),
                         loc='upper left',
                         borderaxespad=0.,
                         frameon=True,
                         fancybox=True,
                         shadow=True)
        
        plt.tight_layout()
        return self._save_plot("storage_dynamics")

    def plot_water_balance(self):
        """
        Create water balance visualization using side-by-side bars.
        
        Returns:
            str: Path to the saved plot file
        """
        if not hasattr(self, 'image_dir'):
            self.image_dir = os.path.join('.', 'figures')
            os.makedirs(self.image_dir, exist_ok=True)
        
        df = self.system.get_water_balance()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Width of bars
        bar_width = 0.35
        
        # Define colors with evaporation included
        colors = {
            'source': '#2196F3',           # Blue
            'supplied demand': '#4CAF50',   # Green
            'sink': '#f44336',             # Red
            'edge losses': '#FF9800',       # Orange
            'reservoir spills': '#9C27B0',  # Purple
            'reservoir ET losses': '#795548', # Brown
            'storage_change': '#607D8B'     # Gray
        }
        
        time_steps = df['time_step']
        
        # Create positive bars (sources)
        source_bars = ax.bar(time_steps, df['source'], 
                           bar_width, label='Source',
                           color=colors['source'], alpha=0.7)
        
        # Create stacked negative bars (sinks)
        components = ['supplied demand', 'sink', 'edge losses', 
                     'reservoir spills', 'reservoir ET losses', 'storage_change']
        labels = ['Supplied Demand', 'Sink', 'Edge Losses', 
                 'Reservoir Spills', 'Reservoir ET', 'Storage Change']
        
        bottom = np.zeros(len(df))
        bars = []
        for comp, label in zip(components, labels):
            bars.append(ax.bar(time_steps + bar_width, df[comp], 
                             bar_width, bottom=bottom,
                             label=label, color=colors[comp], alpha=0.7))
            bottom += df[comp]
            
        # Add balance error as a line
        ax.plot(time_steps + bar_width/2, df['balance_error'],
               label='Balance Error', color='black',
               linestyle='--', linewidth=1, marker='o')
        
        # Customize plot
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Volume (m³)')
        ax.set_title('Water Balance Components')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(self.image_dir, f"{self.name}_water_balance.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path
