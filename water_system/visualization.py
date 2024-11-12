"""
This module provides visualization tools for analyzing water system simulation results.
It includes time series visualizations for flows, storage levels, and demands.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
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
        
    def plot_reservoir_volume(self):
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
        plt.ylabel('Storage Volume [m³]')
        plt.title('Storage Levels Over Time')
        plt.grid(True)
        plt.legend()
        
        return self._save_plot("reservoir_volume")
        
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
        plt.ylabel('Excess Volume [m³]')
        plt.title('Storage Node Spills Over Time')
        plt.grid(True)
        plt.legend()
        
        return self._save_plot("storage_spills")

    def plot_water_levels(self):
        """
        Create a time series plot showing water levels for all storage nodes that have hva data.
        Also shows the bottom and maximum elevation for reference.
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
                # Convert water levels to absolute elevations
                elevations = [node.get_elevation_from_level(level) for level in water_levels]
                
                # Plot water level time series
                plt.plot(range(len(elevations)), elevations, 
                        label=f"{node_id} Water Level", 
                        marker='o',
                        linewidth=0.8,
                        markersize=3,
                        markerfacecolor='white',
                        markeredgewidth=1)
                
                # Plot bottom and maximum elevations as reference lines
                plt.axhline(y=node.hva_data['bottom_elevation'], 
                          linestyle='--', 
                          color='gray', 
                          alpha=0.5,
                          label=f"{node_id} Bottom Elevation")
                          
                plt.axhline(y=node.hva_data['max_elevation'], 
                          linestyle=':', 
                          color='red', 
                          alpha=0.5,
                          label=f"{node_id} Maximum Elevation")
        
        plt.xlabel('Time Step')
        plt.ylabel('Elevation [m.a.s.l.]')  # meters above sea level
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

    def plot_edge_losses(self):
        """
        Create a time series plot showing water losses for all edges in the system.
        Displays both absolute losses and loss percentages.
        
        Returns:
            tuple: Paths to the saved absolute and percentage loss plot files
        """
        # Get all edge-related columns from the DataFrame
        loss_cols = [col for col in self.df.columns if col.startswith('Edge_') and col.endswith('_Losses')]
        loss_pct_cols = [col for col in self.df.columns if col.startswith('Edge_') and col.endswith('_LossPercent')]
        
        if not loss_cols:
            print("No edge loss data found in the system.")
            return None, None
        
        # Plot absolute losses
        plt.figure(figsize=(12, 10))
        
        # Create color map for edges
        colors = plt.cm.tab20(np.linspace(0, 1, len(loss_cols)))
        
        for idx, col in enumerate(loss_cols):
            # Extract edge name from column name for label
            edge_name = col.replace('Edge_', '').replace('_Losses', '')
            edge_name = edge_name.replace('_to_', ' → ')
            
            plt.plot(self.df['TimeStep'], 
                    self.df[col],
                    label=edge_name,
                    color=colors[idx],
                    marker='o',
                    linewidth=0.8,
                    markersize=3,
                    markerfacecolor='white',
                    markeredgewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Water Loss [m³/s]')
        plt.title('Absolute Water Losses Over Edges')
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
        abs_filepath = self._save_plot("edge_losses_absolute")
        
        # Plot loss percentages
        plt.figure(figsize=(12, 10))
        
        for idx, col in enumerate(loss_pct_cols):
            # Extract edge name from column name for label
            edge_name = col.replace('Edge_', '').replace('_LossPercent', '')
            edge_name = edge_name.replace('_to_', ' → ')
            
            plt.plot(self.df['TimeStep'], 
                    self.df[col],
                    label=edge_name,
                    color=colors[idx],
                    marker='o',
                    linewidth=0.8,
                    markersize=3,
                    markerfacecolor='white',
                    markeredgewidth=1)
        
        plt.xlabel('Time Step')
        plt.ylabel('Water Loss [%]')
        plt.title('Percentage Water Losses Over Edges')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference lines for significant loss percentages
        plt.axhline(y=25, color='yellow', linestyle='--', alpha=0.5, label='25% Loss')
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Loss')
        plt.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='75% Loss')
        
        # Enhance legend
        plt.legend(bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.,
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)
        
        plt.tight_layout()
        pct_filepath = self._save_plot("edge_losses_percentage")
        
        return abs_filepath, pct_filepath

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

    def plot_reservoir_flows_and_volume(self):
        """
        Create plots showing inflows, outflows, and storage volume for each reservoir (storage node).
        Uses twin axes to show flows and volume on different scales.
        
        Returns:
            list: Paths to the saved plot files for each reservoir
        """
        # Get all storage nodes from the system
        storage_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], StorageNode)]
        
        if not storage_nodes:
            print("No storage nodes found in the system.")
            return []
        
        plot_paths = []
        
        # Create a separate plot for each reservoir
        for node_id, node in storage_nodes:
            # Create figure with primary and secondary y-axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Get column names for this reservoir
            inflow_col = f"{node_id}_Inflow"
            outflow_col = f"{node_id}_Outflow"
            storage_col = f"{node_id}_Storage"
            excess_vol_col = f"{node_id}_ExcessVolume"
            
            # Define colors for different data series
            colors = {
                'inflow': '#2196F3',    # Blue
                'outflow': '#4CAF50',   # Green
                'storage': '#FF9800',    # Orange
                'excess': '#F44336',     # Red
                'capacity': '#9C27B0'    # Purple
            }
            
            # Plot flows on primary y-axis
            ln1 = ax1.plot(self.df['TimeStep'], self.df[inflow_col], 
                        color=colors['inflow'], label='Inflow',
                        marker='o', linewidth=1.5, markersize=4, markerfacecolor='white')
            ln2 = ax1.plot(self.df['TimeStep'], self.df[outflow_col], 
                        color=colors['outflow'], label='Outflow',
                        marker='s', linewidth=1.5, markersize=4, markerfacecolor='white')
            
            # Plot storage volume on secondary y-axis
            ln3 = ax2.plot(self.df['TimeStep'], self.df[storage_col], 
                        color=colors['storage'], label='Storage Volume',
                        marker='^', linewidth=1.5, markersize=4, markerfacecolor='white')
            
            # Plot capacity line on secondary y-axis
            ln4 = ax2.axhline(y=node.capacity, color=colors['capacity'], 
                            linestyle='--', label=f'Capacity ({node.capacity:,.0f} m³)')
            
            # Plot excess volume if any exists
            if excess_vol_col in self.df.columns and self.df[excess_vol_col].max() > 0:
                ln5 = ax1.plot(self.df['TimeStep'], self.df[excess_vol_col], 
                            color=colors['excess'], label='Excess Volume',
                            marker='x', linewidth=1.5, markersize=4)
                lines = ln1 + ln2 + ln3 + [ln4] + ln5
            else:
                lines = ln1 + ln2 + ln3 + [ln4]
            
            # Calculate and add key statistics
            mean_inflow = self.df[inflow_col].mean()
            mean_outflow = self.df[outflow_col].mean()
            mean_storage = self.df[storage_col].mean()
            min_storage = self.df[storage_col].min()
            max_storage = self.df[storage_col].max()
            storage_utilization = (mean_storage / node.capacity) * 100
            
            stats_text = (
                f'Mean Inflow: {mean_inflow:.2f} m³/s\n'
                f'Mean Outflow: {mean_outflow:.2f} m³/s\n'
                f'Mean Storage: {mean_storage:,.0f} m³\n'
                f'Min Storage: {min_storage:,.0f} m³\n'
                f'Max Storage: {max_storage:,.0f} m³\n'
                f'Storage Utilization: {storage_utilization:.1f}%'
            )
            
            # Add stats text box
            plt.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            # Customize axes
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Flow Rate [m³/s]')
            ax2.set_ylabel('Storage Volume [m³]')
            
            # Format y-axis labels with thousand separators
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Add light grid
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            # Set title
            plt.title(f'Reservoir Flows and Storage Volume - {node_id}')
            
            # Combine legends from both axes
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.9))
            
            # Add shaded regions for storage zones if HVA data is available
            if hasattr(node, 'hva_data') and node.hva_data is not None:
                # Get volume ranges
                ranges = node.get_interpolation_ranges()
                if ranges and 'volume_range' in ranges:
                    min_vol = ranges['volume_range']['min']
                    ax2.axhspan(0, min_vol, color='red', alpha=0.1, label='Dead Storage')
                    ax2.axhspan(min_vol, node.capacity * 0.25, color='orange', alpha=0.1, label='Conservation')
                    ax2.axhspan(node.capacity * 0.25, node.capacity * 0.75, color='green', alpha=0.1, label='Normal Operation')
                    ax2.axhspan(node.capacity * 0.75, node.capacity, color='blue', alpha=0.1, label='Flood Control')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_path = self._save_plot(f"reservoir_flows_and_volume_{node_id}")
            plot_paths.append(plot_path)
            
        return plot_paths

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
                                    f'Cap: {edge.capacity}m³/s\n'
                                    f'L: {edge.length:.1f} km\n')
            else:
                edge_labels[(u, v)] = f'Cap: {edge.capacity}'
        
        nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, 
                                font_size=8, ax=ax)
        
        plt.axis('off')
        plt.tight_layout()
        
        return self._save_plot("network_layout")  
