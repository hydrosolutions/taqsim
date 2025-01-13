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
                            row_data[f"{node_id}_Storage"] = node.storage[time_step]
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
                df.to_csv('water_balance_tot.csv', index=False)
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
        Create and save a static network layout plot for the water system, showing only
        structural information like capacities and lengths.
        Returns the path to the saved image file.
        """
        # Setting node positions based on easting and northing
        pos = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            pos[node] = (node_instance.easting, node_instance.northing)

        # Create figure with a main subplot and space for legend
        fig = plt.figure(figsize=(25, 15))
        gs = plt.GridSpec(1, 20, figure=fig)
        ax = fig.add_subplot(gs[0, :19])  # Main plot takes up most of the space

        #plt.suptitle(f'{self.name} System Network Layout', fontsize=30, y=0.95)
        
        # Define node styling with descriptive names for legend
        node_styles = {
            SupplyNode: {'color': 'skyblue', 'shape': 's', 'name': 'Supply Node'},
            StorageNode: {'color': 'lightgreen', 'shape': 's', 'name': 'Storage Node'},
            DemandNode: {'color': 'salmon', 'shape': 'o', 'name': 'Demand Node'},
            SinkNode: {'color': 'lightgray', 'shape': 's', 'name': 'Sink Node'},
            HydroWorks: {'color': 'orange', 'shape': 'o', 'name': 'Hydroworks'}
        }
        
        node_size = 5000
        
        # Draw nodes and collect legend elements
        legend_elements = []
        for node_type, style in node_styles.items():
            node_list = [node for node, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], node_type)]
            nx.draw_networkx_nodes(self.system.graph, pos, 
                                nodelist=node_list, 
                                node_color=style['color'], 
                                node_shape=style['shape'],
                                node_size=node_size, 
                                alpha=0.6,
                                ax=ax)
            
            # Add to legend elements
            legend_elements.append(plt.scatter([], [], 
                                            c=style['color'],
                                            marker=style['shape'],
                                            s=500,
                                            label=style['name']))

        # Draw edges with width based on capacity
        max_capacity = max(edge_data['edge'].capacity for _, _, edge_data in self.system.graph.edges(data=True))
        
        edge_widths = []
        for _, _, edge_data in self.system.graph.edges(data=True):
            edge = edge_data['edge']
            # Scale width between 1 and 10 based on capacity
            width =1 + (edge.capacity / max_capacity) * 15
            edge_widths.append(width)

        # Draw edges
        nx.draw_networkx_edges(self.system.graph, pos, 
                             edge_color='gray',
                             arrows=True, 
                             arrowsize=65, 
                             width=edge_widths,
                             ax=ax)

        # Update node labels with static information
        labels = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            if isinstance(node_instance, SupplyNode):
                labels[node] = f"{node}"
            elif isinstance(node_instance, DemandNode):
                labels[node] = f"{node}"
            elif isinstance(node_instance, SinkNode):
                labels[node] = f"{node}"
            elif isinstance(node_instance, StorageNode):
                labels[node] = f"{node}\nCap: {node_instance.capacity:,.0f} m³"
            elif isinstance(node_instance, HydroWorks):
                labels[node] = f"{node}"
        
        nx.draw_networkx_labels(self.system.graph, pos, labels, font_size=14, ax=ax)
        
        # Add edge labels showing capacity and length
        edge_labels = {}
        for u, v, d in self.system.graph.edges(data=True):
            edge = d['edge']
            edge_labels[(u, v)] = (f'{edge.capacity} m³/s')
      
        nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, 
                       font_size=14, ax=ax, rotate=False)
  

        # Create the legend on the top right of the plot
        ax.legend(handles=legend_elements,
              loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=22)
        
        plt.axis('off')
        plt.tight_layout()
        
        return self._save_plot("network_layout")
    
    def plot_network_layout_2(self):
        """
        Create and save a static network layout plot for the water system, showing only
        structural information like capacities and lengths.
        Returns the path to the saved image file.
        """
        # Setting node positions based on easting and northing
        pos = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            pos[node] = (node_instance.easting, node_instance.northing)

        # Create figure with a main subplot and space for legend
        fig = plt.figure(figsize=(25, 15))
        gs = plt.GridSpec(1, 20, figure=fig)
        ax = fig.add_subplot(gs[0, :19])  # Main plot takes up most of the space

        #plt.suptitle(f'{self.name} System Network Layout', fontsize=30, y=0.95)
        
        # Define node styling with descriptive names for legend
        node_styles = {
            SupplyNode: {'color': 'skyblue', 'shape': 's', 'name': 'Supply Node'},
            StorageNode: {'color': 'lightgreen', 'shape': 's', 'name': 'Storage Node'},
            DemandNode: {'color': 'salmon', 'shape': 'o', 'name': 'Demand Node'},
            SinkNode: {'color': 'lightgray', 'shape': 's', 'name': 'Sink Node'},
            HydroWorks: {'color': 'orange', 'shape': 'o', 'name': 'Hydroworks'}
        }
        
        node_size = 500
        
        # Draw nodes and collect legend elements
        legend_elements = []
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
            
            # Add to legend elements
            legend_elements.append(plt.scatter([], [], 
                                            c=style['color'],
                                            marker=style['shape'],
                                            s=500,
                                            label=style['name']))

        # Draw edges with width based on capacity
        max_capacity = max(edge_data['edge'].capacity for _, _, edge_data in self.system.graph.edges(data=True))
        
        edge_widths = []
        for _, _, edge_data in self.system.graph.edges(data=True):
            edge = edge_data['edge']
            # Scale width between 1 and 10 based on capacity
            width =1 + (edge.capacity / max_capacity) * 4
            edge_widths.append(width)

        # Draw edges
        nx.draw_networkx_edges(self.system.graph, pos, 
                             edge_color='gray',
                             arrows=True, 
                             arrowsize=35, 
                             width=edge_widths,
                             ax=ax)

        # Update node labels with static information
        labels = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            if isinstance(node_instance, SupplyNode):
                labels[node] = f"{node}"
            elif isinstance(node_instance, DemandNode):
                labels[node] = f"{node}"
            elif isinstance(node_instance, SinkNode):
                labels[node] = f"{node}"
            elif isinstance(node_instance, StorageNode):
                labels[node] = f"{node}\nCap: {node_instance.capacity:,.0f} m³"
            elif isinstance(node_instance, HydroWorks):
                labels[node] = f"{node}"
        
        #nx.draw_networkx_labels(self.system.graph, pos, labels, font_size=14, ax=ax)
        
        # Add edge labels showing capacity and length
        edge_labels = {}
        for u, v, d in self.system.graph.edges(data=True):
            edge = d['edge']
            edge_labels[(u, v)] = (f'{edge.capacity} m³/s')
      
        #nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, font_size=14, ax=ax, rotate=False, bbox=dict(facecolor='white', alpha=0.5))

        # Create the legend on the top right of the plot
        ax.legend(handles=legend_elements,
              loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=22)
        
        plt.axis('off')
        plt.tight_layout()
        
        return self._save_plot("network_layout_hydroworks")

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
            'HW Spills': 'hydroworks spills'
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
        print(f"Total demand:        {total_demand:15,.0f} m³")
        print(f"Satisfied demand:    {total_satisfied:15,.0f} m³")
        print(f"Unmet demand:        {unmet_demand:15,.0f} m³")

        
        # Conservation check
        print_section("Conservation Check")
        total_in = total_source
        total_out = sum(df[comp].sum() for comp in ['supplied demand', 'sink', 'edge losses', 'reservoir spills', 'reservoir ET losses', 'hydroworks spills'])
        total_stored = total_storage_change
        
        print(f"Total in:          {total_in:15,.0f} m³")
        print(f"Total out:         {total_out:15,.0f} m³")
        print(f"Net storage:       {total_stored:15,.0f} m³")
        
        balance = total_in - total_out - total_stored
        print(f"Balance residual:  {balance:15,.0f} m³")

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
            'Res Spills': 'reservoir spills',
            'Res ET': 'reservoir ET losses',
            'HW Spills': 'hydroworks spills'
        }
        
        for label, comp in components.items():
            max_value = df[comp].max()
            timestep = df.loc[df[comp].idxmax(), 'time_step']
            print(f"{label:10s}: {max_value:15,.0f} m³ at timestep {timestep:3.0f}")
        
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
    
    def plot_storage_dynamics(self):
        """
        Create time series plots showing storage dynamics including volumes, elevations, spills, 
        and evaporation losses for each storage node over time in separate subplots.
        
        Returns:
            str: Path to the saved plot file
        """
        
        plt.rcParams.update({'font.size': 13})
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
            )
            
            if spill_col in self.df.columns:
                spill_volume = spills.sum()
                stats_text += f"Total Spill Volume: {spill_volume:,.0f} m³\n"
            
            if node_instance and node_instance.evaporation_losses:
                total_evap = sum(node_instance.evaporation_losses)
                mean_evap = np.mean(node_instance.evaporation_losses)
                stats_text += (f"Total Evaporation Loss: {total_evap:,.0f} m³\n"
                             f"Mean Monthly Evaporation: {mean_evap:,.1f} m³\n")

            
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

    def plot_reservoir_dynamics(self):
        """
        Create a comparison plot showing inflow, outflow, and volume for all reservoirs.
        
        Returns:
            str: Path to the saved plot file
        """
        plt.rcParams.update({'font.size': 13})
        # Find storage nodes in the system
        storage_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], StorageNode)]
        
        if not storage_nodes:
            print("No storage nodes found in the system.")
            return None
            
        # Create figure with subplots
        n_nodes = len(storage_nodes)
        fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 6*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Color scheme
        colors = {
            'inflow': '#2196F3',    # Blue
            'outflow': '#4CAF50',   # Green
            'waterlevel': '#9C27B0'     # Purple
        }
        
        time_steps = range(self.system.time_steps)
        
        # Plot for each reservoir
        for idx, (node_id, node) in enumerate(storage_nodes):
            ax1 = axes[idx]
            ax2 = ax1.twinx()  # Create second y-axis for waterlevel
            
            # Calculate total inflow for each timestep
            inflows = [sum(edge.get_edge_outflow(t) for edge in node.inflow_edges.values())
                    for t in time_steps]
            
            # Calculate total outflow for each timestep
            outflows = [sum(edge.get_edge_inflow(t) for edge in node.outflow_edges.values())
                    for t in time_steps]
            
            # Get waterlevels (excluding last entry which is for next timestep)
            waterlevels = node.water_level[:-1]
            
            # Plot flows on left y-axis
            ax1.plot(time_steps, inflows, color=colors['inflow'], 
                    label='Inflow', linewidth=2)
            ax1.plot(time_steps, outflows, color=colors['outflow'], 
                    label='Outflow', linewidth=2)
            
            # Plot waterlevel on right y-axis
            ax2.plot(time_steps, waterlevels, color=colors['waterlevel'], 
                    label='waterlevel', linewidth=2, linestyle='--')
            
            # Calculate statistics
            stats_text = (
                f"Statistics:\n"
                f"Mean Inflow: {np.mean(inflows):.2f} m³/s\n"
                f"Mean Outflow: {np.mean(outflows):.2f} m³/s\n"
                f"Mean waterlevel: {np.mean(waterlevels):,.0f} m³\n"
                f"Max waterlevel: {np.max(waterlevels):,.0f} m³\n"
                f"Min waterlevel: {np.min(waterlevels):,.0f} m³\n"
                f"waterlevel Change: {waterlevels[-1] - waterlevels[0]:,.0f} m³"
            )
            
            # Add statistics text box
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            # Customize axes
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Flow Rate [m³/s]')
            ax2.set_ylabel('Waterlevel [m asl.]')
            ax1.set_title(f'{node_id} Dynamics')
            ax1.grid(True, alpha=0.3)
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                    loc='center right',
                    bbox_to_anchor=(1.3, 0.5))
        
        plt.tight_layout()
        return self._save_plot("reservoir_inflow_outflow_waterlevel")

    def plot_all_flows(self, water_system):
        """
        Create a comprehensive plot showing all flows in the water system.
        
        Args:
            water_system: WaterSystem instance containing simulation results
            
        Returns:
            str: Path to saved plot file
        """
        # Get data from water balance
        df = water_system.get_water_balance()
        if len(df) == 0:
            print("No simulation data available")
            return None
            
        # Count different types of flows to determine subplot layout
        flow_categories = {
            'Sources': [col for col in df.columns if 'source' in col.lower()],
            'Demands': [col for col in df.columns if 'demand' in col.lower()],
            'Storage': [col for col in df.columns if 'storage' in col.lower()],
            'Losses': [col for col in df.columns if 'losses' in col.lower() or 'spills' in col.lower()]
        }
        
        # Create figure with subplots
        n_rows = len(flow_categories)
        fig, axes = plt.subplots(n_rows, 1, figsize=(15, 5*n_rows), sharex=True)
        
        # Color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # Plot each category
        for idx, (category, columns) in enumerate(flow_categories.items()):
            ax = axes[idx]
            
            # Plot each flow type in this category
            for i, col in enumerate(columns):
                values = df[col] * water_system.dt  # Convert to volumes
                ax.plot(df['time_step'], values, 
                    label=col.replace('_', ' ').title(),
                    color=colors[i],
                    marker='o',
                    markersize=4,
                    linewidth=2)
                
                # Add statistics
                mean_val = values.mean()
                total_val = values.sum()
                max_val = values.max()
                min_val = values.min()
                
                stats_text = (
                    f"Mean: {mean_val:.1f} m³\n"
                    f"Total: {total_val:.1f} m³\n"
                    f"Max: {max_val:.1f} m³\n"
                    f"Min: {min_val:.1f} m³"
                )
                
                # Add stats text box if there's data
                if len(values) > 0:
                    ax.text(0.02, 0.98, stats_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round',
                                facecolor='white',
                                alpha=0.8))
            
            # Customize subplot
            ax.set_title(f'{category} Over Time')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_ylabel('Volume [m³]')
        
        # Set common x-label
        axes[-1].set_xlabel('Time Step')
        
        # Add overall title
        plt.suptitle('Water System Flow Analysis', size=16, y=1.02)
        
        # Add water balance info
        balance_error = df['balance_error'].abs().mean()
        error_text = f"Mean Absolute Balance Error: {balance_error:.2f} m³"
        fig.text(0.98, 0.02, error_text, 
                horizontalalignment='right',
                bbox=dict(boxstyle='round',
                        facecolor='white',
                        alpha=0.8))
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        return self._save_plot("all_flows")

    def plot_water_balance_debug(self, storage_node):
        """
        Create detailed debug plots for water balance analysis of a storage node.
        Shows inflows, outflows, storage changes, and cumulative balance over time.
        
        Args:
            storage_node: StorageNode to analyze
        
        Returns:
            str: Path to saved plot file
        """
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))
        
        # Prepare time series data
        time_steps = range(len(storage_node.storage) - 1)  # Exclude last storage value
        
        # Calculate flow rates and volumes
        inflows = []
        outflows = []
        target_releases = []
        actual_releases = []
        storage_changes = []
        evap_losses = []
        spills = []
        cumulative_balance = []
        
        running_balance = 0
        
        for t in time_steps:
            # Calculate inflow
            total_inflow = sum(edge.get_edge_outflow(t) for edge in storage_node.inflow_edges.values())
            inflows.append(total_inflow)
            
            # Calculate outflow
            total_outflow = sum(edge.get_edge_inflow(t) for edge in storage_node.outflow_edges.values())
            outflows.append(total_outflow)
            
            # Get target release for current water level
            current_level = storage_node.get_level_from_volume(storage_node.storage[t])
            target_release = storage_node.calculate_release(current_level, t)
            target_releases.append(target_release)
            
            # Calculate actual release (might differ from outflow due to capacity constraints)
            actual_release = min(target_release, sum(edge.capacity for edge in storage_node.outflow_edges.values()))
            actual_releases.append(actual_release)
            
            # Get storage change
            storage_change = (storage_node.storage[t+1] - storage_node.storage[t]) / self.system.dt
            storage_changes.append(storage_change)
            
            # Get evaporation loss and spill
            evap_loss = storage_node.evaporation_losses[t] / self.system.dt if t < len(storage_node.evaporation_losses) else 0
            evap_losses.append(evap_loss)
            
            spill = storage_node.spillway_register[t] / self.system.dt if t < len(storage_node.spillway_register) else 0
            spills.append(spill)
            
            # Calculate water balance for this timestep
            balance = (total_inflow - total_outflow - evap_loss - spill - storage_change)
            running_balance += balance * self.system.dt
            cumulative_balance.append(running_balance)
        
        # Plot 1: Flow Rates
        ax1.plot(time_steps, inflows, label='Inflow', color='blue', linewidth=2)
        ax1.plot(time_steps, outflows, label='Actual Outflow', color='green', linewidth=2)
        ax1.plot(time_steps, target_releases, label='Target Release', color='red', linestyle='--')
        ax1.plot(time_steps, actual_releases, label='Capacity-Limited Release', color='orange', linestyle=':')
        
        ax1.set_title('Flow Rates Comparison')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Flow Rate [m³/s]')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Storage Changes and Losses
        ax2.plot(time_steps, storage_changes, label='Storage Change Rate', color='purple', linewidth=2)
        ax2.plot(time_steps, evap_losses, label='Evaporation Loss Rate', color='brown')
        ax2.plot(time_steps, spills, label='Spillway Rate', color='pink')
        
        ax2.set_title('Storage Changes and Losses')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Rate [m³/s]')
        ax2.grid(True)
        ax2.legend()
        
        # Plot 3: Cumulative Water Balance Error
        ax3.plot(time_steps, cumulative_balance, label='Cumulative Balance Error', color='red', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax3.set_title('Cumulative Water Balance Error')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Cumulative Error [m³]')
        ax3.grid(True)
        ax3.legend()
        
        # Add summary statistics
        stats_text = (
            f"Water Balance Summary:\n"
            f"Mean Inflow: {np.mean(inflows):.2f} m³/s\n"
            f"Mean Outflow: {np.mean(outflows):.2f} m³/s\n"
            f"Mean Target Release: {np.mean(target_releases):.2f} m³/s\n"
            f"Mean Storage Change: {np.mean(storage_changes):.2f} m³/s\n"
            f"Mean Evap Loss: {np.mean(evap_losses):.2f} m³/s\n"
            f"Mean Spill: {np.mean(spills):.2f} m³/s\n"
            f"Final Cumulative Error: {cumulative_balance[-1]:.2f} m³"
        )
        
        ax3.text(1.02, 0.5, stats_text,
                transform=ax3.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center')
        
        plt.tight_layout()
        return self._save_plot(f"water_balance_debug_{storage_node.id}")

    def plot_storage_waterbalance(self, storage_node):
        """
        Create a bar plot showing cumulative water balance components for a storage node.
        
        Args:
            storage_node: StorageNode object to analyze
            
        Returns:
            str: Path to saved plot file
        """
        # Calculate cumulative volumes for each component
        time_steps = range(len(storage_node.storage) - 1)  # Exclude last storage value
        dt = self.system.dt
        
        # Input volumes
        inflow_volume = sum(sum(edge.get_edge_outflow(t) for edge in storage_node.inflow_edges.values()) * dt 
                        for t in time_steps)
        initial_storage = storage_node.storage[0]
        
        # Output volumes
        outflow_volume = sum(sum(edge.get_edge_inflow(t) for edge in storage_node.outflow_edges.values()) * dt 
                            for t in time_steps)
        evap_volume = sum(storage_node.evaporation_losses)
        spill_volume = sum(storage_node.spillway_register)
        final_storage = storage_node.storage[-1]
        storage_change = final_storage - initial_storage
        
        # Calculate balance error
        balance_error = (inflow_volume + initial_storage - 
                        (outflow_volume + evap_volume + spill_volume + final_storage))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot inflows and outflows
        inflow_components = {
            'Initial Storage': initial_storage,
            'Total Inflow': inflow_volume
        }
        
        outflow_components = {
            'Final Storage': final_storage,
            'Total Outflow': outflow_volume,
            'Evaporation': evap_volume,
            'Spillway': spill_volume
        }
        
        # Color scheme
        inflow_colors = ['lightblue', 'blue']
        outflow_colors = ['lightblue', 'green', 'red', 'purple']
        
        # Plot inflows (left bar)
        bottom = 0
        for i, (label, volume) in enumerate(inflow_components.items()):
            ax1.bar('Inflow', volume, bottom=bottom, label=label, color=inflow_colors[i])
            # Add volume label in the middle of each segment
            if volume > 0:  # Only add label if segment is visible
                ax1.text('Inflow', bottom + volume/2, f'{volume:.1e}m³\n({(volume/sum(inflow_components.values())*100):.1f}%)',
                        ha='center', va='center')
            bottom += volume
        
        # Plot outflows (right bar)
        bottom = 0
        for i, (label, volume) in enumerate(outflow_components.items()):
            ax1.bar('Outflow', volume, bottom=bottom, label=label, color=outflow_colors[i])
            # Add volume label in the middle of each segment
            if volume > 0:  # Only add label if segment is visible
                ax1.text('Outflow', bottom + volume/2, f'{volume:.1e}m³\n({(volume/sum(outflow_components.values())*100):.1f}%)',
                        ha='center', va='center')
            bottom += volume
        
        # Customize first subplot
        ax1.set_title(f'Water Balance Components for {storage_node.id}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylabel('Volume [m³]')
        
        # Add balance error text
        balance_text = (
            f"Water Balance Error:\n"
            f"{balance_error:.2e} m³\n"
            f"({(balance_error/sum(inflow_components.values())*100):.2f}% of total input)"
        )
        ax1.text(1.4, 0.5, balance_text,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center')
        
        # Plot storage change in second subplot
        storage_components = {
            'Storage Change': storage_change,
            'Balance Error': balance_error
        }
        
        # Color scheme for storage change
        storage_colors = ['blue' if storage_change >= 0 else 'red', 'gray']
        
        # Create bars
        bars = ax2.bar(storage_components.keys(), storage_components.values(),
                    color=storage_colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            label_height = height if height >= 0 else height - (max(storage_components.values()) * 0.05)
            ax2.text(bar.get_x() + bar.get_width()/2, label_height,
                    f'{height:.2e}m³',
                    ha='center', va='bottom')
        
        # Customize second subplot
        ax2.set_title('Storage Change and Balance Error')
        ax2.set_ylabel('Volume [m³]')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        # Add summary statistics
        stats_text = (
            f"Summary Statistics:\n"
            f"Total Input: {sum(inflow_components.values()):.2e} m³\n"
            f"Total Output: {sum(outflow_components.values()):.2e} m³\n"
            f"Storage Change: {storage_change:.2e} m³\n"
            f"({(storage_change/initial_storage*100):.1f}% of initial storage)"
        )
        
        ax2.text(1.05, 0.5, stats_text,
                transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center')
        
        plt.tight_layout()
        return self._save_plot(f"water_balance_bars_{storage_node.id}")

    def plot_monthly_waterbalance(self, storage_node):
        """
        Create monthly bar plots showing water balance components for a storage node.
        
        Args:
            storage_node: StorageNode object to analyze
            
        Returns:
            str: Path to saved plot file
        """
        dt = self.system.dt
        num_months = len(storage_node.storage) - 1
        
        # Initialize monthly volumes
        monthly_volumes = {
            'Inflow': np.zeros(num_months),
            'Outflow': np.zeros(num_months),
            'Evaporation': np.zeros(num_months),
            'Spillway': np.zeros(num_months),
            'Storage Change': np.zeros(num_months)
        }
        
        # Calculate volumes for each month
        for t in range(num_months):
            # Inflow
            monthly_volumes['Inflow'][t] = sum(edge.get_edge_outflow(t) for edge in 
                                            storage_node.inflow_edges.values()) * dt
            
            # Outflow
            monthly_volumes['Outflow'][t] = sum(edge.get_edge_inflow(t) for edge in 
                                            storage_node.outflow_edges.values()) * dt
            
            # Evaporation
            monthly_volumes['Evaporation'][t] = storage_node.evaporation_losses[t]
            
            # Spillway
            monthly_volumes['Spillway'][t] = storage_node.spillway_register[t]
            
            # Storage Change
            monthly_volumes['Storage Change'][t] = storage_node.storage[t+1] - storage_node.storage[t]
        
        # Calculate balance error for each month
        monthly_balance_error = (monthly_volumes['Inflow'] - 
                            monthly_volumes['Outflow'] - 
                            monthly_volumes['Evaporation'] - 
                            monthly_volumes['Spillway'] - 
                            monthly_volumes['Storage Change'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
        
        # Width of bars
        bar_width = 0.35
        
        # Set x positions for bars
        months = range(num_months)
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:num_months]
        
        # Plot positive components (Inflow)
        ax1.bar(months, monthly_volumes['Inflow'], bar_width, 
                label='Inflow', color='blue', alpha=0.7)
        
        # Plot negative components stacked
        bottoms = np.zeros(num_months)
        components = ['Outflow', 'Evaporation', 'Spillway']
        colors = ['green', 'red', 'purple']
        
        for component, color in zip(components, colors):
            values = -monthly_volumes[component]  # Make negative for stacking below axis
            ax1.bar(months, values, bar_width, bottom=bottoms, 
                    label=component, color=color, alpha=0.7)
            bottoms += values
        
        # Plot storage change as a line
        ax1.plot(months, monthly_volumes['Storage Change'], 
                label='Storage Change', color='black', linewidth=2, marker='o')
        
        # Plot balance error in separate subplot
        ax2.bar(months, monthly_balance_error, bar_width, 
                label='Balance Error', color='gray', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Customize first subplot
        ax1.set_title(f'Monthly Water Balance Components for {storage_node.id}')
        ax1.set_ylabel('Volume [m³]')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_labels)
        
        # Customize second subplot
        ax2.set_title('Monthly Water Balance Error')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Error [m³]')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(months)
        ax2.set_xticklabels(month_labels)
        
        # Add summary statistics
        total_error = np.sum(monthly_balance_error)
        total_inflow = np.sum(monthly_volumes['Inflow'])
        error_percent = (total_error / total_inflow) * 100 if total_inflow != 0 else 0
        
        stats_text = (
            f"Water Balance Summary:\n"
            f"Total Inflow: {total_inflow:.2e} m³\n"
            f"Total Outflow: {-np.sum(monthly_volumes['Outflow']):.2e} m³\n"
            f"Total Evaporation: {-np.sum(monthly_volumes['Evaporation']):.2e} m³\n"
            f"Total Spillway: {-np.sum(monthly_volumes['Spillway']):.2e} m³\n"
            f"Net Storage Change: {np.sum(monthly_volumes['Storage Change']):.2e} m³\n"
            f"Total Balance Error: {total_error:.2e} m³\n"
            f"Error as % of Inflow: {error_percent:.2f}%"
        )
        
        ax2.text(1.05, 0.5, stats_text,
                transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center')
        
        plt.tight_layout()
        return self._save_plot(f"monthly_water_balance_{storage_node.id}")

    def plot_release_function(self, storage_node):
        """
        Create a plot showing the release function for a storage node with monthly variations.
        
        Args:
            storage_node: StorageNode object with release parameters
            months (list, optional): List of months (1-12) to plot. If None, plots all months.
                
        Returns:
            str: Path to the saved plot file
        """
        if not hasattr(storage_node, 'release_params'):
            print("Storage node does not have release parameters defined.")
            return None
            
        params = storage_node.release_params
        release_capacity = sum(edge.capacity for edge in storage_node.outflow_edges.values())
        
        # Set up the plot
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))
        
        min_level = storage_node.hva_data['min_waterlevel']
        max_level = storage_node.hva_data['max_waterlevel']
        levels = np.linspace(min_level, max_level, 200)
        
        # Plot release functions
        max_release = 0
        
        releases = []
        for h in levels:
            release = release_capacity
            if params['w']+ (h-params['h2'])* np.tan(params['m2']) < release_capacity:
                release = params['w'] + (h-params['h2'])* np.tan(params['m2'])
            if h <= params['h2']:
                release = params['w']
            if (h-params['h1'])* np.tan(params['m1']) < params['w']:
                release = (h-params['h1'])* np.tan(params['m1'])
            if h <= params['h1']:
                release = 0

            releases.append(release)
            max_release = max(max_release, release)
            
        # Plot main release function
        ax1.plot(levels, releases, linewidth=2)
        
        # Add vertical lines for h1 and h2
        ax1.axvline(x=params['h1'],linestyle='--', alpha=0.3)
        ax1.axvline(x=params['h2'],linestyle=':', alpha=0.3)
        
        # Add horizontal line for base release w
        ax1.axhline(y=params['w'], linestyle=':', alpha=0.3)
        ax1.axhline(y=release_capacity, linestyle='--', alpha=0.3)
            
        # Customize main plot
        ax1.set_xlabel('Water Level [m a.s.l.]')
        ax1.set_ylabel('Release Rate [m³/s]')
        ax1.set_title(f'Monthly Release Functions for {storage_node.id}')
        ax1.grid(True, alpha=0.3)        
        
        plt.tight_layout()
        return self._save_plot(f"release_function_{storage_node.id}")
    
    def plot_demand_satisfaction(self):
        """
        Create time series plots showing target, satisfied, and unmet demand for all demand nodes.
        
        Returns:
            str: Path to the saved plot file
        """
        # Find demand nodes in the system
        demand_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], DemandNode)]
        
        if not demand_nodes:
            print("No demand nodes found in the system.")
            return None
            
        # Create figure with subplots
        n_nodes = len(demand_nodes)
        fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 6*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Color scheme
        colors = {
            'target': '#2196F3',    # Blue
            'satisfied': '#4CAF50',  # Green
            'unmet': '#f44336'      # Red
        }
        
        time_steps = range(self.system.time_steps)
        
        # Plot for each demand node
        for idx, (node_id, node) in enumerate(demand_nodes):
            ax = axes[idx]
            
            # Get demand data
            target_demands = [node.get_demand_rate(t) for t in time_steps]
            satisfied_demands = node.satisfied_demand[:len(time_steps)]
            unmet_demands = [target - satisfied for target, satisfied in 
                            zip(target_demands, satisfied_demands)]
            
            # Plot demands
            ax.plot(time_steps, target_demands, color=colors['target'], 
                    label='Target Demand', linewidth=2)
            ax.plot(time_steps, satisfied_demands, color=colors['satisfied'], 
                    label='Satisfied Demand', linewidth=2)
            ax.fill_between(time_steps, satisfied_demands, target_demands, 
                        color=colors['unmet'], alpha=0.3, 
                        label='Unmet Demand')
            
            # Calculate statistics
            mean_target = np.mean(target_demands)
            mean_satisfied = np.mean(satisfied_demands)
            mean_unmet = np.mean(unmet_demands)
            satisfaction_rate = (mean_satisfied / mean_target * 100) if mean_target > 0 else 0
            
            stats_text = (
                f"Statistics:\n"
                f"Mean Target: {mean_target:.2f} m³/s\n"
                f"Mean Satisfied: {mean_satisfied:.2f} m³/s\n"
                f"Mean Unmet: {mean_unmet:.2f} m³/s\n"
                f"Satisfaction Rate: {satisfaction_rate:.1f}%"
            )
            
            # Add stats text box
            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            # Customize subplot
            ax.set_title(f'{node_id} Demand Satisfaction')
            ax.set_ylabel('Flow Rate [m³/s]')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center right')
            
            if idx == n_nodes - 1:  # Only add xlabel to bottom subplot
                ax.set_xlabel('Time Step')
        
        plt.tight_layout()
        return self._save_plot("demand_satisfaction")

    def plot_reservoir_volumes(self):
        """
        Create time series plots showing volume and water elevation for all reservoir (storage) nodes.
        
        Returns:
            str: Path to the saved plot file
        """
        # Find storage nodes in the system
        storage_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], StorageNode)]
        
        if not storage_nodes:
            print("No storage nodes (reservoirs) found in the system.")
            return None
            
        # Create figure with subplots - one per reservoir
        n_nodes = len(storage_nodes)
        fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 4*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Plot for each reservoir
        for idx, (node_id, node) in enumerate(storage_nodes):
            ax1 = axes[idx]
            ax2 = ax1.twinx()  # Create secondary y-axis for elevation
            
            # Get volume and elevation data
            # Note: storage and water_level arrays have one extra value for the next timestep
            # We'll use all but the last value to match with time steps
            volumes = node.storage[:-1]  # Exclude last storage value
            water_levels = node.water_level[:-1]  # Exclude last water level
            time_steps = range(len(volumes))  # Create matching time steps array
            capacity = node.capacity
            
            # Plot volume time series
            line1 = ax1.plot(time_steps, volumes, color='blue', 
                            label='Storage Volume', linewidth=2)
            
            # Add capacity line
            cap_line = ax1.axhline(y=capacity, color='red', linestyle='--', 
                                label=f'Capacity ({capacity:,.0f} m³)')
            
            # Plot water elevation
            line2 = ax2.plot(time_steps, water_levels, color='green', 
                            label='Water Level', linewidth=2)
            
            # Customize subplot
            ax1.set_title(f'{node_id}')
            ax1.set_ylabel('Volume [m³]')
            ax2.set_ylabel('Water Level [m.a.s.l]')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends from both axes
            lines = line1 + line2 + [cap_line]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        # Add common x-label to bottom subplot
        axes[-1].set_xlabel('Time Step')
        
        plt.tight_layout()
        return self._save_plot("reservoir_volumes")
    
    def plot_sink_outflows(self):
        #set fontsize for all labels to 16
        plt.rcParams.update({'font.size': 14})
        """
        Create time series plots showing system inflows (top) and sink outflows (bottom).
        
        Returns:
            str: Path to the saved plot file
        """
        # Find sink nodes and supply nodes in the system
        sink_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                    if isinstance(data['node'], SinkNode)]
        supply_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], SupplyNode)]
        
        if not sink_nodes:
            print("No sink nodes found in the system.")
            return None
            
        # Create figure with subplots - inflows at top, outflows at bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Color scheme using color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(sink_nodes), len(supply_nodes))))
        
        time_steps = range(self.system.time_steps)
        
        # Plot system inflows (top)
        total_inflows = np.zeros(len(time_steps))
        for idx, (node_id, node) in enumerate(supply_nodes):
            # Get supply rates for each timestep
            inflows = [node.get_supply_rate(t) for t in time_steps]
            total_inflows += np.array(inflows)
            
            # Plot inflow time series
            ax1.plot(time_steps, inflows, color=colors[idx], 
                    label=node_id, linewidth=2, marker='o', markersize=4)
        
        # Plot total system inflow
        ax1.plot(time_steps, total_inflows, color='black', linestyle='--',
                label='Total System Inflow', linewidth=2)
        # Plot a horizonal line at 25
        ax1.axhline(y=25, color='red', linestyle='--', label='Navoi-TPP Demand ')
        
        # Plot sink outflows (bottom)
        total_outflows = np.zeros(len(time_steps))
        for idx, (node_id, node) in enumerate(sink_nodes):
            # Calculate total inflow for each timestep
            flows = [sum(edge.get_edge_outflow(t) for edge in node.inflow_edges.values())
                    for t in time_steps]
            total_outflows += np.array(flows)
            
            # Plot flow time series
            ax2.plot(time_steps, flows, color=colors[idx], 
                    label=node_id, linewidth=2, marker='o', markersize=4)
        
        # Plot total system outflow
        ax2.plot(time_steps, total_outflows, color='black', linestyle='--',
                label='Total System Outflow', linewidth=2)
        
        # Customize inflow plot (top)
        ax1.set_title('System Inflows')
        ax1.set_ylabel('Flow Rate [m³/s]')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Customize outflow plot (bottom)
        ax2.set_title('Sink Node Outflows')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Flow Rate [m³/s]')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        return self._save_plot("sink_outflows_and_inflows")
    
    def plot_hydroworks_flows(self, hydroworks_id):
        """
        Create a plot showing inflows and outflows for a specific HydroWorks node.
        
        Args:
            hydroworks_id (str): ID of the HydroWorks node to visualize
            
        Returns:
            str: Path to the saved plot file
        """
        try:
            # Get the HydroWorks node
            node_data = self.system.graph.nodes[hydroworks_id]
            node = node_data['node']
            
            if not hasattr(node, 'distribution_params'):
                print(f"Node {hydroworks_id} is not a HydroWorks node")
                return None
            
            # Calculate time series for inflows and outflows
            inflows = []
            outflows = {}
            time_steps = range(self.system.time_steps)
            
            # Initialize outflow dictionaries for each target
            for target_id in node.outflow_edges.keys():
                outflows[target_id] = []
            
            # Calculate flows for each time step
            for t in time_steps:
                # Calculate total inflow
                total_inflow = sum(edge.get_edge_outflow(t) for edge in node.inflow_edges.values())
                inflows.append(total_inflow)
                
                # Calculate outflows to each target
                for target_id, edge in node.outflow_edges.items():
                    outflow = edge.get_edge_inflow(t)
                    outflows[target_id].append(outflow)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Plot 1: Time series of flows
            ax1.plot(time_steps, inflows, 'b-', label='Total Inflow', linewidth=2)
            
            # Plot outflows with different colors
            colors = plt.cm.tab20(np.linspace(0, 1, len(outflows)))
            for (target_id, flows), color in zip(outflows.items(), colors):
                ax1.plot(time_steps, flows, '-', color=color, label=f'Outflow to {target_id}', linewidth=2)
            
            # Calculate and display statistics
            mean_inflow = np.mean(inflows)
            stats_text = f"Mean Inflow: {mean_inflow:.2f} m³/s\n"
            total_mean_outflow = 0
            
            for target_id, flows in outflows.items():
                mean_flow = np.mean(flows)
                total_mean_outflow += mean_flow
                stats_text += f"Mean Outflow to {target_id}: {mean_flow:.2f} m³/s\n"
            
            balance_error = mean_inflow - total_mean_outflow
            stats_text += f"Mean Balance Error: {balance_error:.2f} m³/s"
            
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Flow Rate [m³/s]')
            ax1.set_title(f'Flow Rates for {hydroworks_id}')
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 2: Distribution parameters over time
            for target_id, params in node.distribution_params.items():
                ax2.plot(time_steps, params[:len(time_steps)], '-', 
                        label=f'Distribution to {target_id}', linewidth=2)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Distribution Parameter')
            ax2.set_title('Distribution Parameters Over Time')
            ax2.grid(True)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            return self._save_plot(f"hydroworks_flows_{hydroworks_id}")
            
        except Exception as e:
            print(f"Error plotting HydroWorks flows: {str(e)}")
            return None

    def plot_system_demands_vs_inflow(self):
        """
        Create a plot comparing total system inflow against total demands.
        
        Returns:
            str: Path to the saved plot file
        """
        try:
            time_steps = range(self.system.time_steps)
            
            # Calculate total inflow (from supply nodes)
            total_inflows = []
            supply_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], SupplyNode)]
            
            for t in time_steps:
                inflow = sum(node.get_supply_rate(t) for _, node in supply_nodes)
                total_inflows.append(inflow)
            
            # Calculate total demands
            total_demands = []
            demand_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], DemandNode)]
            
            for t in time_steps:
                demand = sum(node.get_demand_rate(t) for _, node in demand_nodes)
                total_demands.append(demand)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Plot 1: Time series
            ax1.plot(time_steps, total_inflows, 'b-', label='Total System Inflow', linewidth=2)
            ax1.plot(time_steps, total_demands, 'r-', label='Total System Demand', linewidth=2)
            
            # Fill the deficit area
            ax1.fill_between(time_steps, total_inflows, total_demands, 
                            where=np.array(total_demands) > np.array(total_inflows),
                            color='red', alpha=0.3, label='Deficit')
            
            # Fill the surplus area
            ax1.fill_between(time_steps, total_inflows, total_demands,
                            where=np.array(total_inflows) > np.array(total_demands),
                            color='green', alpha=0.3, label='Surplus')
            
            # Calculate and display statistics
            mean_inflow = np.mean(total_inflows)
            mean_demand = np.mean(total_demands)
            max_inflow = max(total_inflows)
            max_demand = max(total_demands)
            min_inflow = min(total_inflows)
            min_demand = min(total_demands)
            
            total_inflow_volume = sum(total_inflows) * self.system.dt
            total_demand_volume = sum(total_demands) * self.system.dt
            
            stats_text = (
                f"Statistics:\n"
                f"Mean Inflow: {mean_inflow:.2f} m³/s\n"
                f"Mean Demand: {mean_demand:.2f} m³/s\n"
                f"Max Inflow: {max_inflow:.2f} m³/s\n"
                f"Max Demand: {max_demand:.2f} m³/s\n"
                f"Min Inflow: {min_inflow:.2f} m³/s\n"
                f"Min Demand: {min_demand:.2f} m³/s\n"
                f"Total Inflow Volume: {total_inflow_volume:.2e} m³\n"
                f"Total Demand Volume: {total_demand_volume:.2e} m³\n"
                f"Volume Deficit: {(total_demand_volume - total_inflow_volume):.2e} m³"
            )
            
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Flow Rate [m³/s]')
            ax1.set_title('System Inflow vs Total Demands')
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 2: Demand composition for each time step
            bottom = np.zeros(len(time_steps))
            colors = plt.cm.tab20(np.linspace(0, 1, len(demand_nodes)))
            
            for (node_id, node), color in zip(demand_nodes, colors):
                demands = [node.get_demand_rate(t) for t in time_steps]
                ax2.bar(time_steps, demands, bottom=bottom, label=node_id, color=color)
                bottom += np.array(demands)
            
            # Add inflow line on top of stacked bars
            ax2.plot(time_steps, total_inflows, 'k-', label='Total Inflow', linewidth=2)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Flow Rate [m³/s]')
            ax2.set_title('Demand Composition and Total Inflow')
            ax2.grid(True)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            return self._save_plot("system_demands_vs_inflow")
            
        except Exception as e:
            print(f"Error plotting system demands vs inflow: {str(e)}")
            return None