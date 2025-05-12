"""
This module provides visualization tools for analyzing water system simulation results.
It includes time series visualizations for flows, storage levels, and demands.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd
import seaborn as sns
import numpy as np
import os
import json
import networkx as nx
from datetime import datetime
from .structure import SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
from dateutil.relativedelta import relativedelta

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
        self.image_dir = os.path.join('.', 'model_output/figures')
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
                        inflow = sum(edge.get_edge_flow_after_losses(time_step) for edge in node.inflow_edges.values())
                        outflow = sum(edge.get_edge_flow_before_losses(time_step) for edge in node.outflow_edges.values())

                        row_data[f"{node_id}_Inflow"] = inflow
                        row_data[f"{node_id}_Outflow"] = outflow

                        if isinstance(node, SupplyNode):
                            row_data[f"{node_id}_SupplyRate"] = node.supply_rates[time_step]
                        elif isinstance(node, StorageNode):
                            row_data[f"{node_id}_Storage"] = node.storage[time_step]
                            storage_change = (node.storage[time_step+1] - node.storage[time_step] 
                                            if time_step > 0 else node.storage[0])
                            row_data[f"{node_id}_StorageChange"] = storage_change
                            row_data[f"{node_id}_ExcessVolume"] = node.spillway_register[time_step]
                        elif isinstance(node, DemandNode):
                            row_data[f"{node_id}_Demand"] = node.demand_rates[time_step]
                            row_data[f"{node_id}_SatisfiedDemand"] = node.satisfied_consumptive_demand[time_step]
                            row_data[f"{node_id}_Deficit"] = (node.demand_rates[time_step] - 
                                                            node.satisfied_consumptive_demand[time_step])
                    except Exception as e:
                        print(f"Warning: Error processing node {node_id} at time step {time_step}: {str(e)}")

                # Populate edge data
                for u, v, d in self.system.graph.edges(data=True):
                    edge = d['edge']
                    try:
                        if time_step < len(edge.flow_before_losses):
                            inflow = edge.flow_before_losses[time_step]
                            outflow = edge.flow_after_losses[time_step]
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
                #df.to_csv(f'{self.name}_water_balance.csv', index=False)
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
            StorageNode: {'color': 'lightgreen', 'shape': 'v', 'name': 'Storage Node'},
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
            width =3 #+ (edge.capacity / max_capacity) * 10
            edge_widths.append(width)

        # Draw edges
        nx.draw_networkx_edges(self.system.graph, pos, 
                             edge_color='gray',
                             arrows=True, 
                             arrowsize=55, 
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
                labels[node] = f"{node}"#\nCap: {node_instance.capacity:,.0f} m³"
            elif isinstance(node_instance, HydroWorks):
                labels[node] = f"{node}"
        
        nx.draw_networkx_labels(self.system.graph, pos, labels, font_size=14, ax=ax)
        
        # Add edge labels showing capacity and length
        edge_labels = {}
        for u, v, d in self.system.graph.edges(data=True):
            edge = d['edge']
            edge_labels[(u, v)] = (f'{edge.capacity} m³/s')
      
        '''nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, 
                       font_size=14, ax=ax, rotate=False)'''
  

        # Create the legend on the top right of the plot
        ax.legend(handles=legend_elements,
              loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=22)
        
        plt.axis('off')
        plt.ylim(4360000, 4480000)
        plt.xlim(153000, 384000)
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
      
        nx.draw_networkx_edge_labels(self.system.graph, pos, edge_labels=edge_labels, font_size=14, ax=ax, rotate=False, bbox=dict(facecolor='white', alpha=0.5))

        # Create the legend on the top right of the plot
        ax.legend(handles=legend_elements,
              loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=22)
        
        plt.axis('off')
        plt.tight_layout()
        
        return self._save_plot("network_layout_2")

    def print_water_balance_summary(self):
        """
        Print a comprehensive summary of the water balance results.
        Includes volumes, relative contributions, storage changes and balance error statistics.
        """
        df = self.system.get_water_balance()
        
        # Add deficit column based on flow deficits from sink nodes
        sink_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                    if isinstance(data['node'], SinkNode) and hasattr(data['node'], 'min_flows')]
        
        # Create a list to store total deficits per timestep
        total_deficits = [0] * len(df) if len(df) > 0 else []
        min_flow_requirements = [0] * len(df) if len(df) > 0 else []
        
        # Sum up deficits from all sink nodes with min flow requirements
        for node_id, node in sink_nodes:
            if hasattr(node, 'flow_deficits') and len(node.flow_deficits) > 0:
                for t in range(min(len(node.flow_deficits), len(total_deficits))):
                    total_deficits[t] += node.flow_deficits[t]*self.system.dt
            if hasattr(node, 'min_flows') and len(node.min_flows) > 0:
                for t in range(min(len(node.min_flows), len(min_flow_requirements))):
                    min_flow_requirements[t] += node.min_flows[t]*self.system.dt
        
        # Add the deficit column to the dataframe
        if len(total_deficits) > 0:
            df['minflow deficit'] = total_deficits

        if len(min_flow_requirements) > 0:
            df['minflow requirement'] = min_flow_requirements
        
        wb = pd.DataFrame(df)
        directory = './model_output/water_balances'
        if not os.path.exists(directory):
            os.makedirs(directory)
        wb.to_csv(f'./model_output/water_balances/{self.name}_water_balance.csv', index=False)
        
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
        total_sourcenode = df['source'].sum()
        total_surfacerunoff = df['surfacerunoff'].sum()
        total_source = total_sourcenode + total_surfacerunoff
        print(f"Source Node: {total_sourcenode:,.0f} m³")
        print(f"Surface Runoff: {total_surfacerunoff:,.0f} m³")
        print(f"Total Source: {total_source:,.0f} m³ (100%)")
        
        # Component volumes and percentages
        print_section("Sink Volumes")
        components = {
            'Supplied Demand': 'supplied demand',
            'Sink Outflow': 'sink',
            'Edge Losses': 'edge losses',
            'Res Spills': 'reservoir spills',
            'Res ET Losses': 'reservoir ET losses',
            'HW Spills': 'hydroworks spills',
            'Flow Deficit': 'deficit'  # Add the new deficit component
        }
        
        for label, comp in components.items():
            if comp in df.columns:
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
        print(f"Total demand:        {total_demand:15,.0f} m³")
        print(f"Satisfied demand:    {total_satisfied:15,.0f} m³")
        print(f"Unmet demand:        {unmet_demand:15,.0f} m³")
        
        # Include flow deficit in demand section if available
        if 'deficit' in df.columns:
            total_deficit = df['deficit'].sum()
            print(f"Min flow deficit:        {total_deficit:15,.0f} m³")
        
        # Conservation check
        print_section("Conservation Check")
        total_in = total_source
        sink_components = ['supplied demand', 'sink', 'edge losses', 'reservoir spills', 
                          'reservoir ET losses', 'hydroworks spills']
        total_out = sum(df[comp].sum() for comp in sink_components if comp in df.columns)
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
            'HW Spills': 'hydroworks spills',
            'Deficit': 'deficit'  # Add deficit to maximum values
        }
        
        for label, comp in components.items():
            if comp in df.columns:
                max_value = df[comp].max()
                timestep = df.loc[df[comp].idxmax(), 'time_step']
                print(f"{label:10s}: {max_value:15,.0f} m³ at timestep {timestep:3.0f}")
        
        print("\n" + "=" * 50)

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
            inflows = [sum(edge.get_edge_flow_after_losses(t) for edge in node.inflow_edges.values())
                    for t in time_steps]
            
            # Calculate total outflow for each timestep
            outflows = [sum(edge.get_edge_flow_before_losses(t) for edge in node.outflow_edges.values())
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
        return self._save_plot("reservoir_dynamics")

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
            target_demands = [node.demand_rates[t] for t in time_steps]
            satisfied_demands = node.satisfied_demand_total[:len(time_steps)]
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
            inflows = [node.supply_rates[t] for t in time_steps]
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
            flows = [sum(edge.get_edge_flow_after_losses(t) for edge in node.inflow_edges.values())
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
        return self._save_plot("sink_outflows")
    
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
                total_inflow = sum(edge.get_edge_flow_after_losses(t) for edge in node.inflow_edges.values())
                inflows.append(total_inflow)
                
                # Calculate outflows to each target
                for target_id, edge in node.outflow_edges.items():
                    outflow = edge.get_edge_flow_before_losses(t)
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

    def plot_system_cons_demands_vs_inflow(self):
        """
        Create a plot comparing total system inflow against total demands and minimum flow requirements.
        
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
                inflow = sum(node.supply_rates[t] for _, node in supply_nodes)
                total_inflows.append(inflow)
            
            # Calculate total demands and minimum flows
            total_demands = []
            demand_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], DemandNode)]

            # Get sink nodes with minimum flow requirements
            sink_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], SinkNode)]
            
            for t in time_steps:
                demand = sum(node.demand_rates[t] for _, node in demand_nodes)
                non_consumptive_demand = sum(node.non_consumptive_rate for _, node in demand_nodes)
                total_demands.append(demand-non_consumptive_demand)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Calculate total minimum flow requirements
            total_min_flows = np.zeros(len(time_steps))
            for _, node in sink_nodes:
                if hasattr(node, 'min_flows'):
                    min_flows = [node.min_flows[t] for t in time_steps]
                    total_min_flows += np.array(min_flows)
            
            # Calculate total system requirements (demands + minimum flows)
            total_requirements = np.array(total_demands) + total_min_flows
            
            # Plot 1: Time series
            ax1.plot(time_steps, total_inflows, 'b-', label='Total System Inflow', linewidth=2)
            ax1.plot(time_steps, total_demands, 'purple',  label='Total System Consumptive Demand', linewidth=2, linestyle='--')
            ax1.plot(time_steps, total_requirements, 'r' , label='Total System Requirements\n(Demand + Min Flow)', 
                    linewidth=2)
            
            # Fill the deficit area
            ax1.fill_between(time_steps, total_inflows, total_requirements, 
                            where=total_requirements > np.array(total_inflows),
                            color='red', alpha=0.3, label='Deficit')
            
            # Fill the surplus area
            ax1.fill_between(time_steps, total_inflows, total_requirements,
                            where=np.array(total_inflows) > total_requirements,
                            color='green', alpha=0.3, label='Surplus')
            
            
            # Calculate additional statistics 
            total_inflow_volume = sum(total_inflows) * self.system.dt
            total_demand_volume = sum(total_demands) * self.system.dt
            total_min_flow_volume = sum(total_min_flows) * self.system.dt
            total_req_volume = total_demand_volume + total_min_flow_volume
            
            stats_text = (
                f"Statistics:\n"
                f"Total Inflow: {total_inflow_volume:.2e} m³\n"
                f"Total Consumptive Demand: {total_demand_volume:.2e} m³\n"
                f"Total Min Flow Requirement: {total_min_flow_volume:.2e} m³\n"
                f"Volume Deficit: {(total_req_volume - total_inflow_volume):.2e} m³"
            )
            
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8), 
                    fontsize=12)
            
            ax1.set_ylabel('Flow Rate [m³/s]', fontsize=12)
            ax1.set_title('System Inflow vs Total Consumptive Demands and Minimum Flow Requirements', fontsize=12)
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            # Plot 2: Demand composition and minimum flows for each time step
            bottom = np.zeros(len(time_steps))
            
            # Plot demand nodes
            colors = plt.cm.tab20(np.linspace(0, 1, len(demand_nodes)))
            for (node_id, node), color in zip(demand_nodes, colors):
                demands = [
                    node.demand_rates[t] - node.non_consumptive_rate
                    for t in time_steps
                ]
                ax2.bar(time_steps, demands, bottom=bottom, label=node_id, color=color)
                bottom += np.array(demands)
            
            # Add minimum flow requirements from sink nodes on top
            min_flow_bottom = bottom.copy()
            sink_colors = plt.cm.Purples(np.linspace(0.3, 0.7, len(sink_nodes)))
            
            for (node_id, node), color in zip(sink_nodes, sink_colors):
                if hasattr(node, 'min_flows'):
                    min_flows = [node.min_flows[t] for t in time_steps]
                    ax2.bar(time_steps, min_flows, bottom=min_flow_bottom, 
                        label=f'{node_id} Min Flow', color=color, alpha=0.7)
                    min_flow_bottom += np.array(min_flows)
            
            # Add inflow line on top of stacked bars
            ax2.plot(time_steps, total_inflows, 'k-', label='Total Inflow', linewidth=2)
            
            ax2.set_xlabel('Time Step', fontsize=12)
            ax2.set_ylabel('Flow Rate [m³/s]', fontsize=12)
            ax2.set_title('Consumptive Demand Composition, Minimum Flow Requirements, and Total Inflow', fontsize=12)
            ax2.grid(True)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            plt.tight_layout()
            return self._save_plot("system_consumptive_demands_vs_inflow")
            
        except Exception as e:
            print(f"Error plotting system demands vs inflow: {str(e)}")
            return None
    
    def plot_system_demands_vs_inflow_noevap(self):
        """
        Create a plot comparing total system inflow against total demands and minimum flow requirements.
        
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
                inflow = sum(node.supply_rates[t] for _, node in supply_nodes)
                total_inflows.append(inflow)
            
            # Calculate total demands and minimum flows
            total_demands = []
            demand_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], DemandNode)]

            # Get sink nodes with minimum flow requirements
            sink_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], SinkNode)]
            
            for t in time_steps:
                demand = sum(node.demand_rates[t] for _, node in demand_nodes)
                total_demands.append(demand)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Calculate total minimum flow requirements
            total_min_flows = np.zeros(len(time_steps))
            for _, node in sink_nodes:
                if hasattr(node, 'min_flows'):
                    min_flows = [node.min_flows[t] for t in time_steps]
                    total_min_flows += np.array(min_flows)
            
            # Calculate total system requirements (demands + minimum flows)
            total_requirements = np.array(total_demands) + total_min_flows
            
            # Plot 1: Time series
            ax1.plot(time_steps, total_inflows, 'b-', label='Total System Inflow', linewidth=2)
            ax1.plot(time_steps, total_demands, 'purple',  label='Total System Demand', linewidth=2, linestyle='--')
            ax1.plot(time_steps, total_requirements, 'r' , label='Total System Requirements\n(Demand + Min Flow)', 
                    linewidth=2)
            
            # Fill the deficit area
            ax1.fill_between(time_steps, total_inflows, total_requirements, 
                            where=total_requirements > np.array(total_inflows),
                            color='red', alpha=0.3, label='Deficit')
            
            # Fill the surplus area
            ax1.fill_between(time_steps, total_inflows, total_requirements,
                            where=np.array(total_inflows) > total_requirements,
                            color='green', alpha=0.3, label='Surplus')
            
            
            # Calculate additional statistics 
            total_inflow_volume = sum(total_inflows) * self.system.dt
            total_demand_volume = sum(total_demands) * self.system.dt
            total_min_flow_volume = sum(total_min_flows) * self.system.dt
            total_req_volume = total_demand_volume + total_min_flow_volume
            total_only_deficit = sum(max(0,(a - b)) * self.system.dt for a, b in zip(total_inflows, total_demands))
            total_only_surplus = sum(max(0,(b - a)) * self.system.dt for a, b in zip(total_inflows, total_demands))

            stats_text = (
                f"Statistics:\n"
                f"Total Inflow: {total_inflow_volume:.2e} m³\n"
                f"Total Demand: {total_demand_volume:.2e} m³\n"
                f"Total Min Flow Requirement: {total_min_flow_volume:.2e} m³\n"
                f"Only Deficit (red area): {total_only_deficit:.2e} m³\n"
                f"Only Surplus (green area): {total_only_surplus:.2e} m³\n"
                f"Volume Deficit: {(total_req_volume - total_inflow_volume):.2e} m³"
            )
            
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8), 
                    fontsize=12)
            
            ax1.set_ylabel('Flow Rate [m³/s]', fontsize=12)
            ax1.set_title('System Inflow vs Total Demands and Minimum Flow Requirements', fontsize=12)
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            # Plot 2: Demand composition and minimum flows for each time step
            bottom = np.zeros(len(time_steps))
            
            # Plot demand nodes
            colors = plt.cm.tab20(np.linspace(0, 1, len(demand_nodes)))
            for (node_id, node), color in zip(demand_nodes, colors):
                demands = [node.demand_rates[t] for t in time_steps]
                ax2.bar(time_steps, demands, bottom=bottom, label=node_id, color=color)
                bottom += np.array(demands)
            
            # Add minimum flow requirements from sink nodes on top
            min_flow_bottom = bottom.copy()
            sink_colors = plt.cm.Purples(np.linspace(0.3, 0.7, len(sink_nodes)))
            
            for (node_id, node), color in zip(sink_nodes, sink_colors):
                if hasattr(node, 'min_flows'):
                    min_flows = [node.min_flows[t] for t in time_steps]
                    ax2.bar(time_steps, min_flows, bottom=min_flow_bottom, 
                        label=f'{node_id} Min Flow', color=color, alpha=0.7)
                    min_flow_bottom += np.array(min_flows)
            
            # Add inflow line on top of stacked bars
            ax2.plot(time_steps, total_inflows, 'k-', label='Total Inflow', linewidth=2)
            
            ax2.set_xlabel('Time Step', fontsize=12)
            ax2.set_ylabel('Flow Rate [m³/s]', fontsize=12)
            ax2.set_title('Demand Composition, Minimum Flow Requirements, and Total Inflow', fontsize=12)
            ax2.grid(True)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            
            plt.tight_layout()
            return self._save_plot("system_demands_vs_inflow")
            
        except Exception as e:
            print(f"Error plotting system demands vs inflow: {str(e)}")
            return None
    
    def plot_system_demands_vs_inflow(self):
        """
        Create a plot comparing total system inflow against total demands, minimum flow requirements,
        and reservoir evaporation.
        
        Returns:
            str: Path to the saved plot file
        """
        try:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            time_steps = range(self.system.time_steps)
            
            # Calculate total inflow (from supply nodes)
            total_inflows = []
            supply_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], SupplyNode)]
            # Get the start year and month from the first demand node
            # (assuming all nodes have the same simulation period)
            start_year = None
            start_month = None
            
            # If we couldn't find start_year and start_month, try to get them from the system
            if start_year is None and hasattr(self.system, 'start_year') and hasattr(self.system, 'start_month'):
                start_year = self.system.start_year
                start_month = self.system.start_month
            
            # Create date labels for x-axis
            date_labels = []
            if start_year is not None and start_month is not None:
                for t in range(self.system.time_steps):
                    # Calculate year and month
                    month = start_month + t
                    year = start_year + (month - 1) // 12
                    month = ((month - 1) % 12) + 1  # Adjust month to be between 1-12
                    
                    # Format as year-month
                    date_labels.append(f"{month_names[month-1]}-{year}")
            else:
                # Fall back to timestep numbers if no date information is available
                date_labels = list(range(self.system.time_steps))
            
            for t in time_steps:
                inflow = sum(node.supply_rates[t] for _, node in supply_nodes)
                total_inflows.append(inflow)
            
            # Calculate total demands and minimum flows
            total_demands = []
            demand_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], DemandNode)]

            # Get sink nodes with minimum flow requirements
            sink_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], SinkNode)]
            
            # Get reservoir nodes for evaporation
            reservoir_nodes = [(node_id, data['node']) for node_id, data 
                            in self.system.graph.nodes(data=True) 
                            if isinstance(data['node'], StorageNode)]
            
            for t in time_steps:
                demand = sum(node.demand_rates[t] for _, node in demand_nodes)
                total_demands.append(demand)
            
            # Create figure with one subplot
            plt.rcParams.update({'font.size': 20})
            fig, ax1 = plt.subplots(figsize=(16, 8))
            
            # Calculate total minimum flow requirements
            total_min_flows = np.zeros(len(time_steps))
            for _, node in sink_nodes:
                if hasattr(node, 'min_flows'):
                    min_flows = [node.min_flows[t] for t in time_steps]
                    total_min_flows += np.array(min_flows)
            
            # Calculate total evaporation losses
            total_evap = np.zeros(len(time_steps))
            for _, node in reservoir_nodes:
                if hasattr(node, 'evaporation_losses'):
                    evap_rates = [loss/self.system.dt for loss in node.evaporation_losses[:len(time_steps)]]
                    total_evap += np.array(evap_rates)
            
            # Calculate total system requirements (demands + minimum flows + evaporation)
            total_requirements = np.array(total_demands) + total_min_flows + total_evap
            
            # Plot 1: Time series
            ax1.plot(time_steps, total_inflows, 'b-', label='Inflow', linewidth=2)
            ax1.plot(time_steps, total_demands, 'purple',  label='Demand', linewidth=2, linestyle='--')
            ax1.plot(time_steps, total_min_flows, 'darkorange', label='Min Flow Requirement', linewidth=2, linestyle='--')
            ax1.plot(time_steps, total_evap, 'orchid', label='Reservoir Evaporation', linewidth=2, linestyle='--')
            ax1.plot(time_steps, total_requirements, 'r', 
                    label='Demand + Min Flow Requirement \n + Reservoir Evaporation', linewidth=2)
            
            # Fill the deficit area
            ax1.fill_between(time_steps, total_inflows, total_requirements, 
                            where=total_requirements > np.array(total_inflows),
                            color='red', alpha=0.3, label='Water Deficit')
            
            # Fill the surplus area
            ax1.fill_between(time_steps, total_inflows, total_requirements,
                            where=np.array(total_inflows) > total_requirements,
                            color='green', alpha=0.3, label='Water Surplus')
            
            # Calculate additional statistics 
            total_inflow_volume = sum(total_inflows) * self.system.dt
            total_demand_volume = sum(total_demands) * self.system.dt
            total_min_flow_volume = sum(total_min_flows) * self.system.dt
            total_evap_volume = sum(total_evap) * self.system.dt
            total_req_volume = total_demand_volume + total_min_flow_volume + total_evap_volume
            total_only_deficit = sum(max(0,(b - a)) * self.system.dt for a, b in zip(total_inflows, total_requirements))
            total_only_surplus = sum(max(0,(a - b)) * self.system.dt for a, b in zip(total_inflows, total_requirements))

            stats_text = (
                f"Total Inflow: {total_inflow_volume/1e9:.1f} km³\n"
                f"Total Demand: {total_demand_volume/1e9:.1f} km³\n"
                f"Total Min Flow Requirement: {total_min_flow_volume/1e9:.1f} km³\n"
                f"Total Reservoir Evaporation: {total_evap_volume/1e9:.1f} km³\n\n"
                f"Deficit (red area): {total_only_deficit/1e9:.1f} km³\n"
                f"Surplus (green area): {total_only_surplus/1e9:.1f} km³\n"
                f"Deficit - Surplus: {(total_req_volume - total_inflow_volume)/1e9:.1f} km³"
            )
            
            ax1.text(1.07, 0.4, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8), 
                    fontsize=20)
            
            ax1.set_ylabel('Flow Rate [m³/s]', fontsize=22)
            #ax1.set_title('System Inflow vs Total Demands and Requirements', fontsize=12)
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
            # Set custom x-ticks with year-month labels
            tick_positions = list(time_steps)
            
            # If we have many time steps, use fewer ticks to avoid crowding
            if len(tick_positions) > 12:
                tick_spacing = max(1, len(tick_positions) // 12)
                tick_positions = tick_positions[::tick_spacing]
                tick_labels = [date_labels[i] for i in tick_positions]
            else:
                tick_labels = date_labels
            
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

            
            #plt.tight_layout()
            return self._save_plot("system_demands_vs_inflow")
            
        except Exception as e:
            print(f"Error plotting system demands vs inflow: {str(e)}")
            return None

    def plot_flow_compliance_heatmap(self):
        """
        Create an enhanced heatmap showing flow compliance patterns for all sink nodes
        with minimum flow requirements over time.
        
        Returns:
            tuple: Paths to the saved absolute and percentage deficit plot files
        """
        # Find sink nodes with minimum flow requirements
        sink_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                    if isinstance(data['node'], SinkNode) and hasattr(data['node'], 'min_flows')]
        
        sink_nodes_with_requirements = [(id, node) for id, node in sink_nodes 
                                    if any(flow > 0 for flow in node.min_flows)]
        
        if not sink_nodes_with_requirements:
            print("No sink nodes with minimum flow requirements found in the system.")
            return None
            
        time_steps = range(self.system.time_steps)
        
        # Prepare data for heatmaps
        absolute_deficits = {}
        percentage_deficits = {}
        
        for node_id, node in sink_nodes_with_requirements:
            deficits = node.flow_deficits[:len(time_steps)]
            min_flows = [node.min_flows[t] for t in time_steps]
            
            absolute_deficits[node_id] = deficits
            percentage_deficits[node_id] = [
                (d / m * 100 if m > 0 else 0) 
                for d, m in zip(deficits, min_flows)
            ]
        
        # Create DataFrame for plotting
        abs_df = pd.DataFrame(absolute_deficits)
        pct_df = pd.DataFrame(percentage_deficits)
        
        # Plot absolute deficits
        plt.figure(figsize=(16, 6))
        sns.heatmap(abs_df.T, cmap='YlOrRd', 
                    xticklabels=time_steps,
                    yticklabels=absolute_deficits.keys(),
                    annot=False,
                    cbar_kws={'label': 'Flow Deficit [m³/s]'})
        plt.xlabel('Time Step')
        plt.ylabel('Sink Node')
        plt.title('Absolute Flow Deficits Over Time')
        abs_filepath = self._save_plot("min_flow_requirement_heatmap_absolute")
        
        # Plot percentage deficits
        plt.figure(figsize=(16, 6))
        sns.heatmap(pct_df.T, cmap='YlOrRd', 
                    xticklabels=time_steps,
                    yticklabels=percentage_deficits.keys(),
                    annot=False,
                    vmin=0, vmax=100,
                    cbar_kws={'label': 'Deficit [%]'})
        plt.xlabel('Time Step')
        plt.ylabel('Sink Node')
        plt.title('Percentage Flow Deficits Over Time')
        pct_filepath = self._save_plot("min_flow_requirement_heatmap_percentage")
        
        return abs_filepath, pct_filepath

    def plot_spills(self):
        """
        Create a time series plot showing spills from all hydroworks and reservoir nodes.
        
        Returns:
            str: Path to the saved plot file
        """
        plt.rcParams.update({'font.size': 16})
        
        # Find hydroworks and reservoir nodes in the system
        hydroworks_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                            if isinstance(data['node'], HydroWorks)]
        reservoir_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], StorageNode)]
        
        if not hydroworks_nodes and not reservoir_nodes:
            print("No hydroworks or reservoir nodes found in the system.")
            return None
        
        # Create figure with two subplots - one for hydroworks spills, one for reservoir spills
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14))
        
        # Color scheme using color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(hydroworks_nodes), len(reservoir_nodes))))
        
        time_steps = range(self.system.time_steps)
        
        # Plot spills for each hydroworks node
        total_hydroworks_spills = np.zeros(len(time_steps))
        legend_elements_hydroworks = []
        
        for idx, (node_id, node) in enumerate(hydroworks_nodes):
            if hasattr(node, 'spill_register') and len(node.spill_register) > 0:
                # Convert spill volumes to flow rates
                spill_rates = [spill / self.system.dt for spill in node.spill_register[:len(time_steps)]]
                total_hydroworks_spills += spill_rates
                
                # Plot individual spills
                line = ax1.plot(time_steps, spill_rates, color=colors[idx], 
                        label=node_id, linewidth=2, marker='o', markersize=4)
                legend_elements_hydroworks.append(line[0])
                
                # Calculate statistics for this node
                total_spill_volume = sum(node.spill_register[:len(time_steps)])
                mean_spill_rate = np.mean(spill_rates)
                max_spill_rate = np.max(spill_rates)
                spill_frequency = sum(1 for s in spill_rates if s > 0) / len(time_steps) * 100
                
                print(f"\nSpill Statistics for {node_id}:")
                print(f"Total Spill Volume: {total_spill_volume:,.0f} m³")
                print(f"Mean Spill Rate: {mean_spill_rate:.2f} m³/s")
                print(f"Maximum Spill Rate: {max_spill_rate:.2f} m³/s")
                print(f"Spill Frequency: {spill_frequency:.1f}%")
        
        # Calculate and display total hydroworks system statistics
        total_hydroworks_volume = np.sum(total_hydroworks_spills) * self.system.dt
        mean_hydroworks_rate = np.mean(total_hydroworks_spills)
        max_hydroworks_rate = np.max(total_hydroworks_spills)
        
        stats_text_hydroworks = (
            f"Hydroworks System-wide Statistics:\n"
            f"Total Spill Volume: {total_hydroworks_volume:,.0f} m³\n"
            f"Mean Total Spill Rate: {mean_hydroworks_rate:.2f} m³/s\n"
            f"Maximum Total Spill Rate: {max_hydroworks_rate:.2f} m³/s"
        )
        
        # Add stats text box to hydroworks subplot
        ax1.text(0.02, 0.98, stats_text_hydroworks,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round',
                        facecolor='white',
                        alpha=0.8),
                fontsize=12)
        
        # Customize hydroworks subplot
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Spill Rate [m³/s]')
        ax1.set_title('Hydroworks Spill Rates Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend(handles=legend_elements_hydroworks, loc='upper right')
        
        # Plot spills for each reservoir node
        total_reservoir_spills = np.zeros(len(time_steps))
        legend_elements_reservoir = []
        
        for idx, (node_id, node) in enumerate(reservoir_nodes):
            if hasattr(node, 'spillway_register') and len(node.spillway_register) > 0:
                # Convert spill volumes to flow rates
                spill_rates = [spill / self.system.dt for spill in node.spillway_register[:len(time_steps)]]
                total_reservoir_spills += spill_rates
                
                # Plot individual spills
                line = ax2.plot(time_steps, spill_rates, color=colors[idx], 
                            label=node_id, linewidth=2, marker='o', markersize=4)
                legend_elements_reservoir.append(line[0])
                
                # Calculate statistics for this node
                total_spill_volume = sum(node.spillway_register[:len(time_steps)])
                mean_spill_rate = np.mean(spill_rates)
                max_spill_rate = np.max(spill_rates)
                spill_frequency = sum(1 for s in spill_rates if s > 0) / len(time_steps) * 100
                
                print(f"\nSpillway Statistics for {node_id}:")
                print(f"Total Spillway Volume: {total_spill_volume:,.0f} m³")
                print(f"Mean Spillway Rate: {mean_spill_rate:.2f} m³/s")
                print(f"Maximum Spillway Rate: {max_spill_rate:.2f} m³/s")
                print(f"Spillway Activation Frequency: {spill_frequency:.1f}%")
        
        # Calculate and display total reservoir system statistics
        total_reservoir_volume = np.sum(total_reservoir_spills) * self.system.dt
        mean_reservoir_rate = np.mean(total_reservoir_spills)
        max_reservoir_rate = np.max(total_reservoir_spills)
        
        stats_text_reservoir = (
            f"Reservoir System-wide Statistics:\n"
            f"Total Spillway Volume: {total_reservoir_volume:,.0f} m³\n"
            f"Mean Total Spillway Rate: {mean_reservoir_rate:.2f} m³/s\n"
            f"Maximum Total Spillway Rate: {max_reservoir_rate:.2f} m³/s"
        )
        
        # Add stats text box to reservoir subplot
        ax2.text(0.02, 0.98, stats_text_reservoir,
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round',
                        facecolor='white',
                        alpha=0.8),
                fontsize=12)
        
        # Customize reservoir subplot
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Spillway Rate [m³/s]')
        ax2.set_title('Reservoir Spillway Rates Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend(handles=legend_elements_reservoir, loc='upper right')
        
        plt.tight_layout()
        return self._save_plot("spills")

    def print_flow_compliance_summary(self):
        """
        Print a comprehensive summary of minimum flow compliance for all sink nodes.
        """
        # Find sink nodes with minimum flow requirements
        sink_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                    if isinstance(data['node'], SinkNode) and hasattr(data['node'], 'min_flows')]
        
        sink_nodes_with_requirements = [(id, node) for id, node in sink_nodes 
                                    if any(flow > 0 for flow in node.min_flows)]
        
        if not sink_nodes_with_requirements:
            print("No sink nodes with minimum flow requirements found in the system.")
            return
        
        print("\nMinimum Flow Compliance Summary")
        print("=" * 50)
        
        for node_id, node in sink_nodes_with_requirements:
            print(f"\nNode: {node_id}")
            print("-" * 20)
            
            actual_flows = node.flow_history
            min_flows = [node.min_flows[t] for t in range(len(actual_flows))]
            deficits = node.flow_deficits
            
            # Calculate statistics
            mean_actual = np.mean(actual_flows)
            mean_required = np.mean(min_flows)
            mean_deficit = np.mean(deficits)
            max_deficit = max(deficits)
            total_deficit_volume = node.get_total_deficit_volume(self.system.dt)
            
            # Calculate compliance metrics
            compliant_steps = sum(1 for a, m in zip(actual_flows, min_flows) if a >= m)
            compliance_rate = (compliant_steps / len(actual_flows) * 100)
            
            # Print statistics
            print(f"Mean Actual Flow: {mean_actual:.2f} m³/s")
            print(f"Mean Required Flow: {mean_required:.2f} m³/s")
            print(f"Mean Flow Deficit: {mean_deficit:.2f} m³/s")
            print(f"Maximum Flow Deficit: {max_deficit:.2f} m³/s")
            print(f"Total Deficit Volume: {total_deficit_volume:,.0f} m³")
            print(f"Compliance Rate: {compliance_rate:.1f}%")
            print(f"Compliant Time Steps: {compliant_steps}/{len(actual_flows)}")
            
        print("\n" + "=" * 50)

    def plot_network_overview(self):
        
        # Create a larger figure for detailed visualization
        fig, ax = plt.subplots(figsize=(20, 10.5))
        
        # Setting node positions based on easting and northing
        pos = {}
        for node, data in self.system.graph.nodes(data=True):
            node_instance = data['node']
            pos[node] = (node_instance.easting, node_instance.northing)
        
        # Calculate edge statistics for styling
        edge_capacities = {}
        total_capacity = 0
        max_capacity = 0
        edge_types = {}
        
        for u, v, edge_data in self.system.graph.edges(data=True):
            edge = edge_data['edge']
            capacity = edge.capacity
            edge_capacities[(u, v)] = capacity
            total_capacity += capacity
            max_capacity = max(max_capacity, capacity)
            
            # Categorize edge types
            source_type = type(edge.source).__name__
            target_type = type(edge.target).__name__
            edge_types[(u, v)] = f"{source_type}-to-{target_type}"
        
        # Node styling
        node_sizes = {
            SupplyNode: 1500,
            StorageNode: 2500,
            DemandNode: 1600,
            SinkNode: 2500,
            HydroWorks: 2000,
            RunoffNode: 500
        }
        
        node_colors = {
            SupplyNode: '#2196F3',  # Blue
            StorageNode: '#4CAF50',  # Green
            DemandNode: '#F44336',  # Red
            SinkNode: '#9E9E9E',    # Grey
            HydroWorks: '#FF9800',    # Orange
            RunoffNode: '#8B4513'  
        }
        
        node_shapes = {
            SupplyNode: 's',        # Square
            StorageNode: 'h',       # Hexagon
            DemandNode: 'o',        # Circle
            SinkNode: 'd',          # Diamond
            HydroWorks: 'p',         # Pentagon
            RunoffNode: 's'        # Square
        }

        node_names = {
            SupplyNode: 'Source Node',        # Square
            StorageNode: 'Storage Node',       # Hexagon
            DemandNode: 'Demand Node',        # Circle
            SinkNode: 'Sink Node',          # Diamond
            HydroWorks: 'Hydrowork Node',         # Pentagon
            RunoffNode: 'Surfacerunoff Node'        # Square
        }
        
        # Group nodes by type
        grouped_nodes = {
            node_type: [node for node, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], node_type)]
            for node_type in [SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode]
        }
        
        # Create edge color mapping based on capacity
        norm = Normalize(vmin=0, vmax=max_capacity)
        edge_cmap = LinearSegmentedColormap.from_list('edge_colors', ['#DCE3F5', '#2A57BF', '#0A1D56'])
        
        # Draw edges with width and color based on capacity
        for u, v, edge_data in self.system.graph.edges(data=True):
            edge = edge_data['edge']
            capacity = edge.capacity
            color = edge_cmap(norm(capacity))
            
            # Calculate width based on capacity (square root scaling for better visualization)
            width = 1 + 5 * np.sqrt(capacity / max_capacity)
            
            # Create the edge
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color=color, linewidth=width, alpha=0.7, zorder=1,
                solid_capstyle='round', path_effects=[PathEffects.withStroke(linewidth=width+1, foreground='white', alpha=0.2)])
        
        # Draw nodes by type
        for node_type, nodes in grouped_nodes.items():
            if not nodes:
                continue
                
            nx.draw_networkx_nodes(
                self.system.graph, pos,
                nodelist=nodes,
                node_size=node_sizes[node_type],
                node_color=node_colors[node_type],
                node_shape=node_shapes[node_type],
                edgecolors='white',
                linewidths=1.5,
                alpha=0.9,
                ax=ax
            )
        
        # Add capacity-based edge labels for edges with significant capacity
        capacity_threshold = 0  # Show labels for edges with >10% of max capacity
        edge_labels = {(u, v): f"{edge_data['edge'].capacity:.0f}"
                    for u, v, edge_data in self.system.graph.edges(data=True)
                    if edge_data['edge'].capacity > capacity_threshold}
        
        '''nx.draw_networkx_edge_labels(
            self.system.graph, pos,
            edge_labels=edge_labels,
            font_size=18,
            font_color='black',
            font_weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            ax=ax
        )'''
        
        # Create node labels with key information
        node_labels = {}
        for node_id, data in self.system.graph.nodes(data=True):
            node = data['node']
            label = node_id
            
            
            node_labels[node_id] = label
        
        # Draw node labels with white outline for better visibility
        for node, label in node_labels.items():
            x, y = pos[node]
            txt = ax.text(x, y, label, fontsize=15, ha='center', va='center', 
                        fontweight='bold', zorder=5)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])
        
        # Create a legend
        legend_elements = []
        
        # Node type legend
        for node_type in [SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode]:
            if grouped_nodes[node_type]:  # Only add to legend if this type exists
                legend_elements.append(
                    Line2D([0], [0], marker=node_shapes[node_type], color='w', 
                        markerfacecolor=node_colors[node_type], markersize=15, 
                        label=node_names[node_type])
                )
        
        # Single edge capacity legend element
        legend_elements.append(
            Line2D([0], [0], color=edge_cmap(0.5), linewidth=3,
            label=f'Edge')
        )
        
        # Add the legend
        ax.legend(handles=legend_elements, loc='upper right', fontsize=22,
                framealpha=0.9, fancybox=True, shadow=True)
    

        # Find geographic bounding box
        eastings = [p[0] for p in pos.values()]
        northings = [p[1] for p in pos.values()]
        min_easting, max_easting = min(eastings), max(eastings)
        min_northing, max_northing = min(northings), max(northings)
        
        # Add a 5% padding
        easting_padding = (max_easting - min_easting) * 0.02
        northing_padding = (max_northing - min_northing) * 0.08
        
        # Set axis limits with padding
        ax.set_xlim(min_easting - easting_padding, max_easting + easting_padding)
        ax.set_ylim(min_northing - northing_padding, max_northing + northing_padding)
        
        # Add title
        #plt.title(f"{self.name} Water System Model Overview", fontsize=18, pad=20)
        
        # Turn off axis
        plt.axis('off')
        
        # Optional: Add a scale bar (if geographic coordinates)
        '''scale_bar_length = 25000  # 10% of the x-span
        
        # Add scale bar at the bottom of the plot
        scale_bar_x = min_easting + easting_padding
        scale_bar_y = min_northing + northing_padding
        ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], 
                [scale_bar_y, scale_bar_y], 'k-', lw=3)
        ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y + northing_padding*0.5, 
                f"{scale_bar_length/1000:.0f} km", ha='center', fontweight='bold', fontsize=22)'''
        
        plt.tight_layout()
        
        return self._save_plot("network_overview")   

    def plot_minimum_flow_compliance(self):
        """
        Create time series plots showing actual flows versus minimum flow requirements 
        for all sink nodes that have minimum flow requirements.
        
        Returns:
            str: Path to the saved plot file
        """
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Find sink nodes in the system that have minimum flow requirements
        sink_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                    if isinstance(data['node'], SinkNode) and hasattr(data['node'], 'min_flows')]
        
        sink_nodes_with_requirements = [(id, node) for id, node in sink_nodes 
                                    if any(flow > 0 for flow in node.min_flows)]
        
        if not sink_nodes_with_requirements:
            print("No sink nodes with minimum flow requirements found in the system.")
            return None

        plt.rcParams.update({'font.size': 22})    
        # Create figure with subplots
        n_nodes = len(sink_nodes_with_requirements)
        fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 6*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Color scheme
        colors = {
            'actual': '#2196F3',    # Blue for actual flow
            'minimum': '#FF9800',    # Orange for minimum requirement
            'deficit': '#f44336'     # Red for deficit
        }
        time_steps = range(self.system.time_steps)
        
        # Create date labels for x-axis
        date_labels = []
        if hasattr(self.system, 'start_year') and hasattr(self.system, 'start_month'):
            start_year = self.system.start_year
            start_month = self.system.start_month
            
            for t in range(self.system.time_steps):
                # Calculate year and month
                month = start_month + t
                year = start_year + (month - 1) // 12
                month = ((month - 1) % 12) + 1  # Adjust month to be between 1-12
                
                # Format as year-month
                date_labels.append(f"{month_names[month-1]}-{year}")
        else:
            # Fall back to timestep numbers if no date information is available
            date_labels = list(time_steps)
        
        # Plot for each sink node
        for idx, (node_id, node) in enumerate(sink_nodes_with_requirements):
            ax = axes[idx]
            
            # Get flow data
            min_flows = [node.min_flows[t] for t in time_steps]
            actual_flows = node.flow_history[:len(time_steps)]
            deficits = node.flow_deficits[:len(time_steps)]
            
            # Plot flows
            ax.plot(time_steps, actual_flows, color=colors['actual'], 
                    label='Modelled Flow', linewidth=2)
            ax.plot(time_steps, min_flows, color=colors['minimum'], 
                    label='Min Requirement', linewidth=2)
            
            # Fill deficit areas
            ax.fill_between(time_steps, actual_flows, min_flows, 
                        where=np.array(actual_flows) < np.array(min_flows),
                        color=colors['deficit'], alpha=0.3, 
                        label='Flow Deficit')
            
            # Calculate statistics
            mean_actual = np.mean(actual_flows)
            mean_required = np.mean(min_flows)
            mean_deficit = np.mean(deficits)
            total_deficit_volume = sum(deficits) * self.system.dt
            compliance_rate = (sum(1 for a, m in zip(actual_flows, min_flows) if a >= m) 
                            / len(time_steps) * 100)
            
            stats_text = (
                #f"Statistics:\n"
                #f"Mean Actual Flow: {mean_actual:.2f} m³/s\n"
                #f"Mean Required Flow: {mean_required:.2f} m³/s\n"
                #f"Mean Deficit: {mean_deficit:.2f} m³/s\n"
                f"Total Deficit Volume: {total_deficit_volume:,.0f} m³"
                #f"Compliance Rate: {compliance_rate:.1f}%"
            )
            
            # Add stats text box
            ax.text(0.02, 0.97, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.8))
            
            # Customize subplot
            ax.set_title(f'{node_id} Minimum Flow Requirement')
            ax.set_ylabel('Flow Rate [m³/s]', fontsize=22)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Set custom x-ticks with year-month labels
            tick_positions = list(time_steps)
            
            # If we have many time steps, use fewer ticks to avoid crowding
            if len(tick_positions) > 12:
                tick_spacing = max(1, len(tick_positions) // 12)
                tick_positions = tick_positions[::tick_spacing]
                tick_labels = [date_labels[i] for i in tick_positions]
            else:
                tick_labels = date_labels
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        
        plt.tight_layout()
        return self._save_plot("minimum_flow_compliance")
    
    def plot_demand_deficit_heatmap(self):
        """
        Create enhanced heatmaps showing total demand deficits (both consumptive and non-consumptive)
        across all demand nodes over time.
        """
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Collect total deficits for each demand node
        demand_nodes = [(node_id, data['node']) for node_id, data 
                        in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], DemandNode)]
        
        if not demand_nodes:
            print("No demand nodes found in the system.")
            return
        
        # Create DataFrames for different deficit types
        total_data = pd.DataFrame()
        percentage_data = pd.DataFrame()
        
        # Get the start year and month from the first demand node
        # (assuming all nodes have the same simulation period)
        start_year = None
        start_month = None
        
        # If we couldn't find start_year and start_month, try to get them from the system
        if start_year is None and hasattr(self.system, 'start_year') and hasattr(self.system, 'start_month'):
            start_year = self.system.start_year
            start_month = self.system.start_month
        
        # Create date labels for x-axis
        date_labels = []
        if start_year is not None and start_month is not None:
            for t in range(self.system.time_steps):
                # Calculate year and month
                month = start_month + t
                year = start_year + (month - 1) // 12
                month = ((month - 1) % 12) + 1  # Adjust month to be between 1-12
                
                # Format as year-month
                date_labels.append(f"{month_names[month-1]}-{year}")
        else:
            # Fall back to timestep numbers if no date information is available
            date_labels = list(range(self.system.time_steps))
        
        # Calculate deficits for each node and time step
        for node_id, node in demand_nodes:
            node_total_deficits = []
            node_percentages = []
            
            for t in range(self.system.time_steps):
                total_deficit = (node.demand_rates[t] - node.satisfied_demand_total[t])
                node_total_deficits.append(total_deficit)
                
                # Calculate percentage based on total required flow
                total_required = node.demand_rates[t]
                if total_required > 0:
                    percentage = (total_deficit / total_required) * 100
                else:
                    percentage = 0
                node_percentages.append(percentage)
            
            total_data[node_id] = node_total_deficits
            percentage_data[node_id] = node_percentages
        
        # Plot absolute deficits
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(16, 6))
        sns.heatmap(total_data.T, cmap='YlOrRd', 
                xticklabels=date_labels,
                yticklabels=total_data.columns,
                annot=False,
                cbar_kws={'label': 'Deficit [m³/s]'})
        #plt.xlabel('Period', fontsize=16)
        plt.ylabel('Demand Node', fontsize=18)
        #plt.title('Absolute Water Deficits', fontsize=18)
        plt.xticks(plt.xticks()[0][::3], plt.xticks()[1][::3], fontsize=17, rotation=90)
        plt.yticks(fontsize=17)
        abs_filepath = self._save_plot("deficit_heatmap_absolute")
        
        # Plot percentage deficit
        plt.figure(figsize=(16, 6))
        sns.heatmap(percentage_data.T, cmap='YlOrRd',
                    xticklabels=date_labels,
                    yticklabels=percentage_data.columns,
                    annot=False,
                    cbar_kws={'label': 'Deficit [%]'})
        #plt.xlabel('Period', fontsize=16)
        plt.ylabel('Demand Node', fontsize=16)
        #plt.title('Percentage of Total Unmet Demand', fontsize=18)
        plt.xticks(plt.xticks()[0][::3], plt.xticks()[1][::3], fontsize=14, rotation=90)
        plt.yticks(fontsize=14)
        pct_filepath = self._save_plot("deficit_heatmap_percentage")
        
        return abs_filepath, pct_filepath

    def plot_reservoir_volumes(self):
        """
        Create time series plots showing volume and water elevation for all reservoir (storage) nodes.
        
        Returns:
            str: Path to the saved plot file
        """
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Find storage nodes in the system
        storage_nodes = [(node_id, data['node']) for node_id, data in self.system.graph.nodes(data=True) 
                        if isinstance(data['node'], StorageNode)]
        
        if not storage_nodes:
            print("No storage nodes (reservoirs) found in the system.")
            return None
        
        # Get the start year and month from the first demand node
        # (assuming all nodes have the same simulation period)
        start_year = None
        start_month = None
        
        # If we couldn't find start_year and start_month, try to get them from the system
        if start_year is None and hasattr(self.system, 'start_year') and hasattr(self.system, 'start_month'):
            start_year = self.system.start_year
            start_month = self.system.start_month
        
        # Create date labels for x-axis
        date_labels = []
        if start_year is not None and start_month is not None:
            for t in range(self.system.time_steps):
                # Calculate year and month
                month = start_month + t
                year = start_year + (month - 1) // 12
                month = ((month - 1) % 12) + 1  # Adjust month to be between 1-12
                
                # Format as year-month
                date_labels.append(f"{month_names[month-1]}-{year}")
            
        # Create figure with subplots - one per reservoir
        n_nodes = len(storage_nodes)
        plt.rcParams.update({'font.size': 22})
        fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 6*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Plot for each reservoir
        for idx, (node_id, node) in enumerate(storage_nodes):
            ax1 = axes[idx]
            
            # Get volume and elevation data
            # Note: storage and water_level arrays have one extra value for the next timestep
            # We'll use all but the last value to match with time steps
            volumes = np.array(node.storage[1:])/1e6 if node.storage else np.array([])  # Exclude first volume
            water_levels = node.water_level[1:]  # Exclude first level
            time_steps = range(len(volumes))  # Create matching time steps array
            capacity = node.capacity
            
            # Plot volume time series
            line1 = ax1.plot(time_steps, volumes, color='blue', 
                            label='Storage Volume', linewidth=2)
            
            # Add capacity line
            cap_line = ax1.axhline(y=capacity/1e6, color='red', linestyle='--', 
                                label=f'Maximum Storage Capacity')
            
            
            # Customize subplot
            ax1.set_title(f'{node_id}')
            ax1.set_ylabel(r'Volume [10$^6$ m³]')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends from both axes
            lines = line1 + [cap_line]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')

            tick_positions = list(time_steps)
            
            # If we have many time steps, use fewer ticks to avoid crowding
            if len(tick_positions) > 12:
                tick_spacing = max(1, len(tick_positions) // 12)
                tick_positions = tick_positions[::tick_spacing]
                tick_labels = [date_labels[i] for i in tick_positions]
            else:
                tick_labels = date_labels
            
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        return self._save_plot("reservoir_volumes")

    def plot_objective_function_breakdown(self):

        # Initialize dictionaries to store penalty components
        demand_deficits = {}
        sink_deficits = {}
        hw_spills = {}
        res_spills = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Create date labels for x-axis
        date_labels = []
        if hasattr(self.system, 'start_year') and hasattr(self.system, 'start_month'):
            start_year = self.system.start_year
            start_month = self.system.start_month
            
            for t in range(self.system.time_steps):
                # Calculate year and month
                month = start_month + t
                year = start_year + (month - 1) // 12
                month = ((month - 1) % 12) + 1  # Adjust month to be between 1-12
                
                # Format as year-month
                date_labels.append(f"{month_names[month-1]}-{year}")
        else:
            # Fall back to timestep numbers if no date information is available
            date_labels = list(time_steps)
        
        # Collect demand deficit penalties
        for node_id, node_data in self.system.graph.nodes(data=True):
            node = node_data['node']
            
            # Calculate demand deficit penalties
            if isinstance(node, DemandNode):
                demand = np.array([node.demand_rates[t] for t in range(self.system.time_steps)])
                satisfied = np.array(node.satisfied_demand_total)
                deficit = (demand - satisfied) * self.system.dt
                weighted_deficit = deficit * node.weight
                demand_deficits[node_id] = weighted_deficit
            
            # Calculate sink node minimum flow penalties
            elif isinstance(node, SinkNode):
                if hasattr(node, 'flow_deficits') and len(node.flow_deficits) > 0:
                    deficits = np.array(node.flow_deficits) * self.system.dt
                    weighted_deficit = deficits * node.weight
                    sink_deficits[node_id] = weighted_deficit
            
            # Calculate hydroworks spill penalties
            elif isinstance(node, HydroWorks) and hasattr(node, 'spill_register'):
                if len(node.spill_register) > 0:
                    spills = np.array(node.spill_register)
                    # Apply a penalty weight of 100.0 to match the optimizer's objective function
                    weighted_spills = spills * 100.0
                    hw_spills[node_id] = weighted_spills
            
            # Calculate reservoir spillway penalties
            elif isinstance(node, StorageNode) and hasattr(node, 'spillway_register'):
                if len(node.spillway_register) > 0:
                    spills = np.array(node.spillway_register)
                    # Apply a penalty weight of 100.0 to match the optimizer's objective function
                    weighted_spills = spills * 100.0
                    res_spills[node_id] = weighted_spills
        
        # Create DataFrames for easy plotting
        demand_df = pd.DataFrame(demand_deficits)/1000000000
        sink_df = pd.DataFrame(sink_deficits)/1000000000
        hw_spill_df = pd.DataFrame(hw_spills)/1000000000
        res_spill_df = pd.DataFrame(res_spills)/1000000000
        
        # Calculate totals for each category
        if not demand_df.empty:
            demand_total = demand_df.sum(axis=1)
            total_demand_penalty = demand_total.sum()
        else:
            demand_total = pd.Series(np.zeros(self.system.time_steps))
            total_demand_penalty = 0
            
        if not sink_df.empty:
            sink_total = sink_df.sum(axis=1)
            total_sink_penalty = sink_total.sum()
        else:
            sink_total = pd.Series(np.zeros(self.system.time_steps))
            total_sink_penalty = 0
            
        if not hw_spill_df.empty:
            hw_spill_total = hw_spill_df.sum(axis=1)
            total_hw_spill_penalty = hw_spill_total.sum()
        else:
            hw_spill_total = pd.Series(np.zeros(self.system.time_steps))
            total_hw_spill_penalty = 0
            
        if not res_spill_df.empty:
            res_spill_total = res_spill_df.sum(axis=1)
            total_res_spill_penalty = res_spill_total.sum()
        else:
            res_spill_total = pd.Series(np.zeros(self.system.time_steps))
            total_res_spill_penalty = 0
        
        # Calculate total penalty and percentages
        total_penalty = total_demand_penalty + total_sink_penalty + total_hw_spill_penalty + total_res_spill_penalty
        demand_pct = (total_demand_penalty / total_penalty * 100) if total_penalty > 0 else 0
        sink_pct = (total_sink_penalty / total_penalty * 100) if total_penalty > 0 else 0
        hw_spill_pct = (total_hw_spill_penalty / total_penalty * 100) if total_penalty > 0 else 0
        res_spill_pct = (total_res_spill_penalty / total_penalty * 100) if total_penalty > 0 else 0
        
        # Set up the figure and gridspec
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])
        
        # Plot 1: Stacked bar chart of penalties by time step (large, at the top)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Create stacked bar chart
        bar_width = 0.4
        time_steps = range(self.system.time_steps)
        
        # Convert to numpy arrays for stacking
        demand_values = np.array(demand_total)
        sink_values = np.array(sink_total)
        hw_spill_values = np.array(hw_spill_total)
        res_spill_values = np.array(res_spill_total)
        
        # Create stacked bars
        ax1.bar(np.array(time_steps) - bar_width/2, np.maximum(demand_values, 0), bar_width, label=f'Demand Deficit', color='green')
        ax1.bar(np.array(time_steps) + bar_width/2, np.maximum(sink_values, 0), bar_width, label=f'Min Flow Deficit', color='blue')
      
        # Calculate bottom positions for hydroworks spills
        bottom_hw = demand_values + sink_values
        #ax1.bar(time_steps, hw_spill_values, bar_width, bottom=bottom_hw, label=f'Hydroworks Spills', color='red')
        
        # Calculate bottom positions for reservoir spills
        bottom_res = bottom_hw + hw_spill_values
        #ax1.bar(time_steps, res_spill_values, bar_width, bottom=bottom_res, label=f'Reservoir Spills', color='purple')
        
        # Add total penalty as a line on top
        total_by_timestep = demand_values + sink_values + hw_spill_values + res_spill_values
        #ax1.plot(time_steps, total_by_timestep, 'k-', label='Total Penalty', linewidth=3, marker='o', markersize=8)
        
        ax1.set_title(f'Composition of the Objective Function (OF) value: {self.name}', fontsize=24)
        ax1.set_ylabel('Contribution to OF [km³]', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(fontsize=20, loc='upper right')
        # Set custom x-ticks with year-month labels
        tick_positions = list(time_steps)
            
        # If we have many time steps, use fewer ticks to avoid crowding
        if len(tick_positions) > 12:
            tick_spacing = max(1, len(tick_positions) // 12)
            tick_positions = tick_positions[::tick_spacing]
            tick_labels = [date_labels[i] for i in tick_positions]
        else:
            tick_labels = date_labels
        
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Plot 2: Breakdown of total penalties by type (pie chart)
        ax2 = fig.add_subplot(gs[1, 0])
        components = ['Demand Deficit',  'Minimum Flow\n Requirement Deficit'] #,  'Reservoir Spills'] 'Hydroworks Spills',
        values = [total_demand_penalty,  total_sink_penalty]#,  total_res_spill_penalty] total_hw_spill_penalty,
        colors = ['green',  'blue'] #, 'yellow'] 'red',
        
        # Create pie chart
        wedges, texts, autotexts = ax2.pie(
            values, 
            labels=components,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}, 
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(18)
            #autotext.set_weight('bold')
            autotext.set_backgroundcolor('white')
        
        for text in texts:
            text.set_fontsize(18)
        
        ax2.set_title('Contribution to OF', fontsize=18)
        
        # Plot 3: Detailed breakdown of demand deficit penalties by node
        ax3 = fig.add_subplot(gs[1, 1])
        
        if not demand_df.empty and len(demand_df.columns) > 0:
            # Calculate total penalty by node
            demand_by_node = demand_df.sum().sort_values(ascending=False)
            top_nodes = demand_by_node.head(10)  # Show top 10 nodes
            
            bars = ax3.barh(list(top_nodes.index), top_nodes.values, color='green')
            ax3.set_title('Demand Deficit', fontsize=20)
            ax3.set_xlabel('Contribution to OF [km³]', fontsize=20)
            ax3.tick_params(axis='both', which='major', labelsize=18)
            ax3.grid(True, alpha=0.3, axis='x')
            
        else:
            ax3.text(0.5, 0.5, 'No demand deficit data available', 
                    ha='center', va='center', fontsize=14)
            ax3.set_title('Demand Deficit Penalties by Node', fontsize=18)
        
        # Plot 4: Detailed breakdown of min flow deficit penalties by node
        ax4 = fig.add_subplot(gs[2, 1])
        
        if not sink_df.empty and len(sink_df.columns) > 0:
            # Calculate total penalty by node
            sink_by_node = sink_df.sum().sort_values(ascending=False)
            
            bars = ax4.barh(list(sink_by_node.index), sink_by_node.values, color='blue')
            ax4.set_title('Minimum Flow Requirement Deficit', fontsize=20)
            ax4.set_xlabel('Contribution to OF [km³]', fontsize=20)
            ax4.tick_params(axis='both', which='major', labelsize=18)
            ax4.grid(True, alpha=0.3, axis='x')
            
        else:
            ax4.text(0.5, 0.5, 'No min flow deficit data available', 
                    ha='center', va='center', fontsize=14)
            ax4.set_title('Min Flow Deficit Penalties by Node', fontsize=18)
        
        # Add a summary text box with total objective function value
        summary_text = (
            f"Objective Function (OF) Summary:\n"
            f"Total OF value: {total_penalty:,.2f} km³\n"
            f"Normalized annual OF value:{total_penalty / (len(time_steps)/12):,.2f} km³/a\n\n"
            f"Components:\n"
            f"- Demand Deficit: {total_demand_penalty / 6:,.2f} km³/a\n"
            f"- Min Flow Deficit: {total_sink_penalty / 6:,.2f} km³/a\n"
            f"- Hydroworks Spills: {total_hw_spill_penalty / 6:,.2f} km³/a\n"
            f"- Reservoir Spills: {total_res_spill_penalty / 6:,.2f} km³/a\n"
        )
        
        fig.text(
            0.07, 0.07, 
            summary_text,
            fontsize=20,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        return self._save_plot("objective_function_breakdown")
    
    def create_interactive_network_visualization(self):
        """Creates an offline interactive network visualization using Plotly."""

        def create_frame(timestep):
            node_traces = []
            edge_traces = []
            
            # Create edge traces
            for edge in edges:
                source = next(n for n in nodes if n['id'] == edge['source'])
                target = next(n for n in nodes if n['id'] == edge['target'])
                flow = edge['flows'][timestep] if timestep < len(edge['flows']) else 0
                flow_pct = flow / edge['capacity'] if edge['capacity'] > 0 else 0
                
                color ='#BFDBFE' #if flow_pct < 0.5 else '#2563EB' if flow_pct < 0.8 else '#1E3A8A' if flow_pct < 0.99 else '#8B0000'
                width = 5 + (flow / max_flow * 80) if flow > 0 else 1
                
                edge_trace = go.Scatter(
                    x=[source['easting'], target['easting']],
                    y=[source['northing'], target['northing']],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    showlegend=False
                )
                
                text_trace = go.Scatter(
                    x=[(source['easting'] + target['easting']) / 2],
                    y=[(source['northing'] + target['northing']) / 2],
                    mode='markers',
                    marker=dict(opacity=0),
                    text=[f'{flow:.1f} m³/s'],
                    hoverinfo='text',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
                edge_traces.append(text_trace)
                edge_traces.append(edge_trace)

            # Create node traces by shape
            for shape in ['square', 'hexagon', 'circle', 'diamond', 'pentagon']:
                nodes_of_shape = [n for n in nodes if n['shape'] == shape]
                if nodes_of_shape:
                    node_trace = go.Scatter(
                        x=[n['easting'] for n in nodes_of_shape],
                        y=[n['northing'] for n in nodes_of_shape],
                        mode='markers',
                        marker=dict(
                            symbol=shape,
                            size=20,
                            color=[n['color'] for n in nodes_of_shape],
                            line=dict(width=1, color='black')
                        ),
                        text=[n['id'] for n in nodes_of_shape],
                        hoverinfo='text',
                        name=shape,
                        showlegend=False
                    )
                    node_traces.append(node_trace)
            
            # Add legend traces (only in the first frame)
            if timestep == 0:
                legend_items = [
                    ('Supply Node', 'square', '#2196F3'),
                    ('Storage Node', 'hexagon', '#4CAF50'),
                    ('Demand Node', 'circle', '#F44336'),
                    ('Sink Node', 'diamond', '#9E9E9E'),
                    ('Hydrowork Node', 'pentagon', '#FF9800')
                ]

                for name, symbol, color in legend_items:
                    legend_trace = go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(
                            symbol=symbol,
                            size=20,
                            color=color,
                            line=dict(width=1, color='black')
                        ),
                        name=name,
                        showlegend=True
                    )
                    node_traces.append(legend_trace)

            return edge_traces + node_traces

        # Collect node and edge data
        nodes = []
        edges = []
        max_flow = 0
        
        for node_id, data in self.system.graph.nodes(data=True):
            node = data['node']
            node_type = type(node)

            color = {
                SupplyNode: '#2196F3',     # Blue
                StorageNode: '#4CAF50',    # Green
                DemandNode: '#F44336',     # Red
                SinkNode: '#9E9E9E',       # Grey
                HydroWorks: '#FF9800'      # Orange
            }.get(node_type, 'gray')

            shape = {
                SupplyNode: 'square',
                StorageNode: 'hexagon',
                DemandNode: 'circle',
                SinkNode: 'diamond',
                HydroWorks: 'pentagon'
            }.get(node_type, 'circle')

            nodes.append({
                'id': node_id,
                'easting': float(node.easting),
                'northing': float(node.northing),
                'color': color,
                'shape': shape
            })

        for u, v, data in self.system.graph.edges(data=True):
            edge = data['edge']
            flows = [float(f) for f in edge.flow_after_losses] if edge.flow_after_losses else []
            if flows:
                max_flow = max(max_flow, max(flows))
                
            edges.append({
                'source': u,
                'target': v,
                'capacity': float(edge.capacity),
                'flows': flows
            })

        # Create animation frames
        frames = [go.Frame(
            data=create_frame(t),
            name=f'frame{t}'
        ) for t in range(self.system.time_steps)]

        start_date = datetime(self.system.start_year, self.system.start_month, 1)
        steps = [
            dict(
                method='animate',
                args=[[f'frame{k}'], dict(mode='immediate', frame=dict(duration=0))],
                label=(start_date + relativedelta(months=k)).strftime('%b %Y')  # e.g. "Jan 2025"
            )
            for k in range(self.system.time_steps)
        ]
        # Create initial figure
        fig = go.Figure(
            data=create_frame(0),
            frames=frames,
            layout=go.Layout(
                title=dict(
                    text='ZRB Model',
                    x=0.5,
                    y=0.95
                ),
                width=1200,
                height=800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=1000, redraw=True),
                                            fromcurrent=True,
                                            mode='immediate')]),
                        dict(label='Pause',
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode='immediate')])
                    ],
                    x=0.1,
                    y=1.1
                )],
                
                

                sliders=[dict(
                    currentvalue=dict(
                        prefix='Timestep: ',
                        visible=True,
                        xanchor='right'
                    ),
                    steps=steps,
                )],
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                margin=dict(t=100, r=200)
            )
        )

        # Save as standalone HTML
        save_path = os.path.join(self.image_dir, f"{self.name}_network_vis.html")
        pio.write_html(fig, save_path, auto_open=False, include_plotlyjs=True)
        
        return save_path
