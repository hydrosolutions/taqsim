"""
Modified WaterSystemVisualizer class to ensure compatibility with the updated NodeTypes.
This class properly handles the different node structures and provides visualization methods 
for analyzing water system simulation results.
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
from typing import Dict, List, Optional, Union, Any, Tuple
from .nodes import SupplyNode, StorageNode, DemandNode, SinkNode, HydroWorks, RunoffNode

# Helper function for getting node outflow
def get_node_outflow(node, time_step):
    """
    Helper function to get a node's outflow that works with both old and new node structure.
    
    Args:
        node: A node object (SupplyNode, DemandNode, StorageNode, RunoffNode, or HydroWorks)
        time_step: The time step to get the outflow for
        
    Returns:
        float: The total outflow from the node at the specified time step
    """
    # Import node types here to avoid circular imports
    from .nodes import SupplyNode, StorageNode, DemandNode, RunoffNode, HydroWorks
    
    if isinstance(node, (SupplyNode, StorageNode, DemandNode, RunoffNode)):
        # New structure with single outflow_edge
        if hasattr(node, 'outflow_edge') and node.outflow_edge is not None:
            return node.outflow_edge.flow_before_losses[time_step]
        return 0
    elif isinstance(node, HydroWorks):
        # HydroWorks still has outflow_edges dictionary
        return sum(edge.flow_before_losses[time_step] for edge in node.outflow_edges.values())
    else:
        # Fallback for unknown node types
        if hasattr(node, 'outflow_edges') and node.outflow_edges:
            return sum(edge.flow_before_losses[time_step] for edge in node.outflow_edges.values())
        elif hasattr(node, 'outflow_edge') and node.outflow_edge is not None:
            return node.outflow_edge.flow_before_losses[time_step]
        return 0

# Helper function for getting node outflow capacity
def get_node_outflow_capacity(node):
    """
    Helper function to get a node's outflow capacity that works with both old and new node .structure.
    
    Args:
        node: A node object (SupplyNode, DemandNode, StorageNode, RunoffNode, or HydroWorks)
        
    Returns:
        float: The total outflow capacity from the node
    """
    # Import node types here to avoid circular imports
    from .nodes import SupplyNode, StorageNode, DemandNode, RunoffNode, HydroWorks
    
    if isinstance(node, (SupplyNode, StorageNode, DemandNode, RunoffNode)):
        # New structure with single outflow_edge
        if hasattr(node, 'outflow_edge') and node.outflow_edge is not None:
            return node.outflow_edge.capacity
        return 0
    elif isinstance(node, HydroWorks):
        # HydroWorks still has outflow_edges dictionary
        return sum(edge.capacity for edge in node.outflow_edges.values())
    else:
        # Fallback for unknown node types
        if hasattr(node, 'outflow_edges') and node.outflow_edges:
            return sum(edge.capacity for edge in node.outflow_edges.values())
        elif hasattr(node, 'outflow_edge') and node.outflow_edge is not None:
            return node.outflow_edge.capacity
        return 0

class WaterSystemVisualizer:
    """
    A class for creating time series visualizations of water system simulation results.
    This version is compatible with the updated NodeTypes structure.
    """
    
    def __init__(self, water_system, name=""):
        """
        Initialize the visualizer with a water system.
        
        Args:
            water_system (WaterSystem): The simulated water system to visualize.
            name (str): Optional name for the visualization set
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
                    # Use inflow_edges directly - should work with all node types
                    inflow = sum(edge.flow_after_losses[time_step] for edge in node.inflow_edges.values()) \
                            if hasattr(node, 'inflow_edges') else 0
                    
                    # Use helper function for outflow
                    outflow = get_node_outflow(node, time_step)

                    row_data[f"{node_id}_Inflow"] = inflow
                    row_data[f"{node_id}_Outflow"] = outflow

                    if isinstance(node, SupplyNode):
                        row_data[f"{node_id}_SupplyRate"] = node.supply_rates[time_step]
                    elif isinstance(node, StorageNode):
                        row_data[f"{node_id}_Storage"] = node.storage[time_step]
                        storage_change = (node.storage[time_step+1] - node.storage[time_step] 
                                        if time_step + 1 < len(node.storage) else 0)
                        row_data[f"{node_id}_StorageChange"] = storage_change
                        row_data[f"{node_id}_ExcessVolume"] = node.spillway_register[time_step] if time_step < len(node.spillway_register) else 0
                    elif isinstance(node, DemandNode):
                        row_data[f"{node_id}_Demand"] = node.demand_rates[time_step]
                        row_data[f"{node_id}_SatisfiedDemand"] = node.satisfied_consumptive_demand[time_step] if time_step < len(node.satisfied_consumptive_demand) else 0
                        
                        # Calculate deficit - account for potential length mismatch
                        satisfied = node.satisfied_consumptive_demand[time_step] if time_step < len(node.satisfied_consumptive_demand) else 0
                        row_data[f"{node_id}_Deficit"] = node.demand_rates[time_step] - satisfied
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

    def plot_reservoir_dynamics(self):
        """
        Create a comparison plot showing inflow, outflow, and volume for all reservoirs.
        This version uses the helper functions to work with the updated NodeTypes.
        
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
            inflows = [sum(edge.flow_after_losses[t] for edge in node.inflow_edges.values())
                    for t in time_steps]
            
            # Calculate total outflow for each timestep using helper function
            outflows = [get_node_outflow(node, t) for t in time_steps]
            
            # Get waterlevels (excluding last entry which is for next timestep)
            # Ensure we don't go out of bounds
            waterlevels = node.water_level[:-1] if len(node.water_level) > 1 else node.water_level
            if len(waterlevels) < len(time_steps):
                # Extend with the last value if needed
                waterlevels = list(waterlevels) + [waterlevels[-1]] * (len(time_steps) - len(waterlevels))
            waterlevels = waterlevels[:len(time_steps)]  # Truncate if longer than time_steps
            
            # Plot flows on left y-axis
            ax1.plot(time_steps, inflows, color=colors['inflow'], 
                   label='Inflow', linewidth=2)
            ax1.plot(time_steps, outflows, color=colors['outflow'], 
                   label='Outflow', linewidth=2)
            
            # Plot waterlevel on right y-axis
            ax2.plot(time_steps, waterlevels, color=colors['waterlevel'], 
                   label='Waterlevel', linewidth=2, linestyle='--')
            
            # Calculate statistics
            stats_text = (
                f"Statistics:\n"
                f"Mean Inflow: {np.mean(inflows):.2f} m³/s\n"
                f"Mean Outflow: {np.mean(outflows):.2f} m³/s\n"
                f"Mean Waterlevel: {np.mean(waterlevels):,.0f} m.a.s.l.\n"
                f"Max Waterlevel: {np.max(waterlevels):,.0f} m.a.s.l.\n"
                f"Min Waterlevel: {np.min(waterlevels):,.0f} m.a.s.l.\n"
                f"Waterlevel Change: {waterlevels[-1] - waterlevels[0]:,.0f} m.a.s.l."
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
            ax2.set_ylabel('Waterlevel [m.a.s.l.]')
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

    def plot_spills(self):
        """
        Create a time series plot showing spills from all hydroworks and reservoir nodes.
        This version is compatible with the updated NodeTypes.
        
        Returns:
            str: Path to the saved plot file
        """
        from .nodes import HydroWorks, StorageNode
        
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
                spills = node.spill_register[:len(time_steps)]
                # Extend with zeros if needed
                if len(spills) < len(time_steps):
                    spills = list(spills) + [0] * (len(time_steps) - len(spills))
                spill_rates = [spill / self.system.dt for spill in spills]
                total_hydroworks_spills += spill_rates
                
                # Plot individual spills
                line = ax1.plot(time_steps, spill_rates, color=colors[idx], 
                        label=node_id, linewidth=2, marker='o', markersize=4)
                legend_elements_hydroworks.append(line[0])
                
                # Calculate statistics for this node
                total_spill_volume = sum(spills)
                mean_spill_rate = np.mean(spill_rates)
                max_spill_rate = np.max(spill_rates)
                spill_frequency = sum(1 for s in spill_rates if s > 0) / len(time_steps) * 100
                
                '''print(f"\nSpill Statistics for {node_id}:")
                print(f"Total Spill Volume: {total_spill_volume:,.0f} m³")
                print(f"Mean Spill Rate: {mean_spill_rate:.2f} m³/s")
                print(f"Maximum Spill Rate: {max_spill_rate:.2f} m³/s")
                print(f"Spill Frequency: {spill_frequency:.1f}%")'''
        
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
                spills = node.spillway_register[:len(time_steps)]
                # Extend with zeros if needed
                if len(spills) < len(time_steps):
                    spills = list(spills) + [0] * (len(time_steps) - len(spills))
                spill_rates = [spill / self.system.dt for spill in spills]
                total_reservoir_spills += spill_rates
                
                # Plot individual spills
                line = ax2.plot(time_steps, spill_rates, color=colors[idx], 
                            label=node_id, linewidth=2, marker='o', markersize=4)
                legend_elements_reservoir.append(line[0])
                
                # Calculate statistics for this node
                total_spill_volume = sum(spills)
                mean_spill_rate = np.mean(spill_rates)
                max_spill_rate = np.max(spill_rates)
                spill_frequency = sum(1 for s in spill_rates if s > 0) / len(time_steps) * 100
                
                '''print(f"\nSpillway Statistics for {node_id}:")
                print(f"Total Spillway Volume: {total_spill_volume:,.0f} m³")
                print(f"Mean Spillway Rate: {mean_spill_rate:.2f} m³/s")
                print(f"Maximum Spillway Rate: {max_spill_rate:.2f} m³/s")
                print(f"Spillway Activation Frequency: {spill_frequency:.1f}%")'''
        
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

    def plot_network_overview(self):
        """
        Create a network overview visualization that is compatible with the updated NodeTypes.
        
        Returns:
            str: Path to the saved plot file
        """
        
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
            HydroWorks: '#FF9800',   # Orange
            RunoffNode: '#8B4513'    # Brown
        }
        
        node_shapes = {
            SupplyNode: 's',        # Square
            StorageNode: 'h',       # Hexagon
            DemandNode: 'o',        # Circle
            SinkNode: 'd',          # Diamond
            HydroWorks: 'p',        # Pentagon
            RunoffNode: 's'         # Square
        }

        node_names = {
            SupplyNode: 'Source Node',        
            StorageNode: 'Storage Node',      
            DemandNode: 'Demand Node',        
            SinkNode: 'Sink Node',          
            HydroWorks: 'Hydrowork Node',     
            RunoffNode: 'Surfacerunoff Node'  
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
            width = 1 + 5 * np.sqrt(capacity / max_capacity) if max_capacity > 0 else 1
            
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
        capacity_threshold = 0  # Show labels for edges with >0% of max capacity
        edge_labels = {(u, v): f"{edge_data['edge'].capacity:.0f}"
                    for u, v, edge_data in self.system.graph.edges(data=True)
                    if edge_data['edge'].capacity > capacity_threshold}
        
        # Draw edge labels (capacity) on the figure
        nx.draw_networkx_edge_labels(
            self.system.graph,
            pos,
            edge_labels=edge_labels,
            font_size=14,
            font_color='navy',
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
            ax=ax
        )
        
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
        
        # Turn off axis
        plt.axis('off')
        
        plt.tight_layout()
        
        return self._save_plot("network_overview")   

    def plot_minimum_flow_compliance(self):
        """
        Create time series plots showing actual flows versus minimum flow requirements 
        for all sink nodes that have minimum flow requirements.
        This version is compatible with the updated NodeTypes.
        
        Returns:
            str: Path to the saved plot file
        """
        from .nodes import SinkNode
        
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
            
            # Handle potential length mismatches
            actual_flows = node.flow_history[:len(time_steps)]
            if len(actual_flows) < len(time_steps):
                actual_flows = list(actual_flows) + [0] * (len(time_steps) - len(actual_flows))
            
            deficits = node.flow_deficits[:len(time_steps)]
            if len(deficits) < len(time_steps):
                deficits = list(deficits) + [0] * (len(time_steps) - len(deficits))
            
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
                f"Total Deficit Volume: {total_deficit_volume:,.0f} m³"
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
        Create enhanced heatmaps showing total demand deficits across all demand nodes over time.
        This version is compatible with the updated NodeTypes.
        """
        from .nodes import DemandNode
        
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
                # Get satisfied demand, handling potential length mismatches
                satisfied = node.satisfied_demand_total[t] if t < len(node.satisfied_demand_total) else 0
                total_deficit = (node.demand_rates[t] - satisfied)
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
        plt.ylabel('Demand Node', fontsize=18)
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
        plt.ylabel('Demand Node', fontsize=16)
        plt.xticks(plt.xticks()[0][::3], plt.xticks()[1][::3], fontsize=14, rotation=90)
        plt.yticks(fontsize=14)
        pct_filepath = self._save_plot("deficit_heatmap_percentage")
        
        return abs_filepath, pct_filepath

    def plot_reservoir_volumes(self):
        """
        Create time series plots showing volume and water elevation for all reservoir (storage) nodes.
        This version is compatible with the updated NodeTypes.
        
        Returns:
            str: Path to the saved plot file
        """
        
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
        
        # Create date labels for x-axis (just use time step numbers, no month names)
        date_labels = list(range(self.system.time_steps))
            
        # Create figure with subplots - one per reservoir
        n_nodes = len(storage_nodes)
        plt.rcParams.update({'font.size': 22})
        fig, axes = plt.subplots(n_nodes, 1, figsize=(16, 6*n_nodes), sharex=True)
        if n_nodes == 1:
            axes = [axes]
        
        # Plot for each reservoir
        for idx, (node_id, node) in enumerate(storage_nodes):
            ax1 = axes[idx]
            
            # Get volume data - handle potential length mismatches
            volumes = np.array(node.storage) if len(node.storage) > 1 else np.array([])
            volumes = volumes / 1e6  # Convert to million m³
            
            # Ensure volumes array matches time_steps length
            if len(volumes) < self.system.time_steps + 1:
                # Extend with the last value
                volumes = np.append(volumes, [volumes[-1] if len(volumes) > 0 else 0] * 
                                  (self.system.time_steps - len(volumes)))
            elif len(volumes) > self.system.time_steps + 1:
                volumes = volumes[:self.system.time_steps]
                
            time_steps = range(len(volumes))
            capacity = node.capacity / 1e6  # Convert to million m³
            
            # Plot volume time series
            line1 = ax1.plot(time_steps, volumes, color='blue', 
                           label='Storage Volume', linewidth=2)
            
            # Add capacity line
            cap_line = ax1.axhline(y=capacity, color='red', linestyle='--', 
                                 label=f'Maximum Storage Capacity')
            
            # Customize subplot
            ax1.set_title(f'{node_id}')
            ax1.set_ylabel(r'Volume [10$^6$ m³]')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + [cap_line]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')

            # Set custom x-ticks with time step numbers only
            tick_positions = list(time_steps)
            
            # If we have many time steps, use fewer ticks to avoid crowding
            if len(tick_positions) > 12 and len(date_labels) >= len(tick_positions):
                tick_spacing = max(1, len(tick_positions) // 12)
                tick_positions = tick_positions[::tick_spacing]
                tick_labels = [str(date_labels[i]) for i in tick_positions]
                
                ax1.set_xticks(tick_positions)
                ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        return self._save_plot("reservoir_volumes")

    def plot_network_layout(self):
        """
        Create a network overview visualization that is compatible with the updated NodeTypes.
        
        Returns:
            str: Path to the saved plot file
        """
        
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
            HydroWorks: '#FF9800',   # Orange
            RunoffNode: '#8B4513'    # Brown
        }
        
        node_shapes = {
            SupplyNode: 's',        # Square
            StorageNode: 'h',       # Hexagon
            DemandNode: 'o',        # Circle
            SinkNode: 'd',          # Diamond
            HydroWorks: 'p',        # Pentagon
            RunoffNode: 's'         # Square
        }

        node_names = {
            SupplyNode: 'Source Node',        
            StorageNode: 'Storage Node',      
            DemandNode: 'Demand Node',        
            SinkNode: 'Sink Node',          
            HydroWorks: 'Hydrowork Node',     
            RunoffNode: 'Surfacerunoff Node'  
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
            width = 1 + 8 * np.sqrt(capacity / max_capacity) if max_capacity > 0 else 1
            
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
        capacity_threshold = 200  # Show labels for edges with >0% of max capacity
        edge_labels = {(u, v): f"{edge_data['edge'].capacity:.0f}"
                    for u, v, edge_data in self.system.graph.edges(data=True)
                    if edge_data['edge'].capacity > capacity_threshold}
        
        # Draw edge labels (capacity) on the figure
        nx.draw_networkx_edge_labels(
            self.system.graph,
            pos,
            edge_labels=edge_labels,
            font_size=14,
            font_color='navy',
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
            ax=ax
        )
        
        # Only label SinkNode, DemandNode, and StorageNode
        node_labels = {}
        for node_id, data in self.system.graph.nodes(data=True):
            node = data['node']
            if (
                node_id in SinkNode.all_ids or
                node_id in DemandNode.all_ids #or
                #node_id in StorageNode.all_ids
            ):
                node_labels[node_id] = node_id  # or customize label as needed
      
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
        
        # Turn off axis
        plt.axis('off')
        
        plt.tight_layout()
            
        return self._save_plot("network_layout")

    def print_water_balance_summary(self):
        """
        Print a comprehensive summary of the water balance results.
        Includes volumes, relative contributions, storage changes and balance error statistics.
        """
        df = self.system.get_water_balance()
        number_of_years = (self.system.time_steps)/12
        print(f"Number of years: {number_of_years}")
        
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
        print(f'Number of years:     {number_of_years:,.0f}')
        print(f"Timestep duration:   {self.system.dt:,.0f} seconds")
        
        # Source volumes
        print_section("Source Volumes")
        total_sourcenode = df['source'].sum()
        total_surfacerunoff = df['surfacerunoff'].sum()
        total_source = total_sourcenode + total_surfacerunoff
        print(f"Source Node:    {total_sourcenode/number_of_years:,.0f} m³/a")
        print(f"Surface Runoff:   {total_surfacerunoff/number_of_years:,.0f} m³/a")
        print(f"Total Source:   {total_source/number_of_years:,.0f} m³/a (100%)")
        
        # Component volumes and percentages
        print_section("Sink Volumes")
        components = {
            'Supplied Consumptive Demand': 'supplied consumptive demand',
            'Sink Outflow': 'sink',
            'Edge Losses': 'edge losses',
            'Res Spills': 'reservoir spills',
            'Res ET Losses': 'reservoir ET losses',
            'HW Spills': 'hydroworks spills'
        }
        
        for label, comp in components.items():
            if comp in df.columns:
                total = df[comp].sum()
                percentage = (total / total_source * 100) if total_source > 0 else 0
                print(f"{label:30s}: {total/number_of_years:15,.0f} m³/a ({percentage:6.1f}%)")
        
        # Storage changes
        print_section("Storage")
        print(f"Initial storage:     {df['storage_start'].iloc[0]:15,.0f} m³")
        print(f"Final storage:       {df['storage_end'].iloc[-1]:15,.0f} m³")
        total_storage_change = df['storage_change'].sum()
        storage_percentage = (total_storage_change / total_source * 100) if total_source > 0 else 0
        print(f"Net storage change:  {total_storage_change:15,.0f} m³ ({storage_percentage:6.1f}%)")
        
                # Conservation check
        print_section("Conservation Check")
        total_in = total_source
        sink_components = ['supplied consumptive demand', 'sink', 'edge losses', 'reservoir spills', 
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

        # Sink Nodes
        print_section("Sink Nodes")
        print(f'Total Min Flow Requirement: {df["sink min flow requirement"].sum()/number_of_years:15,.0f} m³/a')
        print(f"Total Sink Outflow:         {df['sink'].sum()/number_of_years:15,.0f} m³/a")
        print(f"Total Sink Deficit:         {df['sink min flow deficit'].sum()/number_of_years:15,.0f} m³/a")
 
        # Demand satisfaction
        print_section("Demand Nodes")
        total_demand = df['demands'].sum()
        total_demand_non_consumptive = df['demands non consumptive'].sum()
        total_consumptive_satisfied = df['supplied consumptive demand'].sum()
        total_non_consumptive_satisfied = df['supplied non consumptive demand'].sum()
        unmet_demand = df['unmet demand'].sum()
        print(f"Total demand:                   {total_demand/number_of_years:15,.0f} m³/a")
        print(f'Of which non consumptive:       {total_demand_non_consumptive/number_of_years:15,.0f} m³/a')
        print(f"Satisfied consumptive demand:   {total_consumptive_satisfied/number_of_years:15,.0f} m³/a")
        print(f"Satisfied non-consumptive:      {total_non_consumptive_satisfied/number_of_years:15,.0f} m³/a")
        print(f"Unmet demand:                   {unmet_demand/number_of_years:15,.0f} m³/a")
        
        # Include flow deficit in demand section if available
        if 'deficit' in df.columns:
            total_deficit = df['deficit'].sum()
            print(f"Min flow deficit:        {total_deficit/number_of_years:15,.0f} m³/a")

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