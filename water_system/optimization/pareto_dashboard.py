import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

class ParetoFrontDashboard3D:
    """
    A class for creating interactive visualizations of Pareto-optimal solutions
    from multi-objective water system optimization.
    """
    
    def __init__(self, pareto_solutions, output_dir="./model_output/dashboard"):
        """
        Initialize the dashboard with Pareto solutions.
        
        Args:
            pareto_solutions (list): List of Pareto-optimal solutions
            output_dir (str): Directory to save dashboard outputs
        """
        self.pareto_solutions = pareto_solutions
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract data once for reuse
        self.df = self._extract_solution_data()
        
    def _extract_solution_data(self):
        """Extract solution data into a DataFrame for plotting"""
        # Check if pareto_solutions exists and has required data
        if not self.pareto_solutions or not isinstance(self.pareto_solutions, list):
            print("No valid Pareto solutions provided")
            return pd.DataFrame()
            
        # Extract data
        data = []
        for sol in self.pareto_solutions:
            # Handle different possible data structures
            if isinstance(sol, dict):
                solution_id = sol.get('id', 0)
                obj_values = sol.get('objective_values', [0, 0, 0])
                
                # Direct attributes for each objective
                regular_deficit = sol.get('demand_deficit', obj_values[0])
                priority_deficit = sol.get('priority_demand_deficit', obj_values[1])
                minflow_deficit = sol.get('minflow_deficit', obj_values[2])
            else:
                # Assume it's an individual from DEAP
                solution_id = getattr(sol, 'id', 0)
                obj_values = sol.fitness.values if hasattr(sol, 'fitness') else [0, 0, 0]
                regular_deficit = obj_values[0]
                priority_deficit = obj_values[1]
                minflow_deficit = obj_values[2]
            
            data.append({
                'Solution': solution_id,
                'Regular Demand Deficit': regular_deficit,
                'Priority Demand Deficit': priority_deficit,
                'Minimum Flow Deficit': minflow_deficit,
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate normalized values for plotting
        df['Normalized Regular'] = df['Regular Demand Deficit'] / df['Regular Demand Deficit'].max()
        df['Normalized Priority'] = df['Priority Demand Deficit'] / df['Priority Demand Deficit'].max()
        df['Normalized MinFlow'] = df['Minimum Flow Deficit'] / df['Minimum Flow Deficit'].max()
        
        # Add composite score for balanced solution identification
        df['Composite Score'] = (df['Normalized Regular'] + df['Normalized Priority'] + df['Normalized MinFlow']) / 3
        
        return df
        
    def create_dashboard(self, filename='pareto_dashboard.html'):
        """
        Create a comprehensive interactive dashboard for Pareto solutions.
        
        Args:
            filename (str): Name of the output HTML file
            
        Returns:
            plotly.graph_objects.Figure: The dashboard figure object
        """
        if self.df.empty:
            print("No data available to create dashboard")
            return None
            
        # Create subplots layout
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=[
                '3D Pareto Front', 
                'Priority vs Regular Demand', 
                'Priority vs Min Flow',
                'Regular Demand vs Min Flow'
            ]
        )
        
        # Find special solutions
        min_regular_idx = self.df['Regular Demand Deficit'].idxmin()
        min_priority_idx = self.df['Priority Demand Deficit'].idxmin()
        min_minflow_idx = self.df['Minimum Flow Deficit'].idxmin()
        balanced_idx = self.df['Composite Score'].idxmin()
        
        # Create 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=self.df['Regular Demand Deficit'],
                y=self.df['Priority Demand Deficit'],
                z=self.df['Minimum Flow Deficit'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.df['Solution'],
                    colorscale='Viridis',
                    colorbar=dict(title='Solution ID'),
                    opacity=0.8
                ),
                text=[f'Solution {i}' for i in self.df['Solution']],
                hovertemplate=
                '<b>Solution %{text}</b><br><br>' +
                'Regular Deficit: %{x:.2e} m³<br>' +
                'Priority Deficit: %{y:.2e} m³<br>' +
                'Min Flow Deficit: %{z:.2e} m³<br>',
                name='Pareto Solutions'
            ), row=1, col=1
        )
        
        # Highlight special points in 3D
        self._add_special_points_3d(fig, min_regular_idx, min_priority_idx, min_minflow_idx, balanced_idx)
        
        # 2D projection: Regular vs Priority
        fig.add_trace(
            go.Scatter(
                x=self.df['Regular Demand Deficit'],
                y=self.df['Priority Demand Deficit'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.df['Solution'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f'Solution {i}' for i in self.df['Solution']],
                hovertemplate=
                '<b>Solution %{text}</b><br><br>' +
                'Regular Deficit: %{x:.2e} m³<br>' +
                'Priority Deficit: %{y:.2e} m³<br>',
                name='Priority vs Regular'
            ), row=1, col=2
        )
        
        # 2D projection: Priority vs MinFlow
        fig.add_trace(
            go.Scatter(
                x=self.df['Priority Demand Deficit'],
                y=self.df['Minimum Flow Deficit'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.df['Solution'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f'Solution {i}' for i in self.df['Solution']],
                hovertemplate=
                '<b>Solution %{text}</b><br><br>' +
                'Priority Deficit: %{x:.2e} m³<br>' +
                'Min Flow Deficit: %{y:.2e} m³<br>',
                name='Priority vs Min Flow'
            ), row=2, col=1
        )
        
        # 2D projection: Regular vs MinFlow
        fig.add_trace(
            go.Scatter(
                x=self.df['Regular Demand Deficit'],
                y=self.df['Minimum Flow Deficit'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.df['Solution'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f'Solution {i}' for i in self.df['Solution']],
                hovertemplate=
                '<b>Solution %{text}</b><br><br>' +
                'Regular Deficit: %{x:.2e} m³<br>' +
                'Min Flow Deficit: %{y:.2e} m³<br>',
                name='Regular vs Min Flow'
            ), row=2, col=2
        )
        
        # Highlight special points in 2D plots
        self._add_special_points_2d(fig, min_regular_idx, min_priority_idx, min_minflow_idx, balanced_idx)
        
        # Update layout
        fig.update_layout(
            title='Multi-objective Optimization Results',
            height=900,
            width=1200,
            scene=dict(
                xaxis_title='Regular Demand Deficit',
                yaxis_title='Priority Demand Deficit',
                zaxis_title='Minimum Flow Deficit'
            ),
            showlegend=False
        )
        
        # Update 2D axes
        fig.update_xaxes(title_text='Regular Demand Deficit', row=1, col=2)
        fig.update_yaxes(title_text='Priority Demand Deficit', row=1, col=2)
        
        fig.update_xaxes(title_text='Priority Demand Deficit', row=2, col=1)
        fig.update_yaxes(title_text='Minimum Flow Deficit', row=2, col=1)
        
        fig.update_xaxes(title_text='Regular Demand Deficit', row=2, col=2)
        fig.update_yaxes(title_text='Minimum Flow Deficit', row=2, col=2)
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Interactive dashboard saved to {output_path}")
        
        return fig
    
    def _add_special_points_3d(self, fig, min_regular_idx, min_priority_idx, min_minflow_idx, balanced_idx):
        """Add highlighted special points to 3D plot"""
        # Best for regular demand
        fig.add_trace(
            go.Scatter3d(
                x=[self.df.loc[min_regular_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_regular_idx, 'Priority Demand Deficit']],
                z=[self.df.loc[min_regular_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='circle'
                ),
                name='Best Regular'
            ), row=1, col=1
        )
        
        # Best for priority demand
        fig.add_trace(
            go.Scatter3d(
                x=[self.df.loc[min_priority_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_priority_idx, 'Priority Demand Deficit']],
                z=[self.df.loc[min_priority_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    symbol='circle'
                ),
                name='Best Priority'
            ), row=1, col=1
        )
        
        # Best for minimum flow
        fig.add_trace(
            go.Scatter3d(
                x=[self.df.loc[min_minflow_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_minflow_idx, 'Priority Demand Deficit']],
                z=[self.df.loc[min_minflow_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(
                    size=8,
                    color='green',
                    symbol='circle'
                ),
                name='Best Min Flow'
            ), row=1, col=1
        )
        
        # Balanced solution
        fig.add_trace(
            go.Scatter3d(
                x=[self.df.loc[balanced_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[balanced_idx, 'Priority Demand Deficit']],
                z=[self.df.loc[balanced_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(
                    size=10,
                    color='purple',
                    symbol='diamond'
                ),
                name='Balanced Solution'
            ), row=1, col=1
        )
    
    def _add_special_points_2d(self, fig, min_regular_idx, min_priority_idx, min_minflow_idx, balanced_idx):
        """Add highlighted special points to 2D plots"""
        # Regular vs Priority plot (row=1, col=2)
        # Best Regular
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[min_regular_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_regular_idx, 'Priority Demand Deficit']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle'),
                name='Best Regular'
            ), row=1, col=2
        )
        
        # Best Priority
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[min_priority_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_priority_idx, 'Priority Demand Deficit']],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='circle'),
                name='Best Priority'
            ), row=1, col=2
        )
        
        # Balanced
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[balanced_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[balanced_idx, 'Priority Demand Deficit']],
                mode='markers',
                marker=dict(size=15, color='purple', symbol='diamond'),
                name='Balanced'
            ), row=1, col=2
        )
        
        # Priority vs MinFlow plot (row=2, col=1)
        # Best Priority
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[min_priority_idx, 'Priority Demand Deficit']],
                y=[self.df.loc[min_priority_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='circle'),
                name='Best Priority'
            ), row=2, col=1
        )
        
        # Best MinFlow
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[min_minflow_idx, 'Priority Demand Deficit']],
                y=[self.df.loc[min_minflow_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(size=15, color='green', symbol='circle'),
                name='Best MinFlow'
            ), row=2, col=1
        )
        
        # Balanced
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[balanced_idx, 'Priority Demand Deficit']],
                y=[self.df.loc[balanced_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(size=15, color='purple', symbol='diamond'),
                name='Balanced'
            ), row=2, col=1
        )
        
        # Regular vs MinFlow plot (row=2, col=2)
        # Best Regular
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[min_regular_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_regular_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle'),
                name='Best Regular'
            ), row=2, col=2
        )
        
        # Best MinFlow
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[min_minflow_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[min_minflow_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(size=15, color='green', symbol='circle'),
                name='Best MinFlow'
            ), row=2, col=2
        )
        
        # Balanced
        fig.add_trace(
            go.Scatter(
                x=[self.df.loc[balanced_idx, 'Regular Demand Deficit']],
                y=[self.df.loc[balanced_idx, 'Minimum Flow Deficit']],
                mode='markers',
                marker=dict(size=15, color='purple', symbol='diamond'),
                name='Balanced'
            ), row=2, col=2
        )
    
    def create_parallel_coordinates(self, filename='parallel_coords.html'):
        """
        Create a parallel coordinates plot for visualizing trade-offs.
        
        Args:
            filename (str): Name of the output HTML file
            
        Returns:
            plotly.graph_objects.Figure: The parallel coordinates figure
        """
        if self.df.empty:
            print("No data available to create parallel coordinates plot")
            return None
        
        # Normalize values for better visualization
        norm_df = self.df.copy()
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            norm_df, 
            color="Composite Score",
            dimensions=['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit'],
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Multi-objective Optimization Results - Parallel Coordinates'
        )
        
        # Update layout
        fig.update_layout(
            font=dict(size=12),
            height=600,
            width=1000
        )
        
        # Save plot
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Parallel coordinates plot saved to {output_path}")
        
        return fig
        
    def create_radar_chart(self, solution_indices=None, filename='radar_chart.html'):
        """
        Create radar chart comparing selected solutions.
        
        Args:
            solution_indices (list): Indices of solutions to compare
            filename (str): Name of the output HTML file
            
        Returns:
            plotly.graph_objects.Figure: The radar chart figure
        """
        if self.df.empty:
            print("No data available to create radar chart")
            return None
            
        # If no indices provided, use best solutions
        if solution_indices is None:
            # Find special solutions
            min_regular_idx = self.df['Regular Demand Deficit'].idxmin()
            min_priority_idx = self.df['Priority Demand Deficit'].idxmin()
            min_minflow_idx = self.df['Minimum Flow Deficit'].idxmin()
            balanced_idx = self.df['Composite Score'].idxmin()
            
            solution_indices = [min_regular_idx, min_priority_idx, min_minflow_idx, balanced_idx]
        
        # Extract solutions to compare
        selected_solutions = self.df.loc[solution_indices]
        
        # Create normalized dataframe for radar chart
        # We invert values so that larger is better
        radar_df = selected_solutions.copy()
        max_vals = radar_df[['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit']].max()
        
        # Invert and normalize values (1 = best, 0 = worst)
        for col in ['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit']:
            radar_df[f'{col}_Radar'] = 1 - (radar_df[col] / max_vals[col])
        
        # Create radar chart
        categories = ['Regular Demand', 'Priority Demand', 'Minimum Flow']
        
        fig = go.Figure()
        
        # Add a trace for each solution
        for _, row in radar_df.iterrows():
            values = [
                row['Regular Demand Deficit_Radar'], 
                row['Priority Demand Deficit_Radar'], 
                row['Minimum Flow Deficit_Radar']
            ]
            
            # Close the polygon by repeating first value
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name=f'Solution {int(row["Solution"])}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Solution Comparison (Higher is Better)"
        )
        
        # Save chart
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Radar chart saved to {output_path}")
        
        return fig
    
    def create_representative_table(self, filename='representative_solutions.html'):
        """
        Create a table of representative solutions.
        
        Args:
            filename (str): Name of the output HTML file
            
        Returns:
            pandas.DataFrame: The representative solutions table
        """
        if self.df.empty:
            print("No data available to create representative table")
            return None
        
        # Find representative solutions
        min_regular_idx = self.df['Regular Demand Deficit'].idxmin()
        min_priority_idx = self.df['Priority Demand Deficit'].idxmin()
        min_minflow_idx = self.df['Minimum Flow Deficit'].idxmin()
        balanced_idx = self.df['Composite Score'].idxmin()
        
        # Create representative solutions table
        representatives = pd.DataFrame({
            'Solution': ['Best Regular', 'Best Priority', 'Best Min Flow', 'Balanced'],
            'Solution ID': [
                self.df.loc[min_regular_idx, 'Solution'], 
                self.df.loc[min_priority_idx, 'Solution'], 
                self.df.loc[min_minflow_idx, 'Solution'], 
                self.df.loc[balanced_idx, 'Solution']
            ],
            'Regular Demand Deficit': [
                self.df.loc[min_regular_idx, 'Regular Demand Deficit'],
                self.df.loc[min_priority_idx, 'Regular Demand Deficit'],
                self.df.loc[min_minflow_idx, 'Regular Demand Deficit'],
                self.df.loc[balanced_idx, 'Regular Demand Deficit']
            ],
            'Priority Demand Deficit': [
                self.df.loc[min_regular_idx, 'Priority Demand Deficit'],
                self.df.loc[min_priority_idx, 'Priority Demand Deficit'],
                self.df.loc[min_minflow_idx, 'Priority Demand Deficit'],
                self.df.loc[balanced_idx, 'Priority Demand Deficit']
            ],
            'Minimum Flow Deficit': [
                self.df.loc[min_regular_idx, 'Minimum Flow Deficit'],
                self.df.loc[min_priority_idx, 'Minimum Flow Deficit'],
                self.df.loc[min_minflow_idx, 'Minimum Flow Deficit'],
                self.df.loc[balanced_idx, 'Minimum Flow Deficit']
            ],
        })
        
        # Calculate trade-offs
        representatives['Regular Trade-off %'] = (representatives['Regular Demand Deficit'] / 
                                               representatives.loc[0, 'Regular Demand Deficit'] - 1) * 100
        representatives['Priority Trade-off %'] = (representatives['Priority Demand Deficit'] / 
                                                representatives.loc[1, 'Priority Demand Deficit'] - 1) * 100
        representatives['MinFlow Trade-off %'] = (representatives['Minimum Flow Deficit'] / 
                                              representatives.loc[2, 'Minimum Flow Deficit'] - 1) * 100
        
        # Create formatted versions for display
        representatives['Regular Deficit (m³)'] = representatives['Regular Demand Deficit'].map('{:,.0f}'.format)
        representatives['Priority Deficit (m³)'] = representatives['Priority Demand Deficit'].map('{:,.0f}'.format)
        representatives['MinFlow Deficit (m³)'] = representatives['Minimum Flow Deficit'].map('{:,.0f}'.format)
        representatives['Regular Trade-off'] = representatives['Regular Trade-off %'].map('{:+.1f}%'.format)
        representatives['Priority Trade-off'] = representatives['Priority Trade-off %'].map('{:+.1f}%'.format)
        representatives['MinFlow Trade-off'] = representatives['MinFlow Trade-off %'].map('{:+.1f}%'.format)
        
        # Create table figure
        display_columns = [
            'Solution', 'Solution ID', 
            'Regular Deficit (m³)', 'Regular Trade-off',
            'Priority Deficit (m³)', 'Priority Trade-off',
            'MinFlow Deficit (m³)', 'MinFlow Trade-off'
        ]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=display_columns,
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[representatives[col] for col in display_columns],
                fill_color='lavender',
                align='right',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Representative Solutions with Trade-offs",
            height=300,
            width=1200
        )
        
        # Save table
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Representative solutions table saved to {output_path}")
        
        # Also save as CSV
        csv_path = os.path.join(self.output_dir, filename.replace('.html', '.csv'))
        representatives.to_csv(csv_path, index=False)
        
        return representatives
        
    def generate_full_report(self):
        """
        Generate a complete set of visualizations for the Pareto front.
        
        Returns:
            dict: Dictionary of generated figures
        """
        figures = {}
        
        # Create main dashboard
        figures['dashboard'] = self.create_dashboard('pareto_dashboard.html')
        
        # Create parallel coordinates plot
        figures['parallel'] = self.create_parallel_coordinates('parallel_coords.html')
        
        # Create radar chart
        figures['radar'] = self.create_radar_chart(filename='radar_chart.html')
        
        # Create representative table
        figures['table'] = self.create_representative_table('representative_solutions.html')
        
        # Create index page to link all visualizations
        self._create_index_page()
        
        return figures
    
    def _create_index_page(self):
        """Create an index HTML page linking to all visualizations"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-objective Optimization Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                h1, h2 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
                .card h2 { margin-top: 0; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .description { color: #666; margin-bottom: 15px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multi-objective Water System Optimization Results</h1>
                
                <div class="card">
                    <h2>Interactive Pareto Front Dashboard</h2>
                    <p class="description">Explore the 3D Pareto front and its 2D projections showing the trade-offs between objectives.</p>
                    <a href="pareto_dashboard.html" target="_blank">Open Dashboard</a>
                </div>
                
                <div class="card">
                    <h2>Parallel Coordinates Plot</h2>
                    <p class="description">Visualize how different solutions compare across all three objectives simultaneously.</p>
                    <a href="parallel_coords.html" target="_blank">Open Parallel Coordinates</a>
                </div>
                
                <div class="card">
                    <h2>Representative Solutions Comparison</h2>
                    <p class="description">View a radar chart comparing key representative solutions.</p>
                    <a href="radar_chart.html" target="_blank">Open Radar Chart</a>
                </div>
                
                <div class="card">
                    <h2>Representative Solutions Table</h2>
                    <p class="description">See a detailed table of representative solutions with trade-off percentages.</p>
                    <a href="representative_solutions.html" target="_blank">Open Solutions Table</a>
                </div>
                
            </div>
        </body>
        </html>
        """
        
        # Save to file
        with open(os.path.join(self.output_dir, 'index.html'), 'w') as f:
            f.write(html_content)
        
        print(f"Index page created at {os.path.join(self.output_dir, 'index.html')}")

class ParetoFrontDashboard4D:
    """
    Dashboard for 4-objective Pareto solutions (including spillage).
    Shows parallel coordinates, radar chart, and solution table.
    """

    def __init__(self, pareto_solutions, output_dir="./model_output/dashboard"):
        self.pareto_solutions = pareto_solutions
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = self._extract_solution_data()

    def _extract_solution_data(self):
        if not self.pareto_solutions or not isinstance(self.pareto_solutions, list):
            print("No valid Pareto solutions provided")
            return pd.DataFrame()
        data = []
        for sol in self.pareto_solutions:
            if isinstance(sol, dict):
                solution_id = sol.get('id', 0)
                obj_values = sol.get('objective_values', [0, 0, 0, 0])
                regular_deficit = sol.get('demand_deficit', obj_values[0])
                priority_deficit = sol.get('priority_demand_deficit', obj_values[1])
                minflow_deficit = sol.get('minflow_deficit', obj_values[2])
                spillage = sol.get('spillage', obj_values[3])
            else:
                solution_id = getattr(sol, 'id', 0)
                obj_values = sol.fitness.values if hasattr(sol, 'fitness') else [0, 0, 0, 0]
                regular_deficit = obj_values[0]
                priority_deficit = obj_values[1]
                minflow_deficit = obj_values[2]
                spillage = obj_values[3]
            data.append({
                'Solution': solution_id,
                'Regular Demand Deficit': regular_deficit,
                'Priority Demand Deficit': priority_deficit,
                'Minimum Flow Deficit': minflow_deficit,
                'Spillage': spillage,
            })
        df = pd.DataFrame(data)
        # Normalized columns for radar/parallel
        for col in ['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit', 'Spillage']:
            max_val = df[col].max()
            df[f'Normalized {col}'] = df[col] / max_val if max_val != 0 else 0
        df['Composite Score'] = (
            df['Normalized Regular Demand Deficit'] +
            df['Normalized Priority Demand Deficit'] +
            df['Normalized Minimum Flow Deficit'] +
            df['Normalized Spillage']
        ) / 4
        return df

    def create_parallel_coordinates(self, filename='parallel_coords_4d.html'):
        if self.df.empty:
            print("No data available to create parallel coordinates plot")
            return None
        fig = px.parallel_coordinates(
            self.df,
            color="Composite Score",
            dimensions=[
                'Regular Demand Deficit',
                'Priority Demand Deficit',
                'Minimum Flow Deficit',
                'Spillage'
            ],
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='4D Pareto Solutions - Parallel Coordinates'
        )
        fig.update_layout(font=dict(size=12), height=600, width=1000)
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Parallel coordinates plot saved to {output_path}")
        return fig

    def create_radar_chart(self, solution_indices=None, filename='radar_chart_4d.html'):
        if self.df.empty:
            print("No data available to create radar chart")
            return None
        if solution_indices is None:
            min_regular_idx = self.df['Regular Demand Deficit'].idxmin()
            min_priority_idx = self.df['Priority Demand Deficit'].idxmin()
            min_minflow_idx = self.df['Minimum Flow Deficit'].idxmin()
            min_spillage_idx = self.df['Spillage'].idxmin()
            balanced_idx = self.df['Composite Score'].idxmin()
            solution_indices = [min_regular_idx, min_priority_idx, min_minflow_idx, min_spillage_idx, balanced_idx]
        selected_solutions = self.df.loc[solution_indices]
        radar_df = selected_solutions.copy()
        max_vals = radar_df[['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit', 'Spillage']].max()
        for col in ['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit', 'Spillage']:
            radar_df[f'{col}_Radar'] = 1 - (radar_df[col] / max_vals[col])
        categories = ['Regular Demand', 'Priority Demand', 'Minimum Flow', 'Spillage']
        fig = go.Figure()
        for _, row in radar_df.iterrows():
            values = [
                row['Regular Demand Deficit_Radar'],
                row['Priority Demand Deficit_Radar'],
                row['Minimum Flow Deficit_Radar'],
                row['Spillage_Radar']
            ]
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name=f'Solution {int(row["Solution"])}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Solution Comparison (Higher is Better)"
        )
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Radar chart saved to {output_path}")
        return fig

    def create_representative_table(self, filename='representative_solutions_4d.html'):
        if self.df.empty:
            print("No data available to create representative table")
            return None
        min_regular_idx = self.df['Regular Demand Deficit'].idxmin()
        min_priority_idx = self.df['Priority Demand Deficit'].idxmin()
        min_minflow_idx = self.df['Minimum Flow Deficit'].idxmin()
        min_spillage_idx = self.df['Spillage'].idxmin()
        balanced_idx = self.df['Composite Score'].idxmin()
        representatives = pd.DataFrame({
            'Solution': ['Best Regular', 'Best Priority', 'Best Min Flow', 'Best Spillage', 'Balanced'],
            'Solution ID': [
                self.df.loc[min_regular_idx, 'Solution'],
                self.df.loc[min_priority_idx, 'Solution'],
                self.df.loc[min_minflow_idx, 'Solution'],
                self.df.loc[min_spillage_idx, 'Solution'],
                self.df.loc[balanced_idx, 'Solution']
            ],
            'Regular Demand Deficit': [
                self.df.loc[min_regular_idx, 'Regular Demand Deficit'],
                self.df.loc[min_priority_idx, 'Regular Demand Deficit'],
                self.df.loc[min_minflow_idx, 'Regular Demand Deficit'],
                self.df.loc[min_spillage_idx, 'Regular Demand Deficit'],
                self.df.loc[balanced_idx, 'Regular Demand Deficit']
            ],
            'Priority Demand Deficit': [
                self.df.loc[min_regular_idx, 'Priority Demand Deficit'],
                self.df.loc[min_priority_idx, 'Priority Demand Deficit'],
                self.df.loc[min_minflow_idx, 'Priority Demand Deficit'],
                self.df.loc[min_spillage_idx, 'Priority Demand Deficit'],
                self.df.loc[balanced_idx, 'Priority Demand Deficit']
            ],
            'Minimum Flow Deficit': [
                self.df.loc[min_regular_idx, 'Minimum Flow Deficit'],
                self.df.loc[min_priority_idx, 'Minimum Flow Deficit'],
                self.df.loc[min_minflow_idx, 'Minimum Flow Deficit'],
                self.df.loc[min_spillage_idx, 'Minimum Flow Deficit'],
                self.df.loc[balanced_idx, 'Minimum Flow Deficit']
            ],
            'Spillage': [
                self.df.loc[min_regular_idx, 'Spillage'],
                self.df.loc[min_priority_idx, 'Spillage'],
                self.df.loc[min_minflow_idx, 'Spillage'],
                self.df.loc[min_spillage_idx, 'Spillage'],
                self.df.loc[balanced_idx, 'Spillage']
            ]
        })
        # Create formatted versions for display
        for col in ['Regular Demand Deficit', 'Priority Demand Deficit', 'Minimum Flow Deficit', 'Spillage']:
            representatives[f'{col} (km³/a)'] = representatives[col].map('{:,.3f}'.format)
        display_columns = [
            'Solution', 'Solution ID',
            'Regular Demand Deficit (km³/a)', 'Priority Demand Deficit (km³/a)',
            'Minimum Flow Deficit (km³/a)', 'Spillage (km³/a)'
        ]
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=display_columns,
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[representatives[col] for col in display_columns],
                fill_color='lavender',
                align='right',
                font=dict(size=11)
            )
        )])
        fig.update_layout(
            title="Representative Solutions (4 Objectives)",
            height=350,
            width=1200
        )
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Representative solutions table saved to {output_path}")
        # Also save as CSV
        csv_path = os.path.join(self.output_dir, filename.replace('.html', '.csv'))
        representatives.to_csv(csv_path, index=False)
        return representatives

    def generate_full_report(self):
        figures = {}
        figures['parallel'] = self.create_parallel_coordinates('parallel_coords_4d.html')
        figures['radar'] = self.create_radar_chart(filename='radar_chart_4d.html')
        figures['table'] = self.create_representative_table('representative_solutions_4d.html')
        self._create_index_page()
        return figures

    def _create_index_page(self):
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>4D Multi-objective Optimization Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                h1, h2 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin-bottom: 20px; }
                .card h2 { margin-top: 0; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .description { color: #666; margin-bottom: 15px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>4D Multi-objective Water System Optimization Results</h1>
                <div class="card">
                    <h2>Parallel Coordinates Plot</h2>
                    <p class="description">Visualize all four objectives and their trade-offs.</p>
                    <a href="parallel_coords_4d.html" target="_blank">Open Parallel Coordinates</a>
                </div>
                <div class="card">
                    <h2>Representative Solutions Comparison</h2>
                    <p class="description">View a radar chart comparing key representative solutions.</p>
                    <a href="radar_chart_4d.html" target="_blank">Open Radar Chart</a>
                </div>
                <div class="card">
                    <h2>Representative Solutions Table</h2>
                    <p class="description">See a detailed table of representative solutions.</p>
                    <a href="representative_solutions_4d.html" target="_blank">Open Solutions Table</a>
                </div>
            </div>
        </body>
        </html>
        """
        with open(os.path.join(self.output_dir, 'index_4d.html'), 'w') as f:
            f.write(html_content)
        print(f"4D index page created at {os.path.join(self.output_dir, 'index_4d.html')}")