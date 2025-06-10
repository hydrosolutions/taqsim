import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from itertools import combinations


class ParetoVisualizer:
    """
    Flexible Pareto dashboard for 2D, 3D, and higher-dimensional problems.
    """

    def __init__(self, pareto_solutions, objective_names=None, output_dir="./model_output/optimization/dashboard"):
        """
        Args:
            pareto_solutions (list): List of Pareto-optimal solutions (dict or DEAP individual)
            objective_names (list): List of objective names (optional, inferred if None)
            output_dir (str): Directory to save dashboard outputs
        """
        self.pareto_solutions = pareto_solutions
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df, self.objective_names = self._extract_solution_data(objective_names)
        self.n_obj = len(self.objective_names)

    def _extract_solution_data(self, objective_names):
        # Guess number of objectives and names
        if not self.pareto_solutions or not isinstance(self.pareto_solutions, list):
            print("No valid Pareto solutions provided")
            return pd.DataFrame(), []
        # Try to infer objectives from first solution
        first = self.pareto_solutions[0]
        if isinstance(first, dict):
            if objective_names is None:
                # Try to get keys that look like objectives
                keys = [k for k in first.keys() if k not in ('id', 'solution', 'Solution', 'objective_values')]
                if 'objective_values' in first:
                    n_obj = len(first['objective_values'])
                    names = [f"Objective {i+1}" for i in range(n_obj)]
                else:
                    names = keys
            else:
                names = objective_names
        else:
            # Assume DEAP individual
            if objective_names is None:
                n_obj = len(first.fitness.values)
                names = [f"Objective {i+1}" for i in range(n_obj)]
            else:
                names = objective_names

        # Extract data
        data = []
        for sol in self.pareto_solutions:
            if isinstance(sol, dict):
                solution_id = sol.get('id', 0)
                if 'objective_values' in sol:
                    obj_values = sol['objective_values']
                else:
                    obj_values = [sol.get(name, 0) for name in names]
            else:
                solution_id = getattr(sol, 'id', 0)
                obj_values = sol.fitness.values if hasattr(sol, 'fitness') else [0]*len(names)
            row = {'Solution': solution_id}
            for i, name in enumerate(names):
                row[name] = obj_values[i] if i < len(obj_values) else 0
            data.append(row)
        df = pd.DataFrame(data)
        # Normalized columns and composite score
        for name in names:
            max_val = df[name].max()
            df[f'Normalized {name}'] = df[name] / max_val if max_val != 0 else 0
        df['Composite Score'] = df[[f'Normalized {name}' for name in names]].mean(axis=1)
        return df, names
    
    def create_pareto_front(self, filename='pareto_dashboard.html'):
        if self.df.empty:
            print("No data available to create dashboard")
            return None
        
        elif self.n_obj == 3:
            return self._pareto_front_3d(filename)
        else:
            return self._pareto_front_nd(filename)
 
    def _pareto_front_3d(self, filename):
        x, y, z = self.objective_names
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=[
                '3D Pareto Front',
                f'{y} vs {x}',
                f'{y} vs {z}',
                f'{x} vs {z}'
            ]
        )
        # 3D scatter for all solutions
        fig.add_trace(go.Scatter3d(
            x=self.df[x], y=self.df[y], z=self.df[z],
            mode='markers',
            marker=dict(size=5, color='#35b779', opacity=0.8),
            text=[f'ID: {sid}' for sid in self.df['Solution']],
            hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>{z}: %{{z:.2e}}<br>',
            name='Pareto Solutions'
        ), row=1, col=1)

        # Highlight best for each objective in 3D
        colors = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
        for i, obj in enumerate(self.objective_names):
            idx = self.df[obj].idxmin()
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=[self.df.loc[idx, x]], y=[self.df.loc[idx, y]], z=[self.df.loc[idx, z]],
                mode='markers', marker=dict(size=12, color=color, symbol='circle'),
                text=[f'Best {obj} (ID: {self.df.loc[idx, "Solution"]})'],
                hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>{z}: %{{z:.2e}}<br>',
                name=f'Best {obj}'
            ), row=1, col=1)


        # 2D projections
        # y vs x
        fig.add_trace(go.Scatter(
            x=self.df[x], y=self.df[y], mode='markers',
            marker=dict(size=10, color='#35b779', showscale=False),
            text=[f'ID: {sid}' for sid in self.df['Solution']],
            hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>',
            name=f'{y} vs {x}'
        ), row=1, col=2)
        # y vs z
        fig.add_trace(go.Scatter(
            x=self.df[y], y=self.df[z], mode='markers',
            marker=dict(size=10, color='#35b779', showscale=False),
            text=[f'ID: {sid}' for sid in self.df['Solution']],
            hovertemplate=f'<b>%{{text}}</b><br>{y}: %{{x:.2e}}<br>{z}: %{{y:.2e}}<br>',
            name=f'{y} vs {z}'
        ), row=2, col=1)
        # x vs z
        fig.add_trace(go.Scatter(
            x=self.df[x], y=self.df[z], mode='markers',
            marker=dict(size=10, color='#35b779', showscale=False),
            text=[f'ID: {sid}' for sid in self.df['Solution']],
            hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{z}: %{{y:.2e}}<br>',
            name=f'{x} vs {z}'
        ), row=2, col=2)

        # Highlight best for each objective and balanced in 2D projections
        for i, obj in enumerate(self.objective_names):
            idx = self.df[obj].idxmin()
            color = colors[i % len(colors)]
            # y vs x
            fig.add_trace(go.Scatter(
                x=[self.df.loc[idx, x]], y=[self.df.loc[idx, y]],
                mode='markers', marker=dict(size=15, color=color),
                text=[f'Best {obj} (ID: {self.df.loc[idx, "Solution"]})'],
                hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>',
                name=f'Best {obj}'
            ), row=1, col=2)
            # y vs z
            fig.add_trace(go.Scatter(
                x=[self.df.loc[idx, y]], y=[self.df.loc[idx, z]],
                mode='markers', marker=dict(size=15, color=color),
                text=[f'Best {obj} (ID: {self.df.loc[idx, "Solution"]})'],
                hovertemplate=f'<b>%{{text}}</b><br>{y}: %{{x:.2e}}<br>{z}: %{{y:.2e}}<br>',
                name=f'Best {obj}'
            ), row=2, col=1)
            # x vs z
            fig.add_trace(go.Scatter(
                x=[self.df.loc[idx, x]], y=[self.df.loc[idx, z]],
                mode='markers', marker=dict(size=15, color=color),
                text=[f'Best {obj} (ID: {self.df.loc[idx, "Solution"]})'],
                hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{z}: %{{y:.2e}}<br>',
                name=f'Best {obj}'
            ), row=2, col=2)

        # Add axis labels to all subplots
        fig.update_scenes(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
            row=1, col=1
        )
        fig.update_xaxes(title_text=x, row=1, col=2)
        fig.update_yaxes(title_text=y, row=1, col=2)
        fig.update_xaxes(title_text=y, row=2, col=1)
        fig.update_yaxes(title_text=z, row=2, col=1)
        fig.update_xaxes(title_text=x, row=2, col=2)
        fig.update_yaxes(title_text=z, row=2, col=2)
        fig.update_layout(
            title='3D Pareto Dashboard',
            height=900, width=1200,
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            ),
            showlegend=False
        )
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"3D dashboard saved to {output_path}")
        return fig

    def _pareto_front_nd(self, filename):
        """
        For n > 3: plot all possible 2D Pareto fronts for all objective combinations.
        Each subplot shows a scatter plot for a unique pair of objectives.
        """
        dims = self.objective_names
        pairs = list(combinations(dims, 2))
        n_pairs = len(pairs)
        n_cols = 2 if n_pairs > 1 else 1
        n_rows = (n_pairs + n_cols - 1) // n_cols

        subplot_titles = [f'{y} vs {x}' for x, y in pairs]
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.15,
            vertical_spacing=0.15
        )

        for idx, (x, y) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            # Pareto front scatter
            fig.add_trace(go.Scatter(
                x=self.df[x], y=self.df[y],
                mode='markers',
                marker=dict(size=10, color='#35b779', showscale=False),
                text=[f'ID: {sid}' for sid in self.df['Solution']],
                hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>',
                name=f'{y} vs {x}'
            ), row=row, col=col)
            # Highlight best for each axis and balanced
            min_x_idx = self.df[x].idxmin()
            min_y_idx = self.df[y].idxmin()
            balanced_idx = self.df['Composite Score'].idxmin()
            fig.add_trace(go.Scatter(
                x=[self.df.loc[min_x_idx, x]], y=[self.df.loc[min_x_idx, y]],
                mode='markers', marker=dict(size=15, color='red'),
                text=[f'ID: {self.df.loc[min_x_idx, "Solution"]}'],
                hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>',
                name=f'Best {x}'
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[self.df.loc[min_y_idx, x]], y=[self.df.loc[min_y_idx, y]],
                mode='markers', marker=dict(size=15, color='blue'),
                text=[f'ID: {self.df.loc[min_y_idx, "Solution"]}'],
                hovertemplate=f'<b>%{{text}}</b><br>{x}: %{{x:.2e}}<br>{y}: %{{y:.2e}}<br>',
                name=f'Best {y}'
            ), row=row, col=col)

            fig.update_xaxes(title_text=x, row=row, col=col)
            fig.update_yaxes(title_text=y, row=row, col=col)

        fig.update_layout(
            title='All 2D Pareto Fronts for Objective Pairs',
            height=400 * n_rows,
            width=500 * n_cols,
            showlegend=False
        )
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"ND dashboard (all 2D Pareto fronts) saved to {output_path}")
        return fig

    def create_representative_table(self, filename='representative_solutions.html'):
        if self.df.empty:
            print("No data available to create representative table")
            return None
        idxs = [self.df[name].idxmin() for name in self.objective_names]
        balanced_idx = self.df['Composite Score'].idxmin()
        labels = [f'Best {name}' for name in self.objective_names] + ['Balanced']
        indices = idxs + [balanced_idx]
        representatives = self.df.loc[indices].copy()
        representatives.insert(0, 'Solution Type', labels)
        # Format for display
        for name in self.objective_names:
            representatives[f'{name}'] = representatives[name].map('{:,.3f}'.format)
        # Remove the Solution column from display
        display_columns = ['Solution Type'] + [f'{name}' for name in self.objective_names]
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
            title="Representative Solutions",
            height=350,
            width=1200
        )
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Representative solutions table saved to {output_path}")
        return representatives

    def create_parallel_coordinates_plot(self, filename='parallel_coordinates.html'):
        """
        Create a parallel coordinates plot for any number of objectives.

        Args:
            filename (str): Name of the output HTML file

        Returns:
            plotly.graph_objects.Figure: The parallel coordinates figure
        """
        if self.df.empty:
            print("No data available to create parallel coordinates plot")
            return None

        fig = px.parallel_coordinates(
            self.df,
            color="Composite Score",
            dimensions=self.objective_names,
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Parallel Coordinates Plot',
            
        )
        fig.update_layout(
            font=dict(size=12),
            height=600,
            width=1000
        )

        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        print(f"Parallel coordinates plot saved to {output_path}")

        return fig
    
    def generate_full_report(self):
        figs = {}
        if self.n_obj < 2:
            print("Not enough objectives for Pareto visualization. At least 2 objectives are required.")
            return None

        figs['pareto_front'] = self.create_pareto_front('pareto_front.html')
        figs['parallel_coordinates'] = self.create_parallel_coordinates_plot('parallel_coordinates.html')
        figs['table'] = self.create_representative_table('representative_solutions_table.html')

        return figs