# Marimo Notebook Guide

Marimo is a reactive notebook framework for creating clear, efficient, and reproducible data analysis workflows.

## Fundamentals

Marimo differs from traditional notebooks (like Jupyter) in key ways:

- **Reactive execution**: Cells execute automatically when their dependencies change
- **No variable redeclaration**: Variables cannot be redeclared across cells
- **DAG structure**: The notebook forms a directed acyclic graph
- **Auto-display**: The last expression in a cell is automatically displayed
- **Reactive UI**: UI elements update the notebook automatically without callbacks

## Cell Structure

Each cell is wrapped in a function decorator:

```python
@app.cell
def _():
    # your code here
    return
```

Marimo automatically handles the function parameters and return statement based on variable dependencies.

## Code Requirements

1. All code must be complete and runnable
2. Import all modules in the first cell, always including `import marimo as mo`
3. Never redeclare variables across cells
4. Ensure no cycles in the dependency graph
5. Never use `global` definitions
6. No comments in markdown or SQL cells

## Reactivity Model

- When a variable changes, all dependent cells automatically re-execute
- UI element values are accessed via `.value` attribute (e.g., `slider.value`)
- UI element values cannot be accessed in the same cell where they're defined
- Variables prefixed with underscore (e.g., `_my_var`) are cell-local and inaccessible from other cells

## Best Practices

### Data Handling

- Use polars for data manipulation
- Implement proper data validation
- Handle missing values appropriately
- Use efficient data structures
- Variables in the last expression are automatically displayed as tables

### Visualization

| Library    | Pattern                                    |
|------------|-------------------------------------------|
| Matplotlib | Use `plt.gca()` as last expression (not `plt.show()`) |
| Plotly     | Return the figure object directly          |
| Altair     | Return the chart object directly; polars DataFrames work directly |

Always include proper labels, titles, and color schemes. Make visualizations interactive where appropriate.

### UI Elements

- Create UI elements in one cell, reference them in later cells
- Use `mo.hstack()`, `mo.vstack()`, and `mo.tabs()` for layouts
- Prefer reactive updates over callbacks
- Group related UI elements together

### SQL (DuckDB)

Use marimo's SQL cells:

```python
df = mo.sql(f"""SELECT * FROM my_table""")
# Or with custom engine:
df = mo.sql(f"""SELECT * FROM my_table""", engine=engine)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Circular dependencies | Reorganize code to remove cycles in the dependency graph |
| UI element value access error | Move value access to a separate cell from definition |
| Visualization not showing | Ensure the visualization object is the last expression |

Run `marimo check --fix` to catch and automatically resolve common formatting issues.

## UI Elements Reference

### Input Elements

| Element | Signature |
|---------|-----------|
| Button | `mo.ui.button(value=None, kind='primary')` |
| Run Button | `mo.ui.run_button(label=None, tooltip=None, kind='primary')` |
| Checkbox | `mo.ui.checkbox(label='', value=False)` |
| Date | `mo.ui.date(value=None, label=None, full_width=False)` |
| Dropdown | `mo.ui.dropdown(options, value=None, label=None, full_width=False)` |
| File | `mo.ui.file(label='', multiple=False, full_width=False)` |
| Number | `mo.ui.number(value=None, label=None, full_width=False)` |
| Radio | `mo.ui.radio(options, value=None, label=None, full_width=False)` |
| Slider | `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)` |
| Range Slider | `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)` |
| Text | `mo.ui.text(value='', label=None, full_width=False)` |
| Text Area | `mo.ui.text_area(value='', label=None, full_width=False)` |

### Data Elements

| Element | Signature |
|---------|-----------|
| Table | `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)` |
| Data Explorer | `mo.ui.data_explorer(df)` |
| DataFrame | `mo.ui.dataframe(df)` |

### Chart Elements

| Element | Signature |
|---------|-----------|
| Altair Chart | `mo.ui.altair_chart(altair_chart)` |
| Plotly | `mo.ui.plotly(plotly_figure)` |

### Container Elements

| Element | Signature |
|---------|-----------|
| Tabs | `mo.ui.tabs(elements: dict[str, mo.ui.Element])` |
| Array | `mo.ui.array(elements: list[mo.ui.Element])` |
| Form | `mo.ui.form(element: mo.ui.Element, label='', bordered=True)` |
| Refresh | `mo.ui.refresh(options: list[str], default_interval: str)` |

## Layout & Utility Functions

| Function | Description |
|----------|-------------|
| `mo.md(text)` | Display markdown |
| `mo.stop(predicate, output=None)` | Stop execution conditionally |
| `mo.output.append(value)` | Append to output (when not last expression) |
| `mo.output.replace(value)` | Replace output (when not last expression) |
| `mo.Html(html)` | Display HTML |
| `mo.image(image)` | Display an image |
| `mo.hstack(elements)` | Stack elements horizontally |
| `mo.vstack(elements)` | Stack elements vertically |
| `mo.tabs(elements)` | Create a tabbed interface |

## Examples

### Markdown Cell

```python
@app.cell
def _():
    mo.md("""
    # Hello world
    This is a _markdown_ **cell**.
    """)
    return
```

### Basic UI with Reactivity

```python
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import numpy as np
    return

@app.cell
def _():
    n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
    n_points
    return

@app.cell
def _():
    x = np.random.rand(n_points.value)
    y = np.random.rand(n_points.value)

    df = pl.DataFrame({"x": x, "y": y})

    chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X('x', title='X axis'),
        y=alt.Y('y', title='Y axis')
    ).properties(
        title=f"Scatter plot with {n_points.value} points",
        width=400,
        height=300
    )

    chart
    return
```

### Data Explorer

```python
@app.cell
def _():
    import marimo as mo
    import polars as pl
    from vega_datasets import data
    return

@app.cell
def _():
    cars_df = pl.DataFrame(data.cars())
    mo.ui.data_explorer(cars_df)
    return
```

### Multiple UI Elements

```python
@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return

@app.cell
def _():
    iris = pl.read_csv("hf://datasets/scikit-learn/iris/Iris.csv")
    return

@app.cell
def _():
    species_selector = mo.ui.dropdown(
        options=["All"] + iris["Species"].unique().to_list(),
        value="All",
        label="Species",
    )
    x_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalLengthCm",
        label="X Feature",
    )
    y_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalWidthCm",
        label="Y Feature",
    )
    mo.hstack([species_selector, x_feature, y_feature])
    return

@app.cell
def _():
    filtered_data = iris if species_selector.value == "All" else iris.filter(pl.col("Species") == species_selector.value)

    chart = alt.Chart(filtered_data).mark_circle().encode(
        x=alt.X(x_feature.value, title=x_feature.value),
        y=alt.Y(y_feature.value, title=y_feature.value),
        color='Species'
    ).properties(
        title=f"{y_feature.value} vs {x_feature.value}",
        width=500,
        height=400
    )

    chart
    return
```

### Conditional Outputs

```python
@app.cell
def _():
    mo.stop(not data.value, mo.md("No data to display"))

    if mode.value == "scatter":
        mo.output.replace(render_scatter(data.value))
    else:
        mo.output.replace(render_bar_chart(data.value))
    return
```

### Interactive Chart with Altair

```python
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    return

@app.cell
def _():
    weather = pl.read_csv("https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv")
    weather_dates = weather.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )
    _chart = (
        alt.Chart(weather_dates)
        .mark_point()
        .encode(
            x="date:T",
            y="temp_max",
            color="location",
        )
    )
    return

@app.cell
def _():
    chart = mo.ui.altair_chart(_chart)
    chart
    return

@app.cell
def _():
    chart.value
    return
```

### Run Button Example

```python
@app.cell
def _():
    import marimo as mo
    return

@app.cell
def _():
    first_button = mo.ui.run_button(label="Option 1")
    second_button = mo.ui.run_button(label="Option 2")
    [first_button, second_button]
    return

@app.cell
def _():
    if first_button.value:
        print("You chose option 1!")
    elif second_button.value:
        print("You chose option 2!")
    else:
        print("Click a button!")
    return
```

### SQL with DuckDB

```python
@app.cell
def _():
    import marimo as mo
    import polars as pl
    return

@app.cell
def _():
    weather = pl.read_csv('https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv')
    return

@app.cell
def _():
    seattle_weather_df = mo.sql(
        f"""
        SELECT * FROM weather WHERE location = 'Seattle';
        """
    )
    return
```
