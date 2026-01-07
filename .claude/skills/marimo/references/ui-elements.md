# Marimo UI Elements Reference

## Input Elements

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

## Data Elements

| Element | Signature |
|---------|-----------|
| Table | `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)` |
| Data Explorer | `mo.ui.data_explorer(df)` |
| DataFrame | `mo.ui.dataframe(df)` |

## Chart Elements

| Element | Signature |
|---------|-----------|
| Altair Chart | `mo.ui.altair_chart(altair_chart)` |
| Plotly | `mo.ui.plotly(plotly_figure)` |

## Container Elements

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
| `mo.output.append(value)` | Append to output |
| `mo.output.replace(value)` | Replace output |
| `mo.Html(html)` | Display HTML |
| `mo.image(image)` | Display an image |
| `mo.hstack(elements)` | Stack elements horizontally |
| `mo.vstack(elements)` | Stack elements vertically |
| `mo.tabs(elements)` | Create a tabbed interface |
| `mo.sql(query, engine=None)` | Execute SQL via DuckDB |

## Common Patterns

### Interactive slider with chart

```python
@app.cell
def _(mo):
    n_points = mo.ui.slider(10, 100, value=50, label="Points")
    n_points
    return (n_points,)

@app.cell
def _(alt, n_points, np, pl):
    df = pl.DataFrame({"x": np.random.rand(n_points.value), "y": np.random.rand(n_points.value)})
    alt.Chart(df).mark_circle().encode(x="x", y="y")
    return
```

### Multiple dropdowns with filtering

```python
@app.cell
def _(mo, df):
    species = mo.ui.dropdown(options=["All"] + df["species"].unique().to_list(), value="All", label="Species")
    x_col = mo.ui.dropdown(options=df.columns, value=df.columns[0], label="X")
    mo.hstack([species, x_col])
    return (species, x_col)

@app.cell
def _(alt, df, species, x_col):
    filtered = df if species.value == "All" else df.filter(pl.col("species") == species.value)
    alt.Chart(filtered).mark_point().encode(x=x_col.value, y="y", color="species")
    return
```

### Run button for expensive operations

```python
@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run Analysis")
    run_btn
    return (run_btn,)

@app.cell
def _(run_btn):
    mo.stop(not run_btn.value, mo.md("Click to run"))
    # expensive computation here
    return
```
