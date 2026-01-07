---
name: marimo
description: Create reactive marimo notebooks for data analysis and visualization. Use when the user asks to create a marimo notebook, work with reactive notebooks, or build interactive data visualizations. User-invocable with /marimo command.
user_invocable: /marimo
---

# Marimo Notebooks

Marimo is a reactive notebook framework. Key differences from Jupyter:

- **Reactive**: Cells auto-execute when dependencies change
- **No redeclaration**: Variables cannot be redeclared across cells
- **DAG structure**: Notebook forms a directed acyclic graph
- **Auto-display**: Last expression in a cell is displayed automatically

## Cell Structure

```python
@app.cell
def _():
    # your code here
    return
```

Marimo handles function parameters and return statements based on variable dependencies.

## Code Requirements

1. All code must be complete and runnable
2. Import all modules in the first cell, always including `import marimo as mo`
3. Never redeclare variables across cells
4. Ensure no cycles in the dependency graph
5. Never use `global` definitions
6. No comments in markdown or SQL cells

## Reactivity Rules

- UI element values accessed via `.value` (e.g., `slider.value`)
- UI values cannot be accessed in the same cell where defined
- Underscore-prefixed variables (e.g., `_my_var`) are cell-local

## Quick Reference

| Task | Pattern |
|------|---------|
| Markdown | `mo.md("# Title")` |
| Matplotlib | Use `plt.gca()` as last expression |
| Plotly/Altair | Return figure/chart object directly |
| SQL (DuckDB) | `df = mo.sql(f"SELECT * FROM table")` |
| Layout | `mo.hstack([a, b])`, `mo.vstack([a, b])`, `mo.tabs({...})` |
| Stop execution | `mo.stop(condition, output)` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Circular dependencies | Reorganize code to remove cycles |
| UI value access error | Move value access to separate cell from definition |
| Viz not showing | Ensure viz object is last expression |

Run `marimo check --fix` to catch and fix common issues.

## Creating a Notebook

Use `scripts/create_notebook.py` to generate a starter notebook:

```bash
uv run scripts/create_notebook.py my_analysis -o notebooks/
```

## Resources

- **UI Elements**: See [references/ui-elements.md](references/ui-elements.md) for complete element signatures and patterns
