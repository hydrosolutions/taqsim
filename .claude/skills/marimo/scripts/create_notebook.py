#!/usr/bin/env python3
"""Generate a starter marimo notebook with proper structure."""

import argparse
import sys
from pathlib import Path

NOTEBOOK_TEMPLATE = '''import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return alt, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # {title}

    {description}
    """)
    return


@app.cell
def _(pl):
    # Load or create your data here
    df = pl.DataFrame({{
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 1, 5, 3],
    }})
    df
    return (df,)


@app.cell
def _(alt, df):
    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x="x",
            y="y",
        )
        .properties(width=400, height=300)
    )
    chart
    return (chart,)


if __name__ == "__main__":
    app.run()
'''


def create_notebook(name: str, output_dir: Path, description: str = "") -> Path:
    """Create a new marimo notebook file."""
    filename = f"{name}.py" if not name.endswith(".py") else name
    output_path = output_dir / filename

    title = name.replace("_", " ").replace("-", " ").title()
    if title.endswith(".Py"):
        title = title[:-3]

    content = NOTEBOOK_TEMPLATE.format(
        title=title,
        description=description or "A reactive marimo notebook.",
    )

    output_path.write_text(content)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new marimo notebook")
    parser.add_argument("name", help="Name of the notebook (without .py extension)")
    parser.add_argument(
        "-o", "--output", default=".", help="Output directory (default: current)"
    )
    parser.add_argument(
        "-d", "--description", default="", help="Notebook description for the header"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_path = create_notebook(args.name, output_dir, args.description)
    print(f"Created: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
