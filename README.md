# Taqsim

Taqsim is a water-modelling layer over the incidence engine. It gives hydrologists a
water-named authoring surface for basin topology, declared time, operating rules,
readable run projections, saved runs, and parameter sweeps.

A `Basin` declaration compiles the water vocabulary to incidence's model and rule IR.
Incidence owns execution, conservation, exhaustive disposition, presence, and exact
replay. Taqsim keeps model authoring and result reading in hydrology terms.

## Maintenance Status

🟢 **Active Development**

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
uv run ruff check
uv run ty check
```

See `AGENTS.md` for project design rules.
