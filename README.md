# Taqsim

Taqsim is a water-modelling layer over the incidence engine. It gives hydrologists a
water-named authoring surface for water system topology, declared time, operating rules,
readable run projections, saved runs, and parameter sweeps.

A `WaterSystem` declaration compiles the water vocabulary to incidence's model and rule IR.
Incidence owns execution, conservation, exhaustive disposition, presence, and exact
replay. Taqsim keeps model authoring and result reading in hydrology terms.

## Typed physical inputs

A source carries its Polars data, interval meaning, unit, cadence, declared data
resolution, and immutable original provenance. The model time axis and conservation
quantum remain separate declarations.

```python
import polars as pl
from datetime import datetime

from taqsim import ConservationQuantum, IntervalVolume, TimeAxis, WaterSystem

inflow = IntervalVolume(
    data=pl.DataFrame(
        {"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [100.0, 80.0]},
        schema={"time": pl.Datetime("us"), "value": pl.Float64},
    ),
    unit="m3",
    cadence="1d",
    data_resolution="0.1 m3",
)
system = WaterSystem(
    time=TimeAxis("2020-01-01", periods=2, frequency="1d"),
    quantum=ConservationQuantum.MILLILITRE,
)
system.source("river", inflow)
system.reach("canal", "river", "farm")
run = system.build().run(bytes(16))
```

Finer inputs must be prepared explicitly with `aggregate_to(system.time)`. Taqsim does
not convert, fill, interpolate, aggregate, or infer time inside `source()`, `build()`, or
`run()`.

Every physical rule scalar is also unit-aware. Use `WaterVolume` and `VolumetricRate`
for water amounts and rates, `WaterDepth` for evaporation depths, `SurfaceArea` for
reservoir areas, `Length` for canal length and width, and
`CanalSeepageCoefficient(..., "sqrt(m3/s)/km")` for the square-root seepage law.
Dimensionless fractions remain numerical. Wrap optimizable physical values as, for
example, `WaterDepth(Parameter("depth", 5.0, (0.0, 10.0)), "mm")`; compiled bounds,
substitutions, sweeps, and solutions retain a compatible named quantity.

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
