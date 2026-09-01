# Taqsim

**Status: 🟢 Active** — Ongoing project — active development.

Taqsim is a Python water-modelling and rule-authoring layer over the
[incidence](https://github.com/CooperBigFoot/incidence) engine. Taqsim names water systems,
physical inputs, topology, and hydrology rules. It compiles them to incidence, which owns
execution semantics: conservation, exhaustive disposition, presence, and exact replay.
Taqsim is a library. It has no CLI or configuration-file workflow.

## Model and run a water system

Declare the real model time and conservation quantum, add typed sources, sinks, and reaches,
then build and run the model. The built model and each run are immutable, reusable values.

<!-- readme-example:start -->
```python
from datetime import datetime

import polars as pl

from taqsim import ConservationQuantum, IntervalVolume, TimeAxis, WaterSystem

inflow = IntervalVolume(
    data=pl.DataFrame(
        {
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "value": [100.0, 80.0],
        },
        schema={"time": pl.Datetime("us"), "value": pl.Float64},
    ),
    unit="m3",
    cadence="1d",
    data_resolution="1 m3",
)

system = WaterSystem(
    name="example-basin",
    time=TimeAxis("2020-01-01", periods=2, frequency="1d"),
    quantum=ConservationQuantum.CUBIC_METRE,
)
system.source("river", inflow)
system.sink("farm")
system.reach("canal", "river", "farm")

model = system.build()
run = model.run(bytes(16))
assert tuple(run.flow("canal").values) == (100.0, 80.0)
assert tuple(run.arrivals("farm").values) == (100.0, 80.0)
```
<!-- readme-example:end -->

The public surface also supports serialisable hydrology rule vocabulary, parameter sweeps,
water-system optimization, and version-checked saved-run caches. A loaded cache supports stored
flow and retained-stock projections. Arrival projections require a live run.

## Invariants that affect changes

- Source data carries interval meaning, units, cadence, resolution, and original provenance.
  Prepare conversions or aggregation explicitly before `build()`; model construction does
  not infer or resample inputs.
- Physical rule scalars use unit-aware types. External source volumes and initial stocks must
  be exactly representable at the declared `ConservationQuantum`.
- Rules are serialisable data compiled to incidence rule IR, not Python callbacks.
- A capacity-limited reach must name its overflow destination. Water never disappears from
  the closed world.
- Results distinguish present zeroes, absent values, and times outside the modelled horizon.

See [`AGENTS.md`](AGENTS.md) for coding doctrine. Treat the public exports in
[`src/taqsim/__init__.py`](src/taqsim/__init__.py), their implementations in
[`src/taqsim/`](src/taqsim/), and the contract tests in [`tests/`](tests/) as the detailed
source of truth.

## Development

The project requires Python 3.12 or later and uses
[uv](https://docs.astral.sh/uv/) exclusively.

```bash
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
uv build
```

Run `uv run ruff format` to apply formatting.
