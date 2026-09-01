# WaterSystem physical and temporal inputs

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/15

## Outcome

Taqsim accepts water inputs only after their physical and temporal meaning is explicit. A user can see the dates, cadence, units, interval semantics, and declared source-data resolution on the input object. Taqsim can then perform an explicitly requested, mechanically determined unit conversion, rate integration, and aggregation into the model timestep. It never guesses, interpolates, or silently changes data during model construction or execution.

The public declaration root is `WaterSystem`, not `Basin`. This is a direct breaking change. There are no compatibility obligations or existing users to preserve.

## Public boundary

### Polars data bound to types that cannot lie

Polars is the canonical numerical carrier for one-dimensional, time-indexed input. A bare `polars.DataFrame` cannot retain physical units, cadence, resolution, or rate-versus-volume meaning, so Taqsim binds the frame to that metadata in a named domain type.

There are distinct public input types for the two physical meanings:

- `IntervalVolume`: each value is the total water volume over its interval;
- `IntervalMeanRate`: each value is the mean volumetric rate over its interval.

Each contains a strict Polars frame with `time` and `value` columns and carries:

- ordered, unique timestamps;
- an explicitly declared cadence;
- a physical unit compatible with its named meaning;
- `data_resolution`, the declared resolution of the values in that object, compatible with its named physical meaning;
- immutable source provenance containing the original input kind, unit, cadence, and declared data resolution.

On a directly constructed input, `data_resolution` and the source provenance describe the same declaration. A transformation changes `data_resolution` to describe its returned values while retaining the original declaration in `source_provenance`.

A timestamp names the interval beginning at that timestamp. The cadence determines its end. A daily value stamped `2026-01-01` covers the calendar day beginning on `2026-01-01`. Taqsim owns the mechanical representation needed to make that convention exact; users do not have to author interval-bound tables for ordinary regular series.

Inputs are complete, finite, and non-negative. Missing values, duplicate or unordered timestamps, gaps, overlaps, ambiguous timestamps, unexpected columns or schemas, and unsupported calendars are refused rather than filled or interpreted.

The normal Gregorian calendar is in scope. Taqsim normalizes ordinary timestamps consistently and refuses timezone ambiguity. Non-Gregorian scientific calendars are outside this Effort.

The expected public shape is:

```python
import polars as pl

from taqsim import (
    ConservationQuantum,
    IntervalVolume,
    TimeAxis,
    WaterSystem,
)

system = WaterSystem(
    time=TimeAxis(start="2020-01-01", periods=2, frequency="1d"),
    quantum=ConservationQuantum.MILLILITRE,
)

river_inflow = IntervalVolume(
    data=pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 2),
                interval="1d",
                eager=True,
            ),
            "value": [100.0, 80.0],
        }
    ),
    unit="m3",
    cadence="1d",
    data_resolution="0.1 m3",
)

system.source("river", river_inflow)
system.sink("farm")
system.reach("canal", "river", "farm")

model = system.build()
run = model.run(bytes(16))
received = run.arrivals("farm")
```

The domain object keeps the Polars data attached to its meaning whenever it is passed between functions. Callers do not repeat units or semantics as separate `source()` arguments.

### Data cadence is not model time

A `WaterSystem` declares its own `TimeAxis`. That axis remains authoritative and is not inferred from any source. Each input declares its own cadence independently. Multiple inputs may begin with different finer cadences, but every `IntervalVolume` passed to `WaterSystem.source()` must already cover and align with the model intervals exactly.

### Explicit conversion and aggregation

`IntervalMeanRate` and finer-cadence `IntervalVolume` inputs provide an explicit operation, illustrated as `aggregate_to`, for preparation against a model axis:

```python
from taqsim import IntervalMeanRate

hourly_inflow = IntervalMeanRate(
    data=pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 2, 23),
                interval="1h",
                eager=True,
            ),
            "value": hourly_values,
        }
    ),
    unit="m3/s",
    cadence="1h",
    data_resolution="0.01 m3/s",
)

daily_inflow = hourly_inflow.aggregate_to(system.time)
system.source("river", daily_inflow)
```

This explicit transformation may:

1. convert supported compatible units;
2. integrate an interval-mean rate over its declared source interval;
3. sum finer interval volumes when those intervals exactly partition each target model interval;
4. mechanically transform `data_resolution` through the same unit conversion and rate integration, while summation on one resolution grid retains that grid resolution;
5. return an aligned `IntervalVolume` with its Polars data, current meaning, and immutable source provenance still attached.

For example, 24 hourly values with a mean rate of `1 m3/s` and `data_resolution="0.01 m3/s"` over one complete day aggregate to one daily interval volume of `86,400 m3`. If the returned volume unit is `m3`, its `data_resolution` is `36 m3`: the original rate resolution integrated over one hour. Its `source_provenance.data_resolution` remains `0.01 m3/s`.

The operation refuses gaps, overlaps, partial target intervals, or any requested transformation that would require interpolation or invented data. It does not disaggregate coarse data into finer model intervals. No conversion or aggregation occurs implicitly in `source()`, `build()`, or `run()`; passing unaligned hourly data directly to a daily system fails.

### Pragmatic physical units

The public boundary supports a pragmatic broad range of common SI and water-modelling volume and volumetric-rate units rather than one hard-coded spelling. Expected examples include cubic metres and their common SI scales, litres and their common SI scales, and those volumes per second, minute, hour, or day. Inputs such as `m3`, `L`, `mL`, `ML`, `m3/s`, `L/s`, and `m3/day` are ordinary cases.

The accepted set need not claim every unit system or spelling. Unknown units, ambiguous units, and dimensions other than water volume or volumetric rate fail clearly. Geographic conversions such as turning a water depth into a volume from a raster or catchment area remain outside this Effort.

### Source-data resolution is not conservation quantum

`data_resolution` records the declared resolution of the values in the current object and has a unit compatible with that object's physical meaning. On a direct input this is the resolution claimed by the source. Taqsim trusts the declaration rather than inferring it from observed values.

An explicit transformation propagates the current resolution mechanically. Unit conversion scales it. Rate integration multiplies it by the exact source-interval duration. Summing values expressed on the same resolution grid retains that grid resolution because a sum of grid multiples remains on that grid. The returned `IntervalVolume.data_resolution` is therefore volume-compatible. Its immutable `source_provenance` separately retains the original kind, unit, cadence, and data resolution, including a rate resolution when the source was an `IntervalMeanRate`. Both current resolution and original provenance remain inspectable after transformation.

Neither resolution field selects the model's arithmetic representation. The model separately requires an explicit `ConservationQuantum`, retaining the four choices established by Effort #14: `1 m3`, `1 L`, `1 mL`, and `1 mm3`. Unit conversion and aggregation transform physical values, but do not derive or silently alter this quantum. Once an input has been prepared for the model axis, Effort #14's exact representability and count-ceiling rules still apply. A transformed interval volume that cannot be represented by the selected conservation quantum is refused.

### Scalar physical inputs

The same rule applies to scalar inputs: anything with physical meaning carries explicit units and enough type information to distinguish an amount from a rate. Initial water, reach capacity, and similar public declarations cannot remain ambiguous bare floats or aliases whose names are the only unit contract. Time-indexed data uses the typed Polars-backed inputs; scalar data uses an appropriate unit-aware scalar quantity. The exact scalar quantity implementation is left to the implementing agent.

## Public vocabulary

The `Basin` term leaves the public API consistently:

| Retired | Replacement |
|---|---|
| `Basin` | `WaterSystem` |
| `BuiltBasin` | `BuiltWaterSystem` |
| `BasinRun` | `WaterSystemRun` |
| `BasinObjective` | `WaterSystemObjective` |
| `BasinSolution` | `WaterSystemSolution` |
| `BasinOptimizeResult` | `WaterSystemOptimizeResult` |
| `optimize_basin()` | `optimize_water_system()` |
| `Resolution` / `resolution=` | `ConservationQuantum` / `quantum=` |

Module paths, public documentation, persistence terminology, error messages, tests, and examples follow the new vocabulary. No deprecated aliases preserve `Basin` or the ambiguous `Resolution` name.

The existing topology and execution flow remains recognizable: users still declare sources, sinks, and reaches, then call `build()`, `run()`, and projections such as `arrivals()`. Output-carrier redesign is not part of this Effort.

## Polars ownership

Public examples and public input APIs use Polars. Callers do not import pandas, pass pandas objects, or use xarray. Polars becomes a runtime dependency because Taqsim owns Polars-backed input values. No new pandas or xarray dependency is introduced for this boundary; an existing direct runtime dependency that has no remaining owner should not be retained merely for compatibility.

This choice is deliberately narrower than a multidimensional scientific-data boundary. Geographic rasters and time-by-location or time-by-depth arrays are outside the Effort. If those carriers become real requirements, they can be designed from those requirements rather than imposed on one-dimensional inputs now.

## Observable success

A delivered implementation demonstrates all of the following through its public API:

- A daily `IntervalVolume` backed by a strict Polars frame can enter a matching `WaterSystem`, build, run, and project results.
- A finer-cadence `IntervalMeanRate` in a compatible common unit can be explicitly integrated, converted, and aggregated to the model axis. A constant `1 m3/s` over a complete day produces `86,400 m3` for that day.
- A finer-cadence `IntervalVolume` aggregates by exact interval membership and summation.
- Source-data resolution remains inspectable and distinct from the model's conservation quantum before and after transformation.
- Two otherwise identical models with different conservation quanta retain the identity and exactness behavior established by #14; source-data resolution does not choose either quantum.
- Unaligned direct input, missing values, temporal gaps or overlaps, partial aggregation windows, disaggregation, incompatible dimensions, unknown units, ambiguous timestamps, and invalid Polars schemas fail with errors that identify the input and reason.
- Scalar water amounts and rates cannot cross the public boundary as unit-ambiguous floats.
- Public examples and input paths require Polars, not pandas or xarray.
- Importing or using the retired `Basin` family and `Resolution` vocabulary fails; the `WaterSystem` family is the sole public vocabulary.
- The existing build, run, persistence, sweep, optimization, projection-presence, and exact-conservation behavior remains intact under the renamed and typed boundary.

## Boundaries

This Effort does not include:

- geographic raster or catchment-area modelling;
- multidimensional scientific array modelling;
- implicit resampling, interpolation, filling, or temporal inference;
- disaggregating coarse observations into finer model intervals;
- non-Gregorian scientific calendars;
- redesigning output carriers;
- compatibility with `Basin` or the old bare-sequence source API;
- changing the exact-conservation arithmetic or quantum choices delivered by #14;
- redesigning fishy.

The implementing agent may choose the unit library, internal validation structure, scalar quantity implementation, and file layout. Those mechanisms must preserve the public behavior and visible distinctions above.
