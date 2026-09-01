# Taqsim modelling layer

Taqsim names water systems, reaches, rules, declared time, physical interval inputs,
and run projections. Polars-backed sources keep their volume or rate meaning, units,
cadence, data resolution, and source provenance attached. Explicit preparation aligns
inputs to a `TimeAxis`; model construction never resamples or guesses. Taqsim compiles
validated declarations for the incidence engine, which owns execution and exact replay.
