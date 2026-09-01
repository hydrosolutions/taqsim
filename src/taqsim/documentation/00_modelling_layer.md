# Taqsim modelling layer

Taqsim names water systems, reaches, rules, declared time, physical interval inputs,
and run projections. Polars-backed sources keep their volume or rate meaning, units,
cadence, data resolution, and source provenance attached. Explicit preparation aligns
inputs to a `TimeAxis`; model construction never resamples or guesses. Taqsim compiles
validated declarations for the incidence engine, which owns execution and exact replay.


Physical rule scalars remain named quantities at the public boundary. Water amounts,
rates, evaporation depths, surface areas, canal lengths and widths, and canal seepage
coefficients each declare compatible units. Physical optimization parameters retain
those descriptors through bounds, substitutions, sweeps, and solutions.
