# Exact conservation by declared quantum

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/14

## Outcome

A taqsim model that builds can run, sweep, optimize, and replay at ordinary non-round parameter values without incidence refusing one-ULP overdrafts or bit-level changes in conserved totals. Conservation remains strict. The system fixes the representation instead of admitting a tolerance: water that exists in conserved state is an integer count of a declared quantum.

This effort spans taqsim and incidence. Taqsim owns the water-specific declaration and error context. Incidence owns the general conserved-state invariant and execution arithmetic.

## Why this is needed

Taqsim compiles rule shapes into incidence expression partitions. Their hydrology is algebraically exhaustive, but `binary64` evaluation is not closed under splitting and recombination. A sum such as `0.1 + 0.2` need not have the same bits as `0.3`. Depending on the expression and stock magnitudes, incidence therefore sees either a tiny negative residual or a one-ULP movement in the conserved total and refuses the run.

The defect affects all six shipped rule shapes. The evidence on #14 records refusal rates from 8% to 47% across seeded in-range parameter draws. A 75-case `CanalLosses` grid refused 45 cases, and an optimizer over `PriorityDistribution` failed in its first generation. Round-valued fixtures hide the defect, so success cannot be established by keeping only the current examples.

## Settled representation

### One quantum per conserved substance

An incidence model document declares a positive quantum for each conserved substance. The quantum is expressed in that substance's declared unit and participates in canonical model identity. Initial stocks, committed transfers, retained stock, replay, and conservation checks operate through exact integer quantum counts. Any model whose conserved total exceeds the exactly countable `2^53` ceiling is refused before execution.

Incidence is authoritative for these invariants even when a model document did not originate in taqsim. Taqsim may reject earlier when it can provide water-specific context.

### Required water resolution in taqsim

For this contained effort, taqsim requires one explicit water resolution from the already established closed set:

- `1 mm3`
- `1 mL`
- `1 L`
- `1 m3`

There is no default. The resolution is compiled into incidence's water quantum and reads back from runs and saved runs.

Source values and initial stocks are external extensive values. They must already be exact multiples of the declared resolution. Taqsim refuses a misaligned value instead of flooring it and identifies the source or initial-stock carrier, its timestep or position, the value, and the quantum. Incidence independently refuses any non-representable conserved amount at its boundary.

Parameters, rates, thresholds, and intensive intermediate calculations remain floating point. Only extensive amounts entering conserved state are quantum-counted.

### Computed branches floor; represented residual remains water

A computed extensive branch amount is floored to a whole quantum before ledger commitment. The error of each independently floored branch is strictly less than one quantum.

After flooring all computed branches, incidence subtracts their integer counts from the available integer count. Any residual whole quanta remain conserved stock in the source compartment and carry into the next timestep. They are not silently assigned to a destination, discarded, or placed in a synthetic loss account.

For an algebraically exhaustive partition with `N` independently floored computed branches, the represented residual can be as large as, but remains strictly below, `N` quanta. The retired workflow's criterion requiring the aggregate residual to be strictly below one quantum per timestep was false. The corrected invariant is exact conservation of all represented counts, a per-branch flooring error below one quantum, and retention of every residual whole quantum.

A sub-quantum fraction produced by floating-point rule arithmetic never enters conserved state. It is not stored, accumulated, or carried as hidden fractional state.

### Observation remains honest

A modelled branch whose computed amount is below one quantum is present with represented value `0.0`. It remains distinguishable from an absent timestep and from a substance or series that was never modelled. The run exposes its declared resolution so the zero's meaning is inspectable.

Do not add a fractional-remainder field or a hidden remainder account. Whole-quantum partition residual is already observable as retained compartment stock through the normal conserved-state model.

## Breaking changes are direct

The project is in active development with no users and no compatibility obligation. Update the model document, model identity, saved-run representation, fixtures, scripts, and public construction calls directly. Do not add a legacy loader, inferred resolution, payload migration, compatibility alias, or fallback semantics.

The broader public input redesign is not part of this effort. Effort #15 owns the `Basin` to `WaterSystem` rename, `xarray.DataArray` inputs, source-data value resolution, data cadence, model timestep, and explicit interval-rate semantics. #14 should leave a strict and coherent quantum boundary for #15 rather than implementing that larger API now.

## Observable evidence of success

The implementation must establish the general property, not only repair one rule shape or one decimal example.

- Every shipped rule shape completes across deterministic, seeded, in-range non-round parameter samples without an invalid-amount or conserved-total refusal.
- The reported `CanalLosses(0.0014, 12.0)` reproduction with flows `[10000, 8000]` completes at a declared resolution. Arrivals, named transfers, and retained whole-quantum stock account exactly for all represented input at every timestep.
- All 75 cases in the recorded `CanalLosses` grid complete.
- The `PriorityDistribution` optimizer proceeds through its first generation and returns a result rather than failing on conservation.
- The seepage-law gate ranges over parameters. Each recorded branch value differs from its floating-point hydraulic target by less than one declared quantum.
- A misaligned source value and a misaligned initial stock are refused before execution with precise location and quantum information.
- A multi-branch case proves that residual whole quanta remain in the source compartment, conserve exactly, and participate in the following timestep.
- A computed result below one quantum reads as present `0.0`, while absent and not-modelled states remain distinct.
- Repeated execution produces identical authoritative log digests.
- The quantum changes model identity, and saved-run identity and resolution readback match the model that produced the run.

Property construction is the proof: committed debits, credits, and retained values are integer counts. Seeded parameter scans are regression evidence that every shipped compiler path uses that construction; they are not a claim to enumerate all real parameters.

## Existing implementation evidence

The retired graph workflow stopped and must not be resumed, but it produced work that a new implementer should inspect and independently verify.

### incidence

Remote commit `0e4283abc3acc1b08d7968b32aaff58568dcf1d4` contains the completed quantum declaration, exact quantum-count transfer arithmetic, residual handling, ceiling checks, and gate repairs from the old `IQ1` and `IQ2` packages. It is published under the old conservation branch namespace in `CooperBigFoot/incidence`.

### taqsim

Local commit `9f3e2b46dc43c492b02c536c2a12ebdd19babe0d` contains the completed required-resolution work and the parameter-ranging seepage gate from the old `TQ1` and `TQ3` packages. It is reachable through local retired-workflow refs but was not published to the taqsim remote.

The old `TQ2` package never completed because its reported-reproduction criterion incorrectly required aggregate retained water to be below one quantum. The old saved-run package never ran. Treat both commits above as implementation evidence, not as automatically trusted final branches: review their diffs, retain only behavior consistent with this vision, integrate them against the intended target branches, and run the repositories' current full gates.

The user's existing taqsim checkout contains unrelated modifications to `CONTEXT.md` and `docs/adr/0005-conservation-is-enforced-by-representation.md`. They are not part of this vision publication and must not be overwritten or assumed to be implementation input.

## Constraints and risks

- Do not weaken incidence conservation with epsilon checks or approximate equality.
- Do not silently floor external source data or initial stocks.
- Do not discard, rename as loss, or arbitrarily route a residual whole quantum.
- Flooring computed branches introduces a downward branch bias below one quantum per branch. Retaining the residual is the deliberate conservation-preserving consequence; tests must expose its accumulation and later availability.
- Conversion from counts back to public floating values must be centralized and deterministic so replay and serialization do not create a second arithmetic authority.
- Both repositories must agree on the model-document and Python-binding revision. A taqsim-only pin change or an incidence-only branch is not a delivered outcome.

## Explicit exclusions

- Floating-point conservation tolerances.
- A general unit system or dimensional-analysis framework.
- Geographic raster resolution.
- Implicit time resampling, interpolation, or rate integration.
- The `WaterSystem` and labeled-data redesign owned by #15.
- Fishy or downstream Zarafshan migration.
- The retired graph workflow and its package mechanics.
