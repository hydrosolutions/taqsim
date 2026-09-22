# Vision: taqsim modelling layer on incidence

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/8

## Historical record provenance

This document preserves the original local vision for this delivered Effort. It is a
historical record repair, not a new implementation request or a claim of retroactive
approval. The original vision text below is unchanged. Its present-tense statements,
paths, versions, constraints and open questions describe the original work, not today's
repository. Delivery evidence and later decisions are separate from those requirements.

Original source: `planning/2026-08-20-taqsim-modelling-layer-on-incidence/vision.md` in taqsim.
Original source SHA-256: `6604ae7b8be9e0a68657fbc6f415ef860ef55fcfcd1868ff2178f12365a1fa73`.
The original local vision has no commit in the available repository history; its frozen
plan and execution records are corroborated by the merged delivery pull requests.

Frozen plan: `planning/2026-08-20-taqsim-modelling-layer-on-incidence/graph.v3.json`.
Frozen-plan SHA-256: `37782cb9e5fdb19e2fb8c01173f7a7fac2dd67335025e8bd51d78b8126b62af0`.
Execution record: `planning/2026-08-20-taqsim-modelling-layer-on-incidence/driver-journal.jsonl`.
Execution-record SHA-256: `510d719f94b150b5b3b6362522e9b763e55f708016b22efecc6a627a7677f1c1`.
The retained final assembly records contain 28 criterion executions,
all with exit status 0. These are historical recorded results, not tests rerun for this repair.

Delivery evidence:
- https://github.com/hydrosolutions/taqsim/pull/13 (merge `018a81342d7e28cf3c172f37ad93cc64f1484182`).
- https://github.com/hydrosolutions/incidence/pull/22 (merge `b61b404e19f7bfc050d3e0e9c5cbd76ca34214c5`).

Decision evidence:
- https://github.com/hydrosolutions/taqsim/blob/9b54535f07758ca317d5f69e7eb1f3731762d73c/docs/adr/0004-disposition-ir-carries-computed-branch-amounts.md

ADR-0004 records a later decision within this Effort: computed branch amounts in the
incidence disposition IR. It is not silently inserted into the original vision. The
frozen v3 graph includes IN2 and the two delivery PRs identify that implementation.
The original vision already excludes matching defective old numbers and treats the
1,012-test suite as a source of scenarios, not expected values; this repair does not
replace that requirement with the earlier issue's scope sketch.

## Original vision (unchanged)

## Goal / Why

taqsim today is a 3,376-line simulation engine that describes itself as event-sourced and is
not. Its state lives in mutable private fields; its log is inert, sparse by omission, undated
and horizon-less. The consequences are live defects, not theoretical ones:

- `node/reach.py:83` records `WaterOutput` only when `net_outflow > 0`, so a dry day and a
  missing day are indistinguishable. fishy computes hydrologic-year statistics over a
  punctured 2,191-day axis as a result.
- `system/water_system.py:477` routes only `WaterOutput` and `WaterDistributed`. `WaterSpilled`
  is recorded and never moved, so a capacity-limited reach or PassThrough **deletes water**
  (taqsim#12). Every canal reach in the Zarafshan basin sets a capacity.
- `node/splitter.py` delivers whatever a split policy returns, so a policy returning
  `amount * 1.5` fabricates water silently (taqsim#11).
- Neither dates nor horizon are carried anywhere; five sites independently hardcode
  `pd.date_range("2017-01-01", ...)`.

incidence now exists (ADR-0001) and is reachable from Python (ADR-0002), with conservation,
exhaustive disposition, presence, and exact replay as structural properties. Its Python
boundary is deliberately unfriendly: it names compartments and substances, not water, and a
hydrology rule is authored as nested plain-data dictionaries.

This vision moves the engine out of taqsim and leaves behind the layer a hydrologist actually
touches: a water-naming authoring surface, a rule vocabulary that compiles to the engine's
rule IR, declared time, readable runs, and a saved-run cache. Both of Tobias Siegfried's open
issues close by construction rather than by a fix.

## Scope — In

1. **A water-naming authoring surface.** `Basin` declares reaches, reservoirs, demands,
   diversions, sources and sinks and compiles to exactly one incidence model document.
   A reach *is* the connection; there is no `Edge` to name.
2. **A rule vocabulary** (ADR-0003). Composable hydrology-flavoured pieces — zone conditions,
   seasonal values, per-second rates, loss cascades — from which a modeller writes a new
   operating policy without editing taqsim. Compiles to the engine's expression tree; never a
   Python callable.
3. **A power operation in incidence**, admitted under the closed-form primitive bar, so canal
   seepage is `alpha * sqrt(Q) * length * time` rather than an interpolated approximation of
   it. Includes exposing it, and the already-present `Subtract`/`Divide`
   (`crates/core/src/rule_expression.rs:102-104`), through the Python binding.
4. **Declared time and units.** A model states start date, timestep length and flow unit
   before it will build; results are indexed by real dates.
5. **Capacity that names its overflow destination.** taqsim refuses a capacity-limited reach
   or structure that does not say where the excess goes.
6. **Runs as values.** `model.run()` returns an immutable run; every reading is a projection
   carrying presence. No `reset()`, no in-place `simulate()`, no per-node event lists.
7. **A saved-run cache.** One file holding the model document and the log's canonical bytes,
   stamped with taqsim, incidence, rule-IR and numerical-semantics versions plus the model
   digest, refusing to load under different versions rather than guessing.
8. **The ctrl-freak optimization path, working.** Parameters come from the rule IR; bounds and
   constraints are declared beside the parameter and held taqsim-side, so retuning a bound
   never changes a model's content-addressed identity. Sweeps hold the compiled model and
   substitute declared parameters only.
9. **The deletion.** `Edge`, the seven node types, the fourteen event types, `LossReason`,
   `Trace`, `node.events`, `events_of_type`, `node.trace`, `reset()`, in-place `simulate()`,
   and the `Strategy`/`__params__`/`__bounds__` reflection machinery.
10. **Fixtures for the six real rule shapes** the Zarafshan basin uses — zone-based reservoir
    release, monthly distribution, priority distribution, e-flow split, reservoir evaporation,
    canal losses — shipping as taqsim's own tests. These are the evidence that the rule
    language is adequate for real hydrology, which ADR-0002 named as its resolution condition.
11. **A rewritten README**, and `AGENTS.md` §0 corrected: taqsim is no longer "an event-sourced
    framework for simulating".

## Scope — Out (explicit non-goals)

- **Porting any downstream project.** zarafshan-taqsim (428 tests), fishy, and chu-swatplus on
  its commit-exact qualified pin all break. Each port is its own later ticket, decided with its
  owner once more of the Program has landed.
- **A compatibility facade over the old API.** The rebuild is a clean break. A shim would have
  to keep `WaterOutput`-on-positive-flow honest, and that shape is the defect being deleted.
- **A written old-to-new migration document.** Prose about code goes stale fast and is now read
  mostly by models, which infer the rest from source. Documentation effort goes into the README
  and into code following the project doctrine.
- **Matching today's numbers.** Today deletes water, so there is no correct old number to match
  for the spill case. Correctness is held to properties — conservation, presence, exact replay —
  plus hydrology cases verifiable independently of taqsim.
- **Rebuilding fishy** (ticket #9) or publishing anything to PyPI (ticket #6).
- **Redesigning ctrl-freak.** It is healthy and stays a dependency.
- **Persistence as an archive format.** A saved run is a cache; the durable artifact is the
  content-addressed model, which reproduces its log exactly.
- **Growing the engine's expression language on demand.** An operation enters incidence only if
  it is a closed-form mathematical primitive naming no substance.

## Constraints

```yaml
repos:
  taqsim:
    path: .
  incidence:
    path: ../incidence
consumption:
  - producer: incidence
    consumer: taqsim
    artifact: power operation in the rule expression language and its Python binding
  - producer: incidence
    consumer: taqsim
    artifact: model document format, compiled model, and authoritative log canonical bytes
```

- **Validation is single-sourced** (ADR-0002). taqsim may not reimplement any legality check the
  core owns; a model document is decoded and validated by exactly one implementation.
- **Rules are data.** No per-timestep callback into Python exists or may be introduced.
- **The rule IR is a versioned wire commitment** and is content-addressed, so adding the power
  operation moves model identity. Nothing downstream depends on it yet, which makes now the
  cheapest moment.
- **Every time-indexed read carries presence.** No taqsim surface returns a bare sequence of
  floats. Modelled zero, absent, and not-modelled stay three distinguishable answers.
- **Closed world.** Every transfer names both endpoints; water never goes nowhere.
- **Project doctrine** (`AGENTS.md` §2): one meaning per module, narrowest arguments, types that
  cannot lie, fail loud. Every module carries a denotation line. `uv` exclusively; `uv run ruff
  format`, `uv run ruff check --fix`, `uv run ty check`.
- **Measured ground for the persistence decision**: one basin-scale run (50 compartments, 20
  forcings, 2,191 daily steps) is 14 s median; reading a series back is 9 ms
  (`incidence/bindings/python/benchmarks/sweep-baseline-v1.json`).

## Acceptance criteria (vision-level "done")

```json
{
  "criteria": [
    {
      "name": "Zone release policy compiles from the vocabulary",
      "input": "Author the Zarafshan three-zone reservoir rule — dead-storage floor, buffer/conservation/flood zones, twelve cyclical monthly release rates in m3/s — in taqsim's rule vocabulary and build the model",
      "observation": "The model builds, and the emitted model document declares twelve substitutable rule parameters for the seasonal release rate"
    },
    {
      "name": "Seepage uses the real hydraulic law",
      "input": "A canal reach with seepage coefficient alpha, length L km, and a flow of Q m3 per timestep",
      "observation": "Recorded seepage equals alpha * sqrt(Q_m3s) * L * seconds_per_timestep to floating-point tolerance, not a piecewise-linear table approximation"
    },
    {
      "name": "Capacity without a destination is refused",
      "input": "Declare a capacity-limited reach that names no overflow destination, and build",
      "observation": "The build refuses, naming the offending reach; no model document is produced and nothing runs"
    },
    {
      "name": "A dry day is present, not missing",
      "input": "Run a model whose reach outflow is exactly zero for a stretch of days inside the horizon, then read that reach's flow",
      "observation": "Those timesteps come back marked present with value 0.0, distinct from timesteps outside the horizon which come back not_modelled"
    },
    {
      "name": "Water cannot be fabricated by a split",
      "input": "Author a diversion whose branches together claim 1.5 times the available stock, and build and run it",
      "observation": "The disposition partition check refuses it; no run turns 100 units of inflow into 150 units of arrivals"
    },
    {
      "name": "Overtopping water lands somewhere named",
      "input": "A capacity-limited reach with a declared overflow destination, driven with inflow above capacity for several timesteps",
      "observation": "Total arrivals across all named endpoints equal total generation; the closed-world balance over the whole run is zero"
    },
    {
      "name": "A saved run refuses a stranger",
      "input": "Save a completed run, hand-edit the stamped incidence version recorded in the file, and load it",
      "observation": "The loader refuses and names the version mismatch; it returns no run and reconstructs nothing"
    },
    {
      "name": "Time is not optional",
      "input": "Build a model that declares no start date",
      "observation": "The build refuses and names the missing declaration"
    },
    {
      "name": "Replay is exact",
      "input": "Run the same built model twice with the same run id",
      "observation": "Both runs produce the identical authoritative log digest"
    },
    {
      "name": "A sweep holds the compiled model",
      "input": "Run a parameter sweep of at least 50 trials over a declared rule parameter",
      "observation": "The model document is decoded and validated exactly once; per-trial only parameter substitutions and a run id cross the boundary"
    }
  ]
}
```

## Decomposition hints

- The power operation in incidence is upstream of everything that compiles a seepage rule, and
  it is the only work outside the primary repository. It is small, self-contained, and can be
  proven in incidence's own test suite before taqsim consumes it.
- The vocabulary is the largest single piece. Its compile target is plain data, so it can be
  proven by asserting on the emitted model document before any run exists.
- Refusals (capacity without destination, missing time declaration) are cheap and catch design
  errors in the authoring surface early; they belong beside the surface that raises them.
- The six rule fixtures are the falsification of the whole vision. Reaching them early is worth
  more than reaching them completely: if the vocabulary cannot express the zone release rule,
  everything downstream changes.
- Deletion of the old engine lands after the new path can express the fixtures, not before, so
  the old tests remain available as a source of scenarios.
- The optimization path depends on the vocabulary declaring parameters, and on nothing else.

## Open questions / risks

- The rule vocabulary will have edges: a policy someone can describe in a sentence and not
  write. When a package hits one, the standing bar decides — a closed-form mathematical
  primitive may be proposed for incidence; a hydrology concept must be composed in taqsim.
  A rule shape outside both is a park, not an invention.
- Node geography (`system/_visualize.py`, `geo.py`, the `location=` argument every Zarafshan
  node passes) was never settled. Whether the new authoring surface carries locations at all is
  undecided, and no acceptance criterion covers it.
- `AGENTS.md` §0 and `src/taqsim/__init__.py` both describe taqsim as an event-sourced
  simulation framework. Both become false and must be corrected as the engine leaves.
- The 1,012-test suite is a source of scenarios, not of expected values. Porting a test that
  encodes today's water-deleting behaviour would encode the defect.
