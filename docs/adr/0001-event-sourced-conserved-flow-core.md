# ADR 0001: Event sourcing is the right core, and is not what taqsim implements

- Status: accepted
- Date: 2026-08-10
- Effort ticket: #5
- Program: #4

## Context

taqsim has been described as event-sourced since 2026-01-08. That description entered the
codebase in a single commit, `e052a7e`, which added 29 files and 3,728 lines with zero
deletions. Its entire justification is one bullet: "Event sourcing: nodes record events
instead of mutating state." Every later statement of the decision — `README.md:19`,
`src/taqsim/__init__.py:2`, `00_philosophy.md:69`, `nodes/01_architecture.md:5`,
`nodes/02_events.md:5`, `edges/01_architecture.md:50` — is that sentence copied forward.
No commit anywhere argues why. The design it replaced, at `3b2e530`, recorded results as
timestep-indexed arrays (`supply_history`, `flow_history`, `flow_deficits`).

The decision was therefore never made, in the sense that no alternative was weighed and no
reason was recorded. This ADR makes it.

### What the code does

Simulation state lives in mutable private fields: `Storage._current_storage`
(`storage.py:24`, mutated at :58, :77, :88) and `Reach._routing_state` (`reach.py:24`,
:61), plus five `_received_this_step` counters. Nothing folds events into state. `reset()`
restores from constructor arguments and clears the log; `to_json` serialises topology and
parameters only; the optimizer clones fresh nodes and re-simulates.

Patching `BaseNode.record` to a no-op, so the durable event list stays permanently empty,
produces bit-identical results on a 7-node system with evaporation, seepage and lag
routing. The load-bearing channel is `_step_outputs` (`base.py:22`), drained by
`take_step_outputs()` and discarded each step; `water_system._route_output` (:477-488)
dispatches on `WaterOutput` and `WaterDistributed` to move water. So the events that do
work are consumed and forgotten, and the events that persist are inert. That is the
inverse of event sourcing.

The honest description is instrumented mutation. The documentation claims otherwise:
`nodes/01_architecture.md:11` states "State is derived from the event history", and
`nodes/02_events.md:150-163` shows a `current_storage()` fold that does not exist in `src/`.

### What it costs

Because nothing enforced completeness, the log rotted:

- `reach.py:83` records `WaterOutput` only when `net_outflow > 0`, so a dry day and a
  missing day are indistinguishable. fishy does not compensate: `_extract.py:13` builds a
  trace and `iari/evaluate.py:38-41` derives the date axis from `trace.timesteps()`, so
  IHA/IARI/DHRAM compute hydrologic-year statistics over a 2,191-day horizon with holes.
  These are wrong numbers in production today.
- 69 densification sites outside tests hand-rebuild dense series.
- Reservoir storage level is unrepresented and reconstructed three incompatible ways
  (`optimization_coursebook.py:445-454` clamped; `explore_tradeoffs.py:180-181` unclamped
  and ignoring spill and evaporation; `analysis.py:332` abandoning the log for live state).
- The log carries neither its horizon nor its dates; two functions independently rescan
  every event for `max(t)+1`, and five sites hardcode `pd.date_range("2017-01-01", ...)`.
- Three event types (`WaterEnteredReach`, `WaterExitedReach`, `WaterInTransit`) have zero
  consumers in any repo.

## Decision

**Event sourcing is the right pattern, for two reasons neither of which was previously
stated, and it will be implemented properly in a new engine called `incidence`.**

1. *The domain is natively event-sourced.* Water is conserved, so stock is the
   time-integral of flux: a compartment's level **is** the fold over its transfers. This is
   physics, not analogy.
2. *Replay is what makes "dumb router, intelligence on top" enforceable.* The goal needs a
   record that is complete and neutral. Completeness has no natural defender — a log kept
   alongside state drifts the moment someone adds a guard, which is exactly what happened.
   Event sourcing supplies the defender: if state must be reconstructible from the log,
   the log is provably sufficient. The invariant, not the vocabulary, is the pattern.

taqsim adopted the vocabulary without the invariant and therefore got none of the benefit.

### The design

- A DAG of **compartments** holding **stocks** of independently conserved **substances**.
  `FiniteCompartment` holds non-negative material stock and may not overdraw;
  `BoundaryAccount` is a signed ledger counterparty (atmosphere, aquifer, crop, catchment,
  sea). The distinction is structural, never a flag.
- The authoritative log is `Genesis`, then `Transfer*`, then an optional `RunCompleted`
  seal. A `Transfer` is atomic and carries a sparse keyed vector of per-substance amounts.
  Absent `RunCompleted`, the log is a resumable prefix.
- **Closed world**: every transfer names both endpoints, so conservation is structural —
  every column of the incidence matrix `B` in `x(t+1) = x(t) + B f(x(t), t)` sums to zero.
  This is double-entry bookkeeping.
- Compartments expose one rule, `outflows(stock, arrivals, history, t)`, returning an
  **exhaustive disposition**: where every substance goes, including what is retained,
  validated as a partition summing to available stock. `retained` is explicitly authored,
  never `available - emitted`.
- **Rules are data** — a serialisable expression tree, not a callable.
- A rule may depend only on current typed inputs, immutable model parameters, and
  deterministic typed projections of logged facts. Cached projection state must be
  disposable and exactly rebuildable. No rule-owned mutable value may be authoritative.
- Delay is emergent: unreleased stock. Compartments evaluate in topological order so
  arrivals are visible within the step, keeping travel time a property of rules rather
  than of how finely the network was subdivided.
- The engine is substance-agnostic: a rule may live in it only if it can be written
  without naming a substance, a unit, or auxiliary data.

### Three replay properties, distinguished

- **Ledger replay**: stocks reconstructed from recorded transfers.
- **Continuation sufficiency**: everything needed to produce the next transfer,
  reconstructed from a log prefix plus the immutable model. This is what private rule
  state breaks.
- **Re-execution reproducibility**: rerunning a model artifact yields the same log.

### Supported class

A run is determined by an immutable, transitively content-addressed model artifact and a
prefix of its authoritative history. Runtime observations, unscheduled operator
interventions, and stochastic inputs are unsupported until represented by explicit typed
facts.

## Alternatives considered

- **Fix emission in place** (always record, including zeros). Cheapest, and fixes the dry
  day. Does not fix storage level, horizon, dates, the four-way fork over "flow at reach
  X", or the 69 densifications, and leaves legal the guard that caused the bug.
- **Dense arrays**, the pre-`e052a7e` design. This is what consumers keep rebuilding by
  hand, and it is a legitimate *projection*. Rejected as the authoritative record because
  nothing then enforces completeness — the same failure, differently shaped.
- **Petri nets** as the formalism. taqsim is expressible as a coloured, continuous, timed
  Petri net, but deterministic synchronous firing deletes conflict, nondeterminism and
  concurrency, which is where Petri net theory pays. Keep the P-invariant vocabulary; take
  the algebra from the incidence matrix.
- **Scalar one-substance-per-transfer records.** Rejected: a reservoir emits two transfers
  to the same destination in one step (policy release and overflow), so joining on
  `(from, to, t)` for concentration is ambiguous. Atomicity requires a shared id and batch
  append, which is a vector envelope in normalised storage.
- **Engine-applied well-mixed advection.** Rejected: the engine cannot distinguish salt
  intentionally retained by evaporation from salt accidentally omitted by a rule author,
  and supplying it silently would embed assumptions about mixing, sampling point,
  proportional withdrawal, phase behaviour and selective uptake into an otherwise
  assumption-free core.

## Consequences

- taqsim, fishy and zarafshan-taqsim are untouched by this decision's first increment;
  incidence is built standalone. Their migration is separate work.
- Installing incidence alone gives a graph engine that models no river. Hydrology lives one
  layer up, by construction.
- Users lose "just write a function" for rules. Novel rules must be expressed in the
  combinator vocabulary or accept a slow escape hatch.
- Adding a substance forces affected rules to be reviewed, because disposition is
  exhaustive. This is the unavoidable price of distinguishing omission from intentional
  retention; no representation can infer author intent.
- Changing a forcing series produces a new model digest and therefore a new run. Historical
  model artifacts must remain retrievable.
- The IR vocabulary's sufficiency for hydrology is reasoned, not demonstrated. One real
  rule set built as a fixture is the test.

## The signature failure mode

Absence conflated with zero, seen three times at three granularities: a value omitted
because it was zero (`reach.py:83`); a substance absent from a log written under an earlier
registry; a run whose final timesteps emitted nothing, indistinguishable from truncation.
Every record in incidence must distinguish "zero" from "not present" from "not modelled".
