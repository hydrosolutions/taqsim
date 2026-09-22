# Vision: incidence core

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/5

## Historical record provenance

This document preserves the original local vision for this delivered Effort. It is a
historical record repair, not a new implementation request or a claim of retroactive
approval. The original vision text below is unchanged. Its present-tense statements,
paths, versions, constraints and open questions describe the original work, not today's
repository. Delivery evidence and later decisions are separate from those requirements.

Original source: `planning/2026-08-10-incidence-core/vision.md` in taqsim.
Original source SHA-256: `c7285de55f9e819d9644d50ea9a0c4db61902dfaf9417b1077ba7782134b35cf`.
The original local vision has no commit in the available repository history; its frozen
plan and execution records are corroborated by the merged delivery pull requests.

Frozen plan: `planning/2026-08-10-incidence-core/graph.v5.json`.
Frozen-plan SHA-256: `53a2aa9ffad6ec9a8ac88928ae35d0237de7d7dea5bc117397230bf599a7eec1`.
Execution record: `planning/2026-08-10-incidence-core/driver-journal.jsonl`.
Execution-record SHA-256: `dabe410f235ea0979a18251ee438521675650e2539cb7880b82552c1425d09af`.
The retained final assembly records contain 24 criterion executions,
all with exit status 0. These are historical recorded results, not tests rerun for this repair.

Delivery evidence:
- https://github.com/hydrosolutions/incidence/pull/19 (merge `25a5bfde8ae9758249138edd62808a4c06886835`).

Decision evidence:
- https://github.com/hydrosolutions/incidence/blob/1a509d991bacc87a13742eb379479f0db21d20a0/docs/adr/0001-event-sourced-conserved-flow-core.md

## Original vision (unchanged)

## Goal / Why

taqsim has been described as event-sourced since 2026-01-08, when the description entered
in a single commit (`e052a7e`, 29 files, 3,728 insertions, zero deletions) justified by one
bullet: "Event sourcing: nodes record events instead of mutating state." No commit anywhere
argues why, and every later statement of the decision is that sentence copied forward.

The code does not implement it. Simulation state lives in mutable private fields
(`Storage._current_storage` at `storage.py:24`, mutated at :58/:77/:88; `Reach._routing_state`
at `reach.py:24`). Nothing folds events into state; `reset()` restores from constructor
arguments and clears the log. Patching `BaseNode.record` to a no-op, so the durable event
list stays permanently empty, produces bit-identical simulation results. The events that do
real work travel a different list, `_step_outputs` (`base.py:22`), which is drained by
`take_step_outputs()` and discarded every step. The load-bearing events are forgotten; the
persisted ones are inert. That is the inverse of event sourcing.

Because nothing enforced completeness, the log rotted. `reach.py:83` records `WaterOutput`
only when `net_outflow > 0`, so a dry day and a missing day are indistinguishable. fishy does
not compensate: `_extract.py:13` builds a trace and `iari/evaluate.py:38-41` derives the date
axis from `trace.timesteps()`, so IHA/IARI/DHRAM compute hydrologic-year statistics over a
2,191-day horizon with holes — wrong numbers in production today. Downstream there are 69
densification sites outside tests, reservoir storage level is reconstructed three mutually
incompatible ways, the log carries neither its horizon nor its dates, and three event types
have zero consumers in any repo.

Event sourcing is nonetheless the right pattern, for two reasons never previously stated.
First, the domain is natively event-sourced: water is conserved, so stock is the
time-integral of flux and a compartment's level *is* the fold over its transfers. Second,
replay is what makes "dumb router, intelligence on top" enforceable — completeness has no
natural defender, and a log kept alongside state drifts the moment someone adds a guard,
which is exactly what happened. If state must be reconstructible from the log, the log is
provably sufficient. The invariant, not the vocabulary, is the pattern.

This vision builds `incidence`: a standalone, substance-agnostic, event-sourced
conserved-flow engine in Rust, where those invariants are structural rather than
aspirational. Named for the incidence matrix (conservation as structure) and for the
incidence of events (the log). Success means a core whose completeness and neutrality
cannot silently rot, and which taqsim can later be rebuilt on.

Rationale is recorded in `docs/adr/0001-event-sourced-conserved-flow-core.md`; vocabulary in
`CONTEXT.md`.

## Scope — In

1. A new repository `incidence`, created as a sibling directory from the `rustplate`
   template (`gh repo create incidence --template CooperBigFoot/rustplate`, then
   `bash init.sh incidence`).
2. The compartment model: a DAG of compartments holding stocks of independently conserved
   substances. `FiniteCompartment` holds non-negative material stock with non-overdraw
   enforced; `BoundaryAccount` is a signed external ledger counterparty (atmosphere,
   aquifer, crop, catchment, sea). The distinction is a structural type difference, never a
   boolean flag or a name convention.
3. The authoritative log: `Genesis` (model digest, run id, semantics version, initial
   stocks), then `Transfer*`, then an optional `RunCompleted` seal (final `t`, transfer
   count, log digest). A `Transfer` is atomic and carries `id`, source, target, `t`, and a
   sparse keyed vector of per-substance amounts. Absence of `RunCompleted` means the log is
   a resumable prefix.
4. Closed-world conservation as structure: every transfer names both endpoints, so every
   column of the incidence matrix `B` in `x(t+1) = x(t) + B f(x(t), t)` sums to zero.
5. Exhaustive disposition at the rule boundary: a rule states where every substance goes,
   including what it retains, validated as a partition summing to available stock.
   `retained` is explicitly authored and never computed as `available - emitted`.
6. Rules as data: a serialisable expression-tree IR and its interpreter, never a callable.
   Vocabulary covers arithmetic, `min`/`max`/`clamp`, table lookup with interpolation,
   deterministic typed projections, and conditional select.
7. The projection layer: engine-managed, disposable, exactly rebuildable projectors
   supplying rules a typed view of history. A projection is any deterministic fold
   `P_t = fold(update, P_initial, facts_<=t)` — bounded lag window, rolling aggregate, or
   finite recurrence summarising an unbounded prefix.
8. Topological evaluation so arrivals are visible within the step, keeping travel time a
   property of rules rather than of network subdivision granularity.
9. The immutable, transitively content-addressed model artifact: topology, substance
   registry, initial stocks and projector states, timestep duration/calendar/horizon, all
   forcing data, rule parameters and IR, units and canonical encoding, IR/interpreter and
   numerical-semantics versions. `Genesis` binds a run permanently to its digest.
10. A hydrology rule set built against incidence purely as a test fixture — Muskingum, lag,
    linear reservoir, evaporation, seepage, a reservoir policy, a demand rule — to
    demonstrate the IR vocabulary is adequate. Not shipped, not published.
11. Executable demonstration of every acceptance criterion below, in Rust.

## Scope — Out (explicit non-goals)

1. Python bindings (PyO3). A separate Effort ticket.
2. Refactoring taqsim onto incidence. A separate Effort ticket.
3. Refactoring fishy onto incidence. A separate Effort ticket.
4. Publication to crates.io or PyPI, and any downstream version pinning.
5. Any change to taqsim, fishy, zarafshan-taqsim, ctrl-freak or taqsim-hydrology. They
   remain untouched and running; nothing is deleted anywhere in this vision.
6. Retiring taqsim's 14 event types, seven node types, `LossReason`, `Edge`, `Trace`, or
   the `Strategy`/`__params__`/`__bounds__` reflection machinery.
7. Shipping a production hydrology rule library.
8. Fixing fishy's sparse-date IHA defect in place.
9. Variable timesteps, stochastic rules, runtime observations, and unscheduled operator
   interventions. Explicitly outside the supported class.
10. Any e-flow method (Q347, Kazakh, WUA, salinity) or scenario-ladder work.
11. Re-surveying the Program Map or minting follow-on tickets.

## Constraints

```yaml
repos:
  taqsim:
    path: .
  incidence:
    path: ../incidence
consumption: []
```

1. Rust, using the `rustplate` template. `incidence` is created as a sibling directory of
   this repository. It was created from the template ahead of this run, because the run
   cannot validate or own a repository that does not yet exist; it is declared above and
   owns every milestone of this vision.
2. The engine is substance-agnostic. A rule may live in it only if it can be written without
   naming a substance, a unit, or auxiliary data. Admissible engine primitives are pure
   partition shapes: retain-all, release-all, fixed-fraction split, exogenous series,
   constant-fraction transfer. Hydrology lives one layer up, by construction.
3. The engine emits no derived quantities and applies no interpretation. The words loss,
   spill, waste, excess, consumed and deficit do not appear in engine vocabulary; every one
   is either a destination or a projection.
4. Three layers stay separate: the engine enforces conservation, disposition totality,
   ordering and replay; rules decide what happens, including inter-substance coupling, and
   run inside the step; the intelligence layer decides what things mean, and runs strictly
   after.
5. The rule law: a rule may depend only on current typed inputs, immutable model parameters,
   and deterministic typed projections of authoritative logged facts. Cached projection
   state must be disposable and exactly rebuildable. No rule-owned mutable value may be
   authoritative. Indexes, memoisation, compiled tables and rolling windows are permitted
   precisely because they are rebuildable.
6. In-memory stock is a running fold, not a re-fold per access; the replay law is a test
   comparing it against the log, not a runtime access path.
7. Substances are conserved extensive quantities only. Counters, indices, hysteresis modes
   and random draws are not substances and must not be encoded as stock.
8. The engine must never apply well-mixed advection as a default; it cannot distinguish
   intentional retention from author omission. Advection is an explicit rule-library helper
   returning a fully resolved vector transfer.
9. Amounts are extensive. Rates, discharges and concentrations are projections.
10. Bit-identical replay requires deterministic floating-point summation order.
11. Every record must distinguish "zero" from "not present" from "not modelled". This is the
    design's signature failure mode, observed three times at three granularities.
12. Changing any forcing value produces a new model digest and therefore a new run; replay
    against a differing artifact is rejected rather than reconciled. Historical model
    artifacts must remain retrievable.
13. Project doctrine in `AGENTS.md` applies: a module means one thing, receives exactly what
    it needs, in types that cannot lie, and dies rather than guess. Each module carries a
    one-line denotation in its docstring.

## Acceptance criteria (vision-level "done")

```json
{
  "criteria": [
    {
      "name": "Ledger replay is exact",
      "input": "Any completed run of a model artifact with storage, routing and loss compartments",
      "observation": "Folding the log from Genesis reproduces every compartment's in-memory stock for every substance, bit-identical, at every timestep"
    },
    {
      "name": "Continuation sufficiency",
      "input": "A log prefix truncated mid-run, plus the immutable model artifact, with no other state carried over",
      "observation": "Resuming from the prefix emits the remaining transfers bit-identically to the uninterrupted run"
    },
    {
      "name": "Books close",
      "input": "Any run containing finite compartments and boundary accounts",
      "observation": "For each substance, total stock summed across all compartments and accounts at the final timestep equals the Genesis total exactly, and every column of the incidence matrix sums to zero"
    },
    {
      "name": "Omitted substance is rejected",
      "input": "A rule whose disposition covers water but omits salt, in a compartment holding salt",
      "observation": "The run aborts with an error naming the compartment, the omitted substance and the timestep, rather than proceeding with salt silently accumulating"
    },
    {
      "name": "Overdraw is rejected",
      "input": "A rule disposing more of a substance than the finite compartment holds",
      "observation": "The run aborts with an error naming the compartment, substance and timestep; no transfer is appended"
    },
    {
      "name": "Dry stretch survives",
      "input": "A model where one compartment releases nothing for 40 consecutive timesteps within the declared horizon",
      "observation": "A dense projection over the run's horizon returns 0.0 for each of those 40 timesteps, distinguishable from timesteps outside the horizon"
    },
    {
      "name": "Truncation is detectable",
      "input": "A run whose final 10 timesteps emit no transfers, read once with its RunCompleted seal and once with the seal removed",
      "observation": "The sealed log reads as complete at its declared final timestep; the unsealed log reads as a resumable prefix; the two are never conflated"
    },
    {
      "name": "Registry absence is not zero",
      "input": "A log written under a substance registry without salt, read under a registry that declares salt",
      "observation": "The reader reports salt as not modelled for that run and refuses to return 0.0 for it"
    },
    {
      "name": "Model digest binds",
      "input": "A completed log replayed against a model artifact identical except for one changed forcing value",
      "observation": "Replay is rejected citing the digest mismatch, rather than silently reconciling or producing results"
    },
    {
      "name": "Rules round-trip",
      "input": "A model whose rules include Muskingum, a table-interpolated evaporation rule and a reservoir policy",
      "observation": "Serialising the model, deserialising it and running it yields a byte-identical transfer log"
    },
    {
      "name": "IR vocabulary covers hydrology",
      "input": "The fixture rule set: Muskingum, lag, linear reservoir, evaporation, seepage, a reservoir operating policy and a demand rule",
      "observation": "Every rule is expressed in the IR combinator vocabulary with no Python or native escape hatch, and runs to completion on a multi-compartment model"
    }
  ]
}
```

## Decomposition hints

1. **Riskiest first: the IR.** The single unproven claim in this vision is that the
   combinator vocabulary covers hydrology. Build the IR and the fixture rule set (scope
   items 6, 7, 10) early enough that an inadequacy forces a design revision rather than a
   rewrite. Muskingum is the sharp case because it needs a deterministic projection over
   prior inflow and outflow; evaporation is the sharp case for table interpolation plus
   forcing lookup.
2. **Types before mechanics.** `FiniteCompartment` vs `BoundaryAccount`, substance registry,
   and the transfer/disposition types are what make the invariants structural. Getting them
   as a boolean flag or a stringly-typed enum silently reintroduces every defect this vision
   exists to remove.
3. **Log and replay before rules.** `Genesis`, `Transfer`, `RunCompleted`, the fold, and the
   replay comparison give every later stage its oracle. Criteria 1, 2 and 3 should be
   passing before rules get interesting.
4. **Disposition validation before any hydrology.** Criteria 4 and 5 are pure engine
   behaviour and need no rule library.
5. **The absence-is-not-zero trio (criteria 6, 7, 8) is one design concern at three
   granularities** — value, run, registry. Treat them together; solving one in isolation
   tends to leave the others.
6. **Model artifact and digest (criterion 9) can land late**, but its transitive coverage
   must be designed with the types, not bolted on: the digest must cover IR and
   numerical-semantics versions, not merely top-level config.
7. **Determinism is a cross-cutting slice.** Fixed summation order affects criteria 1, 2 and
   10; decide it once, early, and state it in the numerical-semantics version.
8. Suggested slices: (a) repo scaffold from rustplate; (b) core types and substance
   registry; (c) log, fold, replay, seal; (d) disposition validation and conservation;
   (e) IR and interpreter; (f) projections and history views; (g) fixture rule set;
   (h) model artifact and digest binding.

## Open questions / risks

1. **The IR vocabulary's sufficiency for hydrology is reasoned, not demonstrated.** Adequacy
   for the fixture rule set is not adequacy for hydrology generally. If a fixture rule
   cannot be expressed, the run must surface this as a design decision rather than adding an
   opaque escape hatch, which would reintroduce non-rebuildable rule state.
2. **This vision builds a core with no consumer**, so its interface meets reality only at the
   taqsim refactor. Interface problems that a co-designed port would have caught may surface
   later. The fixture rule set is the mitigation and is deliberately weaker than a real
   consumer.
3. **Bit-identical replay across the criteria depends on floating-point summation order.**
   If deterministic ordering proves impractical for any aggregation, the run must record the
   tolerance it adopted and which criteria it weakens, rather than quietly relaxing
   "bit-identical" to "approximately equal".
4. **Rules needing long-window projections** may make history access a performance concern.
   Rolling aggregates keep it constant-time; a naive log scan does not. This is a design
   obligation, not a discovery to defer.
5. **Naming a run's numerical-semantics version is load-bearing** for digest binding. If it
   is omitted or under-specified, criterion 9 passes vacuously.
6. No irreversible act is contained in this vision: no publication, no history destroyed, no
   downstream repository touched.
