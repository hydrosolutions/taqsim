# Project domain context

## Canonical terms

### incidence core (the conserved-flow engine)

| Term | Meaning |
|---|---|
| incidence | The standalone, substance-agnostic, event-sourced conserved-flow engine, written in Rust. Named for the incidence matrix (conservation as structure) and for the incidence of events (the log). Knows nothing of water. |
| Compartment | A node that holds stock. Transport delay is a compartment holding stock it has not yet released; there is no separate edge state. |
| FiniteCompartment | A compartment holding non-negative material stock. Non-overdraw is enforced. |
| BoundaryAccount | A signed external ledger counterparty (atmosphere, aquifer, crop, catchment, sea). Non-overdraw does not apply. Structurally a distinct type, never a flag on Compartment. Excluded from projections assuming material inventory, such as concentration. |
| Substance | A conserved kind that flows (water, salt). Each conserves independently. Only conserved extensive quantities are substances; counters, indices, and modes are not. |
| Transfer | The sole movement record: atomic, sparse, keyed-vector, carrying `id`, source, target, `t`, and per-substance amounts. Debits and credits every coordinate independently. The fundamental unit of the model. |
| Genesis | The logged fact establishing initial stocks, model digest, run id, and semantics version. Initial stock is a log fact, never a sidecar field. |
| RunCompleted | The terminal seal recording `final_t`, transfer count, and log digest. Its absence means the log is a resumable prefix, not a finished run. Distinguishes a completed run that emitted nothing late from a truncated one. |
| Stock | The amount of a substance held in a compartment at a time. Always derived: `initial + inflows - outflows`. Never authoritative in a mutable field. |
| Disposition | A rule's exhaustive statement of where every substance goes, including what is retained. Validated as a partition summing to available stock. `retained` is explicitly authored, never computed as `available - emitted`. |
| Projection | Any deterministic typed fold over authoritative facts. Stock, discharge series, concentration, IHA, Q347 are all projections. Implementations may be bounded lag windows, rolling aggregates, or finite recurrences summarising an unbounded prefix. Disposable and exactly rebuildable. |
| Rule | A compartment's `outflows`, expressed as data (an expression tree), never a callable. Depends only on current typed inputs, immutable model parameters, and typed projections of logged facts. |
| Rules as data | The rule interface is a serialisable IR interpreted by the engine, not a closure. Makes rules inspectable, diffable, parameter-extractable, and backend-independent. |
| Model artifact | The immutable, transitively content-addressed input: topology, substance registry, initial stocks and projector states, timestep duration/calendar/horizon, all forcing data, rule parameters and IR, units and encoding, IR/interpreter and numerical-semantics versions. |
| Closed world | Every transfer names both endpoints; boundaries are explicit. Conservation becomes structural: every incidence-matrix column sums to zero. |
| Ledger replay | Reconstructing stocks from recorded transfers. |
| Continuation sufficiency | Reconstructing everything needed to produce the next transfer from a log prefix plus the immutable model. The property private rule state breaks. |
| Re-execution reproducibility | Rerunning a model artifact produces the same complete log. |
| Model document | The public, versioned, plain-data form of a complete model: every compartment, substance, initial stock, forcing series, table, rule expression, disposition and binding in one serialisable value. Distinct from the `Model artifact`, which is the validated in-memory result of decoding one. The document is the only supported way to author a model from outside Rust; the core owns its validation so no consumer reimplements it. |
| Supported class | A run is determined by an immutable content-addressed model and a prefix of its authoritative history. Runtime observations, unscheduled operator interventions, and stochastic inputs are unsupported until represented by explicit typed facts. |

### Absence-is-not-zero

The project's signature failure mode, observed three times at three granularities: a value omitted because it was zero (`reach.py:83`), a substance absent from a log written under an earlier registry, and a run whose final timesteps emitted nothing. Every record must distinguish "zero" from "not present" from "not modelled".

### taqsim (the modelling layer)

| Term | Meaning |
|---|---|
| WaterSystem | taqsim's typed water-naming authoring surface. Declares time, conservation quantum, interval-volume sources, reaches, demands, diversions and their connections, and compiles to one incidence model document. |
| Rule vocabulary | taqsim's composable, hydrology-flavoured pieces (zone conditions, seasonal values, per-second rates) from which a modeller writes a new operating policy without editing taqsim. Compiles to the engine's rule IR. Not a catalogue of fixed rules, and not arbitrary Python: nothing calls back into Python during a step. |
| Declared time | A model states its start date, timestep length and flow unit before it will build. taqsim refuses to build without them, so a result is always indexed by real dates in known units and no consumer re-derives a calendar. |
| Run | The immutable value returned by executing a model. Holds the authoritative log; every reading is a projection over it. Replaces in-place `simulate()` plus `reset()` plus scattered per-node event lists. |
| Capacity | A physical throughput limit on a reach or structure, which must always name where the excess goes — the downstream river, a named escape channel, or an explicit boundary account. taqsim refuses to build a capacity that does not say. Closes taqsim#12 by construction: under a closed world, spilled water cannot go nowhere. |
| Saved run | A file holding the model document and the authoritative log's canonical bytes, stamped with the taqsim, incidence, rule-IR and numerical-semantics versions plus the model digest. A cache, not an archive: a loader whose versions differ refuses to open it and says to re-run, rather than reconstructing a guess. Invents no new wire format; both byte sequences are already public. |
| Closed-form primitive bar | The standing test for admitting an operation into incidence's expression language: it enters only if it is a closed-form mathematical primitive that names no substance. A hydrology concept never enters the engine; it is composed in taqsim's rule vocabulary. |

### Domain (hydrology and e-flow)

| Term | Meaning |
|---|---|
| e-flow (environmental flow) | A prescribed flow left in a river for ecological purposes. In this stack it is a flow obligation the model must deliver, not an ecological outcome it can verify. |
| Naturalisation | Transforming a system with human infrastructure into its natural counterpart, so an altered regime can be scored against an unaltered reference. Implemented by `fishy.naturalize`. |
| Counterfactual | The naturalised reference series the scenario ladder is scored against. Distinct from an unaltered historical record, which the basin does not have. |
| Scenario ladder (S0-S3) | The four compared rungs of the demonstrator: S0 no e-flow, S1 statutory entry floor (S1a Q347 concave, S1b fraction-of-MAF), S2 Kazakh Order 179-NQ baseline, S3 naturalised-regime replication. These labels override all older S-numbering. |
| Q347 | The flow reached or exceeded on 347 of 365 days, i.e. the 95%-exceedance daily flow. Computed on one abscissa-averaged flow-duration curve pooling all daily values; averaging per-year statistics is incorrect. |
| WUA | Weighted usable area. Habitat suitability integrated over a section, expressed against discharge. The WUA-vs-Q curve is unimodal, not monotone. |
| Pressure metric | A measure of how far a regime departs from natural (IHA, IARI, exceedance-class departure). Says nothing about ecological condition. |
| Response metric | A measure of ecological condition. Requires calibration this project does not have; dormant throughout. |
| Provenance | A label on every external series: official, pre_official, synthetic, or proxy. Gates what may enter a deliverable. |
| Uncertainty class | A label identifying which counterfactual member a series came from, so ensemble members are never silently averaged. |
| Screen-only | An output that may flag or challenge a problem but may never assert compliance. The salinity axis is screen-only until the basin salt budget exists. |

### Process

| Term | Meaning |
|---|---|
| Program | The durable container for one multi-vision idea, recorded as a single GitHub issue called the Map. |
| Effort ticket | One ambitious, contained, vision-sized question. Not an implementation task, milestone, or step. |
| Fog | Acknowledged future territory that cannot yet be phrased as one contained question. Lives only in the Map's "Not yet specified" section and is never minted as a ticket. |
| Frontier | The Effort tickets that are currently unblocked and actionable. |

## Aliases to avoid

| Avoid | Use instead | Why |
|---|---|---|
| Loss, lost water | Transfer to a named boundary account | Water is never lost; it went somewhere. "Loss" is an interpretation, and `LossReason` was a taxonomy of destinations demoted to adjectives. |
| Spill, spilled | A second transfer to the same destination | "Spill" labelled an arithmetic branch with a judgement about why. |
| Consumed | Transfer to the crop account | Water in plants still exists. Closed world. |
| Deficit event | A projection over model plus log | `requirement - arrivals` is a subtraction a consumer performs. The engine emitting it was the engine doing the intelligence layer's job. |
| A bare time-indexed series of numbers | A values-and-presence pair | A float per timestep cannot say whether a timestep was a modelled zero, absent, or never modelled. `reach.py:83` plus `iari/evaluate.py:38-41` is what that costs: hydrologic-year statistics over a punctured 2,191-day axis. No binding call returns a time-indexed result without presence alongside it. |
| Waste, excess | Name the destination | Perspective of a beneficiary, not a fact. |
| Node type (Source, Sink, Storage, Demand, Splitter, PassThrough, Reach) | Compartment plus a rule | Seven types were seven `outflows` bodies with their differences promoted to classes and their identical structure hidden. |
| Edge, edge state | Arcs are implicit in transfers and the incidence structure | Edges carry nothing. Delay is compartment stock. |
| Event sourcing (as taqsim implements it today) | Instrumented mutation | State lives in `_current_storage` / `_routing_state`; nothing folds the log. The name was asserted in one commit and never realised. |
| "Events ARE the state; no hidden internal state" | Documentation error | `00_philosophy.md` and `nodes/01_architecture.md` claim replay the code does not implement. |
| Water in transit (as an event) | Reach stock | A projection wearing event clothes; zero consumers. |
| `pip install fishy` | Install from the repository until the package is renamed and published | The PyPI name `fishy` belongs to an unrelated Elder Scrolls fishing bot. |
| S1 = Kazakh, S2 = absolute floor | S1 = Swiss Q347 entry floor, S2 = Kazakh baseline | The pre-2026-07 labelling was inverted. `MODELLING_HANDOVER.md` still uses the old labels and is superseded. |
| Ecological status, river health | Pressure metric, regime departure | This stack emits pressure only. Any status vocabulary implies a calibrated response model that does not exist. |
| GREEN light means healthy river | Delivery-compliance signal | The NCECC streetlight renders whether a prescribed flow was delivered, not ecological condition. |
| taqsim optimizer | ctrl-freak | ctrl-freak is the optimization engine; `taqsim.optimization` is a thin wrapper over it. |
| A compatibility shim for the old taqsim API | A clean break, ported later | A facade would have to keep `WaterOutput`-on-positive-flow honest, and that shape is the defect being deleted. Downstream ports are their own later ticket, decided with the port owners once more of the Program has landed. |
| A written old-to-new migration document | The rebuilt code, its tests, and one clear README | Prose about code goes stale fast and is now read mostly by models, which infer the rest from the source. Documentation effort goes into the README and into code that follows the project doctrine. |
| A rule written as a Python callable | A rule composed in taqsim's rule vocabulary | Rules are data the engine interprets. There is no per-timestep callback into Python, so `Strategy` subclasses with `release()`/`split()`/`calculate()` methods cannot exist. |
| Integer timestep with the calendar supplied by the caller | Declared time on the model | Five sites independently hardcoded `pd.date_range("2017-01-01", ...)` and disagreed. The calendar is model data. |

## Relationships

| Concepts | Relationship |
|---|---|
| incidence, taqsim | incidence is the substance-agnostic engine; taqsim becomes its first consumer, supplying hydrology rules and the Python-facing API. incidence knows nothing of water. |
| Engine, rules, intelligence layer | Three layers. The engine enforces conservation, disposition totality, ordering and replay. Rules decide what happens, including inter-substance coupling, and run inside the step. The intelligence layer decides what it means, and runs strictly after. |
| Optimizer, intelligence layer | The optimizer is the only permitted feedback path: it reads projections and changes parameters between runs. It never participates in a step. |
| Delay, storage | The same phenomenon. Both are stock a compartment has not released. They differ only in whether the release rule is a decision (optimizable) or a law (physics). |
| Petri nets, incidence matrix | taqsim is expressible as a coloured, continuous, timed Petri net, but deterministic synchronous firing deletes the theory that makes Petri nets useful. Take the P-invariant vocabulary; take the algebra from the incidence matrix `x(t+1) = x(t) + B f(x(t), t)`, where every column of `B` sums to zero. |
| Closed world, double-entry bookkeeping | The same invariant. Each transfer debits one account and credits another; the trial balance is conservation. |
| ctrl-freak, taqsim | ctrl-freak is the pure optimizer; taqsim depends on it and wraps NSGA-II in `taqsim/optimization/optimize.py`. |
| taqsim, fishy | fishy is the e-flow intelligence layer built on taqsim; it consumes systems and traces, and taqsim knows nothing of it. |
| taqsim, fishy, zarafshan-taqsim | zarafshan-taqsim is the basin application consuming both. Generic reusable method belongs in fishy; basin policy, calibration, and wiring belong in zarafshan-taqsim. |
| Pressure metric, response metric | Pressure can be computed today; response requires field calibration. A pressure metric may never be presented as a response. |
| Counterfactual, scenario ladder | Every rung is scored against the counterfactual, so the counterfactual's uncertainty propagates into every comparison. |
| Q_eco, Q_quality | The binding obligation is `max(Q_eco, Q_quality)`; quality can raise a flow requirement but never lower it. |
| Program Map, Effort ticket | The Map records program-level discovery and links its tickets; each ticket points back with `Program: #N`. Charting resolves no ticket. |

## Ambiguities

| Topic | Current interpretation | Resolution condition |
|---|---|---|
| The IR combinator vocabulary | Insufficient as shipped. Five of the six rules the Zarafshan basin actually uses compile from the existing operations, but canal seepage is `alpha * sqrt(Q) * length * time` and the language has no power. incidence gains a power operation under the closed-form primitive bar. | Resolved for this rule set by the taqsim rebuild. Adequacy for these six rules is not adequacy for hydrology generally. |
| Whether the engine carries solute or quality state | It does: substances are first-class and independently conserved. Concentration remains a projection, never engine state. | Settled by the incidence design. Supersedes Jazz's open question 8. |
| The Shared-Contracts typed shape | Required semantics fixed: per-reach id, daily index, concentration in mg/L with load derived, provenance, uncertainty class, missing-data policy. Dataclass shape and home still undecided. | Settled during the taqsim refactor onto incidence. |
| Whether Q347 maps to the Uzbek para-38 95%-provision statistic | Assumed equivalent, since 347/365 = 95.07%. | Requires NCECC confirmation. |
| Which water-year classes receive the Kazakh spawning uplift | The drier classes, 75% and 95%. The source wording is self-inverting because *vodnost* and *obespechennost* run in opposite directions. | Confirm against the Order with a Russian-reading reviewer before the numbers are presented. |
| Whether the 50% natural hydrograph caps flow after the spawning uplift | Undecided; exposed as a `cap_after_spawning` parameter rather than guessed. | Order paragraphs 19 and 25 conflict; needs a methodological ruling. |
