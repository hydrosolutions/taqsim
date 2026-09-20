# Coupled conserved transfers

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/22

## Outcome

Provide the generic execution capability that lets transported constituents follow realised carrier transfers while preserving exact accounting, explicit temporal inventory, deterministic replay and reproducible projections. Design from the complete physical-to-assessment contract before deciding engine changes. A proven composition of existing primitives is a valid outcome; a new operation or an engine rewrite is not a precondition.

This is an implementation vision, not implementation evidence. Discovery inspected sources and current Git objects. It did not execute the coupling regression, transport acceptance or inter-library integration. The source-analysis counterexample below must become a real-path failing test before any repair.

## Authority and durable source contract

The sole current commission is `fishy_taqsim_handover_candidate_2026-09-19`, approved by the Program owner on 20 September 2026 as replacing every earlier handover. Its internal working-candidate label does not undo that approval. Earlier delivery records explain existing foundations; they do not add requirements or prohibit justified changes.

This vision is an original, publicly authorised synthesis of the relevant requirements and synthetic acceptance cases. It makes this Effort's contract readable without access to a personal directory. It does not republish the stakeholder report or third-party documents, imply their redistribution licences, adopt policy, or authenticate unresolved law. Preserve the held originals in authorised custody. Subsequent visions must use the same source identity and their applicable operators, not treat this generic-foundation synthesis as a substitute for all foreign-method source clauses.

Original bundle access recorded by the Program: `/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`. All 244 payload checksums were verified during discovery. This location is an access aid, not a runtime dependency or a publicly hosted archive.

| Source identity | SHA-256 |
| --- | --- |
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SHA256SUMS.txt` | `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0` |

The report snapshot derives from `63b115dad1e938ad5240e0475df37523ffd9c949` plus authorised working-tree changes through 19 September; the commit alone does not identify that content. Use `report_snapshot/SNAPSHOT.json` and the bundle checksums. `reference_manifest.json` identifies 97 references, their hashes, provenance and retained URLs. A hash proves identity, not public access or reuse permission.

### Requirement sources

| Source within the bundle | Contract carried into this vision |
| --- | --- |
| `IMPLEMENTATION_BRIEF.md` §§1–2, 5–8 | Component ownership; F1–F11; T1–T7; physical/assessment boundary; implementation return |
| `ACCEPTANCE.md` §§1, 5–6 | C1–C6 support distinctions; 68 assessment fixtures; A1–A13 and supplemental physical cases |
| `report_snapshot/part4_quality_calculation.qmd`, `sec-quality-mixing`, `sec-quality-accounting`, `sec-receptor-quality` | Fixed-boundary screening versus simulation; mass balance; conservative evaporation; dry mass; receptor trajectories |
| `report_snapshot/appendix_computational_requirements.qmd`, C.1 `sec-backbone-exchanges` | Exchange identity, time, process ownership, physical boundaries, convergence and independent reconciliation |
| `report_snapshot/appendices.qmd`, steps 7–9 | Conveyance, supported thermal states, arrival-to-release mapping, receptor balances |
| `report_snapshot/part4_hydraulic_assessment.qmd`, D.9 | Actual elapsed time, predecessors, directional rates, domain and subdaily support |
| `report_snapshot/part4_assembly_contract.qmd`, `sec-component-assembly` | Shared throughflow versus lateral diversion; explicit routing rather than universal sum/max |
| `report_snapshot/part5_implementation_governance.qmd`, `sec-efficiency-reserve` | Diversion, consumption, returns and receptor effects; no automatic basin-savings claim |
| `SOURCE_INDEX.md`, `SOURCE_LIMITATIONS.md`, `SOFTWARE_BASELINE.md` | Source precedence, explicit interpretations, application limits and historical code evidence |

The report governs the proposed Uzbek method. Governing-language primary documents govern independent foreign methods; no country method belongs in the generic engine. Source ambiguity, missing site evidence and unadopted policy remain explicit in their owning assessments.

Public supporting references are Jobson, *Prediction of Traveltime and Longitudinal Dispersion in Rivers and Streams*, USGS WRIR 96-4013 (1996), [original reading source](https://water.usgs.gov/osw/pubs/disp/dispersion.html), especially transport theory and modelling limitations; and US EPA, [Water Quality Analysis Simulation Program](https://www.epa.gov/hydrowq/water-quality-analysis-simulation-program-wasp), the held 2026 description of compartment transport and separate process modules. They support the distinction between conservation, transport assumptions and calibrated physical prediction. They do not mandate WASP, provide local calibration, or supply missing chemistry. The held captures omit some equation images; the commissioned equations below come from report D.4, not guessed image transcriptions.

## Ownership and end-to-end capability map

Implementation repositories: `hydrosolutions/incidence`, with public-boundary proofs in `hydrosolutions/taqsim`. Incidence remains substance-agnostic. Taqsim expresses physical water/constituent processes. Fishy independently calculates requirements and assesses supported observations, imports or simulated results. A runtime simulator is not mandatory for every Fishy calculation.

| Capability | Owner and consequence for this foundation |
| --- | --- |
| T1: constituent identities/forms/units, initial inventories, timed composition/loads | Taqsim #23; engine must support identified conserved quantities and declared quanta without confusing chemical forms or rates with totals |
| T2: realised coupling | #22 generic execution foundation; #23 owns declared physical mixing/process semantics and complete physical acceptance |
| T3: branches, retention, storage and transit | #22 generic state/transfer foundation; #23 maps it to physical compartments, delays, spills and returns |
| T4: evaporation, seepage, abstraction and returns | #23; engine supplies explicit nonnegative destinations and retained conserved stock, not chemistry defaults |
| T5: dry and unknown chemistry | #23 physical supportedness; engine must not destroy retained mass or erase presence distinctions |
| T6: balances and precision | #22 exact bookkeeping and projections; #23 physical local/basin boundaries and independent physical residuals |
| T7: reproducibility and saved physical results | #22 identity/log/replay foundations; #23 durable supported projections and any explicitly offered restart |
| F1: mapping and route selection | Fishy #9/#31; preserve physical body, reach, plan unit and observation/compliance point separately; no inferred designation |
| F2/F10: hydrology, patterns and Uzbek preparation | Fishy #29/#31; retain reference/member identity, actual intervals, predecessor coverage and scientific-use restrictions |
| F3: Swiss prescription and delivery | Fishy #26; physical results distinguish intake release, downstream arrival, gains/losses and supported low-inflow evidence |
| F4: original Kazakh methods | Fishy #25; preserve topology, phase/time resolution and supplied specialist states without importing Uzbek interpretations |
| F5: habitat, potential and receptors | Fishy #28; supported state/pathway trajectories and relation domains, not flow-only assertions about habitat or resident salinity |
| F6/F7: assembly, floors and delivery | Fishy #31 and shared #9 contracts; requirement, floor, availability, deliverability, issued obligation and actual delivery remain separate |
| F8: sanitary duties | Fishy #27; exact schedule windows/control points and independent duties; later surplus cannot erase earlier deficit |
| F9: HYDMOD-F | Fishy #30; preserve event/subdaily support and reference/topology inputs; no Swiss classes in the engine |
| F11: quality/source control | Fishy #24; consume supported physical records; own configured tests, feasible intervals and activation status; complete A13 against #23 |

Callers and specialists own prepared spatial mappings, accepted reference reconstruction, scenario/member orchestration, structural selection, comparisons, operating search, scientific validation and presentation. Authorities own applicable duties, targets, accepted evidence and official decisions. A successful synthetic test neither calibrates a basin nor establishes legal compliance.

Before fixing engine design, produce a requirement → public operation/input → observable result → executed test crosswalk and a capability-to-owner map consistent with this table. The names/signatures and internal architecture remain implementation choices. Do not turn this into an engine-first speculative redesign or implement downstream methods in #22.

## Current execution evidence

Verified Taqsim remote `main`: `bdb2f59be9cd63e79b20b16e2f9f9aebaf8a1fb3`. Its Incidence pin is `ded06a5225b908ebaba5989aa5028df34cbdf026`. Incidence remote `main` inspected at `84f707b8f1dbc2e5c3d18121bdfab959d2021f91`; the core and Python binding diff from the pin is empty. Recheck revisions at implementation start. Local root checkouts were older and contained unrelated changes; they were not used as the current execution baseline.

At the pinned Incidence revision:

- `crates/core/src/execution.rs` evaluates all substance partitions for a finite compartment against the same pre-commit log, then commits the disposition. Substance iteration order cannot make a dependent partition see that compartment's newly computed outgoing carrier allocation.
- Its branch evaluation preserves compatible same-substance/same-quantum authoritative counts, otherwise floors each computed amount independently. Arithmetic expressions lose count provenance. Retention is the exact available count minus allocated counts.
- `crates/core/src/disposition.rs` validates exhaustive per-substance dispositions and omits zero-count transfers. Separate exact closure does not enforce a cross-substance carrier condition.
- `execution_bindings.rs` and `projection.rs` expose forcing/projection/table inputs and incoming/outgoing facts. Their existence is not a demonstrated public staged-commit API or a same-turn realised-branch dependency.
- `quantum_count_authority.rs`, `conservation_closure.rs` and Python `test_disposition_refusal.py` establish intended count/exhaustiveness contracts, not coupled transport acceptance. Test sources were inspected, not executed during discovery.

At the verified Taqsim revision, `src/taqsim/water_system.py::_model_document` hardcodes water in the registry, stocks, bindings and units. It preloads finite source compartments with all future forcing. That is engine bookkeeping, not physical river inventory. `retained()` uses count recurrence; `arrivals()` requires a live run and refuses saved-cache access. Existing saved output is not a checkpoint. Preserve the typed physical/temporal boundary, water regressions and exact quantum contracts delivered by #14/#15.

### Regression to establish before repair

Construct a public model on the actual current interpreter with 1 m³ water at quantum 1 m³ and 1 kg constituent at quantum 0.1 kg, each independently split equally between two destinations. Source analysis predicts zero realised water on both branches but five mass counts on each. Both separate accounts conserve exactly; their coupling is wrong.

This is a proposed discriminant, not an executed result. First add a regression that fails by observing the real transfer/count path. Record actual versus expected output and exact revisions. Do not substitute mocked evaluation, synthetic arithmetic alone, registration of a second substance or finer precision. Also prove a nonzero, non-round allocation whose transported mass must follow realised rather than requested carrier fractions.

## Physical contract the generic capability must support

For a supported completely mixed pool, constituent concentration is `M/V` when `V > 0`. For conservative constituents, the reaction term is zero. Over each declared interval:

- Water change equals incoming water minus outgoing water and evaporation, with every other represented physical exchange included once.
- Constituent change equals incoming carried mass minus outgoing carried mass plus separately declared loads and explicitly represented process/exchange terms.
- A separately declared load excludes mass already carried by a listed water flux. An unexplained residual is diagnostic, never an invented destination or balancing adjustment.

Mix initial stock and incoming stock in the declared order before withdrawal where that is the selected model. Allocate against actual available water and capacity. Derive advected mass from realised carrier transfers under the declared mixing model, not the requested release. Distinguish transfer-interval concentration from final-storage concentration; aggregate mass and water first, then divide, rather than averaging concentrations.

Evaporation of water retains nonvolatile dissolved salt. Well-mixed seepage or abstraction carries mass to its named destination. A return has its own location, timing and composition or a supported coupled process; a generic consumption fraction does not determine return chemistry. Use and transit compartments cannot also be counted as external load injections.

At zero water, aqueous concentration is undefined while mass remains. Rewetting requires a declared supported remobilisation model or an explicit unsupported result. Do not silently dissolve, delete or clip mass. Missing chemistry propagates only to affected constituent results; independently supported water and other constituents remain available. Reject conflicting duplicate mass/concentration input, incompatible chemical basis/units and invalid nonnegative-domain values rather than guessing.

Every destination and retained amount is explicit and nonnegative. Preserve exact counts per declared quantum, finite representability limits, carried remainders and disclosed concentration approximation. Exact numerical closure is distinct from model or measurement uncertainty. If all carrier water leaves but quantisation leaves mass, that mass remains an explicit inventory; its subsequent physical interpretation cannot be concealed.

Balances include in-transit and dry inventories and cancel internal transfers. A physical-basin projection must classify the boundary and exclude future forcing bookkeeping from present physical storage, recognising inflow at its physical entry. Imported/coupled components have one owner per store/process; a replaced component must disable both its incoming and outgoing exchanges. Unknown terms stay unresolved. Coupled iteration, if needed, declares update order, convergence criterion and iteration limit; nonconvergence is not an accepted last iterate.

### Mechanisms to evaluate, not preselected architecture

Compare a generic engine operation using realised carrier counts with staged/coupled execution using authoritative carrier events. Complete mixing, non-round branches, retained remainder and temporal inventory must discriminate the alternatives.

A generic operation may make same-turn coupling and atomic accounting direct, but must define ordering, count provenance, remainder ownership, multiple dependent constituents, canonical identity and replay without embedding water policy or chemistry in Incidence.

A staged design may run conservative carrier movement first and derive constituent transfers from exact destination/time/count events. It must prove that initial and incoming pools mix correctly, retained state recurs, every branch uses realised water, topology mappings remain attributable and nothing is injected twice. Merely adding a topological node does not create an interval delay. A concentration postprocessor without authoritative mass accounting is insufficient. A coupled component's state and configuration must be included in identity and deterministic replay/reconstruction, not hidden mutable side state.

Choose the smallest demonstrated mechanism. Extend the Python boundary only where public use/proof needs it. Existing primitives suffice only if their composed execution passes the same real-path requirements. No final API signature or IR redesign is mandated here.

## Exchange and assessment semantics

Expose supported amounts/rates, transferred mass, retained mass/volume and concentrations at identified outlets, arrivals and sampling compartments. Preserve:

- physical source/receiver and model-object identities, process owner, sign convention and spatial-mapping version;
- constituent identity and chemical/reporting basis, units, exact intervals, sending/arrival times and actual elapsed duration;
- model/input/assumption/precision versions, scenario and reference-member identity;
- presence and original/corrected/infilled/missing state separately from derivation method;
- support/validity, uncertainty, coverage, precision/remainder and balance diagnostics, including required predecessor data.

Do not force every assessment metadata field into the engine: preserve it at the narrow owning boundary. Engine presence and physical/assessment supportedness are different dimensions. Present zero, absent, outside horizon, unmodelled constituent and unsupported quality must remain distinguishable end to end. Signed groundwater exchange, datum-relative stage and directional velocity are legitimate supported domains; they are not negative conserved stock.

Rates integrate over the actual interval once. Aggregation preserves total water/mass without manufacturing subdaily support. A daily mean cannot prove hourly maxima, local velocity, stage or instantaneous concentration. Supported imported hydraulic, thermal or reactive states retain relation domain, location, time, uncertainty and model provenance; generic conservative mixing does not generate them. TDS is not added to its constituent ions; equal numerical units do not establish equal chemical reporting basis.

Fishy's fixed-boundary quality screen is distinct from dynamic transport: `Q = Qb + q > 0`, `C = (B + q Cs)/(Qb + q)`, with background and loads fixed and `q` the additional arrival. Dirty sources and lower constraints can impose upper flow bounds; feasible sets can have open endpoints and no attained minimum. Engine changes must not silently reduce this to a maximum of dilution minima, treat upstream release as arrival, or hold changing return loads constant. Fishy owns the screen and its configured activation, not this Effort.

A supported required failure survives incomplete other checks. Without a supported failure, unresolved required checks prevent a complete pass. Unsupported numerical trajectories remain exploratory. Conservation failure invalidates affected assessment; diagnostics cannot repair unsupported physics. Requirement/floor/availability/deliverability/issued obligation/actual delivery remain separate: requirement 10, deliverability 6 and actual 5 imply obligation 6, ecological deficit 4 and delivery shortfall 1 under the proposed Uzbek rule, without rewriting the requirement or earlier issue. Other methods keep their own duty rules.

## Acceptance contract and ownership

The following synthetic cases preserve `ACCEPTANCE.md` §6 in a self-contained form. Units are interval m³, kg and kg/m³ unless stated. Assume one conservative constituent, complete mixing, no unlisted fluxes and the stated operation order. Values are tests, not policy defaults or site observations.

| ID | Input and expected physical result | #22 proof / downstream ownership |
| --- | --- | --- |
| A1 | Mix 10 at 0.8 and 5 at 0.2, discharge all: 15 water, 9 mass, C=0.6, no retained stock | Generic coupling witness; #23 complete physical public operation |
| A2 | Split A1 into water 6 and 9: mass 3.6 and 5.4, both C=0.6; internal transfers cancel | Generic multi-destination/count witness; #23 basin mapping |
| A3 | Initial 20/4 plus incoming 10/4; mix, request 40: actual release 30/8, water deficit 10, release C=8/30, dry final C undefined | Realised rather than requested coupling witness; #23 process contract |
| A4 | Same pool, withdraw 15: release 15/4, retain 15/4, C=4/15 each | Retained mixed inventory witness |
| A5 | Initial 10/5, evaporate 2: retain 8/5, C=0.625; evaporation carries zero salt | #23 physical process; #22 must permit distinct carrier/dependent dispositions |
| A6 | Initial 10/5, well-mixed seepage 2: destination 2/1; retain 8/4, C=0.5 | #23 physical process, exhaustive generic destinations |
| A7 | Incoming 15/9, downstream capacity 10 with named overflow 5: downstream mass 6, overflow mass 3 | Generic capacity-constrained realised coupling witness |
| A8 | Empty one-interval delay receives 10/5 at t0: no arrival, full transit inventory; deliver once at t1 | Generic temporal-inventory witness; #23 physical delay semantics |
| A9 | Withdraw 10/5 into use compartment, evaporate 4, return remaining water one interval later: 6/5, C=5/6 | Generic use/transit accounting witness; #23 explicit process, not a universal return formula |
| A10 | Initial 2/1 evaporates dry: retain 1 mass, C undefined. Next clean inflow 4: declared complete remobilisation gives C=0.25; absent assumption leaves rewet C unresolved and mass retained | Generic dry inventory/replay support; #23 supported remobilisation |
| A11 | Positive inflow lacks chloride but water/sulphate are known: chloride-dependent results incomplete; independent components supported | #23 unknown propagation; #22 cannot collapse presence or invent zero |
| A12 | 10 m³/s at 500 mg/l for 86,400 s: 864,000 m³, 432,000 kg, C=0.5 | #23 typed preparation and unit/rate semantics |
| A13 | Actual A1 output enters Fishy with aligned location/time/scenario: 600 mg/l fails configured 500 mg/l upper limit; missing other checks remain incomplete | #24 integration against #23; #22 defines/proves the required generic result boundary, not full Fishy acceptance |

#22 owns executed generic proofs supporting A1–A4/A7–A10 and T2/T3/T6/T7. This table deliberately retains downstream cases to constrain the foundation, not to claim this Effort implements the entire physical or assessment stack.

Additional #22 acceptance must cover:

1. The initial real-path zero-water regression and nonzero non-round coupling; repeated allocations, retained/carry-forward mass remainder and multiple independent constituents. Inspect authoritative transfers/counts, not just projected concentrations.
2. Exact local and whole-engine count closure, nonnegative exhaustive destinations, representability/refusal cases and preservation of existing water quantum/regression behaviour. Independently reconstruct physical local/basin balances in the boundary witnesses, including transit, external boundaries and future-forcing exclusion.
3. Complete-mixing and operation-order witnesses that distinguish pre-withdrawal pool from same-interval inflow alone. Distinguish interval transport from end-state concentration and verify volume-weighted aggregation.
4. Stable identical reruns, content identity, authoritative log replay and public count/projection readback. Changed inputs, precision or coupling assumptions must not masquerade as the same model/run.
5. Presence and unsupported-result paths through the public boundary; no zero-fill of missing or unmodelled chemistry. Saved supported projections must preserve their offered meanings and validity metadata. Do not claim arrival persistence merely because current flow caches persist.
6. If restart is offered, split the A8/A9 runs and preserve/reconstruct every coupled water, mass, use, transit, remainder and dry inventory; match uninterrupted results. Otherwise explicitly document saved results as output, not restart.
7. A decision backed by actual complete-mixing/non-round execution for the chosen mechanism and concrete evidence on the rejected alternative. Source analysis alone cannot establish that existing primitives suffice or that an engine extension is necessary.

Downstream acceptance must additionally preserve the supplied assessment fixtures: all 26 `quality_fixtures.json`, 14 `receptor_quality_fixtures.json`, 18 `hydraulic_assessment_fixtures.json` and 10 `floor_uncertainty_fixtures.json` under `report_snapshot/quality_support/`. Foundation-sensitive examples include a dirty-source upper bound; source water versus resident receptor state; transient failure hidden by final pass; bounded duration despite partial coverage; daily means unable to test hourly peaks; unsupported relation domains; missing counterfactual uncertainty; and incompatible nitrite reporting bases. Those suites and the 456 inactive catalogue entries belong to their Fishy Efforts, not #22 implementation acceptance.

The external efficiency comparison 10/4/6 versus 6/4/2 for diversion/consumption/return shows four fewer diversion units but unchanged net consumption. #23 exposes located/timed accounts; callers compare them. Do not label the diversion reduction new basin water. Likewise same-throughflow requirements 3 and 2 combine as 3, whereas lateral diversion 2 plus continuing flow 3 requires 5 only under an explicit zero-loss mapping. Generic execution neither selects policy nor guesses these mappings.

## Completion evidence, boundaries and handoff

Return compatible exact software revisions, public operation/input/result/test crosswalks and reproducible actual-versus-expected results. Declare precision and tolerances separately from exact count closure and scientific uncertainty. Preserve successful, missing, unsupported and infeasible paths. Record the old-code failing regression and the same test passing after repair. Package review, expression compilation, fixture arithmetic and source traceability alone are not implementation acceptance.

The handover's old Fishy source snapshot `f63c7511bc1e5ca2520ddb572a9da31b5eb29c4c` now belongs to the preserved deprecated repository. Its retired Taqsim imports are historical evidence, not an interface to restore. The Program's new Fishy repository is a clean implementation. #23 owns reproducible physical-stack source setup; #9 owns new Fishy source usability; #24 completes realised A13 exchange. There is no PyPI gate and no compatibility obligation to deprecated Fishy.

Out of scope: country methods/policy, targets or chemical defaults; blanket Incidence/Taqsim rewrites; native general dispersion, reactive chemistry, heat, hydraulic or habitat solvers; new optimiser/comparison/reporting engines; the Zarafshan application port; claims of calibration, official source authentication or scientific/legal certification. Supported specialist coupling remains allowed where its owning requirements call for it.

Unresolved application evidence does not block configurable components or complete synthetic success paths. It must remain declared input, supported assumption or explicit unsupported result. There is no unanswered technical decision for the project owner in this vision. Reversible implementation choices belong to the implementing agent, constrained by the demonstrated contracts above.
