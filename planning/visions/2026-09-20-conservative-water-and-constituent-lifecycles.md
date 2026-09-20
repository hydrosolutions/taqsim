# Conservative water and constituent lifecycles

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/23

## Outcome

Extend Taqsim's existing WaterSystem and rule-authoring foundation to simulate realised water and multiple conservative constituents through mixing, allocation, storage, losses, spills, delays and returns. A modeller must be able to inspect independently reconcilable physical accounts and export attributable supported results for subsequent assessment. Requested releases, ecological requirements and assessment thresholds must never create water or constituent mass.

This is the physical modelling Effort, not a blanket rewrite or a Fishy method implementation. Engineering structure, API signatures and any justified generic Incidence additions or coupled specialist component remain implementation decisions. The approved handover already settles the physical requirements; no further technical interview is needed to choose reversible mechanisms.

## Authority and durable source contract

The sole requirements bundle is `fishy_taqsim_handover_candidate_2026-09-19`, approved in Program #4 as replacing every earlier handover. The held source directory is `/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`. Its internal candidate label does not supersede that approval. The following verified SHA-256 identities distinguish the exact source:

| File | SHA-256 |
|---|---|
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SHA256SUMS.txt` | `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0` |

This vision is an originally authored synthesis authorised for repository publication. It does not grant redistribution permission for the stakeholder report or third-party originals. Keep those sources in authorised custody. The local path is an access aid, not a public archive or runtime dependency; hashes do not provide access or permission. If originals are unavailable, obtain the same bundle rather than substitute an older handover. Use `report_snapshot/SNAPSHOT.json` and the payload checksums: the report includes authorised working changes beyond its base Git commit.

Read the bundle's README, brief, acceptance index, source index, limitations and software baseline, then these applicable operators. The contracts below carry the implementation-critical requirements durably; the originals retain attribution and scientific context.

| Source location | Contract carried into this work |
|---|---|
| `IMPLEMENTATION_BRIEF.md` §7 T1–T7 and §8 | Physical lifecycle, exchange, reproducibility and acceptance responsibilities |
| `ACCEPTANCE.md` §6 A1–A12 and supplemental cases; C3/C5 | Discriminating physical tests, exact/unknown/presence distinctions and partial findings |
| `report_snapshot/part4_quality_calculation.qmd#sec-quality-accounting` | Local/basin salt equations, single ownership of carried loads, explicit process accounting, provenance and simulator/assessment separation |
| Same file, `#sec-receptor-quality` | Water and mass trajectories, groundwater counted once, dry inventory, supported mixing and end-state versus exposure distinctions |
| `report_snapshot/appendix_computational_requirements.qmd#sec-backbone-exchanges` | Spatial/time mappings, exchange ownership, update order, uncertainty and volume-preserving aggregation |
| Same file, `#sec-backbone-capability-checks` | Water and individual salt accounts, no double-counted TDS/ions, returns and process ownership |
| `report_snapshot/part_computational_requirements.qmd#sec-backbone-models` | Supported component connections, delays/convergence and independent physical-boundary checks |
| `report_snapshot/part5_implementation_governance.qmd#sec-efficiency-reserve` | Diversion, consumption, recoverable return, groundwater, salt and receptor consequences without a new-net-water verdict |
| `report_snapshot/quality_support/receptor_quality_fixtures.json` | Supporting physical examples: salt-preserving evaporation, dry retained mass, source versus resident concentration and trajectory support; Fishy owns receptor assessment |
| `references/quality_public/usgs_mixing_1996_public_capture.txt`; `epa_wasp_2026_public_capture.txt` | Supporting mixing/process distinctions, not local calibration or mandatory dependencies |

The report controls the proposed Uzbek method and governing-language originals control independent foreign methods. Neither introduces national policy calculations into this Effort. Missing site observations, calibrated relations, accepted targets or authority decisions remain explicit application gaps, not blockers to supported generic and synthetic transport.

## Existing foundation and dependencies

Prerequisite #22 is landed. Discovery inspected Taqsim main `57a234a0b65f01c7bc8a6a15dec2a7bd31482821` and its Incidence pin `665da4e0d81ab28921b8d5d2edbb9be27f4ec612`. Recheck target revisions before implementation rather than treating the bundle's historical software baseline as current.

- `src/taqsim/water_system.py` still compiles and projects water only. Typed interval preparation, capacities, explicit overflow, immutable built models/runs, exact counts and replay are foundations to extend.
- `src/taqsim/vocabulary.py` includes reservoir evaporation, sequential canal losses and allocation/distribution rules. Preserve supported water behaviour and test any required evolution.
- `docs/conserved-transfers.md`, published through PR #33, documents generic `carrier_proportional` count-native coupling. It apportions dependent counts on realised carrier transfers, retains integer remainders and supports partial destination mappings such as water-only evaporation. It is not a physical constituent API.
- Engine selectors expose aggregate compartment/substance/direction/time counts, not arbitrary branch or source provenance. Preserve attributable input identity and supported routing; do not claim source apportionment after mixing from information the engine does not expose.
- The #22 tests supply missing-chemistry/remobilisation metadata externally. Actual selective supportedness propagation and physical remobilisation contracts remain work here.
- Current persistence is a water-output cache, not a checkpoint; arrivals and constituent projections are not yet durable. Engine presence is not physical or scientific supportedness.
- Finite source stores preload future forcing. Their bookkeeping inventories must not appear as river storage.

The #22 delivery and landed receipts are on https://github.com/hydrosolutions/taqsim/issues/22. Their reported test results establish prerequisite evidence, not acceptance of the new physical APIs. The deprecated Fishy imports are not compatibility requirements. The new Fishy foundation (#9) consumes the physical result boundary; the quality Effort (#24) completes A13.

## Required observable behaviour

### T1: explicit physical inputs

Support multiple identified conservative constituents with chemical form/reporting basis, units, initial inventories and timed boundary concentrations or loads. Distinguish rates from interval totals and integrate each exactly once over the actual interval duration. Refuse conflicting duplicate mass/concentration declarations, invalid units/bases, incompatible intervals/locations, nonfinite values and negative values in nonnegative physical domains. Do not collapse missing chemistry into zero. Supported signed states and directional exchanges remain valid in their own domains.

### T2–T3: realised coupling and inventories

Mass follows actual allocation, capacity and losses under a declared mixing and operation order, not requested outflow. Initial scope is compartment mixing with explicit travel delays, not implied stratification or dispersion. Account for mixing points, branches, reservoirs, canals, drains, spills, exports and transit inventories. No constituent advects on an actual-zero-water branch.

Expose transferred water/mass and relevant stored water/mass. Distinguish interval transport concentration from terminal storage concentration; compute ratios only on supported physical denominators. Aggregate mass and water first, then derive volume-weighted concentration. Aggregation cannot manufacture subdaily coverage or remove missing support.

### T4–T5: losses, uses, returns, dry and unknown states

Evaporation removes water without conservative nonvolatile dissolved salt. Seepage and abstraction carry declared mass to identified destinations. Returns have explicit timing and composition or supported physical coupling; neither a generic consumption fraction nor a second external injection may stand in for the return load. Retention, treatment or another process needs explicit accounting and destinations.

A dry compartment retains its salt while aqueous concentration is undefined. Rewetting needs a declared supported remobilisation assumption; without it, retain mass and return unresolved/unsupported dependent concentration rather than assume dissolution. Missing chemistry propagates only to affected results, leaving independent supported water and constituents usable. Do not clip negative stocks or silently dispose of unexplained mass.

### T6: independent physical balances and precision

Expose compartment and physical-basin start/end inventories, inputs, outputs, internal exchanges and residuals for water and each constituent. Include use compartments, transit and dry inventories; cancel internal transfers at basin scale; separate future forcing from physical stock. Natural background salt remains accounted. Do not add TDS and its constituent ions as independent contributions to the same total without a valid declared basis.

Each physical process has one owner. Supplied groundwater and other coupled exchanges carry direction, location, units, interval and identity and enter once. Replacing a process component disables both its old incoming and outgoing exchanges. Explicit coupling order, delays and any iterative convergence/support constraints travel with results. Unknown terms cannot be adjusted to force closure. An unexplained residual is not an invented sink.

Preserve authoritative exact counts and declared quanta. Carry allocation remainders and disclose concentration/quantisation approximations separately from scientific uncertainty. Recompute physical residuals independently of engine bookkeeping totals. Conservation failure invalidates affected results; unsupported balance/domain checks must be controlled and attributable, never a fabricated pass. Exact numerical closure alone does not prove mixing, routing or site validity.

### T7: result exchange, saved projections and reproducibility

Version inputs, assumptions, time axis, precision and software identities so identical runs reproduce accounting. Supported saved water and quality projections round-trip their values and meaning, including required arrivals/outlets and relevant storage. Keep present zero, absent, outside-horizon and unsupported quality distinguishable rather than blindly inheriting an engine sentinel.

Results identify physical locations and mappings, exact intervals, chemical forms/units, scenario/reference/source identity, provenance, validity, uncertainty and balance diagnostics. Preserve predecessor information where supplied and needed for temporal consumers. Source provenance does not imply unsupported post-mixing source attribution. Keep original/corrected/infilled/missing input status distinct from derived model output.

A saved output is not a checkpoint. Restart is optional; if offered, it must restore every water, constituent, transit, use, dry and remainder inventory and reproduce uninterrupted execution. No restart claim may be based on a cache round-trip alone.

### Specialist and efficiency boundaries

Support attributable imports from physical/process models with location, time, chemical basis, validity, uncertainty and version. Generic conservative mixing does not supply oxygen dynamics, pH, temperature, reactive nutrients, stratification or hydraulics. Do not substitute an air-temperature regression for a required reservoir thermal/withdrawal state. WASP is an example, not a required dependency.

Expose diversion, consumptive use, recoverable returns, non-recoverable losses, groundwater exchange and salt accounts at their physical places and times. Callers own comparisons and any savings assessment. Reduced diversion or improved reach timing is not automatically new basin water; preserve downstream/receptor and recharge consequences rather than issue a net-savings verdict.

## Acceptance evidence

Execute the public physical modelling path, not only engine arithmetic, fixtures, source inspection or metadata authored by tests. Return a requirement → public input/operation → observable result → executed test crosswalk, actual-versus-expected outputs, declared tolerances and exact compatible software revisions. Preserve successful, missing, unsupported and infeasible paths. Acceptance is software evidence, not basin calibration, scientific acceptance or official certification.

For the following synthetic cases, volumes are interval m³, mass is kg, concentration is kg/m³; assume complete mixing, one conservative constituent, the stated order and no unlisted process. Tolerances must follow declared water/constituent precision, with independently calculated residuals.

| ID | Physical operation | Required result |
|---|---|---|
| A1 | Mix 10 at 0.8 with 5 at 0.2; discharge all | Water 15, mass 9, concentration 0.6; no retained stock |
| A2 | Split A1 water into 6 and 9 | Mass 3.6 and 5.4; both concentrations 0.6; internal cancellation |
| A3 | Initial water/mass 20/4; add 10/4; request release 40 | Actual 30/8, water deficit 10, release concentration 8/30; dry storage concentration undefined |
| A4 | Same supply as A3; withdraw 15 | Release 15/4 and retain 15/4; each concentration 4/15 |
| A5 | Initial 10/5; evaporate 2 | Retain 8/5, concentration 0.625; evaporated salt zero |
| A6 | Initial 10/5; well-mixed seepage 2 | Destination 2/1; retain 8/4, concentration 0.5 |
| A7 | Incoming 15/9; capacity 10 with declared overflow 5 | Downstream mass 6, overflow mass 3; both destinations accounted |
| A8 | Put 10/5 into an empty one-interval delay at t0 | t0 transit inventory 10/5, no arrival; t1 delivered once |
| A9 | Withdraw 10/5 into explicit use; evaporate 4; return one interval later | Return 6/5, concentration 5/6; visible use/transit inventory, no duplicate external load |
| A10 | Initial 2/1; evaporate all; add clean water 4 next interval | Dry concentration undefined, mass 1 retained; explicit complete remobilisation gives 0.25; without it, unresolved rewet concentration with retained mass |
| A11 | Nonzero inflow, missing chloride, known sulphate and water | Chloride-dependent outputs incomplete; supported sulphate/water remain calculable |
| A12 | 10 m³/s and 500 mg/l over 86,400 s | 864,000 m³, 432,000 kg, concentration 0.5; no double duration multiplication |

Also demonstrate:

- Existing water regressions and multi-constituent independence; several-interval local/basin closure with external boundaries and transit; invalid and conflicting duplicate inputs; unsupported-domain/balance statuses.
- Transfer versus terminal concentration; mass/volume-preserving aggregation; no unsupported subdaily inference; non-round allocation with carried remainders; exact-zero branches; exact counts that would be corrupted by float reinjection.
- Identical reruns and durable supported projections with validity metadata. If restart is implemented, split A8/A9 at a checkpoint and match uninterrupted inventories/delivery.
- Caller-side comparison of diversion/consumption/return 10/4/6 versus 6/4/2: diversion decreases by 4, basin consumption does not. Expose location/time and dependent-receptor implications without calling it new net basin savings.
- Reproducible clean-source setup, compatible dependency pins and usable physical-stack examples. No PyPI publication is required.

A13 belongs to Fishy quality Effort #24: consume realised A1 concentration 600 mg/l against a synthetic upper limit 500 mg/l and report failure while other missing required checks remain incomplete. This Effort must provide the sufficient physical time/location/result boundary; it does not implement Fishy or claim A13 complete. The new Fishy foundation #9 owns its consumer compatibility. Do not restore deprecated APIs to satisfy the historical handover mismatch.

## Limits and remaining uncertainty

Do not implement national/ecological method rules, scenario comparison, a new optimiser, report generator, savings decisions, general native reactive/hydraulic/habitat solvers, or a Zarafshan application port. No basin, national threshold or remobilisation assumption is a hidden default inferred from these synthetic numbers.

Source meanings, site calibration and policy activation remain distinct from physical supportedness. A known supported failure survives other unknown checks; without a known failure, missing required evidence prevents complete satisfaction. An empty required set cannot establish a pass. End-state support does not establish exposure or a full trajectory. These result semantics support downstream assessment without moving its policy decisions into Taqsim.

No unresolved user-authority choice was identified during discovery. Mixing order, explicit supported configuration, component placement and optional restart are engineering decisions within the contracts above. Bring back a genuine source contradiction with its exact operation and alternatives if implementation exposes one; continue unaffected work. This vision authorises no reduction of the contained outcome on timing or resource grounds.
