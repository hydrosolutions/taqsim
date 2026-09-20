# Quality and source-control feasibility

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/24

## Outcome

Implement Fishy's configurable quality assessment and conservative fixed-boundary flow/load feasibility calculations. A modeller can assess one supplied configuration, identify supported failures and missing evidence, calculate the entire feasible interval, and distinguish a physical screen from an activated requirement or official finding. Demonstrate the real physical-to-quality chain through current Taqsim, including saved results, without requiring a simulator for independently supported imported boundaries.

Implementation belongs principally in `hydrosolutions/fishy`, with current `hydrosolutions/taqsim` integration. This vision does not start implementation. It owns F11, all quality cases and catalogue preservation, supplemental source-control/profile/process checks, A13, and applicable C semantics. Receptor assessment belongs to Effort #28; complete Uzbek regime/floor/duty assembly belongs to #31. Supply those consumers with quality results and the constraints needed for final rechecking, not another implementation of their complete workflows.

## Authority and the obsolete-software warning

The sole requirements bundle is `/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`. The Program approves it despite its internal candidate label. Earlier handovers and deprecated Fishy behavior confer no requirements. Read its README, IMPLEMENTATION_BRIEF, ACCEPTANCE, SOURCE_INDEX, SOURCE_LIMITATIONS and SOFTWARE_BASELINE before following the detailed sources below.

The owner explicitly warned that the methodology bundle was developed against an outdated Taqsim. **Its fixtures are evidence to review, not executable tests or implementation contracts to copy blindly.** For each case, distinguish mathematical expectations supported by the governing methodology from obsolete interface, data-shape, operation-order or implementation assumptions. Check units, boundaries, physical support and expected arithmetic before adapting the case to current public operations. Do not recreate retired APIs or preserve historical architecture merely to run old fixtures. Do not change valid methodological expectations merely to fit current software either. Report a fixture-versus-method conflict with its exact evidence; preserve the case and unresolved status rather than silently changing an expected answer, deleting coverage, or claiming it passed. Genuine scientific/source conflicts return to the owner; unaffected work continues.

“Cover all 26 cases” means every original case has an attributable, reviewed disposition and a current-system behavioral test wherever the governing expectation is supported. Unresolved cases cannot count as completed acceptance. Extra checks in ACCEPTANCE remain required; the fixture file is not the full specification.

Recorded SHA-256 identities:

- IMPLEMENTATION_BRIEF.md: `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8`
- ACCEPTANCE.md: `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba`
- SHA256SUMS.txt: `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0`

Verify source identity before use. If unavailable or different, obtain the same authoritative bundle, not a historical substitute. This public vision records derived contracts, hashes and precise source pointers. The full originals remain locally supplied; do not imply repository hosting or upload restricted reports or papers merely to make the planning record self-contained.

## Source records are not active rules

Preserve all 456 inactive source catalogue entries and the eight separate inactive seed rules, including raw cells, original names/formulae, source-local categories, units/reporting basis, qualifications, merged-cell/notes relationships, hazard and harmfulness metadata, and source identity. These are source-table records, not 456 Taqsim fixtures or an adopted national profile. The catalogue covers two held annexes, not every applicable quality duty.

A selected profile explicitly identifies jurisdiction, instrument/version, category/use, location, required parameters, applicability/exclusions, chemical form/fraction and reporting basis, units or supplied supported conversion, operator and strictness, sampling/averaging interval, conditional interpretations, uncertainty/censoring, and required group definitions. Overrides create a new identified profile without changing the source or prior results. A scalar cell does not imply an upper limit. A dash is not zero; a textual prohibition, conditional range or unexplained note is not a guessed numerical threshold. Similar chemical names and matching units do not establish identity; no automatic chemical repair or ion/basis conversion is commissioned.

For supplied public observations, preserve an admission record: publisher and original URL, retrieval and observation dates, station/location and depth, determinand/form/unit and sampling basis, analytical method, reporting/detection limit, quality flags, licence/use restrictions and transformations. Distinguish original measurements from aggregates, model estimates and synthetic inputs. Missing fields remain explicit limitations; annual bulletins cannot become invented daily concentrations or be paired with unrelated daily flows. Conductivity cannot become individual-ion concentrations through an unsupported conversion.

Implement individual and authorised/configured group upper, lower and range tests and supported relative-reference tests. A range contributes both bounds. Preserve required membership: a supported individual failure survives an unknown group member, with incomplete coverage. TDS stays outside ionic sums containing its constituents. Non-detects are not zero. With defensible supplied intervals, pass an upper bound only when the whole interval passes, fail only when the whole interval fails, otherwise return indeterminate; reverse appropriately for lower bounds and retain strict endpoints. Unknown reporting limits remain missing. Group interval bounds require positive denominators and are bounds, not probabilities. Mixing sizing uses declared point-input cases; caller-run uncertainty studies do not turn a point answer into an uncertainty-qualified finding.

Keep source ambiguities explicit rather than selecting policy:

- SanPiN 0083-24 §22 prose and displayed formula disagree at equality. The original HTML element `7354152` retains the strict sign lost by plain-text extraction. Support explicit inclusive/strict scenario interpretations and unresolved source equality.
- SanPiN §§5 and 22 and draft Пакет 7272 §35 have distinct group applicability; do not merge them automatically or infer groups from labels alone.
- The report's cross-source label mapping is provisional. Support the recommended single organoleptic group and qualifier-subgroup alternative as explicitly configured interpretations. Demonstrate the report-required sensitivity comparison in the external acceptance harness; Fishy need not orchestrate comparisons.
- Under the report's DP-QUAL-1 working reading, assigned-category limits size flow while stricter general SanPiN constraints remain separately assessed. Do not silently substitute them into sizing or claim they passed. Official applicability confirmation remains outstanding.
- Do not infer a strictness order across draft А–Д categories, map them automatically to seed I/II categories, or import a foreign classification as an Uzbek category. Other owning methods retain their source-specific interpretations and conflicts.

## Conservative feasible intervals

For one declared interval and completely mixed section, let fixed background water be Qb ≥ 0 in m³/s, constituent background load Bi ≥ 0 in kg/s, and controllable additional **arrival** water q ≥ 0 with Csi ≥ 0 in kg/m³. All other water carriers belong in Qb, their loads in Bi, and separate loads enter once. Exclude q and its load from both background terms.

`Qsection = Qb + q > 0`; `Ci(q) = (Bi + q*Csi)/(Qb + q)`.

Inputs must remain fixed as q varies. Changing returns, source composition, storage, evaporation, withdrawals or travel time needs supported recalculation, a demonstrated reduction to this boundary, or another supported model. A point sample does not prove representative complete mixing.

For upper limit Ti, solve `(Csi - Ti)*q ≤ Ti*Qb - Bi`; for lower limit solve `(Ti - Csi)*q ≤ Bi - Ti*Qb`. A group with Ti > 0 gives `(ΣCsi/Ti - 1)*q ≤ Qb - ΣBi/Ti`. Retain strict source operators. Each positive coefficient gives an upper bound, negative a lower bound, zero either no restriction or impossibility according to its right-hand side and strictness.

Intersect every inequality with nonnegative q, positive total water, supplied operational bounds/capacity and the matched ecological total or base floor: `q ≥ max(0, Qeco - Qb)`. Preserve raw quality-only and combined feasible intervals, open/closed endpoints, binding constraints, predictions and reasons. Distinguish an attained minimum from an unattained infimum. Never invent an epsilon as a smallest compliant release. A dirtier source or lower-limit constraint can give an upper bound; taking the largest dilution minimum is not a general solver. Recheck supplied/final candidates against the original tests after ecological or receptor uplift.

If a quality minimum exists, `Qquality = Qb + qmin` is a conditional total at this boundary, not a newly protected legal/ecological floor. Changing background abstraction or loads requires recalculation. Do not count background water twice or equate arrival flow with upstream release without supported routing. For a dry boundary, zero-flow concentration is undefined and `(0, infinity)` has no attained minimum; an independent positive base constraint can still produce a combined candidate.

Preserve the report's feasible, capacity-limited, source-water-infeasible and indeterminate classifications. Distinguish missing evidence from a known conflicting-constraints failure even if both use the last classification. No priority relaxation may hide a conflict. Reject invalid units/nonfinite values, negative physical inputs and nonpositive group denominators with attributable reasons. Declared numerical tolerances must not rewrite a policy boundary.

## Source control, activation and process support

For a single controllable load at fixed discharge, `Lallow = T*Q - Bother` when nonnegative; required reduction is `max(0, L - Lallow)`. Negative allowance means background alone fails, not a negative permitted discharge. For whole-drain control, substitute one common load factor in all individual/group inequalities and intersect with `[0,1]`, holding drain water fixed. Removing water too requires new water/load balances. This is not optimisation among polluters, a permit, or causal/legal attribution.

Discharge control must be exhausted before residual dilution. Residual dilution serves remaining uncontrollable diffuse loads, not permission to trade away controllable point-source duties. Preserve documented source tags and unmatched mass, but do not equate bookkeeping with proof of responsibility. Require supported local/basin inventories, inputs, outputs, internal cancellation, explicit processes and residuals; never invent a sink. Conservation failure invalidates affected assessments while disclosed precision approximations remain visible.

The report quality component activates only with all five conditions: reach/season feasibility screen; held fisheries targets; approved harmfulness grouping; basin salt budget; and NCECC-approved salt-transport model. Even then it binds only where feasible. Inactive screens remain advisory and retain the base result. Explicit hypothetical activation is permitted with its scenario label, not fabricated official approvals. Retain base, quality and combined findings separately. Active quality may raise a regime requirement or a floor-only route's floor; it cannot create a seasonal requirement, issued obligation or delivery verdict on a floor-only route. Separate supported receptor assembly remains possible while quality is inactive.

Assess supported supplied oxygen, pH, temperature, nutrients and other process outputs against configured tests. Preserve location, interval, chemical basis, support/domain, uncertainty and model provenance. Do not predict their corrective flows with conservative-ion mixing or infer thermal state from air-temperature regression. Native reactive chemistry, oxygen, hydraulic, thermal or habitat solvers and mandatory WASP integration are outside this Effort.

## Current physical integration, not historical compatibility

Discovery inspected Fishy main `53cf80f3e46257bc0b9be883bc772b6b941dc2b3` and Taqsim main `e67b8bda1c976a4d20ccb5baf96263c7dd3be216`. Recheck target branches and dependency pins at implementation start. The local canonical checkouts were behind those targets; existing local edits are unrelated and must be preserved.

Effort #9 delivered Fishy's independent evidence/assessment foundation and explicit physical exchange, not quality-limit A13 acceptance. `src/fishy/physical.py` exposes a prepared location/provenance boundary over Taqsim TransportResult: actual interval-volume-to-rate conversion, incoming/outgoing distinction, immutable physical source/digest, exact summed mass/water concentration, and chemical-form/reporting-basis checks. It preserves dry, absent, missing, outside-horizon and unsupported states and aggregate-resolution limitations. Reuse sound current contracts rather than the old Fishy node/event imports. No final API signatures are frozen here.

Current Taqsim supplies realised conservative transport, exact physical local/basin accounts, incoming/outgoing/storage projections and saved physical documents. Generic constituents still do not supply unsupported reactive/thermal processes. Foundation delivery evidence is at https://github.com/hydrosolutions/taqsim/issues/9#issuecomment-5750531270 and its landed record at https://github.com/hydrosolutions/taqsim/issues/9#issuecomment-5750934021. Those historical test results are not this Effort's acceptance.

A13 must run current Taqsim's realised A1-style mixing through Fishy's public quality operation: 10 m³ at 0.8 kg/m³ plus 5 m³ at 0.2 yields 15 m³/9 kg, hence 600 mg/l, failing an explicitly configured 500 mg/l upper limit. Fishy adds no water or mass. Verify realistic point, direction, interval, duration and unit mappings, live and supported saved paths, and incomplete/failure variants. Aggregate concentration by summed mass/water, not mean concentrations. Transfer concentration and final storage concentration are different observations. Exact accounting precision and measurement/model uncertainty are different concepts. Missing required checks retain incomplete coverage despite this known failure. Also demonstrate the same supported mathematics on imported boundaries without mandatory live Taqsim.

## Acceptance and durable return

Supply a requirement → current public operation/input → observable result → executed test crosswalk, exact compatible revisions, expected and actual outputs, declared tolerances and explicit unresolved dispositions. Separate supplied arithmetic, executed implementation tests, scientific adequacy and official authority. An always-pending implementation or source-traceability-only report is insufficient.

Review and cover every case in `report_snapshot/quality_support/quality_fixtures.json` under the obsolete-software warning above. Preserve ordinary dilution, background already compliant, source equals limit, capacity limitation, dirty-source upper bound, ecological conflict, dry open interval, grouped/individual constraints, lower-plus-upper bounds, TDS separation, missing membership, censoring, units, source control, conservation, inactive status, invalid inputs, strict minimum/capacity equality, zero-coefficient strict failure, equality interpretations, chemical-basis mismatch and uninterpreted source range. Preserve all 456 records and eight disabled seeds without auto-activation.

Add the ACCEPTANCE §5 requirements beyond those cases: relative-reference tests; profile overrides/version isolation; full required membership; fixed-load and drain-factor screens including changed-background recalculation; local/basin closure; original-test rechecks after uplift; five-condition activation and hypothetical/advisory branches; organoleptic sensitivity; and supported/unsupported process-output assessments. Verify applicable C1–C6 distinctions rather than inheriting an obsolete fixture schema. Include empty required-set refusal, known failure plus incomplete coverage, uncertainty-induced indeterminacy, source/scenario isolation and supported independent constituents.

A13 must be actual integration, not synthetic arithmetic or a mocked proxy. Retain current physical and foundation regression behavior. For any implementation bug fix, first demonstrate the failing real path with a regression test, then make that same test pass. Use repository-native environments and rules. Provide usable examples of imported assessment and real physical exchange alongside reproducible acceptance evidence; no registry publication is required.

## Detailed source map and remaining limits

Relative to the authoritative bundle:

- IMPLEMENTATION_BRIEF §§2, 5–8: F11, quality/process boundary, findings, exchange and acceptance responsibilities.
- ACCEPTANCE §§1, 5–6: C semantics, all quality fixtures and supplements, A13 and realistic physical mapping.
- `report_snapshot/part4_quality_calculation.qmd`, `sec-quality-parameters`, `sec-quality-catalogue`, `sec-quality-mixing`, `sec-quality-accounting`: profile/uncertainty rules, exact interval mathematics and load-control/balance operators. Its receptor section informs the integration boundary, not ownership of #28.
- `report_snapshot/part4_recommended_method.qmd`, `sec-quality-coupling`: five conditions, prior source controls, required organoleptic sensitivity and DP-QUAL-1 working interpretation.
- `report_snapshot/part4_assembly_contract.qmd`, component assembly; Appendix A floor-only branches: base/quality/combined distinctions and final rechecks; complete assembly remains #31.
- `report_snapshot/quality_support/parameter_catalogue_README.md`, `parameter_catalogue.json`, `parameter_sources/`, `public_parameter_seed.csv`, `quality_fixtures.json`: preserved source transcriptions, notes, 456 entries, separate seeds and synthetic cases.
- `references/uzbekistan/uz_sanpin_0083_24_water_protection_uz.html`: original typography, especially §§5/22 and Annexes 5–6. Use layout-bearing originals when extraction loses inequalities.
- `references/uzbekistan/uz_paket7272_rules_surface_water_protection_uz.txt` and original members indexed in PAKET_7272_INVENTORY.md: draft §§23–24, 35, 43, Annexes 1/3. Nonbinding source, not adopted permission.
- SOURCE_INDEX and SOURCE_LIMITATIONS: reference provenance, USGS mixing/EPA process support, unresolved applicability and source meanings. Methodological references do not supply local calibration.

Missing observations, accepted targets, category/group authority, calibrated transport and approvals remain application limitations, not fabricated defaults or blockers to supported labelled calculations. No basin certification, national compliance claim, automatic chemistry repair, new optimiser/comparison engine, general process simulator, or implementation of discharge permitting is commissioned.
