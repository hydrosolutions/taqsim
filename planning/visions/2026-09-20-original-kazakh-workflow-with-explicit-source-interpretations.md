# Original Kazakh workflow with explicit source interpretations

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/25

## Outcome and ownership

Implement the original Kazakh environmental-flow workflow in `hydrosolutions/fishy`. A modeller can calculate one explicitly configured basin/section scenario from supported observations, prepared reference products and supplied studies; inspect annual and seasonal results for all four design conditions; and distinguish computed results, missing evidence, source conflicts and scientific/official use restrictions.

This Effort owns F4, Order 179-НҚ paragraphs 4–32 and all four appendices, relevant Order 111 quality classification, and the complete Kazakh section of ACCEPTANCE. It is not merely an annual probability-shift calculator. Cover supported ecological, hydraulic and biological study assessments as well as arithmetic. Demonstrate successful supplied scenarios, not an implementation that always returns pending. Unresolved source clauses cannot count as full source replication, even when a selected interpretation produces a runnable candidate.

Fishy calculates and assesses one supplied configuration. Caller tools prepare reference reconstructions, accepted donor relations, network mappings, scientific criteria and physical/study inputs; orchestrate members, scenarios and comparisons; and present results. Taqsim or supported specialist tools supply physical states when needed. Independently supported calculations have no mandatory runtime simulator dependency. Successive nonconsumptive reach requirements are not separate consumptive demands, extra water sources or automatically issued obligations.

Implementation starts only under a subsequent implementation handoff. This document neither implements the method nor authenticates a legal interpretation.

## Requirements authority and source access

The sole requirements bundle is `/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`. Program #4 approves that exact bundle despite its internal candidate label. Earlier handovers and deprecated Fishy behavior add no requirements. Read README, IMPLEMENTATION_BRIEF, ACCEPTANCE, SOURCE_INDEX, SOURCE_LIMITATIONS and SOFTWARE_BASELINE, then the original sources and detailed operators below.

Recorded SHA-256 identities:

| Bundle file | SHA-256 |
|---|---|
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SHA256SUMS.txt` | `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0` |
| `references/kazakhstan/kz_order179_ecological_flow_2025_ru_gov.pdf` | `c12db953e2a0b2a0c456333a794fb4e0fcdae76214a2c7a191ea54e567a4409f` |
| `references/kazakhstan/kz_order111_water_quality_2025_ru_cawater.pdf` | `c8e0aa22c3c2eb35269e858120ffda7d61de409b3c7e702cb2cb9b9f2c126d38` |
| `report_snapshot/part2_source_interpretations.qmd` | `2ea5663b66d4f87d29c07c4ff6396125c0a86892f524bbcef699d7bdc25b647c` |
| `report_snapshot/appendix_method_reference.qmd` | `254f4f8f2e1716072be91cf2d09758dff1ee66e83e42fde8c5752f1264d8a6f7` |

Verify these identities before use. If access or identity differs, request the same authoritative bundle rather than substitute an earlier handover. This public vision preserves derived contracts, numerical discriminators and exact source pointers. The full report and originals remain locally supplied; do not imply that they are repository-hosted or publish private/restricted material merely to improve access.

Governing-language primary documents control the independent Kazakh method. The report's D.3 register supplies labelled modelling interpretations, not official amendments. Use PDF layout where text extraction loses formulae, inequalities, exponents or notes. The official Ministry Order 179 copy includes all appendices. Preserve enactment and temporal applicability: enactment clause 4 defers methodology paragraph 11 to 1 January 2027; other commencement depends on the source's publication rule. A requested calculation date must not silently imply that every provision was then in force.

Order 111 is a complete 12-page CAWater replica, not authenticated publisher-direct evidence. The held Adilet.kz HTML is a secondary reading aid, not official Adilet. Keep authentication and source currency limits explicit. Presence of a former zan.gov.kz address is not proof of authentication. Do not turn source review or a scenario interpretation into legal certification.

The bundle's software baseline is historical. Review numerical fixtures against the governing method and current public operations rather than copying obsolete API/data-shape assumptions. Preserve valid mathematical expectations; expose an actual source/fixture conflict rather than silently change an expected answer or omit the case.

## Basin, reference and observation routes

Cover mouth-to-source basin/section sequencing, tributary contributions, changes in river type, channel/floodplain/delta/terminal-water context, climate and biological conditions, and supplied water-use, operational and transboundary evidence. Preserve distinct basin, river, water-body, section and observation/calculation/control-point identities and their revisions. Main-river order zero and increasingly numbered tributaries are source labels, not enough to infer connectivity.

Accounting for tributary contributions in basin requirements is distinct from routing releases. Preserve explicit prepared topology, avoid double-counting the same water along successive sections, and leave unsupported routing or donor evidence unresolved. Do not replace a missing physical mapping with a universal sum or maximum.

Support all three observation routes:

- Adequate observations: calculate the supported annual variants using conditionally-natural reference evidence.
- Insufficient observations: consume supported annual/monthly reconstruction and naturalisation evidence, then calculate the annual variants. Do not fabricate a reference reconstruction or infer scientific acceptance from record length alone.
- Absent observations: use an identified, accepted analogue relationship and coefficient evidence. If a coefficient cannot be established, retain the further-study outcome rather than invent a donor, estimator or zero result.

Hydrological, quality, climate, biological and water-use evidence retain their source, interval, location and adequacy for the stated use. Monthly distribution is the normal coarsest source representation; short phases can require daily/weekly support. An annual-only supported result remains useful but partial. A monthly mean cannot establish daily-stage or within-day conditions.

## Annual magnitude and seasonal construction

The four design exceedance classes are 25%, 50%, 75% and 95%. Appendix 2 maps them respectively to conditionally-natural annual volumes at exceedance 50%, 75%, 90% and 97%. Higher exceedance means drier conditions. The natural 50% value is a median, not necessarily an arithmetic mean.

The probability-shift route gives initial ecological volume `W_sigma(c)`. The coefficient route is an alternative expression of this initial allocation, not a second reduction. Under the named median-normalised interpretation, `alpha_c = W_sigma(c) / W_50` and `W_c = alpha_c * W_50`, on one natural-reference basis and at the initial pre-adjustment stage. Require `W_50 > 0`; `0/0` is unidentified. An independently supported zero allocation can remain available through the shift route. Coefficients from a later calculation stage need a separate documented derivation and cannot masquerade as this interpretation. Transfers require donor and accepted transfer evidence.

For design class c, explicitly select design-class shape `s_c(t)` or shifted-class shape `s_sigma(c)(t)`. Each supported shape is nonnegative and has duration-weighted mean one on the receiving calendar. With D the accounting year's seconds and W in m³, calculate `Q(t) = W_sigma(c) * s(t) / D`. Preserve real interval durations, leap days and pattern provenance. Supported imported patterns are permitted; the Uzbek daily-pattern construction is not a required Kazakh algorithm. A missing shape choice need not erase a supported annual magnitude.

Preserve the source's recorded long-term minimum and natural 99% lower/50% upper design hydrographs and annual bounds. These are annual design-condition hydrographs, not daily discharge percentiles. Keep source and input meanings explicit rather than import the entire Uzbek bound-construction procedure. Crossed lower and upper bounds are unresolved before candidate corrections are calculated.

## Explicit source interpretations and ecological conditions

Every relevant interpretation must be explicitly selected and attributable. No universal default resolves the law. Return source version, selected interpretation, supporting evidence and affected results. An absent choice leaves affected operations unresolved while supported intermediate calculations remain available. The implementation must support and test every named branch, not choose one preferred branch for all users.

### Bounds and spawning corrections

For supported uncorrected q, lower/upper schedules L ≤ U, and an explicitly timed finite nonnegative coefficient schedule K, let `B(x) = min(U, max(L, x))`. Support both named candidate operators:

- Correction then bounds: `B(K*q)`.
- Bounds then correction: `K*B(q)`.

Return pre-bound values, each bound adjustment and remaining bound/correction conflicts. The first may suppress a supported ecological increase; the second may exceed a source bound. Neither arithmetic order confers complete source compliance. Retain the interpreted-candidate status until an accepted reconciliation establishes priority. Recompute monthly/annual volumes after corrections; never renormalise the schedule to conceal changed volume. Other hydraulic/study corrections need their own supported relation and combination order.

For paragraph 27(1), support percentage-wording eligibility `c ≤ 75%` and dry-year-wording eligibility `c ≥ 75%`. Equality is included in both. Missing selection is unresolved eligibility, not automatic exclusion or permission. Eligibility alone supplies no species, onset temperature, stage duration or coefficient evidence.

Daily observations support stage-specific migration, spawning and post-spawning coefficients; otherwise use the supported averaged seasonal coefficient for spawning month(s). Preserve the branch and temporal basis. Onset depends on biological water-temperature evidence, not approximate calendar dates. Source annual timing adjustment by up to 15 days does not shorten total duration. Floodplain depth/velocity corrections require supported hydraulic evidence; discharge alone does not establish local stage or velocity.

Preserve Appendix 3 recommended ranges and every Appendix 4 basin row, published average, row identity and unresolved starred heading. Under the listed-value interpretation, use the selected published stage values or published seasonal value. Do not clamp basin values to generic ranges or recompute a printed average. A study-derived average separately identifies stage weights and covered period. An unexplained star cannot become an invented footnote or qualification; affected eligibility remains unresolved where its meaning matters.

### Ungauged donor topology

The receiving-parent interpretation follows actual connectivity to the river receiving the ungauged tributary. Keep the main-river-zero labels separately. Do not select the numerically higher-order tributary or nearest gauge. Require the accepted donor relation and coefficient evidence; missing either preserves the further-study outcome. This interpretation provides no automatic coefficient-estimation method.

### Wider ecological and special-release requirements

Cover the source's level rise/fall, floodplain depth and inundation duration, oxygen/gas, thermal and fish-movement conditions; winter continuity; environmental releases for sensitive downstream areas; and small-river drying, depth and velocity provisions. Preserve mandatory procedures versus recommendations and qualitative study obligations. Missing required relations/criteria remain unsized or unassessed, not passed by a flow-only check.

For regulated waters, the November–December share of 30–50% of observed long-term mean monthly low flow is a recommendation, not a universal mandatory coefficient. The small-river source range 0.20–0.60 m/s does not select one universal velocity threshold; retain its meaning and supported selection. Preserve the 0.1 m minimum-depth provision and distinguish natural seasonal drying from harmful induced drying. Site targets, study acceptance, temperature timing and hydraulic relationships are supplied inputs. No generic dissolved-substance model can supply oxygen, thermal or biological processes by registration alone.

## Order 111 classification and use eligibility

Implement relevant six-class numerical classification and use mapping from the held source, not merely a scalar quality label or citation. Preserve exact units, chemical forms/reporting bases, operators, ranges, qualified cells, table/row identity and notes. Keep the classification's applicability to rivers, canals and in-channel reservoirs, its seas/lakes exclusion (including named terminal waters), and separate sanitary cross-references. A terminal receptor can matter to Order 179 without becoming eligible for Order 111 numerical thresholds.

The class-4 drinking-use conflict stays explicit: Table 1 permits use conditional on intensive treatment, whereas Table 2 marks it negative. Retain both attributed findings. Combined use eligibility is unresolved without an accepted interpretation. A labelled scenario may select descriptive-table mapping or matrix mapping, returning respectively conditional treatment eligibility or non-permission under that mapping. Neither changes numerical limits, proves treatment adequacy nor authenticates the replica.

Inspect PDF exponent/inequality typography, particularly row 70; the secondary HTML corrupts bacterial-ratio values and leaves an annex approval header unfilled. Preserve unresolved transcription status instead of repairing values by guess. Keep sanitary instruments and source applicability distinct; a selected-profile result is not full national compliance.

Reuse supported generic quality/evidence operations where appropriate, but do not impose the proposed Uzbek five-condition quality activation rule or Uzbek use categories on this independent classification. Unsupported oxygen/thermal physics can be supplied by specialist outputs with their validity and provenance; no native general process simulator is commissioned.

## Results, restrictions and revisions

Return the source Appendix 1 monthly and annual flows, volumes and annual shares for all four design conditions, with basin/river/section identity and correct units, including the source million-m³ representation where reported. Distinguish initial annual allocation, corrected schedule, actual recalculated volume and any residual source conflict. Preserve zero versus missing/unsupported/outside-horizon information; a zero annual total must not invent defined shares by division through zero.

Design classes can calculate without actual-year selection. Operational-year adjustments depend on separately supplied current conditions/forecasts and their issue version. Preserve revision triggers from changed basin conditions or new evidence and the source's at-least-five-year study-based review, with temporal applicability retained rather than silently backdated. Existing issued results are not rewritten by a later configuration.

Keep requirement, available water, deliverable water, any supplied issued duty and actual delivery distinct. Do not apply the Uzbek deliverability cap, multi-day issuance gate, advisory spawning semantics or adoption conditions. Assess supported supplied duties under their own basis; an annual allocation alone cannot imply a complete daily regime or successful delivery.

Numerical supportedness, completeness, scientific adequacy and official admissibility remain distinct. A supported required failure survives missing other checks and carries incomplete coverage. Without a supported failure, missing required evidence prevents complete pass. No missing components are renormalised out of a score. Uncertainty and support restrictions remain attributable to the affected points, periods and operations.

## Current repository evidence and integration boundary

Discovery inspected Fishy remote main `0c501def4b552eb46e21d2509740a8df375c9a82` and Taqsim remote main `306de27c68438ce160363fe58328399f0b0cdbf3`. Recheck targets and dependency pins at implementation start. The canonical Fishy working checkout was still the bootstrap scaffold; inspect current Git/GitHub evidence rather than infer current functionality from that stale checkout. Preserve unrelated local changes.

Effort #9 delivered versioned locations/mappings, typed flow/volume and signed state quantities, evidence/use restrictions, supplied-duty checks and optional physical exchange. Effort #24 delivered configurable quality assessment and conservative flow/load feasibility. Relevant Fishy files at the discovery revision include `spatial.py`, `flows.py`, `quantities.py`, `evidence.py`, `duties.py`, `physical.py`, `quality.py` and `quality_activation.py` under `src/fishy/`, with `docs/assessment.md`, `docs/quality.md`, `docs/quality-acceptance.md` and their tests.

Those foundations do not implement Kazakh connectivity, receiving-parent selection, source classification or ecological study rules. Spatial predecessor history means boundary revision, not directed river topology. SignedState permits supplied stage/velocity/temperature; it supplies no physical relationship. Generic quality checks can support configured Order 111 tests, but the existing Uzbek activation/track/category rules are not Kazakh policy. No dependency on the separate Uzbek hydrology or receptor Efforts is needed to evaluate appropriately supplied Kazakh inputs.

Foundation delivery evidence: https://github.com/hydrosolutions/taqsim/issues/9#issuecomment-5750531270 and https://github.com/hydrosolutions/taqsim/issues/24#issuecomment-5751649639. Their test counts are historical evidence, not this Effort's acceptance. The delivered Fishy physical stack pins Taqsim `396ad093c2b6f240e702a3b05aee1fb96a69b3f7` and Incidence `665da4e0d81ab28921b8d5d2edbb9be27f4ec612`; verify actual compatibility if consuming that exchange. Do not recreate deprecated node/event interfaces.

Use established domain vocabulary and the implementation repository's design/testing rules. Architecture, module layout, API signatures and reversible engineering mechanisms remain implementation decisions. This vision does not freeze an illustrative API or require a new configuration framework.

## Acceptance and implementation return

Return a requirement/source clause → public operation/input → observable result → executed test crosswalk. Cover paragraphs 4–32 and all four Order 179 appendices and relevant Order 111 clauses/tables. Distinguish implemented computation, supplied-study assessment and unresolved source interpretation. Report exact software revisions, expected and actual outputs, declared numerical tolerances, support restrictions and remaining source gaps. Citations, package inspection and reviewed synthetic arithmetic alone are not implementation acceptance.

Required numerical and behavioral discriminators include:

- Natural annual P50/P75/P90/P97 volumes 100/80/60/40 million m³ produce initial design allocations 100/80/60/40. Median-normalised coefficients 1/.8/.6/.4 reproduce them without a second reduction. Test positive, zero and invalid denominator cases and independently supported zero allocation.
- Four equal intervals with initial annual mean 10 and design shape [1.6,1.2,.8,.4] versus shifted shape [.4,.8,1.2,1.6] produce [16,12,8,4] versus [4,8,12,16] m³/s, both preserving mean 10. Test real-duration volume conversion and missing choice/unsupported pattern.
- q=8, L=5, U=10, K=1.5 yields 10 for correction then bounds and 12 for bounds then correction, retaining suppressed increase versus upper-bound exceedance. Test crossed bounds, timed corrections and recalculated volume.
- Eligibility at design [50,75,95]% is [true,true,false] versus [false,true,true]. Include missing choice, missing species/temperature/duration, daily-stage and monthly-average branches.
- Retain Aral–Syrdarya migration coefficient 1.20 even though the generic recommended range is 1.10–1.15. Preserve published averages and unresolved stars. Test both listed and separately supported study-derived values without conflating them.
- Receiving-parent topology differs from nearest gauge and numerical-order heuristics. Exercise accepted donor, missing donor/coefficient and further-study routes; verify mouth-to-source processing and no duplicate consumptive accounting across successive reaches.
- Class-4 drinking use differs under descriptive versus matrix mappings, with unresolved combined default. Preserve limits, replica status, scope exclusions and treatment-evidence restrictions.
- Demonstrate an independently runnable complete supplied Kazakh scenario and partial annual, missing, unsupported and infeasible paths. Include study/hydraulic/thermal conditions, winter/small-river/special-release provisions, reporting identities and revision triggers, not just annual arithmetic.
- Run without Uzbek operands or actual-year selection. Changing a hypothetical Uzbek coefficient/quality limit must not alter Kazakh settings or earlier results. Kazakh spawning corrections enter the selected source branch rather than become advisory merely because the Uzbek adaptation is advisory.
- Exercise applicable C1–C6 semantics, including temporal/physical identity, finite and valid quantities, required missing checks, known failure with incomplete coverage, independent scenario/source versions and meaningful partial results. Use current native public paths, not mock substitutes for the failing or integrated behavior.

Provide clear supported-use examples and reproducible acceptance evidence. For a bug fix, first demonstrate the real failing path with a regression test, then make that same test pass. No PyPI publication, full basin pilot or unavailable field data is a prerequisite to synthetic software acceptance.

## Detailed source map and exclusions

All paths below are relative to the authoritative bundle:

- `IMPLEMENTATION_BRIEF.md` §§1–3, 6, 8 and `ACCEPTANCE.md` §§1, 4, Completion evidence: F4, independent-method boundaries, all ambiguity branches and return requirements.
- `references/kazakhstan/kz_order179_ecological_flow_2025_ru_gov.pdf` and matching `.txt`: enactment clause 4; methodology paragraphs 4–11 (basin, sequence, reporting/revision); 12–22 (observations, naturalisation, annual variants, bounds and analogue route); 23–32 (operational adjustment, biological/hydraulic corrections, winter and special/small-river conditions); Appendices 1–4 (reporting, probability shift and coefficient tables). Paragraph 2(6) supplies river-order definitions. Inspect PDF pp.7–8 for inconsistent printed coefficient denominators and pp.12–13 for starred coefficient headings without explanatory bodies.
- `references/kazakhstan/kz_order111_water_quality_2025_ru_cawater.pdf`, matching `.txt`, and `kz_order111_water_quality_2025_ru_adiletkz_offline.html`: numerical classifications, applicability/notes and Tables 1–2. Inspect PDF pp.8–11 for row-70 typography, scope and class-4 conflict; HTML alone is insufficient.
- `report_snapshot/appendix_method_reference.qmd`, `sec-cheat-kz`: original method versus Uzbek adaptation; and `report_snapshot/part2_source_interpretations.qmd`, `sec-source-interpretations`: exact candidate operators and synthetic discriminators reproduced above.
- `SOURCE_INDEX.md`, `SOURCE_LIMITATIONS.md`, `reference_manifest.json`: source origins, hashes, replica limitations and attribution. `evidence/README.md` distinguishes synthetic development findings from site calibration; Uzbek pattern evidence supplies no Kazakh calibration or mandatory algorithm.

Excluded: official resolution/authentication of ambiguous provisions; invented footnotes, targets, coefficients or missing studies; automatic Uzbek answers; a counterfactual reconstruction engine, optimiser, comparison/report generator or native general habitat/hydraulic/reactive/thermal solver; basin ecological certification, permitting and the downstream Zarafshan application port. Scientific and authority inputs remain explicit limitations, not excuses to omit supported calculation branches.
