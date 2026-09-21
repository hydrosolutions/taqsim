# HYDMOD-F assessment

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/30

## Outcome

Fishy independently computes and assesses HYDMOD-F hydrological condition from supported observations, reference estimates, intervention records and prepared spatial or specialist inputs. It covers the official 2011 manual's chapters 2–6 and relevant appendices: intervention inventory and screening, all nine indicators, point-to-reach and catchment propagation, and the source-defined overall hydrology class.

A modeller can inspect each raw metric, its data route and evidence, its indicator class, reach propagation and overall result. A source-screened class 1 is distinguishable from an indicator that could not be assessed. Missing or unsupported evidence cannot manufacture a complete assessment. Supported calculations remain available independently of gaps elsewhere.

This is a hydrological screening method, not a release prescription, biological-status assessment, scientific certification or legal-compliance finding. It can accompany the independent Swiss prescribed-flow assessment without changing that prescription. No automatic uplift follows from a hydrological class.

## Authority and source access

Only `fishy_taqsim_handover_candidate_2026-09-19` supplies the commission. Its authoritative local directory is:

`/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`

Read its README, implementation brief, acceptance index, source index, limitations and software baseline before implementation. The baseline describes historical code, not current architecture. The governing German primary manual controls this foreign method; the report describes its place in the proposed Uzbek workflow but does not replace the Swiss rules.

Relevant anchors:

- `IMPLEMENTATION_BRIEF.md`, F9 and section 3, “Swiss residual-flow workflow and MSK”.
- `ACCEPTANCE.md`, section 4, “HYDMOD-F: all nine indicators”, and applicable common C1–C6 behavior.
- `report_snapshot/appendix_method_reference.qmd#sec-cheat-hydmod`.
- `report_snapshot/part2_source_interpretations.qmd`, “HYDMOD-F: the exact 15% boundary”.
- `SOURCE_LIMITATIONS.md`, Switzerland.
- `references/switzerland/ch_msk_hydrologie_2011_de.pdf`, with its text extraction only as a reading aid.

The source evidence below is retained here so a fresh agent does not need the discovery conversation. The complete private handover is not being republished. Verify the held originals against these SHA-256 identities; if unavailable or different, obtain the same approved source bundle rather than substituting old Fishy code or earlier handovers.

| Source | SHA-256 |
|---|---|
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SOURCE_LIMITATIONS.md` | `c81e23f0f5763e92acd735efadd4aa0cdef1dceef0aae9aa4aca9c6836b90a6d` |
| `report_snapshot/appendix_method_reference.qmd` | `254f4f8f2e1716072be91cf2d09758dff1ee66e83e42fde8c5752f1264d8a6f7` |
| `report_snapshot/part2_source_interpretations.qmd` | `2ea5663b66d4f87d29c07c4ff6396125c0a86892f524bbcef699d7bdc25b647c` |
| `references/switzerland/ch_msk_hydrologie_2011_de.pdf` | `1fddd85c31265a7474d566bb7faab6bbacf571a6428f8702942b1ee1ec987bc2` |

Printed manual page numbers below are two less than the one-based PDF page number. Inspect original PDF layout for equations, inequalities and graphs. Discovery visually checked Figures 17, 19, 21–23, 25, 27–28, 33 and Tables 8–10; this is not executed implementation acceptance. The implementation must verify its own transcription against those primary figures and tables.

## Settled treatment of source gaps

The owner approved implementing the defined cases and leaving the affected screening or classification undetermined where the source supplies no answer. Do not invent a boundary convention, extrapolate graphical classes or silently clamp a value to a plotted edge.

The decision is settled. It requires no further escalation, approval, separate issue, warning campaign or additional decision register. The owner requested a Slack explanation for their superior; sending or acknowledging that message is not an implementation gate. The ordinary computational result must still distinguish an undetermined class from an assessed class. This section supplies the implementing agent with the necessary behavior, not a new reporting obligation.

Specific cases:

- Table 8 E1/E2, printed p. 51, gives flushing-frequency headings `<10`, `<20`, `<40`, `>40` events/year and corresponding significance thresholds `QSpül/MQ >85%, >65%, >50%, >25%`. Exactly 40 belongs to no printed band. Preserve the source values and leave the affected screen undetermined. A useful discriminatory case is 40 events/year with `QSpül/MQ = 0.40`: assigning it to either neighboring band changes significance.
- Figure 25, section 5.8.3, printed p. 71, and Figure 27, section 5.9.3, printed p. 75, classify by coloured regions without assigning shared-boundary equality or unrestricted continuation beyond the finite plots. Reproducibly digitise supported regions and declare transcription precision. Exact shared boundaries, class ambiguity at that precision, and values outside the supported graphical domain do not receive an invented class. Valid interior points must compute successfully.
- In Figure 25 the horizontal variable is `I_h,Schwall = (QSchwall/MQr) * k_EZG`; the vertical variable is `I_P,S/S = V_S/S * k_PR,S/S`. The point `(0.1, 2)` is on a class-1/class-2 edge. Figure 27 uses `I_h,Spül = (QSpül/MQr) * k_PR,Spül * k_t` against flushing events/year; a frequency of 0.05/year is below its plotted minimum 0.1/year. These are source-domain cases, not a reason to reject all graphical assessment.
- Figure 27's frequency of 60/year is different: the prose explicitly requires examining whether hydropeaking should apply from 60 onward. Preserve that source-defined review requirement and supplied applicability judgement; do not treat it as another missing equality convention.
- The already-approved unassessed-upstream-area treatment remains unchanged: below 15%, omit missing areas from the assessed weighted calculation while retaining incomplete coverage; exactly 15%, unassessed with source-equality-unresolved meaning; above 15%, unassessed. Compare valid unrounded areas, not rounded percentages.

These defined undetermined outcomes satisfy the agreed handling of source gaps. They are not a blanket permission to leave required, source-defined capabilities unimplemented.

## Complete source-method behavior

### Inventory, reference and screening

Preserve the manual's near-natural reference under current landscape conditions and its stated exclusions (sections 2.1–2.3). Reference preparation and scientific acceptance remain supplied evidence, not something structural naturalisation alone proves. Keep Swiss regime-type tables and any explicitly supplied local adaptation separately attributable. Do not infer a Swiss type for an Uzbek river. Canals and drains do not automatically acquire a natural-river condition class.

Chapters 3–4 require indicator-specific measured, estimated and regime-reference routes, intervention type and magnitude, relative significance, cumulative effects and an attributable selection of indicators. Implement the source's numerical estimate routes, not only an interface accepting already classified values. Prepared spatial relations and supported event/hydraulic metrics are valid inputs where the source needs specialist evidence.

Table 8 grouping includes nearby same-type interventions on the same watercourse where the larger catchment is at most 15% larger. Preserve its actual strict/non-strict thresholds. Source exclusions, including intervention types with no significance criterion, are not measured proof of no impact. Impounded reaches are inventoried under section 4.2, not assigned a fabricated river class.

Table 9 (printed p. 56) is indicative: recommended, exceptional and indirectly excluded indicator selections differ. Supported situational additions/exclusions and still-significant upstream influences remain possible with attribution. Section 5.11's source-screened class 1 is not missing evidence.

### Nine indicators and their data routes

| Indicator | Required source behavior and discriminatory evidence |
|---|---|
| Mean-flow pattern | Section 5.2, Table 10 and Figure 17, pp. 58–59: `R = 100 * sum(abs(MMQr - MMQb)) / sum(MMQr)`. Preserve regime evidence and the best qualifying class from source regime quantiles or absolute thresholds. Transcribe all 16 regime types and three quantiles from the PDF table. R=36% gives type 3 class 2 and type 10 class 1. |
| Flood frequency | Sections 3.5 and 5.3, Figure 19, p. 61: preserve natural-frequency regime groups, source thresholds and independent instantaneous-event counting. The source threshold is `Q* = 0.6 MHQr`; event separation requires the specified intervening drop and at least five days before a new crossing. Exclude flushing events except the source's ecological artificial-flood exception. Daily-only evidence cannot certify unsupported event frequency. |
| Flood seasonality | Sections 3.5.4 and 5.4, Figure 21, p. 63: use circular timing across the year boundary. Keep direct-reference and reference-ellipse routes distinct; their class thresholds differ. Supported specialist ellipse inputs are permitted without HYDMOD-FIT. |
| Low-flow magnitude | Section 5.5, Figure 22, pp. 64–65: source-specific Q347 and lower hydropeaking-trough substitution where relevant, coefficient-of-variation and better-class exception, discharge-dependent thresholds and interpolation between the 50/200/500/1000 l/s anchors with source endpoint behavior. Increased flow relative to reference follows the source class-1 branch. Do not substitute unrelated Q347 conventions merely because another Fishy module uses the name. |
| Low-flow seasonality | Section 5.6 and Figure 21: preserve circular timing, direct-reference versus ellipse alternatives and their different boundaries. |
| Low-flow duration | Section 5.7, Figure 23, pp. 66–68: mean of annual longest consecutive spells at `Q ≤ Q347r`, not total annual low-flow days. Class intervals are `<20`, `[20,35)`, `[35,50)`, `[50,65)`, `≥65` days. |
| Hydropeaking | Sections 3.7 and 5.8, Figure 25, pp. 69–71: supported subdaily/source routes, representative ten-calendar-week low-flow sampling, daily extrema/ratio statistics, stage-change correction and hydraulic-stress catchment-area correction. Quantile of daily ratios is not ratio of quantiles; discharge rate is not stage rate. Retain raw metrics and graphical support. |
| Flushing and emptying | Section 5.9, Figure 27, pp. 72–75: source event frequency, hydraulic stress, interpolated rise correction and high/mean/low-flow timing factors 0.75/1/1.5. For different flushing types evaluate the combination yielding the worst classification, not their average. Preserve the hydropeaking applicability review from 60 events/year and the settled source-gap treatment. |
| Stormwater discharge | Section 5.10, Figure 28, p. 77: additional instantaneous stormwater events, not natural floods counted twice. Classes use `<0.2`, `[0.2,2)`, `[2,4)`, `[4,8)`, `≥8` events/year. |

Retain source-defined record windows, estimation choices, calendar/time meaning, units, uncertainty and required evidence per route. Daily means cannot prove the absence of within-day hydropeaking or supply missing stage-rate evidence. Missing site judgements remain supplied-input limitations, not invented national defaults.

### Spatial propagation and aggregation

Section 6.1.1, p. 79, defines reach changes at significant interventions, affected tributaries, catchment growth above 15%, groundwater doubling mean flow, and lakes with the specified `1 h * MQ` criterion. Downstream influence ends when all indicators are class 1 or the source `3 h * MQ` lake condition applies. Lakes themselves are not classified as river reaches. Prepared inputs may express these conditions; a GIS or native hydraulic solver is not required.

Section 6.1.2 and Figure 30, pp. 80–81, propagate each indicator independently using the nearest upstream interventions and unaffected intermediate areas at class 1. Avoid double counting nested upstream areas. Preserve the unrounded weighted result, the source's normal rounding to class and original coverage. Validate areas and topology. Apply the settled 14.9/15/15.1% missing-area behavior without allowing a rounded ratio to change the branch. Relevant Appendix A4 characteristic-quantity transfer remains in scope; it is not an off-plot extension of the classification graphs.

Section 6.4 and Figure 33, pp. 87–88, combine the worst indicator class with points 1/2/4/8/12 for classes 1–5. This is not a mean score or simply the worst indicator. Verify every matrix boundary against the primary figure:

| Worst indicator | Overall class from total points |
|---|---|
| 1 | 1 |
| 2 | below 11: 1; otherwise 2 |
| 3 | below 13: 1; below 15: 2; otherwise 3 |
| 4 | below 17: 2; below 23: 3; otherwise 4 |
| 5 | below 25: 3; below 31: 4; otherwise 5 |

A worse hydropeaking class overrides that result. Required missing indicators normally prevent aggregate assessment. Two known class-5 indicators establish overall class 5 even with incomplete coverage; retain that incompleteness. One class 5, two class 3 and six class 1 give 26 points and overall class 4 before any worse hydropeaking override.

## Integration and boundaries

Implementation belongs in `hydrosolutions/fishy`, not Taqsim. Discovery inspected Fishy main `f61b91b9b94f7055d640c608068d098babda8bea` and Taqsim main `0476f9587caa3db334db32b4219725f144280b86`. Recheck current target branches. Local `../fishy` was a stale scaffold checkout, so do not mistake its working tree for current main or overwrite its unrelated files.

Effort #9 is landed. Its durable vision is `planning/visions/2026-09-20-fishy-assessment-foundation.md`, pinned by that issue at `13d4b00b4dd4105312fc141cc3db53f69763fa71`. Current Fishy offers reusable attribution/evidence in `evidence.py`, time/presence in `flows.py`, versioned locations in `spatial.py`, exact quantities, optional live/saved exchange in `physical.py`, and supplied hydraulic-state assessment in `hydraulics.py`. Use suitable existing contracts without freezing these names as mandatory architecture.

No existing HYDMOD implementation was found. Generic evidence aggregation is not Figure 33 aggregation. Prepared location mapping is not the upstream-area algorithm. IHA pulses, annual means and dispersion are not substitutes for the HYDMOD source operators. Existing Swiss condition findings already allow separately supplied HYDMOD findings without a prescription uplift or mandatory import dependency.

Pure observation/import assessment requires neither Taqsim nor HYDMOD-FIT. A live simulator must not become a prerequisite. Preserve source/member/scenario/version, physical reach versus calculation/observation/compliance location, units, intervals, present zero, missing, outside-horizon and unsupported values. Separate numerical support, scientific suitability and official admissibility. Keep invalid inputs distinct from valid inputs whose source classification is undetermined.

Other native MSK modules, automatic prescription increases, national-policy selection, inferred Uzbek regime types, a general GIS/habitat/hydraulic solver, scenario orchestration, optimisation, report generation and PyPI publication are outside this Effort. No compatibility obligation to fishy-deprecated is introduced. Reversible API and implementation design choices belong to the implementing agent.

## Evidence of completion

Deliver a usable independently runnable HYDMOD path, documented with a concise supported example and partial-data behavior. Demonstrate successful inventory-to-indicator-to-reach-to-overall assessment, not an always-undetermined interface or only supplied final-class scoring.

Return the complete F9 and HYDMOD acceptance crosswalk: requirement and manual chapter/figure/table → public operation and input → observable expected/actual result → executed test, with exact software revisions and declared numerical tolerances. Cover all nine indicators, source estimates and supported imports, inventory/screens, graphical transcription, spatial propagation and aggregation. Exercise successful, missing, unsupported and infeasible cases.

Include at least the R=36% regime contrast, all Figure 33 transitions and hydropeaking override, 26-point/class-4 example, two-known-class-5 exception, screened-class-1 versus missing, unrounded 14.9/15/15.1% propagation, longest-spell versus total-days distinction with Q347 equality, independent flood events and flushing exclusions, circular year-end timing, direct-reference versus ellipse thresholds, daily-ratio versus ratio-of-quantiles, and stage-rate versus discharge-rate distinctions. Test the agreed undetermined source cases and successful graph interiors. Source-defined site review at frequency 60 must not disappear into a guessed class.

Retain source attribution and declared transcription precision in ordinary method/acceptance evidence. Do not claim a numerical transcription test establishes local ecological adequacy or official certification. The approved source gaps need no new escalation workflow and do not block completion of the defined scope. No implementation acceptance has been run by publishing this vision.
