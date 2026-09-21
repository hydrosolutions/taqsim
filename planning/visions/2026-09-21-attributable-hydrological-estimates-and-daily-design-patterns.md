# Attributable hydrological estimates and daily design patterns

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/29

## Outcome

Fishy calculates the specified annual hydrological estimates and conditional-analogue daily design patterns, or consumes supported imports, without confusing a computable number with an accepted scientific product. The modeller can inspect reference identity, support, uncertainty, exclusions, transformations and use restrictions for each result. Independent annual-only and imported products remain usable without a complete regime, basin pilot or live simulator.

Implement F2 and the U1, U2/D1–D14 and U10 acceptance obligations. This is the selected proposed Uzbek method, not a new Kazakh statutory algorithm, a WMO-prescribed distribution or an adopted Uzbek standard. Further optimisation of the selected pattern operator is not a prerequisite. This vision authorizes no scientific certification, national policy selection or basin calibration.

Implementation belongs in `hydrosolutions/fishy`. This document belongs in Taqsim as the Program's durable handoff. Engineering structure and final API signatures remain the implementing agent's responsibility.

## Authority and durable source evidence

Only `fishy_taqsim_handover_candidate_2026-09-19` supplies requirements. The owner approved it as superseding every earlier handover. Its internal candidate label does not reverse that decision. Held originals are at:

`/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`

Verified SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SHA256SUMS.txt` | `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0` |

Read the package README, brief, acceptance index, source index, source limitations and software baseline first. Then read the full relevant operators, equations and failure branches:

- `report_snapshot/part4_statistical_estimation.qmd`, D.7, `sec-statistical-estimation`, all five rules and synthetic checks.
- `report_snapshot/part4_design_patterns.qmd`, D.6, `sec-design-patterns`, `sec-pattern-method-selection` and `sec-pattern-development-evidence`.
- `report_snapshot/part4_scientific_acceptance.qmd`, D.8, `sec-scientific-acceptance`: acceptance record, product-specific tests and three scientific statuses.
- `report_snapshot/appendices.qmd`, Appendix A step 5, `sec-recipe-step5`: reference acceptance, rating restrictions and coarse-record meaning.
- `evidence/design_pattern_acceptance.md`, every D1–D14 case and supplemental deterministic checks.
- `report_snapshot/method_development_support/primary_case_metrics.csv` and `RECOMPUTED_SUMMARY.json`: retained development findings, not implementation acceptance.
- `ACCEPTANCE.md`, common C2/C3/C5/C6 and hydrology U1/U2/U10; `IMPLEMENTATION_BRIEF.md`, F2 and section 4 items 1–2.

WMO 1029 (`references/supporting_methods/meth_wmo_1029_low_flow_2008_en.pdf` and `.txt`, especially §§7.3–7.9 and chapters 3, 9, 10), the labelled `usgs_lowflow_frequency_2025_summary.txt`, and `knoben_benchmarks_2019.xml` support statistical appraisal and benchmark interpretation. They do not supply universal acceptance limits or the proposed pattern algorithm. Governing-language originals govern independent foreign profiles; current report operators govern this proposed Uzbek method.

The technical requirements and synthetic witnesses below carry the relevant derived contract durably. The local bundle is not falsely represented as GitHub-hosted. Do not upload private report chapters, original archives, restricted references or private application data to this public repository. Fishy's private status is not blanket permission to copy them either. If the authoritative bundle is unavailable or its identity differs, request that exact source rather than substituting older material. Preserve exact source/section/hash attribution in implementation evidence.

## Existing foundation and boundaries

Discovery inspected Taqsim remote main `4364485e07cc4dc5422305afd45164c6dd0a1845` and Fishy remote main `986acdc62a7984ff9b3629044e0b628f4ef508ff`. Recheck current targets before implementation. The local canonical Fishy checkout was behind remote main and still resembled the scaffold; it is not the current code baseline.

Foundation Effort #9 has landed. Its delivery record is https://github.com/hydrosolutions/taqsim/issues/9#issuecomment-5750531270. Current Fishy already separates physical input, evidence, requirement and assessment. Reuse sound contracts rather than introducing competing meanings:

- `FlowSample` binds location, exact interval, flow/presence/coverage, provenance and uncertainty. Flow validation rejects mixed identities and overlap; aggregation uses actual volume and duration and retains gaps and evidence limitations.
- `daily_discharge` projects supported whole UTC days. It does not infer daily detail from coarse data or impose a universal complete-year gate on all products.
- `Provenance` retains source, scenario, member, software/data/config versions, reference kind, dependencies and limitations. `EvidenceFindings` separates computability, numerical validity, disclosure, scientific adequacy, official admissibility and scoped restrictions. Known failure survives incomplete checks; empty checks cannot pass.
- Sanitary annual nearest-year selection and Swiss pooled-daily Q347 have different definitions. Do not replace either with Uzbek annual interpolation, borrow their acceptance rules, or relabel their outputs.
- Kazakh `SeasonalShape` already represents supplied contiguous duration-weighted mean-one receiving-year shapes and scales once. It does not implement this analogue construction. Its existing requirement for a shape must not override the accepted-zero branch here. Design classes P25/P50/P75/P95 are annual classes, not daily percentiles.
- Study-driven natural requirements and pulses remain independent. A baseline median cap must not suppress supported top-tier high-flow needs.

No new Taqsim routing or transport function follows from this scope. A design bound is neither extra water nor automatically a consumptive demand. Preserve compatibility of independent methods and issued results. The foundation's approved DHRAM exclusion is not reopened.

## Statistical products

### Identity, populations and support

Every product declares reach, units, natural/observed basis, reconstruction member, reference period and climate, accounting calendar, accepted years and exclusions, estimator/profile version, target and intended use. Annual mean discharge is annual volume divided by the actual year's duration. Annual magnitude and duration-specific annual/seasonal minima are separate populations; this effort does not implement the downstream multi-day safeguard.

Retain zeros and ties. Invalid negative/nonfinite discharge and unresolved missing values cannot enter these operators. Supported reconstruction may fill a gap with provenance; zero substitution may not. Correlated donors and simulated members do not create extra independent observed climate years.

For descending accepted annual values `x_1 >= ... >= x_n`, Weibull positions are `p_m = m/(n+1)`. Return an exact-position value or linear interpolation in discharge between its neighbours. Outside `[1/(n+1), n/(n+1)]`, ranking is unavailable, never extrapolated or clamped. Ties remain repeated observations. Source-year membership uses the arithmetic mean of tied rank positions without deleting tied observations.

### Selected fitted candidate and imports

Implement the stationary zero-mixture lognormal candidate when selected. For zero fraction `pi0 = n0/n`, positive log observations `z_i`, use their mean `mu` and maximum-likelihood variance `sigma² = sum((z_i-mu)²)/n_positive`. At target exceedance `0 < P < 1`, let `u = 1-P`. The quantile is zero when `u <= pi0`; otherwise it is `exp(mu + sigma * inverse_normal((u-pi0)/(1-pi0)))`.

The fit requires at least two positive observations and positive log standard deviation, including when the requested quantile would land on the zero atom. All-zero and constant-positive samples cannot claim this mixed fit. Preserve empirical zero estimates without inferring permanent dryness. No observed zeros does not prove zero-flow probability absent. Fitted source-year membership uses midpoint exceedance at atoms, `1 - (F(x-)+F(x))/2`, and `1-F(x)` elsewhere. A low-flow return period uses `u=1/Tr`, `Tr>1`, for the declared minimum population, not daily exceedance.

This is a candidate, not automatic best fit. Supported imports may use other families or nonstationary methods, retaining reproducible derivation and identity. Nonstationary imports name the equation, covariates, evaluation date/scenario, calibration, diagnostics, extrapolation and uncertainty. An unexplained number or model label is insufficient. Untreated genuine trend retains observations and exploratory statistics but cannot become an accepted pooled stationary present-climate estimate.

Freeze diagnostics and intended-use limits before selecting a requirement. Where the report's distribution diagnostic is used, compute `sup |Fn-F|` on both sides of observed jumps, without claiming independent-sample significance for dependent data. A whole-distribution fit or narrow interval does not establish rare-tail validity.

### Uncertainty contract

The report's stationary sampling profile uses circular moving blocks on consecutive complete annual sequences: integer `1 <= L < n`, `B >= 2`, declared generator/version/seed, uniform block starts, chronological length-L blocks concatenated and truncated to n observations, refitting the same estimator. `L=1` requires justified independence; other lengths require dependence support. Missing years cannot silently become adjacent. Related quantities/stations use the same sampled year indices.

For `0 < alpha < 1`, interval endpoints are replicate quantiles at `alpha/2` and `1-alpha/2`, using ascending linear interpolation at index `1+(B-1)v`. Any failed fit or estimate means no interval under this profile; retain every failure count and reason. Never drop failed replicates, substitute zero or imply that increasing B repairs structural failure. Statistical circular sampling grants no permission to wrap a physical hydrograph.

Fishy need not orchestrate resampling, validation or alternative-model studies. External specialist results are valid supported inputs, but their method, dependence and failed-replicate coverage remain enforceable and observable. Sampling intervals are not total uncertainty. Supported linear covariance propagation uses `cᵀΣc`, a compatible positive-semidefinite covariance matrix, and a separate coverage/distribution interpretation before any standard deviation becomes a verdict half-width. Do not invent a general fitting platform or joint nonlinear error model.

## Daily conditional-analogue construction

### Selection and zero target

Use one versioned profile fixed before evaluating requirements: target P, calendar, reference/member and estimator/ties, band h, donor eligibility, support criteria, alignment settings and acceptance tests. Numeric donor/season/shift/support/acceptance values are supplied study settings, not national defaults. The same operator prepares median, 99% and shifted targets without applying a second shift or defining actual-year operational classes.

If the receiving annual magnitude is accepted as zero for a nonnegative reference, produce the zero design hydrograph on a valid calendar before testing positive-source membership or shape support. Its mean-one shape is undefined and unnecessary. Retain magnitude/intermittency evidence and all other restrictions, bounds and safeguards. An unsupported zero estimate does not qualify for this exception.

For positive targets, eligible sources are complete quality-controlled daily natural-reference years or explicit validated reconstructions on the declared climate basis. Donor suitability needs supported hydrological similarity, not proximity or correlation alone. Keep members separate and the eligible reference set fixed independently of the resulting obligation.

Compute source-year probabilities from each source's full accepted annual reference distribution before selecting the neighbourhood, using mean discharge rather than volume ranks across unequal year lengths. Select positive years in the closed clipped band `[max(0,P-h), min(1,P+h)]`, `0 <= h <= 1`, retaining boundary ties. Keep zero years in annual statistics/intermittency evidence, although they cannot supply positive shapes.

Every selected source year contributes once with equal weight. Report donor contributions, source count, distinct climate/accounting-year clusters, selected probability range/centre, nearest distance to P, and separate magnitude/shape extrapolation. P outside the selected probability range is shape extrapolation even if membership is large. No automatic widening, replacement class or pseudo-replication of synchronous droughts is allowed.

### Calendar mapping and alignment

Normalise each source year's daily volumes to annual shares. Supported accounting years start on the first day of the same calendar month and use fixed 86,400-second UTC/fixed-offset days. Other calendars need an explicit supported conversion or remain unsupported. Map within-month cumulative volume by overlaps of normalised source and target day intervals, preserving monthly and annual shares, including leap February. This transformation creates no newly observed daily detail.

A configured melt season crossing the design-year boundary uses documented calendar fallback, not a linear December/January median. Otherwise start with selected calendar-mapped years with identifiable positive seasonal volume. The timing marker is the first cumulative half-volume crossing, with first-attaining boundary on a plateau and fractional within-day interpolation as a declared timing convention. Express markers in elapsed seconds from design-year start. Use their median, midpoint for an even count; each shift is median minus its marker.

Simultaneously exclude all contributions exceeding the declared maximum absolute shift, lacking required adjacent-year context, or retaining zero design-interval volume. Recompute the median and repeat until stable, without reinstating an aligned-set exclusion. Input order must not alter the retained set or result. Record iterations and reasons.

Never wrap year-end data to the beginning or zero-fill unknown edges. Map needed adjacent source years to adjacent design years using the same calendar rule, scaling their volumes relative to the originally selected source year's total rather than independently normalising padding years. Record introduced and displaced volume. Renormalise each complete mapped/aligned contribution to unit total before averaging, using one fixed retained set for the whole design year.

If alignment fails declared support, the calendar-average fallback may use the otherwise eligible original selected years, recording why alignment was unavailable. This differs from reinstatement inside the alignment iteration. It changes alignment only; it cannot cure inadequate dry-year membership, donor suitability or tail evidence. Reject incomplete contributions and zero retained totals, subject to the separate accepted-zero-target branch.

### Shape, scaling and boundaries

For mean daily shares u on N design days, `s(t) = N*u(t)/sum(u)` and `Q(P,t) = receiving_annual_mean(P)*s(t)`. Assert nonnegativity, mean shape one and total volume `receiving_annual_mean*N*86400`. Receiving magnitude, not donor river size, sets scale. Apply it once.

Return membership, mapped/aligned shares, excluded contributions, fallback choice and support diagnostics, alongside shape and scaled schedule. Supported imports retain equivalent temporal, mean/volume, provenance and use meaning. Shape closure does not establish low-flow accuracy or scientific adequacy.

Prepare median and 99% bounds separately. Annual ordering does not imply pointwise daily ordering. Expose crossings without sorting, clipping, blending reference members or silently repairing ordinates. Final baseline assembly/descent, duration-minimum construction and multi-day safeguards belong to #31. Pattern construction does not automatically provide predecessor context for downstream rolling windows.

## Scientific-use findings

Maintain separate numerical validity, disclosure, adequacy for a stated use and official admissibility. Scientific outcomes are `accepted`, `accepted as indicative` and `not accepted`, with scoped reasons and restrictions, never averaged into a score. Official permission pending does not itself mean scientific inadequacy. Short record length alone neither rejects a method nor forces descent; ranking support is route-specific and supported fitted/imported alternatives remain possible.

Acceptance records identify preparer and independent reviewer, scope, frozen tests and limits, metrics/formulas/domains/units, aggregation, equality, applicability, uncertainty and use permissions. Equality at a maximum permitted error passes unless a stricter rule was declared. Missing defensible criteria means acceptance basis missing, not invented defaults; preserve exploratory diagnostics. Changing criteria after seeing validation requires a versioned profile and new or nested withheld validation.

Accept magnitude, daily shape, duration minima and rare-tail applicability separately. Preserve signed and absolute per-case diagnostics, seasonal-share errors, non-wrapping half-volume timing, duration minima and longest strictly-below-threshold spells. Zero-denominator relative metrics remain undefined, with applicable absolute metrics retained. Caller-owned withheld validation keeps climate-year clusters out of both training and validation and includes donor/reference sensitivity where applicable. Less-extreme holdouts, annual volume closure and synthetic software tests cannot certify a 99% tail.

Rating-range sizing prohibitions survive successful fitting and indicative status. Untreated nonstationarity cannot acquire present-climate acceptance. Retain actual dekadal intervals and names: a dekadal percentile is not Q347 or daily Q95. Daily disaggregation requires withheld co-located daily low-tail and spell validation with propagated uncertainty; monthly-total preservation alone is insufficient. Unsupported daily products must not disable independently supported mean/norm/coarse products.

Preserve the exact development limitation: all 21 dry held-out cases in the two-US-record, 50-year-training comparison overestimated the seven-day minimum. Mean shape performance improved over the calendar benchmark in four station–period groups; longer training improved it in three, not all four. These correlated development cases are neither naturalised Uzbek evidence nor untouched independent rare-tail validation. They supply no correction factor or universal tolerance. A successful downstream multi-day safeguard cannot repair a failed or unresolved required daily-pattern acceptance criterion.

## Observable acceptance

Return a requirement → public operation/input → observable result → executed test crosswalk, actual-versus-expected values, declared tolerances and exact software/dependency revisions. Include successful, missing, unsupported and infeasible paths. Do not substitute source traceability, fixture arithmetic, package review or always-pending responses for implementation acceptance.

Required discriminating witnesses include:

| Coverage | Observable result |
| --- | --- |
| U1 empirical | 40/30/20/10 gives P50=25; P99 ranking unavailable, not clamped. |
| U1 ties | 40/20/20/10 retains both 20 observations and gives each source-year probability .5. |
| U1 fit | `[0, exp(-1), 1, exp(1)]` gives pi0=.25, mu=0, variance=2/3; P=.375 gives 1 and P=.75 gives 0. |
| U1 uncertainty/trend | A replicate with one distinct positive value prevents the interval; untreated trend cannot yield accepted present-climate sizing. Supported imports preserve provenance. |
| D1–D3 | Closed band retains .97/.99 boundaries; P=.99 outside selected [.95,.97] flags shape extrapolation; five stations across two climate clusters do not meet a configured three-cluster minimum. |
| D4–D5 | Equal-year shape `[1.5,1.5,1,0]`; receiving mean 8 gives `[12,12,8,0]`, with no second scaling. |
| D6–D7 | February 28-to-29 mapping splits first source-day share as `[28/29,1/29]`; timing markers 100/104 align to 102 with shifts +2/-2. |
| D8–D9 | Missing shifted edge causes exclusion/recomputation or attributable fallback, never wrapping; per-year truncation normalisation gives the supplied `[.625,.125,.125,.125]` average. |
| D10–D11 | Zero source is not a positive-target shape; accepted zero target bypasses shape support. Daily `[0,4,4,0]` crosses median `[3,3,3,3]` despite annual ordering; retain failure without clipping. |
| D12 | Mean 8 over 365/366 days gives 252,288,000 / 252,979,200 m³. |
| D13–D14 | Missing/invalid profile, invalid h/volumes, incompatible members and zero retained total are refused or attributable unresolved outcomes; boundary-crossing melt season uses calendar fallback. |
| Supplemental patterns | Row-order-invariant simultaneous exclusions and median recomputation, plateau/tie conventions, fractional overlap, original-year-scaled adjacent context, fixed contributors, empty-set rejection and logged reasons. |
| U10 | Independently supported short-record/annual/imported/coarse results remain useful, while rating, nonstationarity, daily-equivalence, signed duration-minimum and rare-tail restrictions bind. Retain the 21/21 limitation. |
| Common isolation | Different methods, references, members and scenarios do not contaminate one another; present zero differs from missing; known failure survives incomplete evidence. |

The short synthetic arrays are arithmetic witnesses, not a license to weaken complete-year requirements at a public daily-pattern boundary. Exercise the actual public operations with valid real-calendar inputs as well as the specified operator witnesses. Test configured support and scientific-use gates without adopting synthetic thresholds as policy.

Supply supported user examples for annual estimation, pattern construction and imports, including an indicative/restricted result and a missing-support result. Explain ordinary use and current behavior. Preserve independently runnable Fishy installation without a mandatory simulator. No PyPI publication milestone is required.

## Exclusions and remaining application responsibility

No new reference-reconstruction, forecasting, validation-orchestration, ensemble-comparison or optimisation engine; no automatic naturalisation acceptance; no calibrated basin application; no requirement to install original Kazakh/Swiss software. Do not expand into #31's route selection, complete regime/floor assembly, final safeguard, delivery cap or issued-duty assessment. This effort supplies inspectable hydrological inputs for that work.

Donor eligibility, band widths, seasons, shift limits, climate basis, reconstructions, statistical tail evidence, acceptance tolerances and official decisions remain explicit application inputs. Their absence is an attributable outcome, not a request to invent universal defaults. No unresolved generic operator choice was identified during discovery. Bring any newly found genuine source contradiction back with the exact operation and alternatives; continue unaffected work.
