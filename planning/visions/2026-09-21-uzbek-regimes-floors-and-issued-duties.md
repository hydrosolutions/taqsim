# Uzbek regimes, floors and issued duties

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/31

## Outcome

Fishy completes the proposed Uzbek calculation chain from evidence-aware route selection through uncapped regime assembly, final safeguards, floors, versioned issuance and supported delivery assessment. A modeller can obtain a complete result from sufficient supplied inputs, inspect every intermediate and restriction, and distinguish that successful result from missing, unsupported or infeasible paths. Shortages must not change the requirement to make an operating result appear adequate.

Implement F1/F6/F7/F10 and this Effort's integrated acceptance: U3/M1–M15, U4–U7, U9, the entry supplements, all ten floor-uncertainty fixtures, C4 and combined-duty/source-isolation checks. Study, hydrology and quality capabilities already delivered by other Efforts are inputs to this chain, not substitutes for integrated acceptance. Implementation belongs in `hydrosolutions/fishy`; this vision belongs in Taqsim as the Program handoff. Final APIs and reversible engineering choices belong to the implementer.

This is a configurable proposed method, not adopted Uzbek policy or a calibrated basin application. Hypothetical settings and supported synthetic evidence can exercise it. Missing policy values, scientific acceptance or application evidence remain explicit inputs or unavailable outcomes, never invented national defaults.

## Authority and durable evidence

Only `fishy_taqsim_handover_candidate_2026-09-19` supplies requirements. The owner approved it as superseding every previous handover. Its candidate label does not reverse that decision. Held originals are at:

`/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`

Verified SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SHA256SUMS.txt` | `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0` |

Read README, brief, acceptance index, source index, source limitations and software baseline first. The bundle's software baseline is historical, not current Fishy. Then read the actual equations, branches and failure conditions in:

- `report_snapshot/appendices.qmd`, Appendix A steps 1–12, especially `sec-recipe-fork`, `sec-recipe-step5`, `sec-recipe-step6`, `sec-recipe-step7`, `sec-recipe-step10`, `sec-recipe-step11` and `sec-floor-uncertainty`.
- `report_snapshot/part4_assembly_contract.qmd`, D.10, `sec-member-selection`, `sec-donor-transfer`, `sec-component-assembly` and the complete synthetic calculation.
- `report_snapshot/part4_lowflow_safeguard.qmd`, D.5, `sec-lowflow-safeguard`, and every case and supplement in `evidence/multiday_acceptance.md`.
- `report_snapshot/part4_statistical_estimation.qmd`, `part4_design_patterns.qmd` and `part4_scientific_acceptance.qmd` for hydrological input meaning, estimator identity, separate product acceptance and restrictions.
- `report_snapshot/part4_quality_calculation.qmd`, `part4_hydraulic_assessment.qmd` and the natural/potential/receptor provisions in Appendix A for local activation, final physical checks and imported specialist support.
- `report_snapshot/part1_sanitary_baseline.qmd` and `part1_sanitary_scenarios.qmd` for the separate existing-duty and hypothetical-estimator boundaries.
- `evidence/synthetic_chain.json` and `report_snapshot/quality_support/floor_uncertainty_fixtures.json` for exact numerical witnesses.
- `IMPLEMENTATION_BRIEF.md` sections 2, 4–6 and `ACCEPTANCE.md` common C1–C6, U3–U7/U9 and entry/floor supplements.

The current QMD operators govern the proposed Uzbek method. The original Kazakh Order 179 official PDF, especially paragraphs 17–20 and 25–30, supplies attributed background, not authority to replace the explicit Uzbek readings with foreign branches. WMO 1029 sections 2.3, 5.4.3 and chapter 7 support duration-minimum statistics; the held USGS low-flow file is a labelled authored summary. Neither supplies Uzbek ecological durations, return periods, acceptance limits or the proposed final gate. The Water Code supplies the governing frame; Пакет 7272 remains a nonbinding draft. Scenario success does not authenticate law or adopt a profile.

The derived technical contract and synthetic witnesses below carry relevant evidence durably. Do not upload private report chapters, source archives, restricted originals or application data into this public repository. Fishy's private status is not blanket copying permission either. If the exact bundle is unavailable or its identity differs, request it; do not silently use an older handover. Preserve section and source identities in implementation evidence.

## Existing foundations and integration boundaries

Discovery verified Taqsim main `9972f53d3e1b63f3071f3154cf348672f8441426`. Fishy remote main inspected during discovery was `13d39b7439ef89d397b1ed67116920cfdd508248`. Recheck current targets before implementation. The canonical local Fishy checkout was still at the scaffold and behind main; inspect current Git objects or a safe current checkout, not stale working files.

Dependencies #24 (quality), #28 (study/hydraulic/receptor) and #29 (hydrology) have landed. Foundation #9 and sanitary #27 also supply relevant contracts. Reuse sound domain meanings without treating previous acceptance as proof of this assembly:

- Frozen requirement, floor, availability, deliverability, obligation and delivery records already exist. Existing `assess_duty` returns nominal shortfall and uncertainty separately; its nominal summary is not this Effort's supported-uncertainty floor/but-for verdict. Do not introduce the Uzbek cap inside an independent prescribed-duty assessor.
- Provenance and evidence findings separate reference/scenario/member identity, computability, scientific adequacy, official admissibility and scoped restrictions. Hydrology products have subject-bound scientific assessments; permission on one product cannot silently transfer to a changed final schedule. Low-flow frequency identifies return-period meaning but does not construct duration-minimum populations or implement this gate.
- Quality activation uses the supplied background mapping, source-control ordering, original-test recheck and full feasible intervals. Hypothetical activation does not waive physical support. A later ecological/receptor uplift may violate a quality upper bound.
- Natural study pulses may exceed the baseline median cap. Potential outputs remain floor-only, with winter-share sizing suspended and unresolved ecological conditions retained. Service conveyance does not prove ecological adequacy.
- Assumed sanitary sizing does not activate domestic ecological entry. Supplied sanitary duties remain independently assessable without that estimator. Combined assessment does not establish universal precedence, sum or maximum across unrelated duties.

No new physical simulator or Taqsim transport feature follows from this scope. Supplied observations and supported imports can run without a live simulator. Requirements and safeguards are neither water sources nor automatically consumptive demands.

## Evidence, identities and route selection

Keep physical water body, calculation reach and basin-plan unit as separate versioned identities. Calculation, observation and compliance points have different roles. A provisional or absent plan assignment does not alone disable supported calculation; a later assignment without boundary change differs from a split needing recalculation. Origin/designation, use category, duty basis and replacement history remain explicit.

Natural origin versus constructed origin or valid heavily-modified designation determines track, not use category: an irrigation river remains natural and a drinking-water canal artificial. Undetermined track leaves dependent routing unresolved. Preserve explicit hypothetical assignments. Valid designation prevents a new natural-track calculation, while the prior issued natural version persists until competent supported replacement. Potential routing skips natural reconstruction and the natural tier ladder.

For natural reaches, select the highest specified, resourced, eligible tier whose required statistics are accepted or accepted as indicative and permitted for the stated use. Top tier also needs its priority/trigger basis. Record the highest tier supported by data separately from the selected tier and explain resource, priority and data restrictions. Short series alone do not force descent. Numerical availability, disclosure, adequacy for purpose and official admissibility remain different findings. Rating-range sizing prohibitions and unsupported daily equivalence survive otherwise indicative status. Unknown quality category may limit quality without disabling independent hydrology.

Preserve present-climate natural reference, altered observations and future stress scenarios separately. Structural naturalisation, a fitted value, conserved volume or a passed safeguard does not establish accepted reference hydrology or rare-tail/daily-shape adequacy. Retain the bundled 21/21 dry-case seven-day-minimum overestimation limitation; it supplies no correction factor or new tolerance.

### Entry, natural studies and fallback

Domestic ecological entry follows ordered availability, source applicability and statistic adequacy. In the current profile its enabling contracts are absent, so it remains unavailable. The separate assumed sanitary estimator cannot supply them. Missing availability or scope leads to the Swiss-derived entry option; failure of domestic statistic adequacy follows the prescribed fallback rather than shopping for a different estimator.

Implement the separate Uzbek graduated curve from explicit anchors, breakpoints and marginal rates: piecewise linear, continuous and concave, calculated without display rounding. A returned floor greater than or equal to its input refuses the route; do not cap it to the input. Require the supplied reproducible large-river screen and retain missing-screen status. No numeric screen or adopted table is invented. Label explicit scenario coefficients. The literal Swiss table remains valid in its independent method despite its published jumps and must not be substituted for this curve.

Eligible natural top-tier sizing consumes supported habitat/holistic findings and explicit acceptance/selection criteria, including required high-flow pulses, timing and hydraulic conditions. The baseline median cap cannot suppress those needs. Missing study evidence retains the need and invokes eligible descent; this is required study-input support, not optional Spanish expansion or a native habitat solver.

After route failure, retain its diagnostics and try the next eligible supported tier, then qualified donor transfer, then the presumptive floor, otherwise named pending. Mark failed routes so the same attempt cannot select them again. Presumptive sizing uses a supplied seasonal fraction and defined recipient accepted present-climate natural reference. No computable basis means no invented zero or positive default. Preserve all existing independent duties.

## Four-class baseline

Compute the design family independently of actual-year operational selection. Classes 25/50/75/95 use shifted probabilities 50/75/90/97. Standard class flow is shifted annual natural mean times the supported mean-one shifted shape. A qualified reach coefficient replaces the annual magnitude with `alpha(c) * natural_mean(50%)`, retaining the shifted shape; it never multiplies an already shifted magnitude. Missing alpha does not disable the standard route. Retain coefficient qualification and derivation.

The recorded minimum is one scalar: the lowest daily flow in the accepted naturalised record over its declared period, with day/member/period, missing-data treatment and uncertainty. Preserve supported zero. Do not substitute a daywise minimum series. Natural99 and natural50 bounds are their own-probability annual means times their own supported shapes, not daily percentiles and not shifted again. Scaling uses the actual receiving-year duration and happens once.

For each class and day, first form:

`lower = max(class_flow, natural99, recorded_minimum, applicable_winter)`.

The winter term enters only November–December on regulated natural rivers where the provision and its reference are supplied and applicable; otherwise it is inactive. A labelled scenario can explicitly assume its adoption basis. Do not apply regulated-body winter-share sizing to the potential track.

Test `lower > natural50` before any cap. The current interpretation is strict zero tolerance and reach-wide family failure: one crossing declines the complete family. Do not add an arbitrary tolerance switch. A future nonzero tolerance requires an explicit supported reconciliation of bound terms and failure scope, not just a changed numeric constant.

Only after passing that test form:

`base = max(min(class_flow, natural50), natural99, recorded_minimum, applicable_winter)`.

The cap acts on the class component only and is inert under the current zero-tolerance rule. It must not later cap supported quality/receptor additions or study-based top-tier needs. No obsolete daily `Q_lfr` operand exists. Retain the provisional multi-day check on this ecological base.

Supported Uzbek spawning correction is a separately reported advisory result, with class eligibility, biological timing and coefficient provenance; it does not enter the binding baseline. Original Kazakh spawning branches remain independent.

Actual-year selection needs a supplied versioned, complete non-overlapping exceedance-interval convention with endpoint/tie rules, accounting year and accepted reference. Forecast issue date and assumptions remain distinct from later observed classification. Missing convention leaves selection pending, not a nearest-class default. It does not prevent design-family computation or permit retrospective changes to issued duties.

## Duration-minimum construction and the separate safeguard

Construct or consume attributable duration-specific annual/seasonal minima. Configure positive integer duration d, return period T>1, control section/member, fixed-day boundaries, annual or named continuous seasonal domain, estimator, reference period and uncertainty/adequacy basis. None is a national numeric default. Use 86,400-second UTC/fixed-offset days; convert other representations explicitly with volume and time meaning preserved.

Window flow is total volume divided by duration. Equal complete daily means give their arithmetic mean. Coarse intervals cannot supply a shorter window or unknown fraction; daily means do not establish within-day constancy. On the annual route assign windows by the accounting year of the last daily interval, using real predecessor context across boundaries. Seasonal windows lie wholly within the named continuous season, labelled by its ending year. The same eligibility rules govern reference estimation and candidate testing.

For each complete reference block, retain the minimum across every eligible window. An incomplete block needs explicit supported reconstruction or reported exclusion; it is not a fully observed minimum. Estimate the lower-tail inverse of that minimum population at 1/T, retaining empirical/fitted/imported estimator identity, zeros/ties, dependence, nonstationarity and restrictions. Reuse delivered statistical operators without confusing annual magnitudes with duration minima.

For every configured test and required class, test every eligible candidate window against the threshold. Return minima and locations, per-window shortfalls, threshold, coverage and uncertainty. Point equality passes. With supported bounds, a window is certainly adequate only when its lower mean bound is at least the threshold upper bound, and certainly inadequate when its upper mean bound is below the threshold lower bound. Overlap is indeterminate. Missing uncertainty is not measured certainty; explicit fixed synthetic values may retain their assumptions.

Known violating windows establish failure even with other gaps. No known violation plus missing required windows or inputs is indeterminate; no eligible windows cannot pass. Multiple tests remain separate and all applicable ones must pass, never average into a score.

Keep the provisional ecological result. The final baseline issuance gate is on the completed, active-quality/receptor-assembled uncapped requirement, for every class, before deriving its floor or applying the delivery cap. Keep member checks and repeat applicable checks on the selected family against retained-member evidence. Active supported uplift can repair provisional duration failure; advisory uplift cannot. Neither repairs inconsistent natural bounds or failed scientific pattern acceptance.

Predecessor context must be supplied and justified. No circular wrapping, boundary repetition or zero filling is implicit. An explicit supported hypothetical repeated-year context remains labelled hypothetical. Final failure or indeterminacy declines the complete baseline regime/floor and invokes descent, preserving intermediate results and independent duties. Do not impose this baseline-only gate on donor, entry or potential routes without a separate configured basis.

The test is not a daily floor, schedule-correction algorithm or return-period drought permission. Assess realised operating flow separately; operating failure cannot rewrite the issued requirement.

## Qualified donor transfer

Transfer a qualified donor's pre-quality ecological regime, excluding quality uplift, receptor diversion, sanitary/service duty, delivery cap and advisory spawning. For the same unshifted class c, form `ratio = donor_ecological(c,t) / donor_natural(c,t)` and multiply its supported calendar mapping by the recipient's own natural hydrograph at c. No catchment-area scaling or second probability shift occurs.

Zero donor denominator, including 0/0, is unsupported. Do not fill from adjacent days or zero. Ratios above one remain visible with a transfer-adequacy flag; supported finite ratio times recipient zero is zero. Calendar mapping is the overlap-weighted mean of the dimensionless ratio on normalised within-month intervals, not volume mapping. Require matching accounting-year starts/fixed-day conventions, complete class/period mapping, and retain transformation sensitivity. No additional melt shift.

Select one justified qualified donor before looking at the answer. Qualification covers regime, intermittency, ecological purpose, transferable relationship and recipient evidence. A transferred donor cannot seed another transfer; retain a directed cycle-free register. Alternative donors are separate sensitivity configurations, not pooled members, and cannot replace structurally different recipient reconstructions. Retain donor uncertainty and recipient identity.

Recheck local quality, receptors, duties and natural-floor consistency. A specifically commissioned transfer safeguard remains applicable, but absence of the baseline-only gate cannot falsely disqualify or certify transfer. Failed transfer proceeds to its applicable fallback.

## Component composition and externally selected families

Keep location, interval, physical pathway and constraint meaning attached to every component. Shared lower bounds on the same through-flow combine by maximum. A lateral diversion needs the signed carrier balance and supported travel/storage/loss relation to obtain an upstream equivalent. Missing mapping leaves separate needs visible and integration unresolved, not a universal sum or maximum. Multiple pathways require a supplied selected schedule, not automatic allocation. Non-monotonic/state-dependent relations require supported supplied solutions and verification.

Active quality preserves the route-specific comparison, source-control-first ordering, background mapping and original tests, including upper/open bounds and infeasible intervals. Hypothetical activation is explicit; advisory quality cannot silently become binding. Repeat local quality and receptor/hydraulic checks after any uplift. Supported independent sanitary/foreign duties retain their own rules and compatibility findings.

The caller runs members and performs structural selection. Fishy exposes complete member candidates and diagnostics and assesses the supplied selected family and its provenance. Require one fixed retained member set across all required days/classes, with identical location/calendar/method/parameter/activation basis. Different policy configurations are separate studies, not uncertainty members. Record failed/incomplete exclusions; no daywise or classwise changing member set.

Where reconstruction evidence is required, at least two structurally different accepted members are needed for complete structural selection. One retains its supported screening candidate with selection incomplete; none supplies no selectable result. Do not add this reconstruction requirement to observed-statistic entry, supplied duties or artificial service-conveyance routes that do not need it.

For complete member requirements at each class/day compute median M (middle-pair mean for even count), minimum L and maximum H. Spread is `(H-L)/M` for M>0, zero for all-zero members and infinity for M=0<H. D is the maximum spread over the whole family. At `D <= theta` select the median everywhere; above theta select the upper envelope everywhere. The report's provisional theta is .20, explicitly configured, not an uncertainty confidence limit or adopted policy. Retain M/L/H/D, exclusions, dependence and every attaining contributor/tie. Negative/nonfinite requirements are invalid.

Select uncapped requirements, never delivery-capped values. A selected median/envelope may combine requirements but is not a mosaic natural reconstruction. Keep each member's pre-quality natural-bound test; do not reconstruct new natural bounds or cap later additions. Recheck the selected family against final quality, receptors, applicable habitat/hydraulic/ramping conditions and retained-member baseline safeguard evidence. Intrinsic failure/indeterminacy prevents issue through that route; physical shortage instead preserves requirements and reports deficits.

For floor-only routes use complete final floor series over their declared period, not invented classes or full seasonal obligations. Repeat their own feasibility/quality checks and preserve their ordinary adequacy conditions.

## Floors, issuance and assessment

Natural regime-producing baseline, top-tier and qualified-transfer routes produce an uncapped final requirement. Its scalar floor is the annual minimum of the final selected 95% class after quality/receptor assembly. Require that floor not exceed any day of any required class. A crossing declines the complete family and triggers descent, not a lowered floor, lifted class or convenient class subset. Existing issued versions persist.

Entry, presumptive and potential routes produce only final quality-adjusted floors. They cannot invent a full requirement or newly derived delivery obligation. Potential hierarchy remains supplied habitat, joint velocity/depth, then authenticated service conveyance, preserving residual/convergence/capacity and applicable temporal checks; a last nonconverged iterate is not a result. Separate service, sanitary and receptor schedules remain independently assessable. Floor substitution is not enabled by this proposal.

Retain requirement, floor, physical availability, deliverability, issued obligation and actual delivery as distinct location/interval/versioned quantities. For a newly issued Uzbek obligation, `obligation = min(requirement, deliverability)` and ecological deficit is requirement minus obligation. Later actual flow does not recalculate either obligation or floor. Independent foreign/sanitary duties retain their method-specific rules.

Assess delivery against the immutable issued obligation using supported uncertainty and evidence conditions, retaining raw shortfalls and coverage. A surplus at a different interval does not erase a deficit. Return numeric results separately from admissibility, scenario/observation meaning and causality; do not mistake an existing nominal summary for the required uncertainty-aware finding.

For the separate floor test, keep the issued floor F fixed and compare actual A with `min(F, but_for P)` at the same control point and interval. The delivery cap has no role. Supply finite, nonnegative, ordered supported bounds for A and P, with uncertainty meaning, source and coverage. Missing uncertainty is not a zero-width interval. Supported hypothetical singleton bounds remain explicit assumptions.

Use `G_L = A_L - min(F, P_U)` and `G_U = A_U - min(F, P_L)`. If G_U<0 the result is below throughout support; if G_L>=0 it is not below; otherwise indeterminate. Equality is not strict shortfall. Missing/rejected prerequisite evidence makes the comparison unavailable, not merely overlapping or cause-indeterminate. A supported joint margin can tighten this enclosure with its dependence/method/coverage retained; finite draws alone do not establish enclosing support or joint confidence. Do not infer independence from marginal confidence levels.

Numerical shortfall and attribution to the tested conduct remain separate. A scenario does not establish legal responsibility. Missing designation, metering, admitted evidence or attribution prevents the corresponding official finding while supported technical results remain visible. A not-below floor result concerns this comparator, not every duty.

Version inputs, profiles, references, software, assumptions and results. Recalculation or revision produces a new identified record and retains the prior issued version and reason; it does not mutate history or confer competent replacement authority.

## Observable acceptance

Return actual-versus-expected outputs, declared precision/tolerances, exact compatible software revisions and a requirement → public operation/input → observable result → executed test crosswalk. Include every owned case and supplement, with successful, missing, unsupported and infeasible paths. Traceability, synthetic paper arithmetic and package review are not implementation acceptance. Test the real public boundaries with complete supported calendars and evidence, not weakened miniature substitutes for complete-year requirements.

Required witnesses include:

| Coverage | Observable distinction |
| --- | --- |
| Entry supplements | Configured continuity/concavity, segment junction equality, unrounded evaluation, floor>=input refusal, supplied large-river screen and missing-status branch; independent literal Swiss behavior unchanged. |
| U4 baseline | Four shifted classes, alternative magnitude without double reduction, scalar minimum, own-probability natural99/50, conditional winter, pre-cap crossing, advisory spawning and pending actual-year selection. |
| U3 | Every M1–M15 case and supplemental window/calendar/uncertainty/reference-block check. `[4,8,12,14,12,10,10]` has seven-day mean10 but violates separate floor6 on day1. Active8→10 uplift can repair final duration failure while its provisional failure remains visible. |
| M14 estimator | Empirical duration minima `[1,2,3,4]`, T2, Weibull/interpolation =>2.5. Imported inverse-ECDF2 retains its distinct estimator, never silently substitutes. |
| U5 selection | Members8/12 select12 before deliverability5 gives obligation5/deficit7;9/11 selects10 at theta=.20 equality; swapped contributors produce envelope with identities; all-zero =>0;0/0/3 uses upper branch; one member remains screening. |
| U6 transfer | Ecological2/4 over donor natural4/8 and recipient6/10 =>3/5. Donor quality uplift excluded; local supported quality applied afterward. Zero denominators, unsupported calendars, unqualified/chained/cyclic transfer stay unavailable. |
| U7 composition | Same-water lower bounds3 and2 =>3; lateral2 plus continuing3 =>upstream5 under explicit zero-loss mapping. Missing mapping unresolved. Final95 floor checks every class; crossed family descends without repair. |
| Floor-only | Active quality survives final entry/presumptive/potential floor; no invented regime/obligation or blanket baseline gate. Independent duties remain assessable. |
| C4 | Requirement10/deliverability6/delivery5 =>obligation6, ecological deficit4, raw shortfall1, with immutable earlier values. |
| Floor uncertainty | All ten bundled cases: uncertain but-for changes finding, supported below/not-below, equality at upper margin versus point equality, no natural supply, fixed counterfactual, missing uncertainty, supported dependence and reversed-interval refusal. |
| Isolation/completeness | Different methods, members and scenarios do not contaminate settings or issued results. Known required failure survives other missing checks; missing/empty required coverage cannot pass. Sanitary duties run without assumed sizing or domestic ecological entry. |

### Complete supported U9 chain

Use `evidence/synthetic_chain.json` at its stored precision. Two assumed structurally distinct members have complete daily reference years; A annual means are `10*exp(.3)` in2001 and `10*exp(-.3)` in2002, B is1.5 times A. Supply six preceding reference days at the wet-year value. The explicit h=1, no-melt-alignment scenario yields constant mean-one patterns and acknowledges rare-year shape extrapolation. Winter/spawning are inactive. This is a stipulated accepted synthetic scenario, not scientific validation.

At continuing-river R, active conservative quality with background1 m³/s at9 mg/l, clean additional water and limit1 mg/l needs8 additional or9 total m³/s. A separately supported lateral wetland diversion1 m³/s between upstream U and R raises each upstream candidate by1 under the explicit no-loss/no-delay/no-storage mapping. Selection remains external but its record and final checks are exercised through Fishy.

Expected upper-selected U classes25/50/75/95 are approximately `16.000000000000007`, `13.252172596032329`, `12.112273310225769`, `12.112273310225769` m³/s. Whole-family D is approximately `.3703703703703707`. Member seven-day/T100 thresholds are approximately `4.976270579072646` and `7.464405868608971`. Supply explicit complete predecessor context for each assessed class, not implicit wrapping. All selected classes pass the safeguard and floor invariant. Final floor is approximately `12.112273310225769`.

For class95 deliverability6 and delivery5 give obligation6, ecological deficit approximately `6.112273310225769` and raw delivery shortfall1. Use fixture precision and declared tolerances, not rounded prose. Demonstrate final quality/receptor checks, not just imported expected totals. Also exercise quality-upper-bound failure after uplift, missing context, invalid ratios and known failures within incomplete assessments.

Supply readable public-use examples for supported assembly/issuance, floor-only fallback, missing evidence and independent-duty assessment. Explain current behavior and source limitations. No PyPI publication milestone or basin pilot is required.

## Boundaries and remaining application responsibility

No member/scenario orchestration, optimiser, report generator, counterfactual reconstruction or causal-inference engine; no automatic operating schedule correction or drought permission; no native general habitat/hydraulic/reactive solver; no new mandatory simulator dependency. Do not extend into optional Spanish legal replication, national policy selection, official source authentication or scientific/basin certification.

Duration/season/return period, entry table/screen, presumptive schedule, donor qualification/choice, ecological criteria, uncertainty support, acceptance limits, operational classification and competent decisions remain supplied configuration/evidence. Their absence is a named limitation, not permission to invent values. The current scalar minimum, shifted shape, strict zero tolerance and whole-family failure scope are settled implementation readings. No unresolved generic methodological choice was found in discovery. Bring newly found genuine source contradictions back with the exact operation and alternatives; continue unaffected work.
