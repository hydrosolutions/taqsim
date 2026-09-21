# Sanitary duties and hypothetical seasonal profile

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/27

## Outcome

Fishy can assess supplied sanitary duties without reconstructing their historic sizing, and separately calculate the report's explicitly hypothetical seasonal sanitary profile. Both run independently of ecological sizing and a runtime simulator. A combined sanitary/ecological assessment retains each requirement and its findings instead of inventing one merged duty.

Implementation belongs in `hydrosolutions/fishy`. This vision authorizes no national interpretation, policy adoption, site certification or replacement of an issued duty. Its acceptance ownership is F8, S1–S5 and relevant common C1–C6 isolation, evidence and completeness behavior. Engineering structure and API signatures remain implementation decisions.

## Requirements authority and durable source contract

The sole handover is `fishy_taqsim_handover_candidate_2026-09-19`, approved by the Program owner despite its candidate label. Earlier handovers and deprecated Fishy behavior supply no extra requirements. Source identities:

- `IMPLEMENTATION_BRIEF.md`: SHA-256 `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8`.
- `ACCEPTANCE.md`: SHA-256 `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba`.
- `SHA256SUMS.txt`: SHA-256 `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0`.

Read the brief's F8 and sanitary provisions and acceptance S1–S5. Detailed sources are `report_snapshot/part1_sanitary_baseline.qmd` (`sec-sanitary-baseline`), `part1_sanitary_scenarios.qmd` (`sec-sanitary-scenarios`, D.1), `appendices.qmd` (step 3, `sec-recipe-step3`), and `part4_hydraulic_assessment.qmd` (`sec-hydraulic-assessment`, D.9). Source status is recorded in `SOURCE_INDEX.md` and `SOURCE_LIMITATIONS.md`. The Program records access to the held bundle. The technical contract and synthetic acceptance witnesses below make this vision usable without publishing that bundle. Do not upload private report files or bulk source documents as an implementation convenience.

The held `references/uzbekistan/supporting_discharge_and_entry/su_sanpin_3907_85_reservoirs_ru.txt` is a secondary reproduction, not an authenticated governing original. Its present force and object-specific applicability in Uzbekistan remain unresolved. The government-hosted SanPiN 0315-14 copy is marked unofficial translation. Its register listing and bibliographic reference to 3907-85 do not authenticate the inherited sizing rule. Water Code Article 122 supplies the framework, not a numerical estimator. Preserve these distinctions in maintained documentation and outputs.

## Existing duties remain independent

Accept a supplied schedule with its source instrument, provision, version, units, location, exact intervals, applicability and all required components. Preserve authenticity, currency and applicability separately. Retain the distinct outcomes: authenticated/applicable; located/not applicable; unresolved; none located after a dated documented search. None located is not proof no duty exists. An unauthenticated instrument can be retained and assessed as an explicitly hypothetical input, not silently promoted to binding authority.

Retain instruments with missing schedules or only non-flow clauses. A missing magnitude leaves the dependent flow comparison pending; it does not erase supported hydraulic findings, imply a zero duty or force an invented schedule. Missing interpretation or authority evidence does not prevent assessment of a separately supplied applicable duty.

For matching flow intervals compute positive shortfall `max(duty - delivery, 0)` and integrate against actual interval duration. Never offset a deficit with later surplus. A shortage, availability estimate or later delivery cannot lower the prescribed duty. Distinguish observed comparisons from modelled scenarios and both from legal compliance or attribution.

Preserve all independently issued duties, including conflicting instruments, control points and versions. Do not choose legal precedence by taking a maximum, and do not sum repeated downstream non-consumptive requirements as demands. No cross-location comparison or implicit temporal resampling is valid. Preserve present zero, missing, outside-horizon and unsupported states.

## Hypothetical annual-selection profile

This is an explicit modelling interpretation, not the authenticated SanPiN estimator. Its result is neither an issued duty nor permission to activate the domestic ecological-entry variant.

Supply one identified natural-flow reconstruction at a control point, reference period, provenance and evidence status. Inputs are complete January–December calendar years of finite nonnegative daily mean discharge in m³/s. Each day is 86,400 seconds in UTC or a declared fixed offset; retain leap days. Missing days, duplicates and invalid values stop this calculation. Do not fill gaps, interpolate monthly data or silently discard incomplete years. A replacement period or reconstruction is a new identified input.

Declare summer and winter windows before calculation. Each is a nonempty contiguous interval with inclusive month/day endpoints, entirely within one calendar year, valid in every input year; they cannot overlap. Cross-year windows require another explicitly identified profile, not an extension silently applied here. No national seasonal dates are inferred.

1. For each year `j`, compute annual mean `a_j = sum(daily flow) / number of days`.
2. Rank annual means descending. Equal means receive their occupied positions' average rank `r_j`.
3. With `n` years, assign plotting position `p_j = r_j / (n + 1)`.
4. Fixed target `P = 0.95` must lie inside the actual unrounded `[min(p_j), max(p_j)]`. Otherwise this empirical profile is unsupported. Check support before rounding or nearest-year selection.
5. Select the smallest `abs(p_j - P)`. Equal distances favor larger `p_j`; years tied at that probability favor the earliest calendar year. Retain all tied candidates and the selected year's actual plotting position and annual mean. Use unrounded values throughout comparisons; rounding is display only.
6. Within that selected year, calculate the daily minimum separately for each declared season. Return both values and their dates/windows. The profile supplies no value outside these windows. A complete operating schedule needs separately attributable inputs there.

No interpolation of years or hydrographs, daily flow-duration percentile substitution, minimum across all years, or collapse into a single seasonal value is allowed. The selected year is a nearest supported analogue, not proof of a population quantile. An evidence-supported zero stays zero but does not relax an existing duty.

Nineteen distinct annual ranks first reach `19/20 = 0.95`; ties can reduce even that support. This is mathematical rank support, not a universal minimum record length or proof of scientific adequacy. Retain rating-range, reconstruction, dependence, climate relevance and intended-use restrictions. Do not silently detrend or promote indicative results to accepted sizing inputs.

Support separately identified hypothetical nomination of a year and its seasonal minima without falsely claiming successful empirical annual selection. Supported imported estimates retain their own methods, evidence and windows. Missing interpretation/windows leave derivation pending while supplied-duty assessment remains available.

Callers run alternative reconstructions independently. Preserve each member's selection and seasonal result. Any externally reported min/max range is sensitivity, not an inferred confidence interval; failed members remain visible as incomplete coverage. Do not import ecological structural-selection rules into this profile.

## Hydraulic and qualitative components

The required SanPiN 3907-85 source-clause crosswalk distinguishes:

| Clause | Held provision and required observable behavior |
| --- | --- |
| §4.2 | Minimum flow bound tied to minimum mean-daily discharge in summer/winter low-water periods of a 95%-assurance year under the source regime wording. Retain literal source meaning and the hypothetical status of D.1's chosen interpretation. |
| §4.3 | In reservoir cascades, retain §4.2 and the pre-impoundment minimum velocity requirement; absence of downstream hydropower current is prohibited. Discharge alone cannot prove local velocity or current continuity. |
| §4.4 | Retain maximal possible release uniformity and the prohibition on abrupt within-day changes in downstream levels and velocities. No numerical ramp threshold is supplied. |

The 0.5 m/hour value elsewhere in §3.7.3 concerns beach siting, not a sanitary ramp default.

Consume supported hydraulic states, quantitative criteria and specialist findings without requiring a native hydraulic solver. Findings need source/study identity, assessed component, location/domain, candidate/scenario, variable, units and temporal support. An arbitrary generic pass flag is insufficient to establish a sanitary condition.

Distinguish release discharge, stage with datum, depth with bed reference, directional local velocity and speed magnitude. A relation retains geometry/version, boundary conditions, supported domain, interpolation, uncertainty and acceptance for intended use. Do not extrapolate or replace a missing variable with discharge. Assess current continuity on its own specified domain. Existing Order 179-specific hydraulic rules cannot become sanitary defaults.

The shared quantitative evaluator remains owned by the study-driven assessment Effort #28; this Effort must demonstrate its sanitary use through supported supplied criteria/states or sufficiently attributable imported findings, without making #28 a new delivery dependency. Reuse suitable current capabilities and add only the sanitary boundary required here. Preserve these D.9 semantics wherever this Effort calculates a quantitative check:

- Discrete rate `(x_k - x_previous) / elapsed_time` uses actual strictly positive elapsed time and separate nonnegative supplied rise/fall bounds, including declared strict or inclusive comparisons. An intentionally absent bound differs from a missing required bound.
- Endpoint rates do not prove unseen instantaneous maxima. Differences of interval means concern those means. Daily means cannot certify within-day conditions.
- Missing predecessors and expected-sample gaps remain unassessed; no wrapping or silent gap bridging. A season-boundary transition needs an explicit applicable criterion or supported subdivision, never invented intermediate states.
- For supported state intervals `[l_previous,u_previous]` and `[l_k,u_k]`, conservative rate bounds are `[(l_k-u_previous)/dt, (u_k-l_previous)/dt]`. Preserve tighter supported joint evidence if supplied. Full containment passes, disjointness fails, overlap is indeterminate; preserve endpoint strictness. These are not probabilistic confidence intervals.
- Qualitative conditions without supported criteria or specialist findings stay unassessed. A declared hypothetical numerical condition remains hypothetical.

## Combined findings and completion

Sanitary-only use needs no ecological operands. Sanitary-plus-ecological use returns separate attributable requirements and component findings, with incompatible conditions explicit. Same-water comparison is not a universal maximum or sum, legal precedence decision, or simulator demand construction.

A supported required failure survives unknown components as failure with incomplete coverage. With no known failure, a missing required component prevents complete pass. Empty required sets cannot manufacture satisfaction. Keep numerical support, evidence disclosure, purpose-specific scientific adequacy and official admissibility separate. No scenario finding establishes legal liability, and all specified tests passing only establishes satisfaction on their declared domain.

## Acceptance evidence

Provide a maintained requirement/source-clause → public operation/input → observable result → executed-test crosswalk, with actual versus expected values, declared tolerances and exact compatible source revisions. Synthetic tests prove software behavior, not basin science or authenticated law. A permanently pending implementation is insufficient.

- **S1:** prescribed `[2,3]` and daily delivery `[1.5,3.5]` m³/s return shortfalls `[0.5,0]` and exactly 43,200 m³. No estimator or surplus cancellation. Include missing, zero, unsupported and mismatched location/interval cases.
- **S2:** complete years 2001–2018 each have constant daily flow `2021-year` (20 through 3). In 2019, January 1 is 0.6, July 1 is 0.9, and all other 363 days are `1457/726` m³/s; annual mean is exactly 2. With January–February and July–August windows, select 2019 at 0.95 and return winter 0.6/summer 0.9. Removing 2019 is unsupported (`18/19 < 0.95`); deleting a day is invalid completeness. Exercise ties/earliest-year, support-before-rounding, leap days, exact inclusive windows, overlap/invalid-window rejection and absence of values outside windows. The dates are synthetic inputs, not national defaults.
- **S3:** independently nominated hypothetical year with summer `[4,2,5]` and winter `[6,3,7]` returns separate minima 2 and 3. Preserve that nomination identity and imported-method identity. Missing interpretation/windows cannot disable an independently supplied duty. No universal nineteen-year or complete-year gate on duty/import paths.
- **S4:** executable §4.2–§4.4 crosswalk covers flow, pre-impoundment velocity, cascade continuity, uniformity and within-day change. Supported flow failure plus unknown hydraulics remains failure/incomplete. Flow pass plus unknown hydraulics is not whole-duty pass. Demonstrate supported hydraulic success and failure and refusal of daily-only evidence for within-day certification. A supplied 1.00→1.12 m stage change in 0.5 hour is 0.24 m/hour, failing a hypothetical inclusive rise limit 0.20; the reverse passes a separate fall limit 0.30. Preserve imported equivalent findings without falsely claiming native calculation. Consult D.9 and `quality_support/hydraulic_assessment_fixtures.json` for relevant strict/equality, uncertainty, gap, domain and variable-support cases; complete shared hydraulic-suite ownership remains #28.
- **S5:** run sanitary-only and sanitary-plus-ecological configurations; retain separate requirements, control points, issued versions and conflicts. Changing the hypothetical profile cannot change another source method or prior duty. Shortage cannot lower prescriptions or activate domestic ecological-entry sizing.

Exercise four distinct duty-location outcomes, missing schedules/non-flow-only instruments, invalid finite/nonnegative domains, independent source/member/scenario identity and failed-member completeness. Show standalone supported observations/imports without mandatory Taqsim execution. Current compatible Taqsim delivery imports can reuse the foundation boundary rather than requiring a new simulator.

## Current repository evidence and boundaries

During discovery, Fishy GitHub `main` and fetched `origin/main` were `389dcdac7a34585ee179c8b2acdb7b65584095f5`; the local canonical checkout was still the older scaffold. Inspect current Git target objects rather than assuming its working tree represents current implementation. Recheck target revisions before implementation.

Foundation Effort #9 landed supplied-duty/evidence semantics at Fishy `53cf80f3e46257bc0b9be883bc772b6b941dc2b3`, with delivery evidence on its issue. Its documented 264 passing tests concern that revision, not validation of today's complete tree. `SuppliedDuty`, `assess_duty`, evidence findings, exact intervals and live/saved Taqsim exchange are reusable evidence, not a mandate to freeze their API. The existing 43,200 m³ witness does not establish all F8/S1–S5 acceptance.

No sanitary estimator or sanitary-specific source crosswalk was found on the inspected target. Swiss pooled Q347 and Kazakh seasonal allocation are different operators and must not substitute for annual sanitary selection. Existing country-specific hydraulic evidence patterns may inform design but confer no sanitary source authority.

Out of scope: authenticating missing national meaning, selecting legal precedence or national seasons, reconstructing site documents, replacing issued duties, domestic ecological-entry implementation, native general hydraulic simulation, operating optimisation, caller scenario comparison, PyPI publication and implementation of other country methods. Missing scientific/application evidence remains explicit; it is not a reason to invent defaults or block independently supported calculations.
