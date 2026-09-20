# Independent Swiss residual-flow workflow

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/26

## Outcome

Implement Fishy's independently runnable Swiss residual-flow calculation and assessment: establish attributable Q347, calculate the literal statutory starting minimum, assess safeguards and exceptions, retain a separate interest-balancing decision, translate downstream needs into a supported intake prescription, and assess delivery without rewriting that prescription. This is F3 of the approved handover, not an Uzbek profile or an automatic permitting system.

The modeller calculates requirements, optionally simulates an independently configured physical system, then assesses observations or supported imported/modelled results. Pure Swiss calculations and supplied-duty assessments require neither a live simulator nor implemented HYDMOD. A complete supported synthetic path must work; an always-pending interface is insufficient. Missing scientific or authority evidence must remain visible without disabling unrelated supported results.

Implementation belongs in `hydrosolutions/fishy`. This document authorizes no implementation by its publication alone. Reversible API and internal architecture choices remain with the implementing agent.

## Authority and durable source evidence

Only `fishy_taqsim_handover_candidate_2026-09-19` supplies commission requirements. The owner approved it despite its internal candidate label; all previous handovers are superseded. Exact supplied directory:

`/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`

Verified identities:

| File | SHA-256 |
| --- | --- |
| IMPLEMENTATION_BRIEF.md | 0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8 |
| ACCEPTANCE.md | 851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba |
| SHA256SUMS.txt | 7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0 |

Read README, IMPLEMENTATION_BRIEF §3 (Swiss), ACCEPTANCE §4 (Swiss prescription), SOURCE_INDEX, SOURCE_LIMITATIONS and SOFTWARE_BASELINE. Follow `report_snapshot/appendix_method_reference.qmd#sec-cheat-ch`. The brief defines the commission; German primary sources govern this independent foreign method. English translations are reading aids. FOEN 2000 is supporting guidance, not authority to override amended law. Its printed page numbers are two less than the PDF/text-extraction page markers.

The public primary-source identities and clause-level paraphrases below carry the relevant evidence durably without publishing private reports, archives or application material. Publisher URLs identify held versions, not a new legal-currency certification. Use original PDF layouts where extraction loses typography. If the supplied bundle is unavailable or differs, obtain that same bundle rather than substitute deprecated sources. Do not publish restricted material merely to make tests self-contained; use own synthetic witnesses and public-source citations.

- [ch_foen_wegleitung_restwassermengen_2000_de.pdf](https://www.bafu.admin.ch/dam/de/sd-web/DURPl8AmZvgE/angemessene_restwassermengenwiekoennensiebestimmtwerdenwegleitun.pdf)
  SHA-256: `c62afb26847e5bc2e8591423d2bdfad98bc95144f770c9055a642f8720cc46c6`.

- [ch_gschg_waters_protection_act_2025_de.pdf](https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/1992/1860_1860_1860/20250801/de/pdf-a/fedlex-data-admin-ch-eli-cc-1992-1860_1860_1860-20250801-de-pdf-a.pdf)
  SHA-256: `c7393639a38b9bd5e6e8c9b253fe799f42ecb34dd0606adf3b6dbc0c14db4b46`.

- [ch_gschv_waters_protection_ordinance_2025_de.pdf](https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/1998/2863_2863_2863/20251201/de/pdf-a/fedlex-data-admin-ch-eli-cc-1998-2863_2863_2863-20251201-de-pdf-a.pdf)
  SHA-256: `1aced4652f93ea8cf2530dacebac3e2dfc59ddf6c0136127494ac20c3cef352a`.

In the sections below, **Act** means that GSchG consolidation; **Ordinance** means that GSchV consolidation; **Guide** means the FOEN guide. These source choices and declared interpretations must remain attributable in outputs.

## Approved source interpretations

Two narrowly scoped modelling decisions were approved during discovery. Neither authenticates a legal interpretation or grants permission.

1. **Art. 30(b), 1,000 l/s limit.** The Act's wording can attach “together with other abstractions” to both limits. Guide chapter 3, printed pp. 20 and 23, explicitly applies 1,000 l/s per abstraction. Support both named readings: aggregate 1,000 l/s, or per-abstraction 1,000 l/s. In both, the combined abstraction must satisfy the 20% Q347 limit. There is no automatic default. Without a supplied selection retain the branch findings and unresolved combined eligibility, not an inferred permission. A selected reading retains its source and scenario/authority status. The Guide's instantaneous percentage condition must not silently become a daily or seasonal mean.
2. **Art. 32(a), altitude endpoints.** For the expressly labelled modelling scenario, interpret the non-fish-water band as inclusive: 1,500 ≤ elevation ≤ 1,700 m. The source says “between” without mathematical endpoint notation; the older Guide predates this amendment. Without that declared interpretation or applicable supplied authority evidence, exact-endpoint eligibility stays unresolved. This does not relax Q347 <50 l/s, the 1,000 m reach limit, non-fish evidence, or decision requirements. The other altitude branch remains strictly above 1,700 m.

A short message to the owner's superior communicates these decisions. No separate decision report or legal-authentication exercise is requested. They appear here only so implementation does not silently lose them.

## Required behaviour and clause coverage

### Scope, routes and Q347

Cover Act Arts. 4(h–l), 29–36 and 59, and relevant Ordinance Arts. 33–35, including 33a. Distinguish new/renewed permits, existing-concession remediation, and assessment of an already prescribed duty. Existing concessions are not automatically assigned the Art. 31 table. Recognise the separate Arts. 80 ff. route without implementing a full remediation-derivation method in this Effort. An existing duty remains assessable without reconstructing its historical sizing.

Act Art. 4 defines Q347 by flow reached or exceeded on average 347 days per year over ten years, without substantial influence from impoundment, abstraction or additions. Permanent flow means Q347 >0. Residual flow downstream and release at the intake (*Dotierwassermenge*) are different quantities. A raw table calculation at zero is not eligibility for a permanent-flow permit or authority to prescribe 50 l/s.

Arts. 29–30 require applicability evidence for beyond-common-use abstraction and significant lake/groundwater effects. Cover limited-abstraction routes: Art. 30(b)'s combined ≤20% Q347 plus the explicitly selected ≤1,000 l/s reading; Art. 30(c)'s drinking-water annual-mean limits of ≤80 l/s from a spring and ≤100 l/s from groundwater. Keep source, purpose, aggregation and time basis distinct. Ordinance Art. 33 distinguishes permanence at the intake from mixed perennial/nonperennial downstream sections: Art. 30 conditions apply to perennial sections, and a nonperennial intake still requires applicable nature/fisheries protection measures rather than automatic permission.

Compute Q347 from a pooled multi-year flow-duration distribution, not the mean of annual Q347 values. Declare the actual reference period, daily calendar/leap-day treatment, record selection, interpolation and missing-data conventions. Do not relabel coarse means or repeated monthly values as daily low-flow evidence. Preserve uncertainty, human influences, trend/representativeness findings and numerical versus scientific acceptance. Support attributable Art. 59 estimates from observations/model calculations independently of the direct daily-record route.

Guide chapter 7, printed pp. 82–86, distinguishes ≥10-year records, shorter records needing representativeness evidence, reconstruction and preliminary estimates. Its conditional final verification guidance (at least three measurement years where the table minimum has little/no uplift, with justified exceptions) is not an unconditional statutory duration gate. Preserve the guidance version, applicability, verification/exception evidence and acceptance decision. Preliminary estimates may run labelled scenarios, not silently become final accepted evidence. Do not adopt the Guide's old pragmatic <10 l/s infiltration convention as a universal replacement for the Act's Q347 >0 definition.

### Literal Art. 31(1) starting minimum

For q=Q347 in l/s, preserve this literal source table:

| q domain (l/s) | Starting residual minimum (l/s) |
| --- | --- |
| 0 ≤ q ≤60 | 50 |
| 60 < q <160 | 50 + 0.8(q−60) |
| 160 ≤ q <500 | 130 + 0.44(q−160) |
| 500 ≤ q <2,500 | 280 + 0.31(q−500) |
| 2,500 ≤ q <10,000 | 900 + 0.213(q−2,500) |
| 10,000 ≤ q <60,000 | 2,500 + 0.15(q−10,000) |
| q ≥60,000 | 10,000 |

Retain the 279.6→280 and 2,497.5→2,500 anchor jumps, unrounded arithmetic, and values above Q347. Do not smooth, cap or substitute the continuous Uzbek graduated curve. Keep numerical table output separate from legal applicability and final prescription.

### All five Art. 31(2) safeguards

Assess each separately with attributable site evidence, supported relationships and alternative measures. Increase the minimum where a requirement is unmet and cannot be met by other measures; no universal uplift coefficient or automatic addition of successive total requirements.

- (a) Prescribed surface-water quality despite abstraction and existing wastewater inputs.
- (b) Groundwater recharge sufficient for dependent drinking-water abstraction and no substantial impairment of agricultural-soil water balance.
- (c) Rare habitats and communities preserved, or equivalent replacement subject to the statutory qualifications.
- (d) Water depth needed for free fish passage. A qualified guide recommendation such as 20 cm is not a universal statutory depth.
- (e) Spawning/rearing functions where Q347 ≤40 l/s and elevation is below 800 m; include affected downstream sections below that elevation even when the intake is higher (Guide §4.4, p. 46).

Support supplied seasonal/event needs and alternative measures rather than reducing all safeguards to one scalar. Site relationships, acceptance criteria and specialist findings are inputs, not invented hydraulic/habitat equations. Missing required safeguards remain unresolved; a supported failure survives other missing evidence. Economic balancing does not silently erase a safeguard.

### Every Art. 32 exception and separate Art. 33 balancing

A lower minimum requires a supplied authorised decision or an explicit hypothetical decision, plus the applicable conditions. Preserve decision scope, period, reach and evidential status.

| Exception | Required conditions |
| --- | --- |
| 32(a) | Q347 <50 l/s; exception reach limited to 1,000 m below intake; water above 1,700 m, or non-fish water in the explicitly interpreted 1,500–1,700 m band. |
| 32(b) | Non-fish water; reduced residual flow must not fall below 35% of Q347. |
| 32(bbis) | At most 1,000 m below intake; low ecological potential; natural functions not substantially impaired. Ordinance 33a includes present significance and significance after proportionate restoration, not present degradation alone. |
| 32(c) | Limited topographically connected area; adequate compensating protection in the same area; Federal Council-approved protection/use plan. Ordinance 34 requires plan, adequacy explanation and binding arrangements for the concession duration. Already legally required environmental measures cannot count as compensation. |
| 32(d) | Emergency and temporary abstraction, with identified basis and period; not ordinary unsupported scarcity. |

Retain normal protection beyond a limited exception reach. Art. 33 balancing remains a separate requirement, including after an exception (Guide §§4.5–4.6). It considers public, source-region/applicant economic and energy interests against landscape, habitats/biodiversity/fish reproduction, long-term water quality, future groundwater/drinking/land-use/vegetation and irrigation interests. Consume the applicant's alternative-abstraction/mitigation evidence and attributed decision; do not implement arbitrary weights or an optimiser. A supported final total is not added to the preceding minimum.

### Arts. 34–36: intake prescription and delivery

Apply analogous protection for significant lake/groundwater effects. Consume supported routing/balances and site relationships to connect downstream residual needs, tributaries, losses, other abstractions and protection points to the intake release. Protect the affected reach, not just the intake; storage effects may extend below a powerhouse return (Guide §§4.3 and 4.9). A topology map or tributary sum alone is not a routing relationship. Unsupported mappings remain unresolved, not a universal sum/subtraction rule.

Keep the Art. 35 intake schedule, downstream needs and other protective measures separately identified. Temporal variation must preserve applicable Arts. 31–32 minima. Retain specialist consultation and federal hearing evidence where hydropower gross power is >300 kW; equality is not included. Ordinance 35 covers the residual-water report within EIA, and the cantonal opinion/draft made available to FOEN on the federal-hearing/non-EIA route. These are supplied process/decision facts, not automatic legal approval.

Art. 36 accepts measurement proof, or a water balance where measurement burden is unreasonable. Only supported proof of temporarily lower inflow activates the all-inflow treatment. Preserve nominal prescription, evidenced inflow, justified time-specific duty and actual delivery as separate values. Missing inflow proof cannot create relief. Tributary gains do not retroactively change an issued intake duty. Do not apply the Uzbek deliverability cap.

Link attributable MSK/HYDMOD and other specialist condition findings to the same reach/scenario, while keeping release compliance and condition findings separate. A hydrological class cannot automatically produce an uplift or biological verdict.

## Evidence semantics and existing foundations

Preserve source/version, method/interpretation, member/scenario, physical body/reach/control-point identity, exact intervals, units and uncertainty. Keep present zero, absent, outside-horizon and unsupported distinct. Numerical availability, scientific adequacy for a stated use, disclosure and official admissibility are separate. No known failure plus missing required checks is indeterminate; known required failure plus missing checks remains failure with incomplete coverage. Empty required sets cannot establish a complete pass. Scenario evidence is not observed compliance or attribution.

At discovery, Fishy target `f7bf7b55428524c8f89375f6519462ce44225d39` contains landed foundations from Efforts 9, 24 and 25. The local canonical checkout remained an old scaffold; inspect/fetch the real target before implementing rather than treating stale files as the architecture. Taqsim target at publication preparation is `5413b81`. Prior delivered compatible optional pins are Taqsim `396ad093c2b6f240e702a3b05aee1fb96a69b3f7` and Incidence `665da4e0d81ab28921b8d5d2edbb9be27f4ec612`; reverify actual chosen revisions.

Relevant existing Fishy source at that target: `quantities.py`, `spatial.py`, `time.py`, `flows.py`, `evidence.py`, `duties.py`, `physical.py`, plus `docs/assessment.md`, `docs/ecological-conditions.md` and `docs/basin.md`. Exact quantities, scoped evidence, fixed supplied duties, uncancelled shortfalls and optional live/saved physical projections already exist. Quality and supplied ecological-state checks may be reused where semantically valid; do not import Kazakh defaults or country decisions. Existing topology/tributary aggregation explicitly is not routing. No Swiss public calculation module was present at discovery.

A supported supplied-input path must run with Taqsim absent. Where Taqsim results are consumed, preserve authoritative counts/support, physical and temporal mappings, real-duration volume-to-flow conversion and balance failures. Supported imports remain sufficient; no mandatory simulator or new physical solver is implied.

## Observable acceptance

Deliver a revision-pinned requirement → source clause → public operation/input → observable result → executed test crosswalk for all clauses above and every Swiss test in ACCEPTANCE §4. Report actual/expected values and declared numerical tolerances. Source traceability, reviewed arithmetic and this vision are not executed implementation acceptance.

Required witnesses include:

- Pooled multi-year Q347 demonstrably differs from mean annual Q347; ten-year meaning, declared conventions, supported Art. 59 import, preliminary/accepted verification and justified exception paths are exercised without a blanket duration rejection.
- Every table anchor and immediately adjacent value, below 60 and above 60,000 l/s, both jumps and output above Q347. Invalid negative/nonfinite inputs cannot silently produce valid results.
- Q347 160 → table 130 → supported safeguard minimum 180 → separately supported final 220 l/s, not 130+180+220. All five safeguards, alternative measures, missing evidence and seasonal/site-supported needs are exercised.
- Every exception's eligible/ineligible boundary, absent authorisation and labelled hypothetical decision. Include q=50 exclusion; q=40/elevation=800 safeguard boundaries; altitude endpoints under the approved labelled interpretation; 1,000 m reach and 35% residual thresholds; compensation/approval restrictions. Test both Art. 30(b) readings using two 700 l/s intakes within the combined 20% allowance, plus the no-selection outcome. Exercise other permit routes and process thresholds without treating passing arithmetic as a permit.
- Nominal intake prescription 220, evidenced inflow 150, delivery 120 l/s → time-specific duty 150 and shortfall 30, retaining nominal 220. Without evidence, no invented exemption. Include gains/losses giving intake release different from downstream residual need and no retroactive change from tributary gains.
- Same prescribed-release compliance with different supplied hydropeaking/seasonality findings, without HYDMOD dependency or class-to-uplift conversion.
- A complete independently runnable supported scenario, an already prescribed duty whose sizing is unavailable, and missing/unsupported/infeasible paths. Required failure must survive incomplete coverage. Demonstrate method/configuration isolation and preservation of earlier issued values.

Maintain usable examples and clear documentation of public inputs, calculation limits and scenario decisions. Run repository-native tests and checks on actual compatible revisions; prior delivery test counts are not proof of this work. Scientific/official certification is not a software acceptance outcome.

## Boundaries

No automatic cantonal judgement, official source authentication, Swiss-to-Uzbek legal transfer, national policy selection, biological certification, native general habitat/hydraulic/reactive solver, counterfactual reconstruction engine, optimiser, report generator or new comparison engine. No full Arts. 80–83 remediation derivation. No implementation of all MSK modules or dependence on Effort 30 to calculate Swiss releases. No PyPI publication or deprecated-Fishy compatibility requirement. Missing local studies and approvals remain explicit application inputs rather than a reason to invent defaults or withhold supported calculations.
