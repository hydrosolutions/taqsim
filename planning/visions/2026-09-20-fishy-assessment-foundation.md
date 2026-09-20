# Fishy assessment foundation

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/9

## Outcome

Build the new `hydrosolutions/fishy` library from its fresh pyplate scaffold. A modeller can calculate supported diagnostics and assess a supplied duty against identified observations, supported imports, or current Taqsim results. Simulation and assessment remain separate: calculate requirements using the owning method, simulate an independently configured operating system if needed, then assess its supported results. Requirements neither create water nor automatically become consumptive demands.

This vision commissions implementation in `hydrosolutions/fishy`, not a rewrite of Taqsim or preservation of `fishy-deprecated`. The planning record remains in Taqsim alongside its Program. No deprecated API compatibility, package-registry publication, or Zarafshan application port is required. Public signatures and reversible engineering mechanisms are not frozen here.

## Source authority and required reading

The sole handover requirements source is the owner-approved directory:

`/Users/nicolaslazaro/Downloads/fishy_taqsim_handover_candidate_2026-09-19/`

Its internal candidate label does not supersede the Program's approval. All earlier handovers and deprecated feature plans are historical evidence only. Read `README.md`, `IMPLEMENTATION_BRIEF.md`, `ACCEPTANCE.md`, `SOURCE_INDEX.md`, `SOURCE_LIMITATIONS.md`, and `SOFTWARE_BASELINE.md` before implementation. Verify these SHA-256 identities:

| File | SHA-256 |
|---|---|
| `IMPLEMENTATION_BRIEF.md` | `0ad077a69f27af74dd6cf340830438576d020d66b9bf5a8d568bb84eb5c757c8` |
| `ACCEPTANCE.md` | `851fc64c35e827fa3a072750f760108ddd5f445684ba65f85bb4037758b047ba` |
| `SHA256SUMS.txt` | `7dcf598b23a282a88a12af6ca2aee89795df27a119842ed17603403398dc97f0` |

Within that directory, read these actual report sections, not just the brief:

- `report_snapshot/appendices.qmd`: Appendix A steps 1–3 (prepared mapping, origin/designation, independent duties), step 4 (reference reconstruction and disclosure), and the floor-uncertainty boundary. Full route-specific evaluators belong to their method Efforts.
- `report_snapshot/appendix_computational_requirements.qmd`: C.1 exchanges, C.2 capability checks, C.3 passport records. These establish location, interval, process ownership, validity and version meaning; they do not commission a report generator or full passport application here.
- `report_snapshot/part_computational_requirements.qmd`: component responsibilities and separate natural/managed/future cases.
- `report_snapshot/part4_scientific_acceptance.qmd`, `sec-scientific-acceptance`: separate numerical validity, disclosure, scientific adequacy for the stated use and official admissibility.
- `report_snapshot/part4_recommended_method.qmd`, `sec-reference-regime`: natural processes and reference uncertainty.
- `IMPLEMENTATION_BRIEF.md` §§1–2, 4.1, 6–8 and `ACCEPTANCE.md` C1–C6; use S1 as the supplied-duty arithmetic witness without pulling sanitary estimation into this foundation.

This vision durably records the relevant foundation contracts below. Full handover originals remain locally supplied, not falsely described as repository-hosted. If originals are unavailable or checksums differ, obtain the same material rather than substituting a superseded archive. Do not upload restricted report material or papers merely to make a public planning record self-contained.

### User-supplied diagnostic papers: mandatory reading

The owner supplied these two PDFs during discovery specifically so implementing agents can read them. Their identities were checked from the actual PDFs, not inferred from old Fishy citations. Local paths are access locations on the supplied workstation; DOI links provide durable bibliographic identity, not a guarantee of unrestricted access elsewhere.

1. **Greco, Michele; Arbia, Francesco; Giampietro, Raffaele (2021).** *Definition of Ecological Flow Using IHA and IARI as an Operative Procedure for Water Management*. **Environments 8(8), 77**. DOI: <https://doi.org/10.3390/environments8080077>.
   - Supplied file: `/Users/nicolaslazaro/Downloads/environments-08-00077-v2 (1).pdf`
   - SHA-256: `ed4e0cee87b1ec1797d57f82034dab2d770564a7353be8b9af435ad26413b398`; 54,670,175 bytes.
   - Read §2.3, the IHA/IARI formulas and tables, the calculation examples and cited source [40] (ISPRA 2011). Verify the equations against PDF layout, not a potentially damaged text extraction.
   - Correct the historical citation: the third author is **Giampietro**, not Gioia; this is an *Environments* article, not a proceedings paper.
   - The paper's monthly case study and ecological-flow derivation are not authorization to add an Italian release-prescription method to this Effort or to apply monthly evidence to all daily IHA parameters. Trace IARI's original definition, aggregation and thresholds through ISPRA where needed; do not equate the old 33-parameter arithmetic average with an independently verified source method without evidence.
2. **Black, A. R.; Rowan, J. S.; Duck, R. W.; Bragg, O. M.; Clelland, B. E. (2005).** *DHRAM: a method for classifying river flow regime alterations for the EC Water Framework Directive*. **Aquatic Conservation: Marine and Freshwater Ecosystems 15, 427–446**. DOI: <https://doi.org/10.1002/aqc.707>.
   - Supplied file: `/Users/nicolaslazaro/Downloads/2005Blacketal.AC.pdf`
   - SHA-256: `8ea9d281675cdd488bc463c451b309fc25f036d66e73d66a09584293db5e5d44`; 267,274 bytes.
   - Read the parameter definitions, summary indicators, threshold and classification tables, within-day/intermittency adjustments, data-availability routes and worked case studies. Check original tables visually where extraction loses alignment or symbols.
   - Source-based hydrological alteration/risk classes are not measured biological status or legal compliance.

**IHA source gap remains explicit.** Richter, B. D.; Baumgartner, J. V.; Powell, J.; Braun, D. P. (1996), *A method for assessing hydrologic alteration within ecosystems*, *Conservation Biology* 10(4), 1163–1174, is cited in the handover bibliography and both diagnostic discussions, but its full text was not supplied or verified during discovery. Obtain a lawful readable copy and the applicable official Nature Conservancy IHA manual, or establish sufficient authoritative manual coverage for the implemented 33-parameter profile and record the still-missing paper. Record source versions and precise definition/table/test crosswalks. Do not claim that a citation, old code, or these two papers alone establishes every IHA convention. Missing access blocks unsupported method claims, not unrelated foundation work. No source-completeness claim is made by publishing this vision.

These papers inform verification of already-required diagnostics; they do not expand the commission to every method or application they discuss. If primary sources genuinely leave a consequential interpretation unresolved, record the operation and alternatives and seek an explicit decision rather than inventing a formula or silently retaining old behaviour.

## Natural references are supplied, not automatically constructed

Do not carry forward automatic `naturalize()` in this Effort. Fishy consumes identified reference series or model results prepared and assessed externally. Comparing reference and managed flows remains required; generating the reference is a separate responsibility.

The report explicitly recognizes model-consistent reconstruction but retains natural channel losses, groundwater exchange and other natural processes. Switching off all losses is a simplification whose effect must be tested, not an upper bound on natural flow. Removing demands alone neither removes regulation nor reconstructs nature. Present-climate natural reference, naturalised historical benchmark and future-climate stress series remain distinct.

The old helper removed losses and replaced storage/demands with unrestricted passage. Those are unapproved physical assumptions, not requirements. A reconstructed series does not acquire scientific or official acceptance because it was generated by a structural transformation, has zero numerical balance residual, or matches a calibration record. Where a method requires two structurally different references, two parameter configurations of one model do not satisfy that condition. Fishy preserves reference identity, dependence, limitations and supplied acceptance findings; it does not orchestrate reconstruction, calibration, member selection or validation studies.

## Shared foundation contracts

### Identity, mappings and duties

Accept prepared, versioned physical water-body, calculation-reach and basin-plan-unit identities separately. Distinguish calculation sections, observation points, compliance points and model output mappings. Support multiple sections per reach and shared sections where declared. Missing plan assignment alone does not prevent supported reach calculations. Later assignment without a physical boundary change differs from splitting a reach, which requires new attributable results and preserved predecessor history.

Origin/designation governs natural versus potential track; use/category governs its own dependent checks. An irrigation river does not become a canal. Unknown category cannot disable independent hydrology. Carry unresolved, pending, expired and other source-defined designation states rather than inferring a designation. F1 foundation represents and explains supported prepared classification decisions; full method-specific route ladders remain with their owning Efforts.

Keep existing duties independently attributable by provision, applicability, quantity, units, interval, location and version. A duty found but inapplicable, an unresolved duty and none found after documented search are different states; no located duty is not proof that none exists. Do not silently reconcile conflicting instruments, infer category strictness or merge prescriptions by an assumed universal sum/maximum. Earlier issued results and original source profiles do not change when a caller changes a scenario.

### Physical and temporal meaning

Inputs and results carry source, scenario, reference member, mapping and software/data/configuration versions, units, exact interval, calendar, coverage, uncertainty, validity and relevant predecessor data. Observation correction/infill status remains distinct from simulated or reconstructed production method.

Convert interval water volumes to mean discharge using actual duration. Temporal aggregation must preserve volume and disclose lost resolution; daily means cannot certify unseen within-day conditions. Preserve leap days, partial coverage, excluded warm-up and missing predecessors. Complete-year requirements apply only to operators that need them, not every Fishy input.

Preserve present zero, missing, absent, outside-horizon, dry undefined concentration and unsupported evidence. Reject invalid units/basis, conflicting duplicates, incompatible mappings/intervals, nonfinite values and negatives outside a declared nonnegative domain. Do not prohibit supported signed stage, velocity or exchanges. Numerical counts/quanta and disclosed approximations are not scientific uncertainty. Conservation or domain failure invalidates affected assessment paths, without disabling independent supported constituents or water results.

Keep requirement, floor, physical availability, deliverability, issued obligation and actual delivery distinct. A generic supplied-duty check must not import Uzbek issuance caps into sanitary or foreign duties. The C4 10/6/5 example and final Uzbek issuance belong to the later assembly method; this foundation supplies the distinct immutable carriers and evidence semantics, not a hidden full regime implementation.

### Evidence and findings

Keep computability, disclosure, purpose-specific scientific adequacy and official admissibility separate. Preserve `accepted`, `accepted as indicative` and `not accepted` findings with the specific product, reach, member, period, use and reasons. Short records alone neither reject all methods nor confer acceptance; rating-range and other reason-specific restrictions still apply. Official pending is not scientific rejection. Scenarios and illustrations are not observations or official findings.

Required checks combine as follows: all supported passes yield complete pass; a supported required failure survives unknown checks and reports incomplete coverage; without a known failure, an unknown required check yields indeterminate. Empty required sets, omitted reaches, missing group members and failed computations cannot yield complete satisfaction. Do not silently skip failed reaches and report the surviving average as complete. An annual-only or otherwise partial product remains useful but is not a complete regime or passport.

## Required diagnostics and physical integration

Re-establish all required IHA, IARI and DHRAM diagnostics with explicit source-defined profiles on supported dated data, independently of a simulator. Use current Taqsim results through an explicit physical/time/location boundary, not retired `Trace`, node or event imports. Reference and impacted inputs remain identifiable and compatible; diagnostic comparisons are not a general scenario-comparison engine.

Independently verify parameter membership, means versus medians, rolling-window boundaries, leap/calendar treatment, pulse thresholds and runs, extrema ties, circular timing, rise/fall/reversal definitions, dispersion and coefficient-of-variation conventions, zero reference denominators, zero-width IQR and supplementary evidence. Historical code includes ad hoc degenerate-value handling and false-default supplementary flags; neither becomes a source rule through reuse. Unknown subdaily or cessation evidence must not silently become false. Preserve attributable unsupported outcomes when a source offers no justified operator. Do not add a legacy simplified threshold profile solely because it exists in deprecated code.

At discovery, current Taqsim `main` was `1c7aea708e2878047b792b859a9bec17789c3734`; its Incidence pin was `665da4e0d81ab28921b8d5d2edbb9be27f4ec612`. Effort #23 is landed, with physical implementation PR <https://github.com/hydrosolutions/taqsim/pull/35>. Read its pinned `docs/conservative-lifecycles.md`, `src/taqsim/physical_results.py`, tests and public example. Recheck actual target revisions before implementation; handover baseline commits describe an older source inspection, not current compatibility.

The physical surface now supplies incoming/outgoing/storage views, counts/quanta, amounts, constituent support/reasons, balances, metadata and durable saved projections including arrivals. Choose the physically appropriate view explicitly rather than accepting the default outgoing view. Available numeric water does not override unsupported water-quality status. Concentration aggregation uses summed mass divided by summed water, not an unweighted mean; storage is not interchangeable with interval transfers. Imported process results require their declared validity and account ownership. Saved projections are not restart checkpoints.

The new Fishy repository started from scaffold commit `619bf43e15aa4eed021cb38959061a149329ee29`. Scaffold tests are not diagnostic acceptance. Deprecated code/tests may help locate mathematical cases but are neither architecture nor source authority. Reproducible clean-source installation and compatibility examples are required; PyPI publication is excluded.

## Observable completion evidence

Deliver maintained public-path tests and examples, with actual versus expected values, justified tolerances, exact software revisions, and a requirement → public operation/input → observable result → executed test crosswalk.

- Demonstrate all three diagnostics on sufficient reference/impacted records with independently justified numerical expectations, source-table/example checks where available, and discriminating boundary/degenerate cases. A zero or constant series must retain its source-supported meaning rather than hide division failures.
- Demonstrate a real Taqsim-to-Fishy path with explicit location/view, interval-volume conversion and supplied-duty assessment, plus equivalent supported observation/import input without running Taqsim. Exercise live and supported saved results and their metadata/presence meanings.
- Use supplied duties `[2, 3]` and daily delivery `[1.5, 3.5]` m³/s to obtain shortfalls `[0.5, 0]` and 43,200 m³, with no later-surplus cancellation. Preserve duty versions and distinguish scenario prediction from observed compliance. Include present zero, missing, outside-horizon and unsupported variants.
- Exercise C1–C6 foundation semantics: prepared mappings and classification, source/scenario isolation, evidence/completeness, distinct physical and duty quantities, exact time/unit meaning and useful partial outputs. Record which route-specific cases remain owned by the method Efforts rather than claiming full Program acceptance here.
- Cover successful, missing, unsupported and infeasible supplied cases; a system that always returns pending is insufficient. Do not require unavailable basin field data for synthetic software acceptance or label synthetic acceptance scientific certification.
- Provide clean-source setup/use instructions, compatible pinned source revisions and executable modeller-facing examples. Run native tests, formatting, lint and type checks through `uv`, following repository instructions. For bug fixes, prove the real failing path before changing it.

A13's quality-limit assessment is owned by #24, not completed by this foundation merely because physical data crosses the boundary. Full Swiss, Kazakh, sanitary-estimation, HYDMOD, Uzbek hydrology/assembly, quality, hydraulic and receptor operators belong to their respective Efforts. This foundation must enable their independently runnable contracts without creating a monolithic all-method orchestrator.

## Boundaries and remaining risks

No automatic naturalisation, counterfactual engine, scientific-validation orchestrator, optimiser, report generator, GIS delineation, national-policy selection, legal authentication, general hydraulic/habitat/reactive solver or deprecated API compatibility layer is commissioned. No implementation is authorized by publication of this document alone.

The known source-access gap is IHA primary/manual coverage; IARI additionally requires tracing its cited ISPRA definition where necessary. These are explicit verification work, not permission to guess or silently narrow the required diagnostics. Record any genuinely unresolved source interpretation and continue independent supported work. Source compliance, numerical implementation acceptance, scientific suitability and official use remain separate claims.
