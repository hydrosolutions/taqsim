# Application validation: 1 October 2026

This dated record complements the [conservative lifecycle guide](conservative-lifecycles.md):
it describes tested application sizes and unresolved acceptance gates. It is not
an exhaustive feature certification or field calibration.

The campaign used Taqsim `54c00a8784704dd12ef74de38cd5bb881df30f6d`,
Incidence `665da4e0d81ab28921b8d5d2edbb9be27f4ec612`, and Fishy
`881660e8901987d63d89caf61ef0d30991000301`. Results concern those revisions;
future changes need their own acceptance evidence.

## What the applications establish

| Application | Outcome and boundary |
|---|---|
| Exported Zarafshan basin: 61 nodes, 70 connections | One January day passed integer water/salt accounting, comparison with the original model, actual Taqsim-to-Fishy projections, metadata refusals and save/load preservation. Independent review confirmed the same water model/log across stages. This does not establish a continuous season. [T03] |
| Branching reservoir/use/return corridor | Nine combinations of three chemistry variants and 7/30/92-day horizons passed 12,322 known mass comparisons, 836 unknown-state comparisons and 774 Fishy constituent-day findings. Competing requests, a saline tributary, one-/three-day returns and a rejoin were exercised. Mass expectations are conditioned on realised native water, not a second water solver. [T06] |
| Public multiyear workflow | Eight reduced reservoir applications exercised native `run`, physical output, outgoing/arrival/retained projections, save/load and Fishy assessment. Four completed all 1,096 days of 2018–2020. Independently reconstructed daily water and Fishy arithmetic passed; reloaded results matched exactly. The reservoirs were separate systems with model-derived inflows and hypothetical priorities. [T08] |
| Resolution and process isolation | Seventeen reduced cases used joint water/mass quanta of 1 m³/1 g, 1 L/1 mg and 1 mL/1 µg. All 6,223 known integer balances closed; unknown balances stayed unknown. Five concurrent replicas matched eight scientific files each byte-for-byte. [T09] |

Resolution comparisons agreed for all 180 classifications whose declared
engineering envelopes lay entirely on one side of the target. Another 96 envelopes
crossed the target: unchanged observed classifications do not establish robustness
near a threshold. These envelopes are application acceptance criteria, not universal
rounding bounds. The paired changes do not identify separate water/mass effects.
Separate-process success does not test thread safety, shared-output writes or
optimizer multiworker execution. A saved-run cache preserves results; it is not a
restart checkpoint.

## Acceptance gates that remain open

Optimization accounting and repeat-seed reproducibility passed for two 30-day
scenarios, 121-point grids and eight NSGA-II invocations. Review checked 2,838 native
candidate evaluations and 100 returned-solution occurrences. However, the half-full
scenario **failed** the strict grid-nondominance quality gate. Drought results passed
nondominance but had incomplete, seed-dependent coverage. Neither outcome establishes
a global Pareto frontier; search quality must be checked separately from accounting.
These results alone do not establish a product correctness defect. [T10]

Two reviewed findings remain recorded: [#47](https://github.com/hydrosolutions/taqsim/issues/47)
observed one litre retained instead of spilled relative to exact allocation, while
conservation remained intact; [#48](https://github.com/hydrosolutions/taqsim/issues/48)
concerns discoverability of canonical quantized float construction. No fix acceptance
is claimed here. [T05, T12]

The continuous 92-day full basin timed out twice at the 7,200-second execution
budget. The retry built successfully, but native execution produced no water result;
conservation validation, seasonal temperature and Fishy assessment were never reached.
Reduced-system successes do not clear [Incidence #28](https://github.com/hydrosolutions/incidence/issues/28).
Six-year basin runs/optimization and real large SWAT+ imports remain unestablished. [T11]

## Evidence and reuse

The [evidence excerpt](evidence/application-validation-2026-10-01.json) includes
selected original result/review fields, source identifiers and SHA-256 hashes,
plus archive identities. Source paths identify files in the originating
`hydrosolutions/zarafshan-taqsim` campaign, not files in this checkout.
This excerpt is **not a complete reproduction harness**; the separately retained
archives contain the scripts, inputs, full reviews and preserved attempts. Counts
overlap across scenarios and checks and must not be summed into exhaustive coverage.
Use these scopes and failed gates when designing acceptance for a new application.
