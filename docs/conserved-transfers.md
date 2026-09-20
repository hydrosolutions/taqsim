# Conserved transfers through the Incidence boundary

Incidence can allocate a dependent conserved quantity from the **realised carrier
counts** in the same compartment and timestep. Taqsim's transport tests exercise
that generic capability through the public Python binding. `WaterSystem` remains
a water-only authoring API. These tests do not introduce physical constituent,
chemistry, evaporation or return-flow APIs.

## Declare the coupling

Use a `carrier_proportional` disposition for the dependent substance. Each mapping
names one of its branches and an ordinary carrier branch in the same compartment:

```json
{
  "rule_ir_version": "v1",
  "numerical_semantics_version": "v1",
  "partition": {
    "kind": "carrier_proportional",
    "carrier": "water",
    "branches": [
      {"branch": "carried-mass", "carrier_branch": "release"}
    ]
  }
}
```

The public model document declares substances, unit quanta, initial inventories,
rules, transfer bindings, forcing and calendar. Both mapped branches must have the
same destination. Mappings can omit carrier branches. Mass not allocated by the
mapped branches remains in the compartment. Duplicate carrier mappings,
self-coupling and chaining dependent carriers are refused.

For dependent available count `D`, carrier available count `W`, and realised
carrier branch count `w`, the allocated dependent count is `floor(D*w/W)` when
`W > 0`. A zero carrier pool allocates no dependent stock. Each branch is computed
against the same pre-withdrawal pool, including initial and current incoming stock.
The retained count is the exact pool count minus all allocated counts. The
calculation uses integer authority, not rounded projected amounts.

An unmapped water-only branch can represent one step of a separately declared
physical process. Incidence itself does not decide which substances evaporate,
how returns acquire composition, or whether dry salt remobilises. Those are
physical modelling responsibilities.

## Read counts before deriving ratios

```python
model = incidence.compile_model(document)
run = model.run(bytes(range(16)))
water = run.transfer_count_series("outlet", "water", direction="incoming")
mass = run.transfer_count_series("outlet", "salt", direction="incoming")
```

Multiply each integer count by that substance's declared quantum to report an
amount. A supported transfer concentration divides transferred mass by transferred
water. End-storage concentration instead uses the final stored pair. A dry
compartment has undefined aqueous concentration even when it retains mass.
Aggregate mass and water first, then divide. Averaging concentrations without
water-volume weights changes the answer.

`transfer_count_series` folds stored counts directly. `transfer_series` reports
floating-point amounts. A projected float can lose the distinction between
neighboring large counts; it must not be decoded back into count authority.

Selectors aggregate by compartment, substance, direction and time. They do not
identify the original branch label or a particular upstream source inside a
receiver's aggregate. Two branches to one destination produce one aggregate
reading. The shared-destination test checks that aggregate without claiming
per-branch readback. The balance witnesses use explicit store sets and source
accounts whose declared exits all enter that set. They do not infer source
attribution from merged receiver readings or parse opaque log bytes.

## Time, inventory and boundaries

An extra topological compartment does not create an interval delay. The transit
witness uses an explicit one-step lag and retains both quantities before arrival.
The use/return witness uses its use compartment as the one-interval holding store;
its lagged return is capped by the remaining water after the earlier water-only
loss. It adds no second travel delay.

Count balances reconcile initial inventory plus incoming transfers minus outgoing
transfers at each store and timestep. Whole-engine balances include finite stocks
and named boundary accounts. Physical-basin balances instead classify which stores
are physical and when transfers cross their boundary. Preloaded future forcing is
not present river storage. Dry mass, use stock and in-transit inventory remain in
the appropriate account.

## Presence is not physical supportedness

The public Incidence boundary distinguishes:

| Query | Value | Presence |
| --- | --- | --- |
| Modelled zero transfer | `0` | `present` |
| After the model horizon | `None` | `absent` |
| Substance outside the registry | `None` | `not_modelled` |

Taqsim's calendar-indexed water series labels out-of-horizon dates `not_modelled`.
An unknown reach lookup is refused. Preserve query and horizon context when
crossing these existing boundaries; matching enum strings are not sufficient.

Missing chemistry and unsupported remobilisation in the tests are **test-owned
physical metadata**, not new Taqsim features. The missing-chloride witness does not
register chloride with a fictitious zero inventory. Known water and sulphate run
independently; the chloride query remains unmodelled and its physical reason stays
with the caller. Retained-mass bookkeeping without a supported remobilisation
model does not establish a clean-water concentration or a supported physical pass.

## Identity, replay and saved output

The complete coupling declaration contributes to model identity. Removing a
mapping changes the model digest even when every realised transfer is zero.
Initial inventory and precision changes also produce different identities.
Identical model and run ID reproduce canonical log bytes and digest.
`replay_against` refuses a different model artifact.

The existing Taqsim v4 saved artifact is a **water-output cache, not a checkpoint**.
It preserves offered water flow, retained-water stock, presence, time, quantum,
model identity, engine version and opaque authoritative log bytes. The test uses
the public `WaterSystemRun` constructor with a generic completed run to verify
only those water projections. A loaded artifact does not expose arrival or
constituent projections. Preserving its opaque log does not add such projections
or resume execution. Generic mass/count readback is tested live and by rerunning
the declared model. Durable supported physical quality results belong to the
physical constituent API work.

## Executable acceptance crosswalk

Run `uv run pytest tests/transport` with the committed Incidence dependency.

Verified with Incidence `665da4e0d81ab28921b8d5d2edbb9be27f4ec612` and Python
3.12.11: 33 transport cases and all 108 Taqsim tests passed. The locked dependency
was built from its Git revision without a local source override. `ruff check`,
`ruff format --check` and `ty check` also passed.

Fixtures use interval m3, kg and kg/m3. Most water quanta are 1 m3 and mass quanta
0.1 kg; the repeated multiconstituent fixture uses unit mass counts, and the
large-count witness uses 1e-6 kg. Exact count comparisons have **zero tolerance**.
Derived non-terminating concentration ratios use floating-point comparison only
for presentation, not accounting. These arithmetic tolerances are not measurement
uncertainty or scientific acceptance criteria.

| Requirement | Public input/operation | Expected and asserted result | Test |
| --- | --- | --- | --- |
| A1 | Merge 10/8 and 5/1; release all | 15/9 transferred; no retained stock; aggregate C=.6 | `test_realised_mixed_pool[A1-mixing]` |
| A2 | Same pool; branches 6 and 9 | 6/3.6 and 9/5.4 | `test_realised_mixed_pool[A2-branches]` |
| A3 | Initial 20/4 plus 10/4; cap request40 at available | Actual30/8, not40 or incoming-only chemistry; zero final inventory | `test_realised_mixed_pool[A3-actual-not-requested]` |
| A4 | Same pool; withdraw15 | Release15/4; retain15/4 | `test_realised_mixed_pool[A4-retention]` |
| A7 | Incoming15/9; capacity10 plus overflow5 | Named destinations10/6 and5/3 | `test_realised_mixed_pool[A7-capacity-overflow]`, `test_capacity_overflow_is_the_residual_of_the_available_mixed_pool` |
| Zero-carrier regression | Water1, mass1; half requests at 1 m3/0.1 kg | Carrier0/0; dependent0/0; mass retained1 | `test_zero_realised_carrier_cannot_advect_mass` |
| Non-round/repeated/multiple | Initial water2, salt4, tracer11; incoming water[3,2,0], salt[6,3,0], tracer[7,5,0] | Water left[1,2,1]/right[2,1,0]; salt left[2,3,3]/right[4,1,0]; tracer left[3,6,4]/right[7,3,0]; retained counts checked each step | `test_nonround_repeated_allocations_keep_remainders_and_independent_constituents` |
| Shared destination | Carrier pool3, dependent5 counts; two branches realise1 each | Dependent aggregate2 counts, not merged-floor3; retain3 | `test_same_destination_preserves_sum_of_distinct_branch_allocations` |
| A8 | Supply10/5 to explicit one-interval lag | Arrival water[0,10,0], mass[0,5,0]; transit initial step10/5 then0 | `test_a8_one_interval_transit_is_inventory_not_a_topological_label` |
| Sparse projection | Separate water/mass lags over mixed-substance events | Each named substance arrives once at t1, without cross-substance projection refusal | `test_sparse_multisubstance_lag_reads_only_its_named_substance` |
| A9 | Withdraw10/5, remove water4, return after one-interval hold | Return6/5 at t1 only; use stock6/5 at t0; no duplicate external load | `test_a9_use_inventory_returns_once_after_water_only_loss` |
| A10 supported | Initial2/1 dries; clean inflow4 next step with declared remobilisation | Dry retains1; later release4/1, C=.25 | `test_a10_dry_mass_survives_then_declared_complete_remobilisation` |
| A10 unsupported | Same forcing, no supported remobilisation | Mass1 remains; test-owned aqueous result unsupported, never an inferred zero | `test_a10_unsupported_remobilisation_keeps_inventory_not_invented_concentration` |
| Dry quantisation remainder | Water3/mass1 released in water branches1 and2, then clean water3 | First-step mass branches.3/.6; dry residue.1; next release carries residue | `test_all_carrier_leaves_but_fractional_mass_stays_as_dry_inventory` |
| Selective disposition / C meanings | Initial10/5; release4 plus water-only loss2 | Transfer C=.5; final stock4/3, C=.75 | `test_transfer_concentration_differs_from_final_storage_after_selective_loss` |
| Aggregation / result for A13 handoff | Interval outputs10/8 and5/1 | Sum15/9, C=.6=600 mg/l; not unweighted mean.5 | `test_volume_weighted_aggregation_uses_mass_and_volume_not_mean_concentration` |
| Physical-basin exclusion | Future supply101/101, actual entries[1,100] | t0 physical pool1/1 versus engine stock101/101; future supply excluded | `test_future_forcing_counts_are_excluded_at_the_physical_entry_boundary` |
| Count precision | Four exact source mass counts2197111022144419 merge | Total8788444088577676 and exact1:3 split preserved despite ambiguous projected float | `test_large_merged_dependent_counts_never_reenter_through_float_amounts` |
| Refusals | Invalid domains, counts/quanta, allocation or branch/carrier mappings | Attributable ValueError; no clipping or balancing adjustment | `test_coupled_invalid_models_or_runs_refuse`, `test_unrepresentable_generic_count_domain_refuses` |
| A11 boundary prerequisite | Supported water/sulphate; chloride unmodelled with missing-input reason | Known components available; chlorideNone/not_modelled; water zero remains present | `test_presence_and_missing_quality_keep_independently_known_components` |
| Content identity | Same execution; change mapping, precision or initial stock | Stable identical log; changed digest and replay refusal | `test_model_identity_binds_coupling_mapping_precision_and_initial_input` |
| Saved output scope | Public completed run wrapped as water results, saved and loaded | Water flow/retention/log preserved; arrivals refuse; no mass projection or restart | `test_existing_saved_water_projections_do_not_offer_mass_arrivals_or_restart` |

`model_cases.assert_balances` independently reconciles public counts in the local,
whole-engine and declared physical-basin accounts. The original water quantum,
capacity, time, replay and saved-output tests remain compatibility checks.

### Source and ownership

This crosswalk implements the generic execution foundation in
https://github.com/hydrosolutions/taqsim/issues/22, following the approved
[coupled-transfer vision](../planning/visions/2026-09-20-coupled-conserved-transfers.md).
Its source requirements are the approved 19 September 2026 brief T2/T3/T6/T7 and
acceptance A1–A4/A7–A10, report D.4 mixing/accounting/receptor operators and
Appendix C.1 exchange reconciliation. The vision records source identities;
private originals are not included or required at runtime.

The zero-water counterexample originally produced carrier counts[0,0] and mass
counts[5,5] through the legacy independent-partition interpreter. Independent
partitions still mean independent allocation; the explicit coupled declaration
is the repair for advective use. A staged public composition reproduced small
complete-mixing cases, but re-entering large authoritative counts through float
forcing lost one count. The selected generic operation keeps exact carrier and
dependent authority together. Incidence's regression evidence records the
old-code failure and the repair; the Taqsim boundary tests exercise the resulting
public operation without private interpreter hooks.

Taqsim's physical constituent work owns typed composition/load inputs, chemical
basis, supported mixing/loss/return/remobilisation models, physical spatial
mapping, unknown propagation and durable physical quality outputs. Fishy owns
configured quality tests and the realised A13 exchange against that physical API,
plus country methods and assessment fixtures. Generic conservation neither
calibrates those models nor selects policy. A daily interval total supplies no
hourly peak evidence, and reduced diversion alone proves no net basin saving.
