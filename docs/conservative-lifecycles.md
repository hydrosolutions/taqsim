# Conservative water and constituent lifecycles

Attach `ConservativeTransport` to an ordinary `WaterSystem`. Water rules determine
realised transfers. Chemistry follows those transfers, not requested releases.
Run the example with:

```sh
uv sync --locked
uv run python examples/conservative_lifecycles.py
uv run pytest tests/test_conservative_lifecycles.py -q
```

The example mixes 10 m³ carrying 8 kg with 5 m³ carrying 1 kg. Its outlet is
15 m³, 9 kg and 0.6 kg/m³ (600 mg/l). These are synthetic checks, not site calibration.

## Inputs and results

Identify each constituent by chemical form, reporting basis and mass quantum.
Use `Mass` for interval totals, `MassRate` for rates, `Concentration` for boundary
concentrations, or `MassCounts` for authoritative integers. A `Composition` holds
independent constituent declarations. Omitted chemistry is unknown, not zero.
Initial inventory requires mass, not concentration or rate. Initial empty reaches
without chemistry start at zero mass. Initial wet reaches need chemistry.

Prepare water rates explicitly with `IntervalMeanRate.aggregate_to(time)`.
Chemistry rates integrate once over the actual model interval. Source compositions
must have exactly one record per model interval. Original water unit, cadence,
resolution and rate-versus-volume provenance remain attached to physical results. Conflicting mass/concentration
inputs and unknown locations fail rather than silently override each other.

`run.physical` offers:

- `amount(location, step, substance="water", view="incoming")`: Decimal m³ or kg.
- `concentration(location, step, constituent, view="incoming")`: exact Fraction
  in kg/m³, or `None` when no supported ratio exists.
- `sample(...)`: exact water and mass counts, `water_quality`, per-constituent
  quality and reasons. Invalid support does not erase the diagnostic numeric amount.
- `quality_state(location, step, constituent, view="incoming")`: `ABSENT` for an
  unregistered constituent, `OUTSIDE_HORIZON` for a registered constituent outside
  the horizon, `DRY` for a zero denominator, or the actual supported/missing/
  unsupported state. The saved constituent registry preserves this distinction.
- `aggregate(location, start, stop, view="incoming")`: sum whole-interval amounts
  before calling `sample_concentration`. It does not infer subdaily values.
- `balances` and `basin_balances`: start + incoming − outgoing − end in counts.
- `process_balances`: independently recomputed supplied `ProcessAccount` residuals,
  separate from the simulated physical accounts. Unknown terms stay incomplete.
  `metadata["process_account_exchange_checks"]` separately compares supplied terms
  with mapped realised exchanges. A closed imported account can still disagree
  with the actual exchange; neither check overwrites the other.

Use `incoming` at outlets and arrival points, `outgoing` for reach transfers, and
`storage` for terminal inventory. Storage concentration is not a whole-interval
exposure or necessarily the concentration of an earlier transfer. Unknown locations
raise `KeyError`; outside-horizon samples have an explicit quality state. Zero water
has undefined aqueous concentration. Missing chemistry remains distinct from both.

`EvaporateThenRelease` removes salt-free water before mixing the release.
`TravelDelay(1)` preserves each incoming parcel for one interval; it is not a
well-mixed delay tank. `Hold` exposes retained stock. Dry salt remains inventory.
Rewetting requires explicit `Remobilisation.COMPLETE` for complete dissolution.
Without it, dependent concentration and downstream support remain unresolved.

## Executable crosswalk

All tests below use the public water/chemistry configuration and actual run output.
Volumes are m³ and masses kg. Most tests declare 0.001 m³ water and 0.000001 kg
mass quanta and compare exact values, with no floating tolerance. Concentrations
compare exact fractions. Per-branch integer floors retain unallocated mass.

| Requirement | Test name fragment | Independent expected output |
|---|---|---|
| A1 | `a1_complete_mixing` | 15/9, concentration 3/5; empty final stock |
| A2 | `a2_branch_allocations` | 6/3.6 and 9/5.4 |
| A3/A4 | `a3_a4_realised_release` | Request 40: 30/8, deficit 10; request 15: 15/4 released and retained |
| A5/A6 | `a5_a6_and_post_evaporation_order` | Evaporate 2: retain 8/5; seep 2: destination 2/1, retain 8/4 |
| Operation order | Same test | Evaporate 2 then release 4: release 4/2.5, retain 4/2.5 |
| A7 | `a7_capacity_routes_spill` | Downstream 10/6; spill 5/3 |
| A8 | `a8_variable_chemistry_parcels_and_aggregate` | Arrivals 0/0, 10/8, 5/1; aggregate concentration 3/5 |
| A9 | `a9_located_use_delayed_return_and_caller_comparison` | Use evaporation 4/0; next-interval return 6/5 |
| A10 | `a10_drying_rewetting_and_dependent_outflow` | Dry 0/1; refill 4/1; explicit dissolution 1/4, otherwise unresolved |
| A11 | `a11_missing_chloride_does_not_remove_sulphate_support` | Sulphate concentration 2/5 remains supported; chloride missing |
| A12 | `a12_rate_integration_once_on_leap_day` | 864000/432000 daily; 36000/18000 hourly; twice daily amounts over two days |
| Remainder | `nonround_remainder` | 3 water/10 mass split 1/2 carries 3/6; 1 retained at 1 kg quantum |
| Exact authority | `exact_counts_above_float_precision` | Four input masses of 2⁵³+1 counts split into one and three input masses exactly |
| Persistence | `delay_projection_roundtrip_and_presence` | Entire physical document unchanged after save/load |
| Dry initial stock | `initial_dry_inventory` | Saline refill gives 4/2; supported 1/2 only with explicit dissolution |
| Selective support | `zero_realised_branch` | Zero branch cannot contaminate independent 4/2 discharge |
| Physical boundary | `future_boundary_water` | Initial basin inventory zero; only current forcing enters stock |

Every supported case independently recomputes local and basin residuals. Several
cases span multiple intervals and include transit. The efficiency example compares
10/4/6 with 6/4/2: diversion and return both decrease by 4; consumption stays 4.
The result does not declare new net basin water or assess ecological thresholds.

## Water authority and physical accounting

Incidence remains the single water authority. Taqsim compiles ordinary water rules
and reads their realised transfers through Incidence's native
`transfer_count_series` interface. It does not reconstruct counts from displayed
floats or rerun a separate water allocator.

The compiler gives each physical branch a pass-through observation compartment.
This preserves distinct allocations even when branches share a destination. The
observers forward exact available counts in the same interval. Their implementation
stocks, like preloaded future forcing, are excluded from physical basin inventory.
Tests compare observed and unobserved water flows, storage, arrivals and exact
counts, including large counts whose displayed floats coincide.

Constituent accounting uses those realised water counts. Incidence's generic
`carrier_proportional` rule uses one simultaneous dependent-pool denominator.
That operation alone cannot express evaporation followed by release from the
concentrated pool, or keep unresolved dry salt separate from fresh mobile mass.
The physical accounting layer therefore applies the declared sequential or
simultaneous mixing rule with integer mass arithmetic. It neither changes water
movement nor claims a new engine solver. Complete-mixing branches use integer
floors; every unallocated mass count stays in inventory. Numerical remainder and
scientific uncertainty are reported separately.

## Supplemental T1–T7 evidence

The following crosswalk covers input contracts, support propagation and result
semantics beyond the A-series examples. Result-level tests construct exact result
objects to exercise diagnostic rules; they are not substituted for the public
run-path acceptance tests above. Compiler observation tests inspect the actual
engine count interface in addition to public water outputs.

| Contract | Test module and witness | Expected and observed result |
|---|---|---|
| T1 typed units and integration | `test_constituent_inputs.py`: `units_and_rate_integrate_once`, `invalid_units`, `nonnegative_finite_domain` | 500 mg/l becomes 0.5 kg/m³; 432000 kg integrated once; incompatible units and negative/nonfinite amounts refused |
| T1 identity and conflicts | Same module: `conflicts_duplicate_and_missing_status`, `invalid_chemical_identity_or_basis`, `network_and_time_binding` | Duplicate/conflicting declarations, unregistered constituents, mismatched basis/location/horizon refused |
| T1/T7 build boundary | `test_transport_support.py`: `invalid_chemistry_bindings_fail_at_build_not_during_execution` | Bad locations, horizons and imported state intervals fail at build |
| T2/T3 single water authority | `test_transport_water_authority.py`: `observation_preserves_original_water_authority`, `parameter_substitution_preserves_water_authority` | All built-in rule configurations, including reservoir evaporation and canal losses, retain identical flows, stocks and arrivals; substitutions retain equivalence |
| T2/T6 branch identity and counts | Same module: `shared_destination_branches_stay_individually_observable`, `observers_preserve_merged_exact_counts_when_float_projections_collide` | Shared endpoint branches retain 20 million and 80 million counts separately; observed large counts equal unobserved counts exactly |
| T3 delay | Same module: `travel_delay_real_retention_and_schedule` | Input 10,20,0 produces arrivals 0,10,20 and transit 10,20,0 |
| T4 dry versus mobile inventory | `test_transport_support.py`: `unresolved_old_dry_mass_does_not_immobilise_fresh_advected_mass` | Old dry 1 kg remains; fresh 2 kg leaves with 4 m³; dependent concentration remains unsupported, balances close |
| T4/T5 zero-water support | Same module: `actual_zero_water_unsupported_boundary_cannot_taint_supported_stream` | Unsupported dry source adds no advected unknown load; supported outlet stays 4 m³/2 kg at 0.5 kg/m³ |
| T5 independent domain support | Same module: `unsupported_input_retains_numeric_mass_and_independent_constituent` | Unsupported chloride retains 5000 counts; sulphate retains supported 2000 counts and concentration 1/5 |
| T5 initial unknown | Same module: `unspecified_initial_chemistry_depends_on_real_initial_water` | Empty initial stock does not invent missing mass; wet unspecified stock remains missing |
| T6 numerical precision | Same module: `physical_output_discloses_input_quantisation_separately` | 0.0019 kg at 0.001 kg quantum gives 1 count and explicit discarded fraction 9/10 quantum; basin residual zero |
| T6 process ownership | `test_transport_water_authority.py`: `process_replacement_disables_both_old_water_directions` | Old incoming and outgoing water both zero; old 3 m³ stock remains; replacement outputs 4 m³/2 kg once |
| T6 validity | `test_physical_results.py`: `balance_independent_residual_missing_and_failure`, `water_balance_failure_invalidates_quality`, `known_balance_failure_survives_unknown_basin_balance` | Residual independently evaluated; unknown is incomplete, nonzero is failed; known failure survives unrelated missing coverage |
| T5/T6 selective diagnostics | Same module: `incomplete_basin_account_does_not_poison_supported_local_observation`, `incomplete_local_balance_preserves_known_incoming_and_zero_outgoing` | Missing basin/local terms do not overwrite independently supported observations |
| T7 attributed specialist values | `test_transport_support.py`: `imported_signed_stage_and_temperature_survive_saved_physical_result` | Imported stage −1.25 m and withdrawal temperature 16.4 °C retain domain, model version and uncertainty through save/load; neither is predicted by mixing |
| T1/T7 imported chemical basis | `test_transport_support.py`: `imported_registered_chemistry_cannot_bypass_basis_unit_or_domain` | Registered chemical imports reject conflicting form/basis, non-concentration units and negative conservative concentration |
| T7 mapping and temporal context | Same module: `supplied_mapping_predecessor_and_exact_software_revision_round_trip` | Mapping version, physical water body, supplied step −1 signed velocity and exact component revision survive unchanged |
| T7 result meaning | `test_physical_results.py`: `dry_absent_unknown_and_outside_horizon_remain_distinct`, `physical_document_retains_exchange_identity_and_all_metadata`, `public_physical_run_saved_roundtrip` | Distinct presence/support states, attributed exchanges, exact counts and full public physical documents survive projection roundtrip |
| T7 legacy cache | Same module: `v4_water_cache_migrates_without_inventing_arrivals` | Old cached water remains available; absent historic arrival data is not fabricated |
| T6 imported account closure | `test_transport_support.py`: `failed_supplied_process_account_invalidates_only_dependent_species`, `unknown_supplied_process_account_never_fabricates_supported_closure` | Chloride account residual 1000 is failed; unknown residual stays `None`/`INCOMPLETE`; numeric mass remains, sulphate stays supported; simulated basin closure is not substituted for imported closure |
| T6 mapped exchange consistency | Same module: `closed_imported_account_must_match_realised_exchange` | Closed imported chloride account with 4000 outgoing counts versus realised 5000 gives difference −1000 and unsupported dependent output; matching 5000 remains supported |
| T5/T6 water validity | Same module: `failed_water_process_account_invalidates_all_carried_constituents`; `test_physical_results.py`: `failed_water_account_retains_numeric_water_with_explicit_invalid_support` | Failed imported water residual 1 leaves numeric water 10 and salt counts visible but sets water and both carried constituents unsupported |
| T5 delayed support recovery | `test_transport_support.py`: `delay_storage_recovers_known_mass_after_unknown_parcel_departs` | Unknown first parcel leaves; retained second parcel recovers supported 4 m³/2 kg, later delivers once, and terminal inventory is zero |
| T3 unsupported delay initialization | Same module: `travel_delay_refuses_initial_dry_mass_without_dated_cohort` | Initial dry salt without a dated delay cohort is refused, not assigned an invented arrival |
| T7 explicit query semantics | `test_physical_results.py`: `public_quality_state_distinguishes_unregistered_dry_outside_and_unsupported` | Unregistered `ABSENT`, known outside-horizon `OUTSIDE_HORIZON`, zero `DRY` and domain `UNSUPPORTED` remain distinct after save/load |
| T6/T7 separate imported diagnostics | Same module: `process_balance_is_separate_and_invalidates_affected_output`; `test_constituent_inputs.py`: `supplied_process_accounts_are_exact_independent_diagnostics` | Independent imported counts and residuals remain explicit; only dependent result support is invalidated |

Executed together after the process-account and quality-query additions: **127 passed** with:

```sh
uv run pytest tests/test_conservative_lifecycles.py tests/test_constituent_inputs.py tests/test_transport_support.py tests/test_physical_results.py tests/test_transport_water_authority.py -q
```

This is focused physical-stack evidence, not a claim that every project regression
or scientific application is covered by these five modules. The A-series suite
contributes 27 cases to this run. The runnable example also produces its asserted
15 m³/9 kg outlet and zero water/salt basin residuals.

## Reproducibility and limits

`run.save(path)` and `WaterSystemRun.load(path)` preserve projections, including
exact counts, chemistry support and transit states. They are not restart checkpoints.
Identical declarations and run identifiers reproduce physical result digests.
Result metadata records input declarations, assumptions, water-model identity and
software versions, the exact water execution digest and transport semantics version.
Supplied `SoftwareIdentity` records retain exact external source revisions. Source identities do not imply post-mixing source apportionment.

The dependency baseline is Taqsim 0.1.4 and Incidence commit
`665da4e0d81ab28921b8d5d2edbb9be27f4ec612`; use the repository lockfile for the full
environment. This document describes synthetic conservative transport only.
The physical graph must be acyclic. Unsupported rule semantics and cyclic coupling
are refused; no last iterate is reported as converged. Supplied specialist state
units and domain stay attached. A state named for a registered constituent requires
its exact chemical basis and concentration units.
Reactive chemistry, reservoir thermal structure, site validity and Fishy assessment
are separate responsibilities. Exact closure does not establish scientific validity.
