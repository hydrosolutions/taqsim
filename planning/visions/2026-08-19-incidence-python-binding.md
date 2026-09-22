# Vision: incidence python binding

Program: https://github.com/hydrosolutions/taqsim/issues/4
Effort: https://github.com/hydrosolutions/taqsim/issues/7

## Historical record provenance

This document preserves the original local vision for this delivered Effort. It is a
historical record repair, not a new implementation request or a claim of retroactive
approval. The original vision text below is unchanged. Its present-tense statements,
paths, versions, constraints and open questions describe the original work, not today's
repository. Delivery evidence and later decisions are separate from those requirements.

Original source: `planning/2026-08-19-incidence-python-binding/vision.md` in taqsim.
Original source SHA-256: `1fa27236f6e17f632bc2e8c78e99956b427b864445f1927aa2845eeb3cafeb1e`.
The original local vision has no commit in the available repository history; its frozen
plan and execution records are corroborated by the merged delivery pull requests.

Frozen plan: `planning/2026-08-19-incidence-python-binding/graph.v3.json`.
Frozen-plan SHA-256: `a6fa0535ce83f678b05135ce6b37090e397e2cfed91f49104a17f58ea34ae9d4`.
Execution record: `planning/2026-08-19-incidence-python-binding/driver-journal.jsonl`.
Execution-record SHA-256: `285f5cc3bcbb27be254b046ae01bf0905d440ebbc3de1563d87c4f9d47f9f878`.
The retained final assembly records contain 21 criterion executions,
all with exit status 0. These are historical recorded results, not tests rerun for this repair.

Delivery evidence:
- https://github.com/hydrosolutions/incidence/pull/20 (merge `29a2f0cae10924858c11906791fad87331197a5d`).
- https://github.com/hydrosolutions/incidence/pull/21 (merge `90398c8e625b7d9dcb915107f586bb78bd111cbf`).

Decision evidence:
- https://github.com/hydrosolutions/incidence/blob/90398c8e625b7d9dcb915107f586bb78bd111cbf/docs/adr/0002-python-binding-is-a-thin-boundary.md

## Original vision (unchanged)

## Goal / Why

`incidence` is a finished conserved-flow engine that nothing outside Rust can use.

On `main` at `25a5bfd` it is 10,557 lines of library source with 206 passing tests, no `todo!`
or `unimplemented!` anywhere in `crates/core/src`, and every one of its own eleven acceptance
criteria discharged against the composed assembly: exact ledger replay, continuation sufficiency
from a truncated prefix, closed books with every incidence-matrix column summing to zero,
rejection of omitted substances and overdraw, the absence-is-not-zero trio at all three
granularities, digest binding, IR round-trip, and a seven-rule hydrology fixture running to
completion in one model.

None of that is reachable. `ModelArtifact` does not deserialise; the only way to construct one is
`ModelArtifact::builder(...)` fed by typed fallible Rust calls. The only model anyone has ever
built is `HydrologyModelDocument` — 524 lines in `crates/core/tests/support/hydrology.rs`, private
to the test target, invented by the fixture author precisely because no public whole-model form
exists. Its own doc comment states the requirement this vision fills: *"Every field needed by the
artifact builder is data in this value; decoding never chooses a native rule implementation."*

The incidence vision named this gap as its own unmitigable risk: *"This vision builds a core with
no consumer, so its interface meets reality only at the taqsim refactor. Interface problems that a
co-designed port would have caught may surface later. The fixture rule set is the mitigation and
is deliberately weaker than a real consumer."* This vision collects that debt. Python is the real
consumer, and every day #8 waits is a day the engine's interface goes untested by anything that
must actually live with it.

Success means a Python caller can describe a complete model as plain data, hand it over once, run
it, and read results back — with the core owning every validation, so exactly one implementation
decides whether a model is legal. It also means the signature failure mode of this project cannot
cross the boundary: `reach.py:83` recorded an output only when the amount exceeded zero, making a
dry day and a missing day indistinguishable, and `iari/evaluate.py:38-41` then derived its date
axis from what was present, computing hydrologic-year statistics over a punctured 2,191-day
horizon. incidence made that unwritable in Rust through `ValueState`. This vision makes it
unwritable in Python.

Vocabulary is in `CONTEXT.md`, including the `Model document` entry this vision introduces and the
prohibition on bare time-indexed series.

## Scope — In

1. A **public, versioned model document format** in `incidence-core`: the serialisable plain-data
   form of a complete model — compartments, boundary accounts, substance registry, initial stocks,
   calendar, horizon, units, forcing series, interpolation tables, rule expressions, dispositions,
   rule parameters, and execution bindings — with validated decoding into `ModelArtifact`. This
   promotes the private `HydrologyModelDocument` shape into supported public surface and is the
   only supported way to author a model from outside Rust.
2. Deletion of the private `HydrologyModelDocument` from `crates/core/tests/support/hydrology.rs`,
   with the fixture rebuilt on the public format so exactly one document type exists.
3. A **PyO3 binding crate inside the `incidence` repository**, built with maturin, versioned and
   released in lockstep with the core.
4. Python model authoring: typed document construction plus pure-data expression combinator
   helpers (`add`, `mul`, `min`, `max`, `clamp`, `select`, `param`, `input`, `forcing`,
   `projection`, table lookup) that build document nodes, never closures.
5. Run execution across the boundary exactly once per run — whole model in, whole log out, no
   per-timestep callback: `compile`, `run`, and access to the resulting sealed log.
6. Result reading in which **presence is unavoidable**: every time-indexed read returns values
   alongside a per-timestep presence array distinguishing modelled zero, absent, and not modelled.
   No call in the binding returns a bare time-indexed sequence of numbers.
7. A **parameter-substitution path** for optimizer sweeps: a decoded model held across trials, run
   with substituted declared rule parameters, recomputing a genuinely fresh model digest per trial
   without re-transmitting forcing data. Substitution is restricted to declared rule parameters;
   forcing values, topology, stocks and calendar are refused.
8. **Total panic containment at the boundary**: every rejection surfaces as an ordinary Python
   `Exception` subclass carrying the core's own error text. Measured on 2026-08-19 against a PyO3
   0.22 probe: a Rust panic does *not* abort the interpreter as first assumed — PyO3 converts it
   to `pyo3_runtime.PanicException`, which derives from `BaseException` and **not** from
   `Exception`, so it escapes every `except Exception` handler in the caller. For an NSGA-II sweep
   wrapping each trial in `except Exception`, that is an optimization run dying hours in on a
   Rust-internal message no caller can act on. No boundary function may allow a panic to escape.
9. A measured sweep-cost baseline for a basin-scale model, recorded with the machine it was
   measured on, sufficient to decide whether the substitution path in item 7 earns its existence.
10. The seven-rule hydrology fixture — Muskingum, lag, linear reservoir, evaporation, seepage,
    reservoir policy, demand — authored from Python and proven to produce a byte-identical log to
    the Rust fixture.
11. Type stubs and a wheel buildable for CPython 3.12+, matching taqsim's `requires-python`.

## Scope — Out (explicit non-goals)

1. Any hydrology-friendly or water-naming Python API. `CONTEXT.md` assigns the Python-facing
   modelling API to taqsim; a friendlier surface here would have to name water to be friendly,
   breaking the engine's substance-neutrality constraint, and would compete with the layer #8 must
   build. The binding wears incidence's own vocabulary: compartments, substances, transfers,
   dispositions.
2. **Persistence.** Writing a model or a log to disk and reading it back. Deliberately deferred to
   ticket #8 and recorded there as a comment on 2026-08-19, on two grounds: a storage format is a
   second permanent wire commitment that should be shaped by a real consumer's needs, and nobody
   has measured whether re-running is even painful. Nothing is made harder by waiting — the log
   carries a `RunCompleted` seal and log digest, the artifact is content-addressed.
3. Publication to PyPI or crates.io, and reserving the `incidence` name on either. Publication
   belongs to ticket #6. The name is unclaimed on PyPI as of 2026-08-19 and that risk is
   knowingly accepted; the stack is an internal tool and public presence is secondary.
4. Refactoring taqsim onto incidence (#8) or fishy onto incidence (#9).
5. Any change whatsoever to `taqsim`, `fishy`, `zarafshan-taqsim`, `ctrl-freak` or
   `taqsim-hydrology`. They remain untouched and running; taqsim's 1,012 tests are not expected to
   change because taqsim is not touched.
6. Retiring taqsim's 14 event types, seven node types, `LossReason`, `Edge`, `Trace`, or the
   `Strategy`/`__params__`/`__bounds__` reflection machinery. All of that is #8's.
7. Proving the IR adequate for real hydrology as opposed to the seven fixture rules. `CONTEXT.md`
   carries this as an open ambiguity whose resolution condition is a real consumer; #8 resolves it.
8. Any e-flow method (Q347, Kazakh, WUA, salinity) or scenario-ladder work.
9. Any change to the engine's semantics, guarantees, or numerical behaviour. This vision exposes
   incidence; it does not alter what incidence computes.
10. Fixing the PCE run-machinery friction observed during the incidence-core run. Tooling, not
    engine, and explicitly kept out.
11. Re-surveying the Program Map or minting follow-on tickets.

## Constraints

```yaml
repos:
  taqsim:
    path: .
  incidence:
    path: ../incidence
consumption: []
```

1. All code changes land in `incidence`. `taqsim` is declared because the vision and its planning
   artifacts live here, and is not modified by any package.
2. The binding lives inside the `incidence` repository as a crate built with maturin, not in a
   separate repository. It and the model document format are two halves of one promise and must
   version together.
3. The engine stays substance-agnostic. No binding symbol, type, docstring or error message may
   name water, a hydrological unit, or any substance. Substance identifiers are caller-supplied
   strings throughout.
4. The model document format is a **public versioned wire commitment**. Once taqsim writes models
   in it, its shape is owed compatibility. It carries an explicit format version alongside the four
   version stamps incidence already has (`RuleIrVersion`, `InterpreterVersion`,
   `NumericalSemanticsVersion`, `CanonicalEncodingVersion`).
5. Expression construction on the Python side produces plain data. A combinator helper returns a
   document node; it never returns a closure, and no native or Python escape hatch may be
   reachable from the IR.
6. Validation lives in the core, never in the binding. The binding may not re-implement any check
   the core performs; a Python-side check that duplicates a Rust rule is a defect, because the two
   drift the first time either changes.
7. No binding call returns a time-indexed sequence of numbers without presence alongside it. See
   the `CONTEXT.md` alias entry; this is what makes the `reach.py:83` defect unwritable.
8. No Rust panic may escape a boundary function. PyO3 converts a panic into
   `pyo3_runtime.PanicException`, a `BaseException` that is not an `Exception`; it therefore slips
   past `except Exception` in caller code and carries a Rust-internal message rather than a domain
   error. Boundary functions catch and translate; they never let a panic through. The workspace
   already denies `unwrap_used` and `expect_used` in library crates, which removes the commonest
   sources; indexing and arithmetic panics remain and must be handled.
9. Python-side criterion commands must not use bare `uv run` after `maturin develop`. Measured on
   2026-08-19: plain `uv run` re-syncs the environment and silently replaces the maturin-built
   extension, so the module imports without the symbols just built. Use `uv run --no-sync` or the
   virtualenv's interpreter directly. A criterion command that gets this wrong fails against stale
   bytes rather than the code under test.
10. Parameter substitution is restricted to declared rule parameters and must recompute the model
   digest. It may never become a path for mutating forcing data, topology, stocks, or calendar,
   because that would break the guarantee that a digest identifies a run.
11. CPython 3.12+, matching taqsim's `requires-python = ">=3.12"`.
12. Workspace clippy lints hold: `unwrap_used`, `expect_used`, `print_stdout` and `print_stderr`
    are denied in library crates.
13. Project doctrine in `AGENTS.md` applies in both repositories: a module means one thing,
    receives exactly what it needs, in types that cannot lie, and dies rather than guess. Each
    module carries a one-line denotation in its docstring.
14. No irreversible act is contained in this vision: nothing published, no name claimed, no history
    destroyed, no downstream repository touched.

## Acceptance criteria (vision-level "done")

```json
{
  "criteria": [
    {
      "name": "Python authors the fixture",
      "input": "The seven-rule hydrology fixture — Muskingum, lag, linear reservoir, evaporation, seepage, reservoir policy and demand — expressed as a Python model document instead of the Rust HydrologyModelDocument, and executed through the binding",
      "observation": "The resulting log is byte-identical to the log the Rust fixture produces for the same run id, and replays successfully against its own artifact"
    },
    {
      "name": "No panic reaches Python",
      "input": "A corpus of deliberately malformed model documents — NaN and infinite literals, an expression tree deeper than the 128-node limit, a cyclic topology, unknown substance and compartment identifiers, duplicate compartment identifiers, and negative initial stocks — each submitted from a caller that wraps the call in a plain except Exception handler, in one interpreter process",
      "observation": "Every document is caught by that except Exception handler and carries the core's own error text; no input produces pyo3_runtime.PanicException, which derives from BaseException rather than Exception and would therefore escape the caller's handler and kill a running sweep"
    },
    {
      "name": "Omission is refused across the boundary",
      "input": "A model document whose rule disposes water but omits salt, in a compartment holding salt, passed to the binding's compile entry point",
      "observation": "The call raises naming the compartment, the omitted substance and the timestep; no artifact is returned and no log is produced"
    },
    {
      "name": "Presence survives the crossing",
      "input": "A run in which one compartment releases nothing for 40 consecutive timesteps inside the declared horizon, read from Python across the full horizon, together with a read of a substance the run's registry never declared",
      "observation": "The 40 timesteps read as modelled zeros distinguishable from positions outside the horizon, and the undeclared substance reads as not-modelled and never yields 0.0"
    },
    {
      "name": "No bare time series exists",
      "input": "The binding's complete public surface, enumerated",
      "observation": "No public call returns a time-indexed sequence of numbers without a per-timestep presence array alongside it"
    },
    {
      "name": "Substitution cannot smuggle",
      "input": "A parameter substitution naming a forcing series, a topology element, an initial stock or a calendar field rather than a declared rule parameter of the named compartment",
      "observation": "The call is refused with an error naming what was addressed and why it is not substitutable, and the held model is left unchanged and still runnable"
    },
    {
      "name": "Digest still binds through Python",
      "input": "Two runs launched from Python off one held model, differing only in one substituted declared rule parameter",
      "observation": "The two model digests differ, and replaying either run's log against the other run's artifact is rejected citing the digest mismatch"
    },
    {
      "name": "Sweep cost is bounded",
      "input": "One basin-scale model document, then 1,000 runs varying only declared rule parameters, measured against a naive baseline that re-sends the whole document per trial on the same machine",
      "observation": "Total decode-and-validate work stays flat as trial count rises rather than growing linearly with it, no forcing series crosses the boundary after the first trial, and the measured baseline is recorded with its machine"
    }
  ]
}
```

## Decomposition hints

1. **The document format is the spine.** Everything else consumes it. Land the public format and
   validated decoding in `incidence-core` first, and rebuild the Rust hydrology fixture on it in
   the same package so the format is proven against the hardest model that exists before any
   Python is written. Deleting the private `HydrologyModelDocument` in that same package is what
   guarantees exactly one document type rather than two that drift.
2. **Panic containment is not a late hardening pass.** It is a property of how every boundary
   function is written, so establish the error-mapping discipline with the first binding entry
   point rather than auditing for it afterwards.
3. **Presence-carrying reads and raw transfer access are one design concern.** The rule is that
   the log is fully readable but nothing time-indexed loses presence; solving them separately
   tends to leave a bare-array convenience method behind.
4. **Measure before building the substitution path.** Criterion "Sweep cost is bounded" is the only
   one whose result can tell the run to *delete* work: if the naive per-trial document is already
   fast enough at basin scale, the correct response is to drop the held-model path and the
   "Substitution cannot smuggle" guard rather than ship them. Sequence the measurement so that
   answer is still actionable.
5. **The Python fixture is the real interface test.** It is the first time anything outside Rust
   authors a complete model, and it is where an inadequate document format shows up. Reaching it
   early converts a design flaw into a revision rather than a rewrite.
6. Suggested slices: (a) public model document format, validated decoding, and the Rust fixture
   rebuilt on it; (b) binding crate scaffold, maturin wheel, error mapping and panic containment;
   (c) document construction and expression combinators from Python; (d) run execution and log
   access; (e) presence-carrying projection reads; (f) the Python hydrology fixture and byte-identical
   log proof; (g) sweep measurement, then the held-model substitution path if the measurement
   justifies it.

## Open questions / risks

1. **The sweep-cost measurement may invalidate scope item 7.** If naive per-trial document
   submission is already fast enough at basin scale, the held-model substitution path and its
   smuggling guard are unnecessary complexity and the run must delete them rather than ship them.
   The run must not treat the substitution path as a foregone conclusion.
2. **The sweep-cost criterion is machine-sensitive.** It compares against a baseline rather than an
   absolute number, and the baseline must be measured on the same machine in the same run. A
   recorded absolute threshold would be meaningless on other hardware.
3. **How a rule addresses the stock it is disposing was not settled during discovery.** The Rust
   fixture wires named inputs through explicit input bindings; a sketch during the grill assumed an
   implicit accessor. The run must read the interpreter's input model and follow it rather than
   invent an implicit form.
4. **Two entry points may collapse into one.** The sketch carried both a compile-to-artifact path
   and a hold-model path. If the sweep measurement removes the held-model path, one of them
   disappears; the run should not freeze both into public surface before that measurement lands.
5. **The document format is a permanent commitment made before its main consumer exists.** taqsim
   (#8) is the first real writer of these documents and has not been built. The Rust hydrology
   fixture and the Python fixture are the mitigation and are both weaker than a real consumer, so
   format inadequacies discovered during #8 are a live possibility. The explicit format version is
   what makes that survivable.
6. **Promoting the fixture document type changes a test-support file that currently passes.** The
   Rust fixture must be rebuilt on the public format and must continue to produce the same log; a
   silent behavioural change there would invalidate the byte-identical comparison the Python
   fixture depends on.
