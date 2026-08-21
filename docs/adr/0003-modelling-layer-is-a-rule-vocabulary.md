# ADR-0003: The modelling layer is a rule vocabulary, not a rule catalogue

## Status

Accepted

- Date: 2026-08-20
- Effort ticket: #8
- Program: #4

## Context

incidence interprets rules as data — a serialisable expression tree — and never calls back
into Python during a step. ADR-0002 placed the water-naming, hydrologist-facing layer above
the binding, in taqsim, but did not say what that layer offers when someone needs a rule
taqsim does not already ship.

The question is not hypothetical. The Zarafshan basin defines four custom strategies of its
own in `zarafshan_taqsim/strategies/`, none written by taqsim. The densest of them,
`ZRBReleaseRule` (`strategies/release.py:75`), is a Python method with three storage zones,
twelve cyclical monthly parameters, an m³/s→m³ conversion and five optimizable values under
an `Ordered(v1, v2)` constraint. Under rules-as-data that method cannot exist in any form.

Two directions were available.

A **fixed catalogue** — taqsim ships a closed set of hydrology rules, and a basin needing
anything else waits for a taqsim release. Small surface, easy to document, and a hard wall
the first time a basin evolves. The basin's own four custom strategies are evidence that the
catalogue is not closed and never was.

A **rule vocabulary** — taqsim offers hydrology-flavoured composable pieces (zone
conditions, seasonal values, per-second rates) that compile to the engine's expression tree.
Anything expressible in the vocabulary works without editing taqsim; anything outside it
still fails, but the failure is about a vocabulary rather than about Python.

Surveying the six rule shapes the basin actually uses settled a second question. Five compile
from operations the engine already has — `Subtract` and `Divide` exist in
`crates/core/src/rule_expression.rs:102-104` and are merely unexposed in the binding. One
does not: canal seepage is `α · √Q · length · time` (`strategies/loss.py:259`), and the
expression language has no power operation. The choices were to approximate √Q with an
interpolation table in taqsim, or to extend the engine.

## Decision

**taqsim exposes a rule vocabulary. A modeller composes a new operating policy from taqsim's
own pieces without editing taqsim, and taqsim compiles it to the engine's rule IR. No rule is
ever a Python callable.**

**incidence gains a power operation**, so seepage is the published hydraulic relation rather
than a piecewise-linear approximation of it. That extension is in scope for ticket #8.

The extension is bounded by a standing bar, which is the durable half of this decision:

> An operation enters incidence's expression language only if it is a closed-form
> mathematical primitive that names no substance. A hydrology concept never enters the
> engine; it is composed in taqsim's rule vocabulary.

Without that bar, "add one more operation" repeats until the expression language is a
programming language, at which point rules-as-data has bought nothing.

Parameter bounds and constraints are declared beside the parameter in taqsim's vocabulary and
held taqsim-side. They are not shipped into the model document, so retuning a bound never
changes a model's content-addressed identity.

## Consequences

The vocabulary is the largest single piece of design in the taqsim rebuild, and it will have
edges: there will be policies someone can describe in a sentence and not write. That is the
accepted cost of rules staying inspectable, diffable and parameter-extractable data.

The vocabulary is also the test ADR-0002 asked for. It named the taqsim rebuild as the
resolution condition for whether the rule language is adequate for real hydrology; the six
Zarafshan rule shapes ship as taqsim's test fixtures, and their compiling is the evidence.
Adequacy for those six is not adequacy for hydrology generally.

Adding an operation is a version bump on a wire commitment: the rule IR is public and
content-addressed, so model identity moves with it. Nothing downstream depends on it yet —
fishy is not rebuilt — which makes now the cheapest possible moment.

Because a rule can no longer be a callable, every `Strategy` subclass in every downstream
project must be rewritten in the vocabulary. That is real porting work, and it is deliberately
not this ticket's: the ports are a later ticket, decided with their owners.
