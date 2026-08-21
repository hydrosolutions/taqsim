# ADR-0004: The disposition IR carries computed branch amounts

## Status

Accepted

- Date: 2026-08-21
- Effort ticket: #8
- Program: #4

## Context

ADR-0003 admitted the power operation into incidence under a standing bar:

> An operation enters incidence's expression language only if it is a closed-form
> mathematical primitive that names no substance. A hydrology concept never enters the
> engine; it is composed in taqsim's rule vocabulary.

That bar governs the *expression language*. Building the rule vocabulary hit something it
does not reach.

Two of the six real Zarafshan rule shapes — priority distribution and e-flow split — assign a
*different computed amount* to each of *several named branches* within one exhaustive
partition. Priority distribution gives one destination up to a stated amount and the
remainder to others; an e-flow split stages a natural-flow share and then a remainder share.
Neither is a constant fraction, and neither is single-branch.

incidence's disposition IR at that point offered five shapes: retain all, release all to one
branch, split by constant fractions across several branches, send one branch an exogenous
series, or transfer a constant fraction to one branch. Multiple named branches existed only
where every amount was a constant `Fraction`; the only non-constant amount, an exogenous
series, went to exactly one branch. The intersection the vocabulary needed — distinct
expression-derived amounts across several named branches — did not exist.

This is not an operation in the expression tree. It is a shape in the disposition IR, so
ADR-0003's bar is silent on it, and the vision's scope-out of "growing the engine's
expression language on demand" does not literally cover it either. Both were written about a
different subsystem.

## Alternatives

**Compose it in taqsim.** The vocabulary would emit several ordinary partitions and reconcile
them itself. Refused on a hard constraint rather than a preference: exhaustive disposition and
conservation are properties the core owns, and ADR-0002 established that validation is
single-sourced — taqsim may not reimplement a legality check the core owns, and exactly one
implementation decides whether a model is legal. Staging partitions taqsim-side means taqsim
deciding whether the aggregate is exhaustive and conserving, which is precisely that check.

**Extend the disposition IR.** Chosen. Partition structure is not an expression operation, and
structure is exactly what the core owns exclusively. Putting the shape where the conservation
check already lives keeps one implementation deciding legality.

## Decision

`PartitionNode` gains `ExpressionPartition { branches: Vec<ExpressionBranch> }`, where each
named branch takes an amount from its own rule expression.

The standing bar is extended, not weakened. Alongside ADR-0003's rule for the expression
language:

> A structural shape enters incidence's disposition IR only when the alternative would make a
> consumer reimplement a legality check the core owns.

That condition is narrow by construction. It is not satisfied by convenience, by a shape being
awkward to express, or by a consumer preferring not to write the composition — only by the
core's own ownership of the check being at stake. Without it, "one more partition shape"
repeats until the disposition IR is a catalogue of hydrology, which is the failure ADR-0003
exists to prevent.

Conservation holds identically to the fixed-fraction case, and was proven in both directions:
a partition whose branch expressions together exceed the available stock is refused, and a
partition whose amounts are conservative but decimal is accepted. The second half was found by
a gate, not by the authored criteria — the first implementation refused legitimate decimal
partitions, so the property was over-strict before it was correct.

## Consequences

The rule IR is a public, content-addressed wire commitment, so adding this shape moves model
identity, exactly as the power operation did. Nothing downstream depends on it yet — fishy is
not rebuilt and the Zarafshan port has not started — which makes now the cheapest moment, and
the same reasoning that admitted the power operation admits this.

Two extensions to incidence landed under one ticket, in two different subsystems, each with
its own bar. A third proposal should be read against both, and against the fact that this
ticket already spent the cheap moment twice.
