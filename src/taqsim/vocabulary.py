"""compile_rule : WaterRule × RuleContext → RulePlan   (pure)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import incidence

type Expression = dict[str, object]


def _binary(kind: str, lhs: Expression, rhs: Expression) -> Expression:
    return {
        "rule_ir_version": "v1",
        "numerical_semantics_version": "v1",
        "expression": {"kind": kind, "lhs": lhs["expression"], "rhs": rhs["expression"]},
    }


def subtract(lhs: Expression, rhs: Expression) -> Expression:
    """Compile subtraction, which is part of the IR but has no binding helper."""
    return _binary("subtract", lhs, rhs)


def divide(lhs: Expression, rhs: Expression) -> Expression:
    """Compile division, which is part of the IR but has no binding helper."""
    return _binary("divide", lhs, rhs)


@dataclass(frozen=True)
class Parameter:
    """A substitutable scalar rule value; bounds stay outside the model document."""

    name: str
    value: float
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must not be empty")
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError(f"parameter {self.name!r} has reversed bounds")


type Scalar = float | int | Parameter
type SeasonalScalar = Scalar | tuple[Scalar, ...]


@dataclass
class RuleContext:
    """The narrow compilation state shared by one rule and its calendar."""

    owner: str
    start: datetime
    steps: int
    timestep: timedelta
    parameters: dict[str, float] = field(default_factory=dict)
    forcings: dict[str, list[float]] = field(default_factory=dict)
    tables: list[dict[str, object]] = field(default_factory=list)

    @property
    def available(self) -> Expression:
        return incidence.projection(f"{self.owner}-available", "extensive")

    def scalar(self, value: Scalar) -> Expression:
        if isinstance(value, Parameter):
            previous = self.parameters.setdefault(value.name, float(value.value))
            if previous != float(value.value):
                raise ValueError(f"parameter {value.name!r} has conflicting values")
            return incidence.param(value.name)
        return incidence.literal(float(value))

    def seasonal(self, value: SeasonalScalar, label: str) -> Expression:
        if not isinstance(value, tuple):
            return self.scalar(value)
        if not value:
            raise ValueError(f"seasonal value {label!r} must not be empty")
        if len(value) not in {1, 12}:
            raise ValueError(f"seasonal value {label!r} must contain one or twelve monthly values")
        if len(value) == 1:
            return self.scalar(value[0])
        forcing_id = f"{self.owner}-{label}-month"
        months: list[float] = []
        for step in range(self.steps):
            moment = self.start + self.timestep * step
            months.append(float(moment.month))
        self.forcings[forcing_id] = months
        month = incidence.forcing(forcing_id)
        result = self.scalar(value[-1])
        for index in range(11, 0, -1):
            result = incidence.select(
                incidence.compare("equal", month, incidence.literal(float(index))),
                self.scalar(value[index - 1]),
                result,
            )
        return result


@dataclass(frozen=True)
class RulePlan:
    """Absolute branch amounts and their named water destinations."""

    branches: tuple[tuple[str, str, Expression], ...]


class WaterRule(Protocol):
    """A serialisable hydrology rule shape, never a timestep callback."""

    def compile(self, context: RuleContext, downstream: str) -> RulePlan: ...


def monthly_parameters(
    name: str, values: tuple[float, ...], bounds: tuple[float, float] | None = None
) -> tuple[Parameter, ...]:
    """Name twelve substitutable monthly values deterministically."""
    if len(values) != 12:
        raise ValueError("monthly parameters require exactly twelve values")
    return tuple(Parameter(f"{name}-{month:02d}", value, bounds) for month, value in enumerate(values, 1))


@dataclass(frozen=True)
class ZoneRelease:
    """A dead-floor, buffer, conservation, and flood reservoir policy."""

    dead_storage: SeasonalScalar
    buffer_limit: SeasonalScalar
    conservation_limit: SeasonalScalar
    release_rate_m3s: SeasonalScalar
    buffer_fraction: SeasonalScalar = 0.2
    flood_fraction: SeasonalScalar = 1.0

    def compile(self, context: RuleContext, downstream: str) -> RulePlan:
        available = context.available
        dead = context.seasonal(self.dead_storage, "dead-storage")
        buffer = context.seasonal(self.buffer_limit, "buffer-limit")
        conservation = context.seasonal(self.conservation_limit, "conservation-limit")
        release_rate = context.seasonal(self.release_rate_m3s, "release-rate")
        seconds = incidence.literal(context.timestep.total_seconds())
        target = incidence.mul(release_rate, seconds)
        buffered = incidence.mul(context.seasonal(self.buffer_fraction, "buffer-fraction"), target)
        flood = incidence.add(
            target,
            incidence.mul(
                context.seasonal(self.flood_fraction, "flood-fraction"),
                incidence.max(incidence.literal(0.0), subtract(available, conservation)),
            ),
        )
        zoned = incidence.select(
            incidence.compare("less_than_or_equal", available, dead),
            incidence.literal(0.0),
            incidence.select(
                incidence.compare("less_than_or_equal", available, buffer),
                buffered,
                incidence.select(incidence.compare("less_than_or_equal", available, conservation), target, flood),
            ),
        )
        release = incidence.min(
            incidence.max(incidence.literal(0.0), subtract(available, dead)),
            incidence.max(incidence.literal(0.0), zoned),
        )
        return RulePlan((("release", downstream, release),))


ZoneReleasePolicy = ZoneRelease


@dataclass(frozen=True)
class MonthlyDistribution:
    """An exhaustive monthly allocation among named destinations."""

    ratios: dict[str, tuple[Scalar, ...]]

    def compile(self, context: RuleContext, downstream: str) -> RulePlan:
        del downstream
        if not self.ratios:
            raise ValueError("monthly distribution requires at least one destination")
        branches = tuple(
            (
                f"to-{destination}",
                destination,
                incidence.mul(context.available, context.seasonal(values, f"{destination}-ratio")),
            )
            for destination, values in self.ratios.items()
        )
        return RulePlan(branches)


@dataclass(frozen=True)
class PriorityDistribution:
    """Serve one named destination first, then distribute the remainder."""

    priority_destination: str
    priority_amount: SeasonalScalar
    remainder_ratios: dict[str, Scalar]

    def compile(self, context: RuleContext, downstream: str) -> RulePlan:
        del downstream
        priority = incidence.min(context.available, context.seasonal(self.priority_amount, "priority-amount"))
        remainder = incidence.max(incidence.literal(0.0), subtract(context.available, priority))
        branches = [(f"to-{self.priority_destination}", self.priority_destination, priority)]
        branches.extend(
            (f"to-{destination}", destination, incidence.mul(remainder, context.scalar(ratio)))
            for destination, ratio in self.remainder_ratios.items()
        )
        return RulePlan(tuple(branches))


@dataclass(frozen=True)
class EFlowSplit:
    """Reserve capped environmental flow, then allocate the remainder."""

    natural_ratios: dict[str, Scalar]
    remainder_ratios: dict[str, Scalar]
    eflow_fraction: Scalar = 0.2
    eflow_cap: Scalar = 1e300

    def compile(self, context: RuleContext, downstream: str) -> RulePlan:
        del downstream
        environmental = incidence.min(
            incidence.mul(context.available, context.scalar(self.eflow_fraction)), context.scalar(self.eflow_cap)
        )
        remainder = incidence.max(incidence.literal(0.0), subtract(context.available, environmental))
        branches = [
            (f"to-{destination}", destination, incidence.mul(environmental, context.scalar(ratio)))
            for destination, ratio in self.natural_ratios.items()
        ]
        branches.extend(
            (f"to-{destination}", destination, incidence.mul(remainder, context.scalar(ratio)))
            for destination, ratio in self.remainder_ratios.items()
        )
        return RulePlan(tuple(branches))


EFlowSplitPolicy = EFlowSplit


@dataclass(frozen=True)
class ReservoirEvaporation:
    """Monthly evaporation depth over area interpolated from stored water."""

    rates_mm: tuple[Scalar, ...]
    volume_area: tuple[tuple[float, float], ...]
    destination: str = "evaporation"

    def compile(self, context: RuleContext, downstream: str) -> RulePlan:
        if len(self.volume_area) < 2:
            raise ValueError("reservoir evaporation requires at least two volume-area points")
        table_id = f"{context.owner}-surface-area"
        context.tables.append(
            {
                "id": table_id,
                "numerical_semantics_version": "v1",
                "boundary_policy": "clamp_to_endpoint",
                "abscissae": [point[0] for point in self.volume_area],
                "ordinates": [point[1] for point in self.volume_area],
            }
        )
        area = incidence.table_lookup(table_id, context.available)
        loss = incidence.min(
            context.available,
            incidence.mul(
                incidence.mul(context.seasonal(self.rates_mm, "evaporation-rate"), incidence.literal(0.001)), area
            ),
        )
        delivered = incidence.max(incidence.literal(0.0), subtract(context.available, loss))
        return RulePlan((("evaporation", self.destination, loss), ("release", downstream, delivered)))


EvaporationLossRule = ReservoirEvaporation


@dataclass(frozen=True)
class CanalLosses:
    """A sequential canal loss cascade including the real square-root seepage law."""

    seepage_coefficient: Scalar
    length_km: float
    seepage_destination: str = "seepage"
    evaporation_mm: SeasonalScalar = 0.0
    width_m: float = 0.0
    evaporation_destination: str = "evaporation"
    operational_fraction: Scalar = 0.0
    operational_destination: str = "operational-loss"

    def compile(self, context: RuleContext, downstream: str) -> RulePlan:
        seconds = incidence.literal(context.timestep.total_seconds())
        q_m3s = divide(context.available, seconds)
        seepage = incidence.min(
            context.available,
            incidence.mul(
                incidence.mul(
                    incidence.mul(
                        context.scalar(self.seepage_coefficient), incidence.power(q_m3s, incidence.literal(0.5))
                    ),
                    incidence.literal(self.length_km),
                ),
                seconds,
            ),
        )
        after_seepage = incidence.max(incidence.literal(0.0), subtract(context.available, seepage))
        evaporation = incidence.min(
            after_seepage,
            incidence.mul(
                context.seasonal(self.evaporation_mm, "canal-evaporation"),
                incidence.literal(0.001 * self.length_km * 1000.0 * self.width_m),
            ),
        )
        after_evaporation = incidence.max(incidence.literal(0.0), subtract(after_seepage, evaporation))
        operational = incidence.mul(after_evaporation, context.scalar(self.operational_fraction))
        delivered = incidence.max(incidence.literal(0.0), subtract(after_evaporation, operational))
        return RulePlan(
            (
                ("seepage", self.seepage_destination, seepage),
                ("evaporation", self.evaporation_destination, evaporation),
                ("operational-loss", self.operational_destination, operational),
                ("release", downstream, delivered),
            )
        )


CanalLossRule = CanalLosses
