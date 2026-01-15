from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from taqsim.constraints import Constraint
    from taqsim.node.base import BaseNode

ParamValue = float
ParamBounds = tuple[float, float]


class LossReason(str):
    """A typed string representing a loss reason."""

    __slots__ = ()


EVAPORATION = LossReason("evaporation")
SEEPAGE = LossReason("seepage")
OVERFLOW = LossReason("overflow")
INEFFICIENCY = LossReason("inefficiency")
CAPACITY_EXCEEDED = LossReason("capacity_exceeded")


def summarize_losses(events: list) -> dict[str, float]:
    """Group losses by reason."""
    totals: dict[str, float] = {}
    for e in events:
        totals[e.reason] = totals.get(e.reason, 0) + e.amount
    return totals


class Strategy:
    """Mixin for operational strategies with tunable parameters.

    Concrete strategies should:
    1. Inherit from Strategy
    2. Be frozen dataclasses
    3. Declare __params__ listing optimizable field names
    """

    __params__: ClassVar[tuple[str, ...]] = ()
    __bounds__: ClassVar[dict[str, ParamBounds]] = {}
    __constraints__: ClassVar[tuple["Constraint", ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        valid = set(cls.__params__)
        for c in cls.__constraints__:
            invalid = set(c.params) - valid
            if invalid:
                raise TypeError(f"{cls.__name__}: constraint references unknown params: {invalid}")

    def params(self) -> dict[str, ParamValue]:
        """Return current parameter values."""
        return {name: getattr(self, name) for name in self.__params__}

    def bounds(self, node: "BaseNode") -> dict[str, ParamBounds]:
        """Return parameter bounds, optionally derived from node properties."""
        return dict(self.__bounds__)

    def constraints(self, node: "BaseNode") -> tuple["Constraint", ...]:
        """Return constraints for this strategy."""
        return self.__constraints__

    def with_params(self, **kwargs: ParamValue) -> Self:
        """Create new instance with updated parameters (immutable)."""
        invalid = set(kwargs) - set(self.__params__)
        if invalid:
            raise ValueError(f"Unknown parameters: {invalid}")
        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Describes a single tunable parameter in the system."""

    path: str  # e.g., "dam.release_rule.rate"
    value: float  # scalar value
