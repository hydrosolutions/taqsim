from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from taqsim.constraints import Constraint
    from taqsim.node.base import BaseNode

from taqsim.constraints import BoundViolationError, ConstraintViolationError

ParamValue = float | tuple[float, ...]
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
    __time_varying__: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        valid = set(cls.__params__)
        for c in cls.__constraints__:
            invalid = set(c.params) - valid
            if invalid:
                raise TypeError(f"{cls.__name__}: constraint references unknown params: {invalid}")
        invalid_tv = set(cls.__time_varying__) - valid
        if invalid_tv:
            raise TypeError(f"{cls.__name__}: __time_varying__ references unknown params: {invalid_tv}")

    def __post_init__(self) -> None:
        """Validate parameter values against bounds and constraints."""
        for param in self.__params__:
            if param not in self.__bounds__:
                continue
            value = getattr(self, param)
            lo, hi = self.__bounds__[param]
            if param in self.__time_varying__:
                if not isinstance(value, tuple):
                    raise TypeError(f"Time-varying param '{param}' must be tuple, got {type(value).__name__}")
                for i, v in enumerate(value):
                    if not (lo <= v <= hi):
                        raise BoundViolationError(f"{param}[{i}]", v, (lo, hi))
            else:
                if not (lo <= value <= hi):
                    raise BoundViolationError(param, value, (lo, hi))

        if self.__constraints__:
            # Determine timesteps from time-varying params
            timesteps = 1
            for tv_param in self.__time_varying__:
                value = getattr(self, tv_param)
                if isinstance(value, tuple):
                    timesteps = len(value)
                    break

            for t in range(timesteps):
                values_t = self._values_at_timestep(t)
                for constraint in self.__constraints__:
                    relevant = {p: values_t[p] for p in constraint.params}
                    if not constraint.satisfied(relevant):
                        if timesteps > 1:
                            raise ConstraintViolationError(
                                constraint,
                                {f"{k}[{t}]": v for k, v in relevant.items()},
                            )
                        raise ConstraintViolationError(constraint, relevant)

    def params(self) -> dict[str, ParamValue]:
        """Return current parameter values."""
        return {name: getattr(self, name) for name in self.__params__}

    def _values_at_timestep(self, t: int) -> dict[str, float]:
        """Extract scalar values for all params at a given timestep."""
        result: dict[str, float] = {}
        for name, value in self.params().items():
            if isinstance(value, tuple):
                result[name] = value[t]
            else:
                result[name] = value
        return result

    def bounds(self, node: "BaseNode") -> dict[str, ParamBounds]:
        """Return parameter bounds, optionally derived from node properties."""
        return dict(self.__bounds__)

    def constraints(self, node: "BaseNode") -> tuple["Constraint", ...]:
        """Return constraints for this strategy."""
        return self.__constraints__

    def time_varying(self) -> tuple[str, ...]:
        """Return names of time-varying parameters."""
        return self.__time_varying__

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
