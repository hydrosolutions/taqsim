from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from taqsim.constraints import Constraint
    from taqsim.node.base import BaseNode

from taqsim.constraints import BoundViolationError, ConstraintViolationError
from taqsim.time import Frequency, Timestep

ParamValue = float | tuple[float, ...]
ParamBounds = tuple[float, float]


class LossReason(str):
    """A typed string representing a loss reason."""

    __slots__ = ()


EVAPORATION = LossReason("evaporation")
SEEPAGE = LossReason("seepage")
OVERFLOW = LossReason("overflow")
INEFFICIENCY = LossReason("inefficiency")


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
    __cyclical__: ClassVar[tuple[str, ...]] = ()
    __cyclical_freq__: ClassVar[dict[str, "Frequency"]] = {}
    __tags__: ClassVar[frozenset[str]] = frozenset()
    __metadata__: ClassVar[Mapping[str, object]] = {}

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
        invalid_cyc = set(cls.__cyclical__) - set(cls.__time_varying__)
        if invalid_cyc:
            raise TypeError(f"{cls.__name__}: __cyclical__ params must be in __time_varying__: {invalid_cyc}")
        if cls.__cyclical__ and not cls.__cyclical_freq__:
            raise TypeError(f"{cls.__name__}: __cyclical_freq__ is required when __cyclical__ is non-empty")
        if cls.__cyclical_freq__:
            if set(cls.__cyclical_freq__.keys()) != set(cls.__cyclical__):
                raise TypeError(
                    f"{cls.__name__}: __cyclical_freq__ keys must match __cyclical__: "
                    f"got {set(cls.__cyclical_freq__.keys())}, expected {set(cls.__cyclical__)}"
                )
            for name, freq in cls.__cyclical_freq__.items():
                if not isinstance(freq, Frequency):
                    raise TypeError(
                        f"{cls.__name__}: __cyclical_freq__['{name}'] must be a Frequency, got {type(freq).__name__}"
                    )

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
            # Non-cyclical params determine full simulation length; cyclical params cycle
            timesteps = 1
            for tv_param in self.__time_varying__:
                value = getattr(self, tv_param)
                if isinstance(value, tuple) and tv_param not in self.__cyclical__:
                    timesteps = max(timesteps, len(value))
            # If all time-varying params are cyclical, use max cycle length
            if timesteps == 1:
                for tv_param in self.__cyclical__:
                    value = getattr(self, tv_param)
                    if isinstance(value, tuple):
                        timesteps = max(timesteps, len(value))

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
                if name in self.__cyclical__:
                    result[name] = value[t % len(value)]
                else:
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

    def cyclical(self) -> tuple[str, ...]:
        """Return names of cyclical time-varying parameters."""
        return self.__cyclical__

    def param_at(self, name: str, t: Timestep) -> float:
        value = getattr(self, name)
        if not isinstance(value, tuple):
            return value
        if name in self.__cyclical__:
            param_freq = self.__cyclical_freq__[name]
            idx = (t.index * param_freq // t.frequency) % len(value)
            return value[idx]
        return value[t.index]

    def cyclical_freq(self) -> dict[str, Frequency]:
        return dict(self.__cyclical_freq__)

    def tags(self) -> frozenset[str]:
        """Return tags for this strategy type."""
        return self.__tags__

    def metadata(self) -> Mapping[str, object]:
        """Return metadata for this strategy type."""
        return self.__metadata__

    def with_params(self, **kwargs: ParamValue) -> Self:
        """Create new instance with updated parameters (immutable)."""
        invalid = set(kwargs) - set(self.__params__)
        if invalid:
            raise ValueError(f"Unknown parameters: {invalid}")
        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Describes a single tunable parameter in the system."""

    path: str  # e.g., "dam.release_policy.rate"
    value: float  # scalar value
