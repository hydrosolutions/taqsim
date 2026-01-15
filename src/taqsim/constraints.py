from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Constraint(Protocol):
    """Protocol for constraint types that enforce relationships between parameters."""

    @property
    def params(self) -> tuple[str, ...]: ...

    def repair(
        self,
        values: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]: ...

    def satisfied(self, values: dict[str, float], tol: float = 1e-9) -> bool: ...

    def is_feasible(self, bounds: dict[str, tuple[float, float]]) -> bool: ...


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """Fully resolved constraint specification for repair functions."""

    constraint: Constraint
    prefix: str  # e.g., "dam.release_rule"
    param_paths: dict[str, str]  # {"r1": "dam.release_rule.r1", ...}
    param_bounds: dict[str, tuple[float, float]]  # {"r1": (0.0, 1.0), ...}


class ConstraintInfeasibleError(ValueError):
    """Raised when constraint cannot be satisfied within bounds."""

    pass


class BoundViolationError(ValueError):
    """Raised when a parameter value violates its declared bounds."""

    def __init__(self, param: str, value: float, bounds: tuple[float, float]):
        self.param = param
        self.value = value
        self.bounds = bounds
        lo, hi = bounds
        super().__init__(f"Parameter '{param}' value {value} outside bounds [{lo}, {hi}]")


class ConstraintViolationError(ValueError):
    """Raised when parameter values violate a constraint."""

    def __init__(self, constraint: "Constraint", values: dict[str, float]):
        self.constraint = constraint
        self.values = values
        param_str = ", ".join(f"{k}={v}" for k, v in values.items())
        super().__init__(f"{type(constraint).__name__} violated: {param_str}")


@dataclass(frozen=True, slots=True)
class SumToOne:
    params: tuple[str, ...]
    target: float = 1.0

    def repair(
        self,
        values: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        result = dict(values)

        # Backwards compat - no bounds
        if bounds is None:
            param_values = [max(0.0, values[p]) for p in self.params]
            total = sum(param_values)

            if total == 0.0:
                n = len(self.params)
                normalized = [self.target / n] * n
            else:
                normalized = [v / total * self.target for v in param_values]

            for p, v in zip(self.params, normalized, strict=True):
                result[p] = v

            return result

        # Bounds-aware repair
        # 1. Clamp to bounds
        for p in self.params:
            lo, hi = bounds[p]
            result[p] = max(lo, min(hi, max(0.0, values[p])))

        current = sum(result[p] for p in self.params)

        if abs(current - self.target) < 1e-9:
            return result

        delta = self.target - current

        if delta > 0:  # Need to increase
            room = {p: bounds[p][1] - result[p] for p in self.params}
        else:  # Need to decrease
            room = {p: result[p] - bounds[p][0] for p in self.params}

        total_room = sum(max(0.0, r) for r in room.values())

        if total_room < 1e-9:
            # Can't satisfy - return best effort (already clamped)
            return result

        for p in self.params:
            if room[p] > 0:
                share = room[p] / total_room * delta
                result[p] = result[p] + share

        return result

    def satisfied(self, values: dict[str, float], tol: float = 1e-9) -> bool:
        return abs(sum(values[p] for p in self.params) - self.target) < tol

    def is_feasible(self, bounds: dict[str, tuple[float, float]]) -> bool:
        lower_sum = sum(bounds[p][0] for p in self.params)
        upper_sum = sum(bounds[p][1] for p in self.params)
        return lower_sum <= self.target <= upper_sum


@dataclass(frozen=True, slots=True)
class Ordered:
    low: str
    high: str

    @property
    def params(self) -> tuple[str, ...]:
        return (self.low, self.high)

    def repair(
        self,
        values: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        result = dict(values)
        low_val = values[self.low]
        high_val = values[self.high]

        # Already satisfied
        if low_val <= high_val:
            return result

        # Backwards compat - simple swap
        if bounds is None:
            result[self.low] = high_val
            result[self.high] = low_val
            return result

        # Bounds-aware repair
        low_min, low_max = bounds[self.low]
        high_min, high_max = bounds[self.high]

        # Strategy 1: Try simple swap if both stay in bounds
        if low_min <= high_val <= low_max and high_min <= low_val <= high_max:
            result[self.low] = high_val
            result[self.high] = low_val
            return result

        # Strategy 2: Find overlap region
        overlap_min = max(low_min, high_min)
        overlap_max = min(low_max, high_max)

        if overlap_min <= overlap_max:
            # Overlap exists - push both to midpoint
            mid = (overlap_min + overlap_max) / 2
            result[self.low] = max(low_min, min(low_max, mid))
            result[self.high] = max(high_min, min(high_max, mid))
            return result

        # Strategy 3: No overlap - check if there's a valid gap
        if low_max < high_min:
            # low bounds entirely below high bounds - push to boundaries
            result[self.low] = low_max
            result[self.high] = high_min
            return result

        # Strategy 4: Impossible case (low_min > high_max)
        # Return best effort - minimize violation
        result[self.low] = low_min
        result[self.high] = high_max
        return result

    def satisfied(self, values: dict[str, float], tol: float = 1e-9) -> bool:
        return values[self.low] <= values[self.high] + tol

    def is_feasible(self, bounds: dict[str, tuple[float, float]]) -> bool:
        return bounds[self.low][0] <= bounds[self.high][1]
