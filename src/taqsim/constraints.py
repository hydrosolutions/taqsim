from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Constraint(Protocol):
    @property
    def params(self) -> tuple[str, ...]: ...

    def repair(self, values: dict[str, float]) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class SumToOne:
    params: tuple[str, ...]

    def repair(self, values: dict[str, float]) -> dict[str, float]:
        result = dict(values)
        param_values = [max(0.0, values[p]) for p in self.params]
        total = sum(param_values)

        if total == 0.0:
            n = len(self.params)
            normalized = [1.0 / n] * n
        else:
            normalized = [v / total for v in param_values]

        for p, v in zip(self.params, normalized, strict=True):
            result[p] = v

        return result


@dataclass(frozen=True, slots=True)
class Ordered:
    low: str
    high: str

    @property
    def params(self) -> tuple[str, ...]:
        return (self.low, self.high)

    def repair(self, values: dict[str, float]) -> dict[str, float]:
        result = dict(values)
        low_val = values[self.low]
        high_val = values[self.high]

        if low_val > high_val:
            result[self.low] = high_val
            result[self.high] = low_val

        return result
