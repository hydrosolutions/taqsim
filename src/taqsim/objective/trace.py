from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class HasTimestep(Protocol):
    t: int


@dataclass(frozen=True, slots=True)
class Trace:
    _data: dict[int, float]

    @classmethod
    def from_events[E: HasTimestep](
        cls,
        events: list[E],
        field: str = "amount",
        reduce: Callable[[float, float], float] | None = None,
    ) -> Trace:
        reduce_fn = reduce if reduce is not None else lambda a, b: a + b
        data: dict[int, float] = {}
        for event in events:
            t = event.t
            value = getattr(event, field)
            if t in data:
                data[t] = reduce_fn(data[t], value)
            else:
                data[t] = value
        return cls(_data=data)

    @classmethod
    def from_dict(cls, data: dict[int, float]) -> Trace:
        return cls(_data=dict(data))

    @classmethod
    def constant(cls, value: float, timesteps: range | list[int]) -> Trace:
        return cls(_data=dict.fromkeys(timesteps, value))

    @classmethod
    def empty(cls) -> Trace:
        return cls(_data={})

    def __getitem__(self, t: int) -> float:
        return self._data[t]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(sorted(self._data.keys()))

    def timesteps(self) -> list[int]:
        return sorted(self._data.keys())

    def values(self) -> list[float]:
        return [self._data[t] for t in sorted(self._data.keys())]

    def items(self) -> list[tuple[int, float]]:
        return [(t, self._data[t]) for t in sorted(self._data.keys())]

    def to_dict(self) -> dict[int, float]:
        return dict(self._data)

    def map(self, fn: Callable[[float], float]) -> Trace:
        return Trace(_data={t: fn(v) for t, v in self._data.items()})

    def filter(self, predicate: Callable[[int, float], bool]) -> Trace:
        return Trace(_data={t: v for t, v in self._data.items() if predicate(t, v)})

    def __add__(self, other: Trace | float) -> Trace:
        if isinstance(other, Trace):
            common = set(self._data.keys()) & set(other._data.keys())
            return Trace(_data={t: self._data[t] + other._data[t] for t in common})
        return Trace(_data={t: v + other for t, v in self._data.items()})

    def __radd__(self, other: float) -> Trace:
        return Trace(_data={t: other + v for t, v in self._data.items()})

    def __sub__(self, other: Trace | float) -> Trace:
        if isinstance(other, Trace):
            common = set(self._data.keys()) & set(other._data.keys())
            return Trace(_data={t: self._data[t] - other._data[t] for t in common})
        return Trace(_data={t: v - other for t, v in self._data.items()})

    def __rsub__(self, other: float) -> Trace:
        return Trace(_data={t: other - v for t, v in self._data.items()})

    def __mul__(self, other: Trace | float) -> Trace:
        if isinstance(other, Trace):
            common = set(self._data.keys()) & set(other._data.keys())
            return Trace(_data={t: self._data[t] * other._data[t] for t in common})
        return Trace(_data={t: v * other for t, v in self._data.items()})

    def __rmul__(self, other: float) -> Trace:
        return Trace(_data={t: other * v for t, v in self._data.items()})

    def __truediv__(self, other: Trace | float) -> Trace:
        if isinstance(other, Trace):
            common = set(self._data.keys()) & set(other._data.keys())
            return Trace(_data={t: self._data[t] / other._data[t] for t in common})
        return Trace(_data={t: v / other for t, v in self._data.items()})

    def __rtruediv__(self, other: float) -> Trace:
        return Trace(_data={t: other / v for t, v in self._data.items()})

    def __neg__(self) -> Trace:
        return Trace(_data={t: -v for t, v in self._data.items()})

    def __pow__(self, exponent: float) -> Trace:
        return Trace(_data={t: v**exponent for t, v in self._data.items()})

    def sum(self) -> float:
        return sum(self._data.values()) if self._data else 0.0

    def mean(self) -> float:
        if not self._data:
            raise ValueError("cannot compute mean of empty Trace")
        return sum(self._data.values()) / len(self._data)

    def max(self) -> float:
        if not self._data:
            raise ValueError("cannot compute max of empty Trace")
        return max(self._data.values())

    def min(self) -> float:
        if not self._data:
            raise ValueError("cannot compute min of empty Trace")
        return min(self._data.values())

    def cumsum(self, initial: float = 0.0) -> Trace:
        acc = initial
        result: dict[int, float] = {}
        for t in sorted(self._data.keys()):
            acc += self._data[t]
            result[t] = acc
        return Trace(_data=result)
