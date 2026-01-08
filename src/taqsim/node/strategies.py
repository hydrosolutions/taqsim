from typing import Protocol, runtime_checkable

from .events import LossReason


@runtime_checkable
class ReleaseRule(Protocol):
    def release(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float: ...


@runtime_checkable
class SplitStrategy(Protocol):
    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]: ...


@runtime_checkable
class LossRule(Protocol):
    def calculate(self, storage: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]: ...
