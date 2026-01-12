from typing import Protocol, runtime_checkable

from taqsim.common import LossReason


@runtime_checkable
class EdgeLossRule(Protocol):
    def calculate(
        self,
        flow: float,
        capacity: float,
        t: int,
        dt: float,
    ) -> dict[LossReason, float]: ...
