from dataclasses import dataclass, field

from .base import BaseNode
from .events import DeficitRecorded, WaterConsumed, WaterDistributed, WaterReceived
from .strategies import SplitStrategy
from .timeseries import TimeSeries


@dataclass
class Demand(BaseNode):
    requirement: TimeSeries
    targets: list[str] = field(default_factory=list)
    split_strategy: SplitStrategy | None = field(default=None)
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.split_strategy is None:
            raise ValueError("split_strategy is required")

    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        self._received_this_step += amount
        return amount

    def consume(self, available: float, t: int, dt: float) -> tuple[float, float]:
        required = self.requirement[t] * dt
        consumed = min(available, required)
        remaining = available - consumed

        self.record(WaterConsumed(amount=consumed, t=t))

        if consumed < required:
            deficit = required - consumed
            self.record(DeficitRecorded(required=required, actual=consumed, deficit=deficit, t=t))

        return (consumed, remaining)

    def distribute(self, amount: float, t: int) -> dict[str, float]:
        if not self.targets or amount <= 0:
            return {}
        allocation = self.split_strategy.split(amount, self.targets, t)  # type: ignore[union-attr]
        for target_id, alloc_amount in allocation.items():
            self.record(WaterDistributed(amount=alloc_amount, target_id=target_id, t=t))
        return allocation

    def update(self, t: int, dt: float) -> None:
        consumed, remaining = self.consume(self._received_this_step, t, dt)
        self.distribute(remaining, t)
        self._received_this_step = 0.0
