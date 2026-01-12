from dataclasses import dataclass, field

from .base import BaseNode
from .events import DeficitRecorded, WaterConsumed, WaterOutput, WaterReceived
from .timeseries import TimeSeries


@dataclass
class Demand(BaseNode):
    requirement: TimeSeries
    _received_this_step: float = field(default=0.0, init=False, repr=False)

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

    def update(self, t: int, dt: float) -> None:
        consumed, remaining = self.consume(self._received_this_step, t, dt)
        if remaining > 0:
            self.record(WaterOutput(amount=remaining, t=t))
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset demand node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
