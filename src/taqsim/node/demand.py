from dataclasses import dataclass, field

from .base import BaseNode
from .events import DeficitRecorded, WaterConsumed, WaterOutput, WaterReceived
from .timeseries import TimeSeries


@dataclass
class Demand(BaseNode):
    requirement: TimeSeries
    consumption_fraction: float = 1.0
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.consumption_fraction <= 1.0:
            raise ValueError(f"consumption_fraction must be between 0.0 and 1.0, got {self.consumption_fraction}")

    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        self._received_this_step += amount
        return amount

    def consume(self, available: float, t: int, dt: float) -> tuple[float, float]:
        required = self.requirement[t] * dt
        met = min(available, required)

        # Split met demand: consumed leaves system, returned goes downstream
        consumed = met * self.consumption_fraction
        returned = met - consumed
        excess = available - met

        self.record(WaterConsumed(amount=consumed, t=t))

        if met < required:
            deficit = required - met
            self.record(DeficitRecorded(required=required, actual=met, deficit=deficit, t=t))

        return (met, returned + excess)

    def update(self, t: int, dt: float) -> None:
        consumed, remaining = self.consume(self._received_this_step, t, dt)
        if remaining > 0:
            self.record(WaterOutput(amount=remaining, t=t))
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset demand node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
