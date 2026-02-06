from dataclasses import dataclass, field

from taqsim.common import INEFFICIENCY
from taqsim.time import Timestep

from .base import BaseNode
from .events import DeficitRecorded, WaterConsumed, WaterLost, WaterOutput, WaterReceived
from .timeseries import TimeSeries


@dataclass
class Demand(BaseNode):
    requirement: TimeSeries
    consumption_fraction: float = 1.0
    efficiency: float = 1.0
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.consumption_fraction <= 1.0:
            raise ValueError(f"consumption_fraction must be between 0.0 and 1.0, got {self.consumption_fraction}")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency must be greater than 0.0 and at most 1.0, got {self.efficiency}")

    def receive(self, amount: float, source_id: str, t: Timestep) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t.index))
        self._received_this_step += amount
        return amount

    def consume(self, available: float, t: Timestep) -> tuple[float, float]:
        required = self.requirement[t]

        # How much we need to withdraw to meet the full requirement
        withdrawal_needed = required / self.efficiency

        # Actual withdrawal is limited by available
        withdrawal = min(available, withdrawal_needed)

        # What actually gets delivered after losses
        delivered = withdrawal * self.efficiency

        # Loss due to inefficiency
        loss = withdrawal - delivered
        if loss > 0:
            self.record(WaterLost(amount=loss, reason=INEFFICIENCY, t=t.index))

        # Now apply consumption_fraction to delivered amount
        consumed = delivered * self.consumption_fraction
        returned = delivered - consumed
        excess = available - withdrawal

        self.record(WaterConsumed(amount=consumed, t=t.index))

        if delivered < required:
            deficit = required - delivered
            self.record(DeficitRecorded(required=required, actual=delivered, deficit=deficit, t=t.index))

        return (withdrawal, returned + excess)

    def update(self, t: Timestep) -> None:
        consumed, remaining = self.consume(self._received_this_step, t)
        if remaining > 0:
            self.record(WaterOutput(amount=remaining, t=t.index))
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset demand node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
