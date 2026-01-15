from dataclasses import dataclass, field

from .base import BaseNode
from .events import WaterOutput, WaterPassedThrough, WaterReceived, WaterSpilled


@dataclass
class PassThrough(BaseNode):
    capacity: float | None = None  # None = unlimited
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        self._received_this_step += amount
        return amount

    def update(self, t: int, dt: float) -> None:
        amount = self._received_this_step
        if amount > 0:
            if self.capacity is not None and amount > self.capacity:
                passed = self.capacity
                spilled = amount - self.capacity
                self.record(WaterSpilled(amount=spilled, t=t))
            else:
                passed = amount
            self.record(WaterPassedThrough(amount=passed, t=t))
            self.record(WaterOutput(amount=passed, t=t))
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset pass-through node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
