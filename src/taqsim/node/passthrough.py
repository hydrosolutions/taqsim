from dataclasses import dataclass, field

from .base import BaseNode
from .events import WaterOutput, WaterPassedThrough, WaterReceived


@dataclass
class PassThrough(BaseNode):
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        self._received_this_step += amount
        return amount

    def update(self, t: int, dt: float) -> None:
        amount = self._received_this_step
        if amount > 0:
            self.record(WaterPassedThrough(amount=amount, t=t))
            self.record(WaterOutput(amount=amount, t=t))
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset pass-through node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
