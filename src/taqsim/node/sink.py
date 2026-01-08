from dataclasses import dataclass

from .base import BaseNode
from .events import WaterReceived


@dataclass
class Sink(BaseNode):
    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        return amount

    def update(self, t: int, dt: float) -> None:
        pass
