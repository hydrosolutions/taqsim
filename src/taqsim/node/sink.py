from dataclasses import dataclass

from taqsim.time import Timestep

from .base import BaseNode
from .events import WaterReceived


@dataclass
class Sink(BaseNode):
    def receive(self, amount: float, source_id: str, t: Timestep) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t.index))
        return amount

    def update(self, t: Timestep) -> None:
        pass
