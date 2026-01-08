from dataclasses import dataclass

from .base import BaseNode
from .events import WaterGenerated, WaterOutput
from .timeseries import TimeSeries


@dataclass
class Source(BaseNode):
    inflow: TimeSeries

    def generate(self, t: int, dt: float) -> float:
        amount = self.inflow[t] * dt
        self.record(WaterGenerated(amount=amount, t=t))
        return amount

    def update(self, t: int, dt: float) -> None:
        generated = self.generate(t, dt)
        if generated > 0:
            self.record(WaterOutput(amount=generated, t=t))
