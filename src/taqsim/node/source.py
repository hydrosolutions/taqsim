from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import BaseNode
from .events import WaterGenerated, WaterOutput
from .timeseries import TimeSeries

if TYPE_CHECKING:
    from taqsim.time import Timestep


@dataclass
class Source(BaseNode):
    inflow: TimeSeries

    def generate(self, t: Timestep) -> float:
        amount = self.inflow[t]
        self.record(WaterGenerated(amount=amount, t=t.index))
        return amount

    def update(self, t: Timestep) -> None:
        generated = self.generate(t)
        if generated > 0:
            self.record_output(WaterOutput(amount=generated, t=t.index))
