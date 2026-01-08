from dataclasses import dataclass, field

from .base import BaseNode
from .events import WaterDistributed, WaterGenerated
from .strategies import SplitStrategy
from .timeseries import TimeSeries


@dataclass
class Source(BaseNode):
    inflow: TimeSeries
    targets: list[str] = field(default_factory=list)
    split_strategy: SplitStrategy = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.split_strategy is None:
            raise ValueError("split_strategy is required")

    def generate(self, t: int, dt: float) -> float:
        amount = self.inflow[t] * dt
        self.record(WaterGenerated(amount=amount, t=t))
        return amount

    def distribute(self, amount: float, t: int) -> dict[str, float]:
        if not self.targets or amount <= 0:
            return {}
        allocation = self.split_strategy.split(amount, self.targets, t)
        for target_id, alloc_amount in allocation.items():
            self.record(WaterDistributed(amount=alloc_amount, target_id=target_id, t=t))
        return allocation

    def update(self, t: int, dt: float) -> None:
        generated = self.generate(t, dt)
        self.distribute(generated, t)
