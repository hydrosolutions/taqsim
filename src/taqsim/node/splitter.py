from dataclasses import dataclass, field

from .base import BaseNode
from .events import WaterDistributed, WaterReceived
from .strategies import SplitStrategy


@dataclass
class Splitter(BaseNode):
    split_strategy: SplitStrategy | None = field(default=None)
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.split_strategy is None:
            raise ValueError("split_strategy is required")

    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        self._received_this_step += amount
        return amount

    def distribute(self, amount: float, t: int) -> dict[str, float]:
        if not self.targets or amount <= 0:
            return {}
        allocation = self.split_strategy.split(self, amount, t)  # type: ignore[union-attr]
        for target_id, alloc_amount in allocation.items():
            self.record(WaterDistributed(amount=alloc_amount, target_id=target_id, t=t))
        return allocation

    def update(self, t: int, dt: float) -> None:
        self.distribute(self._received_this_step, t)
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset splitter node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
