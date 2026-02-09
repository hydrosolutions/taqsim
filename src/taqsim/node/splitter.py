from dataclasses import dataclass, field

from taqsim.time import Timestep

from .base import BaseNode
from .events import WaterDistributed, WaterReceived
from .strategies import SplitPolicy


@dataclass
class Splitter(BaseNode):
    split_policy: SplitPolicy | None = field(default=None)
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.split_policy is None:
            raise ValueError("split_policy is required")

    def receive(self, amount: float, source_id: str, t: Timestep) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t.index))
        self._received_this_step += amount
        return amount

    def _distribute(self, amount: float, t: Timestep) -> dict[str, float]:
        if not self.targets or amount <= 0:
            return {}
        allocation = self.split_policy.split(self, amount, t)  # type: ignore[union-attr]
        for target_id, alloc_amount in allocation.items():
            self.record_output(WaterDistributed(amount=alloc_amount, target_id=target_id, t=t.index))
        return allocation

    def update(self, t: Timestep) -> None:
        self._distribute(self._received_this_step, t)
        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset splitter node for a fresh simulation run."""
        super().reset()
        self._received_this_step = 0.0
