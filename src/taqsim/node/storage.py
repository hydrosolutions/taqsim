from dataclasses import dataclass, field

from .base import BaseNode
from .events import (
    WaterLost,
    WaterOutput,
    WaterReceived,
    WaterReleased,
    WaterSpilled,
    WaterStored,
)
from .strategies import LossRule, ReleaseRule


@dataclass
class Storage(BaseNode):
    capacity: float
    initial_storage: float = 0.0
    release_rule: ReleaseRule | None = field(default=None)
    loss_rule: LossRule | None = field(default=None)
    _current_storage: float = field(init=False, repr=False)
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.initial_storage < 0:
            raise ValueError("initial_storage cannot be negative")
        if self.initial_storage > self.capacity:
            raise ValueError("initial_storage cannot exceed capacity")
        if self.release_rule is None:
            raise ValueError("release_rule is required")
        if self.loss_rule is None:
            raise ValueError("loss_rule is required")
        self._current_storage = self.initial_storage

    @property
    def storage(self) -> float:
        return self._current_storage

    def receive(self, amount: float, source_id: str, t: int) -> float:
        self.record(WaterReceived(amount=amount, source_id=source_id, t=t))
        self._received_this_step += amount
        return amount

    def store(self, amount: float, t: int, dt: float) -> tuple[float, float]:
        available_space = self.capacity - self._current_storage
        stored = min(amount, available_space)
        spilled = amount - stored

        self._current_storage += stored
        self.record(WaterStored(amount=stored, t=t))

        if spilled > 0:
            self.record(WaterSpilled(amount=spilled, t=t))

        return (stored, spilled)

    def lose(self, t: int, dt: float) -> float:
        losses = self.loss_rule.calculate(self._current_storage, self.capacity, t, dt)
        total_loss = sum(losses.values())

        if total_loss > self._current_storage:
            scale = self._current_storage / total_loss if total_loss > 0 else 0
            losses = {reason: amount * scale for reason, amount in losses.items()}
            total_loss = self._current_storage

        for reason, amount in losses.items():
            if amount > 0:
                self._current_storage -= amount
                self.record(WaterLost(amount=amount, reason=reason, t=t))

        return total_loss

    def release(self, inflow: float, t: int, dt: float) -> float:
        raw_release = self.release_rule.release(self._current_storage, self.capacity, inflow, t, dt)
        actual_release = max(0.0, min(raw_release, self._current_storage))

        if actual_release > 0:
            self._current_storage -= actual_release
            self.record(WaterReleased(amount=actual_release, t=t))

        return actual_release

    def update(self, t: int, dt: float) -> None:
        inflow = self._received_this_step

        # 1. Store (handles spillway)
        stored, spilled = self.store(inflow, t, dt)

        # 2. Losses
        self.lose(t, dt)

        # 3. Release
        released = self.release(inflow, t, dt)

        # 4. Record output (released + spilled goes downstream)
        total_outflow = released + spilled
        if total_outflow > 0:
            self.record(WaterOutput(amount=total_outflow, t=t))

        self._received_this_step = 0.0

    def reset(self) -> None:
        """Reset storage to initial state for a fresh simulation run."""
        super().reset()
        self._current_storage = self.initial_storage
        self._received_this_step = 0.0
