from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from taqsim.common import CAPACITY_EXCEEDED
from taqsim.time import Timestep

from .events import (
    EdgeEvent,
    WaterDelivered,
    WaterLost,
    WaterReceived,
)
from .losses import EdgeLossRule

if TYPE_CHECKING:
    from taqsim.objective import Trace


@dataclass
class Edge:
    id: str
    source: str
    target: str
    capacity: float
    loss_rule: EdgeLossRule | None = field(default=None)
    targets: list[str] = field(default_factory=list)
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    events: list[EdgeEvent] = field(default_factory=list, init=False, repr=False)
    _received_this_step: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.source:
            raise ValueError("source cannot be empty")
        if not self.target:
            raise ValueError("target cannot be empty")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.loss_rule is None:
            raise ValueError("loss_rule is required")

    def record(self, event: EdgeEvent) -> None:
        self.events.append(event)

    def events_of_type[T: EdgeEvent](self, event_type: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, event_type)]

    def trace[T: EdgeEvent](self, event_type: type[T], field: str = "amount") -> "Trace":
        from taqsim.objective import Trace

        return Trace.from_events(self.events_of_type(event_type), field=field)

    def clear_events(self) -> None:
        self.events.clear()

    def receive(self, amount: float, t: Timestep) -> float:
        self.record(WaterReceived(amount=amount, t=t.index))
        self._received_this_step += amount
        return amount

    def update(self, t: Timestep) -> float:
        received = self._received_this_step

        # 1. Capacity check
        excess = 0.0
        if received > self.capacity:
            excess = received - self.capacity
            self.record(WaterLost(amount=excess, reason=CAPACITY_EXCEEDED, t=t.index))
            received = self.capacity

        # 2-3. Calculate and record losses
        losses = self.loss_rule.calculate(self, received, t)
        total_loss = sum(losses.values())

        # Scale losses if they exceed available flow
        if total_loss > received:
            scale = received / total_loss if total_loss > 0 else 0
            losses = {reason: amount * scale for reason, amount in losses.items()}
            total_loss = received

        for reason, amount in losses.items():
            if amount > 0:
                self.record(WaterLost(amount=amount, reason=reason, t=t.index))

        # 4. Calculate delivered
        delivered = received - total_loss

        # 5. Record delivery
        self.record(WaterDelivered(amount=delivered, t=t.index))

        # 6. Reset
        self._received_this_step = 0.0

        # 7. Return
        return delivered

    def reset(self) -> None:
        """Reset edge to initial state for a fresh simulation run.

        Clears accumulated events and step accumulator.
        """
        self.clear_events()
        self._received_this_step = 0.0

    def _fresh_copy(self) -> "Edge":
        return Edge(
            id=self.id,
            source=self.source,
            target=self.target,
            capacity=self.capacity,
            loss_rule=self.loss_rule,
            targets=self.targets,
            tags=self.tags,
            metadata=self.metadata,
        )
