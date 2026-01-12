from dataclasses import dataclass, field

from taqsim.node.timeseries import TimeSeries

from .events import (
    CapacityExceeded,
    EdgeEvent,
    FlowDelivered,
    FlowLost,
    FlowReceived,
    RequirementUnmet,
)
from .losses import EdgeLossRule


@dataclass
class Edge:
    id: str
    source: str
    target: str
    capacity: float
    requirement: TimeSeries | None = None
    loss_rule: EdgeLossRule | None = field(default=None)
    targets: list[str] = field(default_factory=list)

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

    def clear_events(self) -> None:
        self.events.clear()

    def receive(self, amount: float, t: int) -> float:
        self.record(FlowReceived(amount=amount, t=t))
        self._received_this_step += amount
        return amount

    def update(self, t: int, dt: float) -> float:
        received = self._received_this_step

        # 1. Capacity check
        excess = 0.0
        if received > self.capacity:
            excess = received - self.capacity
            self.record(CapacityExceeded(excess=excess, t=t))
            received = self.capacity

        # 2-3. Calculate and record losses
        losses = self.loss_rule.calculate(received, self.capacity, t, dt)
        total_loss = sum(losses.values())

        # Scale losses if they exceed available flow
        if total_loss > received:
            scale = received / total_loss if total_loss > 0 else 0
            losses = {reason: amount * scale for reason, amount in losses.items()}
            total_loss = received

        for reason, amount in losses.items():
            if amount > 0:
                self.record(FlowLost(amount=amount, reason=reason, t=t))

        # 4. Calculate delivered
        delivered = received - total_loss

        # 5. Check requirement
        if self.requirement is not None:
            required = self.requirement[t] * dt
            if delivered < required:
                self.record(
                    RequirementUnmet(
                        required=required,
                        actual=delivered,
                        deficit=required - delivered,
                        t=t,
                    )
                )

        # 6. Record delivery
        self.record(FlowDelivered(amount=delivered, t=t))

        # 7. Reset
        self._received_this_step = 0.0

        # 8. Return
        return delivered

    def reset(self) -> None:
        """Reset edge to initial state for a fresh simulation run.

        Clears accumulated events and step accumulator.
        """
        self.clear_events()
        self._received_this_step = 0.0
