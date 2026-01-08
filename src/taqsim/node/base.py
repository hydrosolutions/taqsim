from dataclasses import dataclass, field

from .events import NodeEvent


@dataclass
class BaseNode:
    id: str
    events: list[NodeEvent] = field(default_factory=list, init=False, repr=False)
    _targets: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def targets(self) -> list[str]:
        return self._targets

    def _set_targets(self, targets: list[str]) -> None:
        self._targets = targets

    def record(self, event: NodeEvent) -> None:
        self.events.append(event)

    def events_at(self, t: int) -> list[NodeEvent]:
        return [e for e in self.events if e.t == t]

    def events_of_type[T: NodeEvent](self, event_type: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, event_type)]

    def clear_events(self) -> None:
        self.events.clear()

    def update(self, t: int, dt: float) -> None:
        raise NotImplementedError("Subclasses must implement update()")
