from dataclasses import dataclass, field, fields

from taqsim.common import Strategy

from .events import NodeEvent


@dataclass
class BaseNode:
    id: str
    location: tuple[float, float] | None = field(default=None, kw_only=True)  # (lat, lon) in WGS84
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

    def strategies(self) -> dict[str, Strategy]:
        """Return all Strategy-typed fields (operational policies only).

        Auto-discovers fields that inherit from Strategy. Physical models
        like LossRule do not inherit from Strategy and are excluded.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if isinstance(getattr(self, f.name), Strategy)}

    def reset(self) -> None:
        """Reset node to initial state for a fresh simulation run.

        Clears accumulated events. Subclasses should override to reset
        additional state, calling super().reset() first.
        """
        self.clear_events()
