from collections.abc import Callable

from taqsim.objective.objective import Direction, Objective


class ObjectiveRegistry:
    _factories: dict[str, Callable[..., Objective]]
    _direction: Direction

    def __init__(self, direction: Direction) -> None:
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(self, "_factories", {})

    @property
    def direction(self) -> Direction:
        return self._direction

    def register(self, name: str, factory: Callable[..., Objective]) -> None:
        if name in self._factories:
            raise ValueError(f"Objective '{name}' is already registered")
        self._factories[name] = factory

    def __getattr__(self, name: str) -> Callable[..., Objective]:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "(none)"
            raise AttributeError(f"Unknown objective '{name}'. Available: {available}")
        return self._factories[name]

    def list_available(self) -> list[str]:
        return list(self._factories.keys())
