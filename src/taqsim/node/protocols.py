from typing import Protocol, runtime_checkable


@runtime_checkable
class Generates(Protocol):
    def generate(self, t: int, dt: float) -> float: ...


@runtime_checkable
class Receives(Protocol):
    def receive(self, amount: float, source_id: str, t: int) -> float: ...


@runtime_checkable
class Stores(Protocol):
    @property
    def storage(self) -> float: ...

    @property
    def capacity(self) -> float: ...

    def store(self, amount: float, t: int, dt: float) -> tuple[float, float]: ...


@runtime_checkable
class Loses(Protocol):
    def lose(self, t: int, dt: float) -> float: ...


@runtime_checkable
class Consumes(Protocol):
    def consume(self, amount: float, t: int, dt: float) -> tuple[float, float]: ...


@runtime_checkable
class Gives(Protocol):
    def distribute(self, amount: float, t: int) -> dict[str, float]: ...
