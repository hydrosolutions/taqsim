from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from taqsim.time import Timestep


@runtime_checkable
class Generates(Protocol):
    def generate(self, t: "Timestep") -> float: ...


@runtime_checkable
class Receives(Protocol):
    def receive(self, amount: float, source_id: str, t: "Timestep") -> float: ...


@runtime_checkable
class Stores(Protocol):
    @property
    def storage(self) -> float: ...

    @property
    def capacity(self) -> float: ...

    def store(self, amount: float, t: "Timestep") -> tuple[float, float]: ...


@runtime_checkable
class Loses(Protocol):
    def lose(self, t: "Timestep") -> float: ...


@runtime_checkable
class Consumes(Protocol):
    def consume(self, amount: float, t: "Timestep") -> tuple[float, float]: ...
