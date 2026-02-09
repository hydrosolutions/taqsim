from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from taqsim.time import Timestep


@runtime_checkable
class Receives(Protocol):
    def receive(self, amount: float, source_id: str, t: "Timestep") -> float: ...
