from typing import TYPE_CHECKING, Protocol, runtime_checkable

from taqsim.common import LossReason

if TYPE_CHECKING:
    from taqsim.edge.edge import Edge
    from taqsim.time import Timestep


@runtime_checkable
class EdgeLossRule(Protocol):
    def calculate(self, edge: "Edge", flow: float, t: "Timestep") -> dict[LossReason, float]: ...
