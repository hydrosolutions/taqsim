from typing import TYPE_CHECKING, Protocol, runtime_checkable

from taqsim.common import LossReason

if TYPE_CHECKING:
    from taqsim.edge.edge import Edge


@runtime_checkable
class EdgeLossRule(Protocol):
    def calculate(self, edge: "Edge", flow: float, t: int, dt: float) -> dict[LossReason, float]: ...
