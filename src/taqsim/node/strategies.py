from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from taqsim.common import LossReason

if TYPE_CHECKING:
    from taqsim.node.splitter import Splitter
    from taqsim.node.storage import Storage
    from taqsim.time import Timestep


@runtime_checkable
class ReleasePolicy(Protocol):
    def release(self, node: "Storage", inflow: float, t: "Timestep") -> float: ...


@runtime_checkable
class SplitPolicy(Protocol):
    """Protocol for water distribution strategies at Splitter nodes.

    The split() method determines how incoming water is distributed
    among downstream nodes.

    Returns:
        dict mapping downstream node IDs to water amounts.
        Keys must be valid node IDs that the splitter connects to.

    Example:
        def split(self, node: Splitter, amount: float, t: Timestep) -> dict[str, float]:
            return {"irrigation": amount * 0.6, "thermal": amount * 0.4}
    """

    def split(self, node: "Splitter", amount: float, t: "Timestep") -> dict[str, float]: ...


@runtime_checkable
class LossRule(Protocol):
    def calculate(self, node: "Storage", t: "Timestep") -> dict[LossReason, float]: ...


@dataclass(frozen=True)
class NoLoss:
    def calculate(self, node: "Storage", t: "Timestep") -> dict[LossReason, float]:
        return {}
