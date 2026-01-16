from typing import TYPE_CHECKING, Protocol, runtime_checkable

from taqsim.common import LossReason

if TYPE_CHECKING:
    from taqsim.node.splitter import Splitter
    from taqsim.node.storage import Storage


@runtime_checkable
class ReleaseRule(Protocol):
    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float: ...


@runtime_checkable
class SplitRule(Protocol):
    """Protocol for water distribution strategies at Splitter nodes.

    The split() method determines how incoming water is distributed
    among downstream nodes.

    Returns:
        dict mapping downstream node IDs to water amounts.
        Keys must be valid node IDs that the splitter connects to.

    Example:
        def split(self, node: Splitter, amount: float, t: int) -> dict[str, float]:
            return {"irrigation": amount * 0.6, "thermal": amount * 0.4}
    """

    def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]: ...


@runtime_checkable
class LossRule(Protocol):
    def calculate(self, node: "Storage", t: int, dt: float) -> dict[LossReason, float]: ...
