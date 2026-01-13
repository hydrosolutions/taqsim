from typing import TYPE_CHECKING, Protocol, runtime_checkable

from taqsim.common import LossReason

if TYPE_CHECKING:
    from taqsim.node.splitter import Splitter
    from taqsim.node.storage import Storage


@runtime_checkable
class ReleaseRule(Protocol):
    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float: ...


@runtime_checkable
class SplitStrategy(Protocol):
    def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]: ...


@runtime_checkable
class LossRule(Protocol):
    def calculate(self, node: "Storage", t: int, dt: float) -> dict[LossReason, float]: ...
