from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim.common import Strategy
from taqsim.node.timeseries import TimeSeries
from taqsim.testing import ConstantLoss
from taqsim.time import Timestep

if TYPE_CHECKING:
    from taqsim.node import Splitter, Storage


@pytest.fixture
def simple_timeseries() -> TimeSeries:
    return TimeSeries([10.0] * 12)


@pytest.fixture
def varying_timeseries() -> TimeSeries:
    return TimeSeries([float(i * 10) for i in range(12)])


@dataclass(frozen=True)
class FakeReleasePolicy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("fraction",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"fraction": (0.0, 1.0)}
    fraction: float = 0.5

    def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
        return node.storage * self.fraction


@dataclass(frozen=True)
class FakeSplitPolicy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ()
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {}

    def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


@pytest.fixture
def fake_release_policy() -> FakeReleasePolicy:
    return FakeReleasePolicy()


@pytest.fixture
def fake_split_policy() -> FakeSplitPolicy:
    return FakeSplitPolicy()


@pytest.fixture
def fake_loss_rule() -> ConstantLoss:
    return ConstantLoss()
