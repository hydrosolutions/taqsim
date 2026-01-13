from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason, Strategy
from taqsim.node.timeseries import TimeSeries

if TYPE_CHECKING:
    from taqsim.node import Splitter, Storage


@pytest.fixture
def simple_timeseries() -> TimeSeries:
    return TimeSeries([10.0] * 12)


@pytest.fixture
def varying_timeseries() -> TimeSeries:
    return TimeSeries([float(i * 10) for i in range(12)])


@dataclass(frozen=True)
class FakeReleaseRule(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("fraction",)
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {"fraction": (0.0, 1.0)}
    fraction: float = 0.5

    def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
        return node.storage * self.fraction


@dataclass(frozen=True)
class FakeSplitStrategy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ()
    __bounds__: ClassVar[dict[str, tuple[float, float]]] = {}

    def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


class FakeLossRule:
    def __init__(self, evap_rate: float = 0.01, seepage_rate: float = 0.005):
        self.evap_rate = evap_rate
        self.seepage_rate = seepage_rate

    def calculate(self, node: "Storage", t: int, dt: float) -> dict[LossReason, float]:
        return {
            EVAPORATION: node.storage * self.evap_rate * dt,
            SEEPAGE: node.storage * self.seepage_rate * dt,
        }


@pytest.fixture
def fake_release_rule() -> FakeReleaseRule:
    return FakeReleaseRule()


@pytest.fixture
def fake_split_strategy() -> FakeSplitStrategy:
    return FakeSplitStrategy()


@pytest.fixture
def fake_loss_rule() -> FakeLossRule:
    return FakeLossRule()
