from dataclasses import dataclass
from typing import ClassVar

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason, Strategy
from taqsim.node.timeseries import TimeSeries


@pytest.fixture
def simple_timeseries() -> TimeSeries:
    return TimeSeries([10.0] * 12)


@pytest.fixture
def varying_timeseries() -> TimeSeries:
    return TimeSeries([float(i * 10) for i in range(12)])


@dataclass(frozen=True)
class FakeReleaseRule(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ("fraction",)
    fraction: float = 0.5

    def release(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
        return storage * self.fraction


@dataclass(frozen=True)
class FakeSplitStrategy(Strategy):
    __params__: ClassVar[tuple[str, ...]] = ()

    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


class FakeLossRule:
    def __init__(self, evap_rate: float = 0.01, seepage_rate: float = 0.005):
        self.evap_rate = evap_rate
        self.seepage_rate = seepage_rate

    def calculate(self, storage: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
        return {
            EVAPORATION: storage * self.evap_rate * dt,
            SEEPAGE: storage * self.seepage_rate * dt,
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
