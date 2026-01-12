import pytest

from taqsim.common import LossReason
from taqsim.edge import Edge
from taqsim.node import Sink, Source, Splitter, Storage
from taqsim.node.timeseries import TimeSeries


class FakeSplitStrategy:
    def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


class FakeReleaseRule:
    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    def release(self, storage: float, dead_storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
        return storage * self.fraction


class FakeLossRule:
    def calculate(self, storage: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
        return {}


class FakeEdgeLossRule:
    def __init__(self, losses: dict[LossReason, float] | None = None):
        self._losses = losses if losses is not None else {}

    def calculate(self, flow: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
        return self._losses


@pytest.fixture
def fake_split_strategy() -> FakeSplitStrategy:
    return FakeSplitStrategy()


@pytest.fixture
def fake_release_rule() -> FakeReleaseRule:
    return FakeReleaseRule()


@pytest.fixture
def fake_loss_rule() -> FakeLossRule:
    return FakeLossRule()


@pytest.fixture
def fake_edge_loss_rule() -> FakeEdgeLossRule:
    return FakeEdgeLossRule()


@pytest.fixture
def simple_timeseries() -> TimeSeries:
    return TimeSeries([10.0] * 12)


def make_source(
    id: str = "source",
    inflow: TimeSeries | None = None,
    location: tuple[float, float] | None = None,
) -> Source:
    if inflow is None:
        inflow = TimeSeries([100.0] * 12)
    return Source(id=id, inflow=inflow, location=location)


def make_sink(
    id: str = "sink",
    location: tuple[float, float] | None = None,
) -> Sink:
    return Sink(id=id, location=location)


def make_splitter(
    id: str = "splitter",
    split_strategy: FakeSplitStrategy | None = None,
    location: tuple[float, float] | None = None,
) -> Splitter:
    if split_strategy is None:
        split_strategy = FakeSplitStrategy()
    return Splitter(id=id, split_strategy=split_strategy, location=location)


def make_storage(
    id: str = "storage",
    capacity: float = 1000.0,
    initial_storage: float = 500.0,
    release_rule: FakeReleaseRule | None = None,
    loss_rule: FakeLossRule | None = None,
    location: tuple[float, float] | None = None,
) -> Storage:
    if release_rule is None:
        release_rule = FakeReleaseRule()
    if loss_rule is None:
        loss_rule = FakeLossRule()
    return Storage(
        id=id,
        capacity=capacity,
        initial_storage=initial_storage,
        release_rule=release_rule,
        loss_rule=loss_rule,
        location=location,
    )


def make_edge(
    id: str = "edge",
    source: str = "source",
    target: str = "sink",
    capacity: float = 1000.0,
    loss_rule: FakeEdgeLossRule | None = None,
) -> Edge:
    if loss_rule is None:
        loss_rule = FakeEdgeLossRule()
    return Edge(
        id=id,
        source=source,
        target=target,
        capacity=capacity,
        loss_rule=loss_rule,
    )
