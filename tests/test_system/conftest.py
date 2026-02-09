from typing import Any

import pytest

from taqsim.edge import Edge
from taqsim.node import Sink, Source, Splitter, Storage
from taqsim.node.timeseries import TimeSeries
from taqsim.testing import NoLoss
from taqsim.time import Timestep


class FakeSplitPolicy:
    def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)


class FakeReleasePolicy:
    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
        return node.storage * self.fraction


@pytest.fixture
def fake_split_policy() -> FakeSplitPolicy:
    return FakeSplitPolicy()


@pytest.fixture
def fake_release_policy() -> FakeReleasePolicy:
    return FakeReleasePolicy()


@pytest.fixture
def fake_loss_rule() -> NoLoss:
    return NoLoss()


@pytest.fixture
def simple_timeseries() -> TimeSeries:
    return TimeSeries([10.0] * 12)


def make_source(
    id: str = "source",
    inflow: TimeSeries | None = None,
    location: tuple[float, float] | None = None,
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Source:
    if inflow is None:
        inflow = TimeSeries([100.0] * 12)
    return Source(
        id=id,
        inflow=inflow,
        location=location,
        tags=tags or frozenset(),
        metadata=metadata or {},
    )


def make_sink(
    id: str = "sink",
    location: tuple[float, float] | None = None,
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Sink:
    return Sink(
        id=id,
        location=location,
        tags=tags or frozenset(),
        metadata=metadata or {},
    )


def make_splitter(
    id: str = "splitter",
    split_policy: FakeSplitPolicy | None = None,
    location: tuple[float, float] | None = None,
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Splitter:
    if split_policy is None:
        split_policy = FakeSplitPolicy()
    return Splitter(
        id=id,
        split_policy=split_policy,
        location=location,
        tags=tags or frozenset(),
        metadata=metadata or {},
    )


def make_storage(
    id: str = "storage",
    capacity: float = 1000.0,
    initial_storage: float = 500.0,
    release_policy: FakeReleasePolicy | None = None,
    loss_rule: NoLoss | None = None,
    location: tuple[float, float] | None = None,
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Storage:
    if release_policy is None:
        release_policy = FakeReleasePolicy()
    if loss_rule is None:
        loss_rule = NoLoss()
    return Storage(
        id=id,
        capacity=capacity,
        initial_storage=initial_storage,
        release_policy=release_policy,
        loss_rule=loss_rule,
        location=location,
        tags=tags or frozenset(),
        metadata=metadata or {},
    )


def make_edge(
    id: str = "edge",
    source: str = "source",
    target: str = "sink",
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Edge:
    return Edge(
        id=id,
        source=source,
        target=target,
        tags=tags or frozenset(),
        metadata=metadata or {},
    )
