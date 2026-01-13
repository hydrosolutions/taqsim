from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class FakeEvent:
    amount: float
    t: int


@dataclass(frozen=True)
class FakeEventWithDeficit:
    deficit: float
    t: int


@pytest.fixture
def sample_events() -> list[FakeEvent]:
    return [FakeEvent(10.0, 0), FakeEvent(20.0, 1), FakeEvent(30.0, 2)]


@pytest.fixture
def duplicate_events() -> list[FakeEvent]:
    return [FakeEvent(10.0, 0), FakeEvent(5.0, 0), FakeEvent(20.0, 1)]
