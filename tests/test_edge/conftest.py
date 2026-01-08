import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason


class FakeEdgeLossRule:
    """Fake implementation of EdgeLossRule protocol for testing."""

    def __init__(self, losses: dict[LossReason, float] | None = None):
        self._losses = losses if losses is not None else {}

    def calculate(
        self, flow: float, capacity: float, t: int, dt: float
    ) -> dict[LossReason, float]:
        return self._losses


class ProportionalEdgeLossRule:
    """Loss rule that calculates losses as fraction of flow."""

    def __init__(self, loss_fraction: float = 0.1):
        self.loss_fraction = loss_fraction

    def calculate(
        self, flow: float, capacity: float, t: int, dt: float
    ) -> dict[LossReason, float]:
        loss = flow * self.loss_fraction
        return {SEEPAGE: loss}


@pytest.fixture
def fake_edge_loss_rule() -> FakeEdgeLossRule:
    return FakeEdgeLossRule()


@pytest.fixture
def fake_edge_loss_rule_with_losses() -> FakeEdgeLossRule:
    return FakeEdgeLossRule(losses={EVAPORATION: 5.0, SEEPAGE: 3.0})


@pytest.fixture
def proportional_loss_rule() -> ProportionalEdgeLossRule:
    return ProportionalEdgeLossRule(loss_fraction=0.1)
