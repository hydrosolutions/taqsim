from typing import TYPE_CHECKING

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.testing import NoEdgeLoss, ProportionalEdgeLoss
from taqsim.time import Timestep

if TYPE_CHECKING:
    from taqsim.edge import Edge


class FakeEdgeLossRule:
    """Fake implementation of EdgeLossRule protocol for testing with custom losses."""

    def __init__(self, losses: dict[LossReason, float] | None = None):
        self._losses = losses if losses is not None else {}

    def calculate(self, edge: "Edge", flow: float, t: Timestep) -> dict[LossReason, float]:
        return self._losses


@pytest.fixture
def fake_edge_loss_rule() -> NoEdgeLoss:
    return NoEdgeLoss()


@pytest.fixture
def fake_edge_loss_rule_with_losses() -> FakeEdgeLossRule:
    return FakeEdgeLossRule(losses={EVAPORATION: 5.0, SEEPAGE: 3.0})


@pytest.fixture
def proportional_loss_rule() -> ProportionalEdgeLoss:
    return ProportionalEdgeLoss(loss_fraction=0.1)
