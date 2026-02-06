from typing import TYPE_CHECKING

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.edge.losses import EdgeLossRule
from taqsim.time import Frequency, Timestep

if TYPE_CHECKING:
    from taqsim.edge.edge import Edge


class TestEdgeLossRuleProtocol:
    def test_class_with_calculate_satisfies_protocol(self):
        class ValidLoss:
            def calculate(self, edge: "Edge", flow: float, t: Timestep) -> dict[LossReason, float]:
                return {EVAPORATION: flow * 0.01}

        assert isinstance(ValidLoss(), EdgeLossRule)

    def test_class_without_calculate_does_not_satisfy(self):
        class NoCalculate:
            pass

        assert not isinstance(NoCalculate(), EdgeLossRule)

    def test_fake_edge_loss_rule_satisfies_protocol(self, fake_edge_loss_rule):
        assert isinstance(fake_edge_loss_rule, EdgeLossRule)

    def test_calculate_returns_dict_with_loss_reasons(self, fake_edge_loss_rule):
        from taqsim.edge.edge import Edge

        class ValidLoss:
            def calculate(self, edge: "Edge", flow: float, t: Timestep) -> dict[LossReason, float]:
                return {
                    EVAPORATION: flow * 0.01,
                    SEEPAGE: flow * 0.005,
                }

        edge = Edge(
            id="test",
            source="src",
            target="tgt",
            capacity=2000.0,
            loss_rule=fake_edge_loss_rule,
        )
        rule = ValidLoss()
        result = rule.calculate(edge, 1000.0, Timestep(0, Frequency.MONTHLY))
        assert result[EVAPORATION] == 10.0
        assert result[SEEPAGE] == 5.0

    def test_proportional_loss_rule_satisfies_protocol(self, proportional_loss_rule):
        assert isinstance(proportional_loss_rule, EdgeLossRule)

    def test_proportional_loss_rule_calculates_correctly(self, fake_edge_loss_rule, proportional_loss_rule):
        from taqsim.edge.edge import Edge

        edge = Edge(
            id="test",
            source="src",
            target="tgt",
            capacity=200.0,
            loss_rule=fake_edge_loss_rule,
        )
        result = proportional_loss_rule.calculate(edge, 100.0, Timestep(0, Frequency.MONTHLY))
        assert SEEPAGE in result
        assert result[SEEPAGE] == 10.0


class TestProtocolNonSatisfaction:
    def test_class_with_wrong_method_name_does_not_satisfy(self):
        class WrongMethod:
            def compute_loss(self, edge: "Edge", flow: float, t: Timestep) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), EdgeLossRule)

    def test_empty_class_does_not_satisfy(self):
        class EmptyClass:
            pass

        assert not isinstance(EmptyClass(), EdgeLossRule)
