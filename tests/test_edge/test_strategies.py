from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.edge.losses import EdgeLossRule


class TestEdgeLossRuleProtocol:
    def test_class_with_calculate_satisfies_protocol(self):
        class ValidLoss:
            def calculate(self, flow: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
                return {EVAPORATION: flow * 0.01}

        assert isinstance(ValidLoss(), EdgeLossRule)

    def test_class_without_calculate_does_not_satisfy(self):
        class NoCalculate:
            pass

        assert not isinstance(NoCalculate(), EdgeLossRule)

    def test_fake_edge_loss_rule_satisfies_protocol(self, fake_edge_loss_rule):
        assert isinstance(fake_edge_loss_rule, EdgeLossRule)

    def test_calculate_returns_dict_with_loss_reasons(self):
        class ValidLoss:
            def calculate(self, flow: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
                return {
                    EVAPORATION: flow * 0.01 * dt,
                    SEEPAGE: flow * 0.005 * dt,
                }

        rule = ValidLoss()
        result = rule.calculate(1000.0, 2000.0, 0, 1.0)
        assert result[EVAPORATION] == 10.0
        assert result[SEEPAGE] == 5.0

    def test_proportional_loss_rule_satisfies_protocol(self, proportional_loss_rule):
        assert isinstance(proportional_loss_rule, EdgeLossRule)

    def test_proportional_loss_rule_calculates_correctly(self, proportional_loss_rule):
        result = proportional_loss_rule.calculate(100.0, 200.0, 0, 1.0)
        assert SEEPAGE in result
        assert result[SEEPAGE] == 10.0


class TestProtocolNonSatisfaction:
    def test_class_with_wrong_method_name_does_not_satisfy(self):
        class WrongMethod:
            def compute_loss(self, flow: float, capacity: float, t: int, dt: float) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), EdgeLossRule)

    def test_empty_class_does_not_satisfy(self):
        class EmptyClass:
            pass

        assert not isinstance(EmptyClass(), EdgeLossRule)
