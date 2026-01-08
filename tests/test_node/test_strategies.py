from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node.strategies import LossRule, ReleaseRule, SplitStrategy


class TestReleaseRuleProtocol:
    def test_class_with_release_satisfies_protocol(self):
        class ValidRelease:
            def release(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
                return storage * 0.5

        assert isinstance(ValidRelease(), ReleaseRule)

    def test_class_without_release_does_not_satisfy(self):
        class NoRelease:
            pass

        assert not isinstance(NoRelease(), ReleaseRule)

    def test_fake_release_rule_satisfies_protocol(self, fake_release_rule):
        assert isinstance(fake_release_rule, ReleaseRule)

    def test_release_returns_float(self):
        class ValidRelease:
            def release(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
                return min(storage, inflow)

        rule = ValidRelease()
        result = rule.release(100.0, 200.0, 50.0, 0, 1.0)
        assert result == 50.0


class TestSplitStrategyProtocol:
    def test_class_with_split_satisfies_protocol(self):
        class ValidSplit:
            def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
                return {target: amount / len(targets) for target in targets}

        assert isinstance(ValidSplit(), SplitStrategy)

    def test_class_without_split_does_not_satisfy(self):
        class NoSplit:
            pass

        assert not isinstance(NoSplit(), SplitStrategy)

    def test_fake_split_strategy_satisfies_protocol(self, fake_split_strategy):
        assert isinstance(fake_split_strategy, SplitStrategy)

    def test_split_returns_dict(self):
        class ValidSplit:
            def split(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
                return {target: amount / len(targets) for target in targets}

        strategy = ValidSplit()
        result = strategy.split(100.0, ["a", "b"], 0)
        assert result == {"a": 50.0, "b": 50.0}


class TestLossRuleProtocol:
    def test_class_with_calculate_satisfies_protocol(self):
        class ValidLoss:
            def calculate(self, storage: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
                return {EVAPORATION: storage * 0.01}

        assert isinstance(ValidLoss(), LossRule)

    def test_class_without_calculate_does_not_satisfy(self):
        class NoCalculate:
            pass

        assert not isinstance(NoCalculate(), LossRule)

    def test_fake_loss_rule_satisfies_protocol(self, fake_loss_rule):
        assert isinstance(fake_loss_rule, LossRule)

    def test_calculate_returns_dict_with_loss_reasons(self):
        class ValidLoss:
            def calculate(self, storage: float, capacity: float, t: int, dt: float) -> dict[LossReason, float]:
                return {
                    EVAPORATION: storage * 0.01 * dt,
                    SEEPAGE: storage * 0.005 * dt,
                }

        rule = ValidLoss()
        result = rule.calculate(1000.0, 2000.0, 0, 1.0)
        assert result[EVAPORATION] == 10.0
        assert result[SEEPAGE] == 5.0


class TestProtocolNonSatisfaction:
    def test_class_with_wrong_method_name_does_not_satisfy_release_rule(self):
        class WrongMethod:
            def release_water(self, storage: float, capacity: float, inflow: float, t: int, dt: float) -> float:
                return 0.0

        assert not isinstance(WrongMethod(), ReleaseRule)

    def test_class_with_wrong_method_name_does_not_satisfy_split_strategy(self):
        class WrongMethod:
            def divide(self, amount: float, targets: list[str], t: int) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), SplitStrategy)

    def test_class_with_wrong_method_name_does_not_satisfy_loss_rule(self):
        class WrongMethod:
            def compute_loss(self, storage: float, capacity: float, t: int, dt: float) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), LossRule)

    def test_empty_class_satisfies_no_protocols(self):
        class EmptyClass:
            pass

        instance = EmptyClass()
        assert not isinstance(instance, ReleaseRule)
        assert not isinstance(instance, SplitStrategy)
        assert not isinstance(instance, LossRule)
