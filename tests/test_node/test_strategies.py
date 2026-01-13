from typing import TYPE_CHECKING

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node.strategies import LossRule, ReleaseRule, SplitRule

if TYPE_CHECKING:
    from taqsim.node.splitter import Splitter
    from taqsim.node.storage import Storage


class TestReleaseRuleProtocol:
    def test_class_with_release_satisfies_protocol(self):
        class ValidRelease:
            def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
                return node.storage * 0.5

        assert isinstance(ValidRelease(), ReleaseRule)

    def test_class_without_release_does_not_satisfy(self):
        class NoRelease:
            pass

        assert not isinstance(NoRelease(), ReleaseRule)

    def test_fake_release_rule_satisfies_protocol(self, fake_release_rule):
        assert isinstance(fake_release_rule, ReleaseRule)

    def test_release_returns_float(self, fake_release_rule, fake_loss_rule):
        from taqsim.node.storage import Storage

        class ValidRelease:
            def release(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
                return min(node.storage, inflow)

        storage = Storage(
            id="test",
            capacity=200.0,
            initial_storage=100.0,
            release_rule=fake_release_rule,
            loss_rule=fake_loss_rule,
        )
        rule = ValidRelease()
        result = rule.release(storage, 50.0, 0, 1.0)
        assert result == 50.0


class TestSplitStrategyProtocol:
    def test_class_with_split_satisfies_protocol(self):
        class ValidSplit:
            def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
                return {target: amount / len(node.targets) for target in node.targets}

        assert isinstance(ValidSplit(), SplitRule)

    def test_class_without_split_does_not_satisfy(self):
        class NoSplit:
            pass

        assert not isinstance(NoSplit(), SplitRule)

    def test_fake_split_rule_satisfies_protocol(self, fake_split_rule):
        assert isinstance(fake_split_rule, SplitRule)

    def test_split_returns_dict(self, fake_split_rule):
        from taqsim.node.splitter import Splitter

        class ValidSplit:
            def split(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
                return {target: amount / len(node.targets) for target in node.targets}

        splitter = Splitter(id="test", split_rule=fake_split_rule)
        splitter._set_targets(["a", "b"])
        strategy = ValidSplit()
        result = strategy.split(splitter, 100.0, 0)
        assert result == {"a": 50.0, "b": 50.0}


class TestLossRuleProtocol:
    def test_class_with_calculate_satisfies_protocol(self):
        class ValidLoss:
            def calculate(self, node: "Storage", t: int, dt: float) -> dict[LossReason, float]:
                return {EVAPORATION: node.storage * 0.01}

        assert isinstance(ValidLoss(), LossRule)

    def test_class_without_calculate_does_not_satisfy(self):
        class NoCalculate:
            pass

        assert not isinstance(NoCalculate(), LossRule)

    def test_fake_loss_rule_satisfies_protocol(self, fake_loss_rule):
        assert isinstance(fake_loss_rule, LossRule)

    def test_calculate_returns_dict_with_loss_reasons(self, fake_release_rule, fake_loss_rule):
        from taqsim.node.storage import Storage

        class ValidLoss:
            def calculate(self, node: "Storage", t: int, dt: float) -> dict[LossReason, float]:
                return {
                    EVAPORATION: node.storage * 0.01 * dt,
                    SEEPAGE: node.storage * 0.005 * dt,
                }

        storage = Storage(
            id="test",
            capacity=2000.0,
            initial_storage=1000.0,
            release_rule=fake_release_rule,
            loss_rule=fake_loss_rule,
        )
        rule = ValidLoss()
        result = rule.calculate(storage, 0, 1.0)
        assert result[EVAPORATION] == 10.0
        assert result[SEEPAGE] == 5.0


class TestProtocolNonSatisfaction:
    def test_class_with_wrong_method_name_does_not_satisfy_release_rule(self):
        class WrongMethod:
            def release_water(self, node: "Storage", inflow: float, t: int, dt: float) -> float:
                return 0.0

        assert not isinstance(WrongMethod(), ReleaseRule)

    def test_class_with_wrong_method_name_does_not_satisfy_split_rule(self):
        class WrongMethod:
            def divide(self, node: "Splitter", amount: float, t: int) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), SplitRule)

    def test_class_with_wrong_method_name_does_not_satisfy_loss_rule(self):
        class WrongMethod:
            def compute_loss(self, node: "Storage", t: int, dt: float) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), LossRule)

    def test_empty_class_satisfies_no_protocols(self):
        class EmptyClass:
            pass

        instance = EmptyClass()
        assert not isinstance(instance, ReleaseRule)
        assert not isinstance(instance, SplitRule)
        assert not isinstance(instance, LossRule)
