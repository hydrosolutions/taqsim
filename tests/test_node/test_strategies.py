from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node import NoLoss, NoRelease, NoSplit, Splitter, Storage
from taqsim.node.strategies import LossRule, ReleasePolicy, SplitPolicy
from taqsim.time import Frequency, Timestep

if TYPE_CHECKING:
    pass


class TestReleasePolicyProtocol:
    def test_class_with_release_satisfies_protocol(self):
        class ValidRelease:
            def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
                return node.storage * 0.5

        assert isinstance(ValidRelease(), ReleasePolicy)

    def test_class_without_release_does_not_satisfy(self):
        class NoRelease:
            pass

        assert not isinstance(NoRelease(), ReleasePolicy)

    def test_fake_release_policy_satisfies_protocol(self, fake_release_policy):
        assert isinstance(fake_release_policy, ReleasePolicy)

    def test_release_returns_float(self, fake_release_policy, fake_loss_rule):
        from taqsim.node.storage import Storage

        class ValidRelease:
            def release(self, node: "Storage", inflow: float, t: Timestep) -> float:
                return min(node.storage, inflow)

        storage = Storage(
            id="test",
            capacity=200.0,
            initial_storage=100.0,
            release_policy=fake_release_policy,
            loss_rule=fake_loss_rule,
        )
        rule = ValidRelease()
        result = rule.release(storage, 50.0, Timestep(0, Frequency.MONTHLY))
        assert result == 50.0


class TestSplitPolicyProtocol:
    def test_class_with_split_satisfies_protocol(self):
        class ValidSplit:
            def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
                return {target: amount / len(node.targets) for target in node.targets}

        assert isinstance(ValidSplit(), SplitPolicy)

    def test_class_without_split_does_not_satisfy(self):
        class NoSplit:
            pass

        assert not isinstance(NoSplit(), SplitPolicy)

    def test_fake_split_policy_satisfies_protocol(self, fake_split_policy):
        assert isinstance(fake_split_policy, SplitPolicy)

    def test_split_returns_dict(self, fake_split_policy):
        from taqsim.node.splitter import Splitter

        class ValidSplit:
            def split(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
                return {target: amount / len(node.targets) for target in node.targets}

        splitter = Splitter(id="test", split_policy=fake_split_policy)
        splitter._set_targets(["a", "b"])
        strategy = ValidSplit()
        result = strategy.split(splitter, 100.0, Timestep(0, Frequency.MONTHLY))
        assert result == {"a": 50.0, "b": 50.0}


class TestLossRuleProtocol:
    def test_class_with_calculate_satisfies_protocol(self):
        class ValidLoss:
            def calculate(self, node: "Storage", t: Timestep) -> dict[LossReason, float]:
                return {EVAPORATION: node.storage * 0.01}

        assert isinstance(ValidLoss(), LossRule)

    def test_class_without_calculate_does_not_satisfy(self):
        class NoCalculate:
            pass

        assert not isinstance(NoCalculate(), LossRule)

    def test_fake_loss_rule_satisfies_protocol(self, fake_loss_rule):
        assert isinstance(fake_loss_rule, LossRule)

    def test_calculate_returns_dict_with_loss_reasons(self, fake_release_policy, fake_loss_rule):
        from taqsim.node.storage import Storage

        class ValidLoss:
            def calculate(self, node: "Storage", t: Timestep) -> dict[LossReason, float]:
                return {
                    EVAPORATION: node.storage * 0.01,
                    SEEPAGE: node.storage * 0.005,
                }

        storage = Storage(
            id="test",
            capacity=2000.0,
            initial_storage=1000.0,
            release_policy=fake_release_policy,
            loss_rule=fake_loss_rule,
        )
        rule = ValidLoss()
        result = rule.calculate(storage, Timestep(0, Frequency.MONTHLY))
        assert result[EVAPORATION] == 10.0
        assert result[SEEPAGE] == 5.0


class TestProtocolNonSatisfaction:
    def test_class_with_wrong_method_name_does_not_satisfy_release_policy(self):
        class WrongMethod:
            def release_water(self, node: "Storage", inflow: float, t: Timestep) -> float:
                return 0.0

        assert not isinstance(WrongMethod(), ReleasePolicy)

    def test_class_with_wrong_method_name_does_not_satisfy_split_policy(self):
        class WrongMethod:
            def divide(self, node: "Splitter", amount: float, t: Timestep) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), SplitPolicy)

    def test_class_with_wrong_method_name_does_not_satisfy_loss_rule(self):
        class WrongMethod:
            def compute_loss(self, node: "Storage", t: Timestep) -> dict[str, float]:
                return {}

        assert not isinstance(WrongMethod(), LossRule)

    def test_empty_class_satisfies_no_protocols(self):
        class EmptyClass:
            pass

        instance = EmptyClass()
        assert not isinstance(instance, ReleasePolicy)
        assert not isinstance(instance, SplitPolicy)
        assert not isinstance(instance, LossRule)


class TestNoRelease:
    def test_satisfies_release_policy_protocol(self):
        assert isinstance(NoRelease(), ReleasePolicy)

    def test_is_frozen_dataclass(self):
        nr = NoRelease()
        with pytest.raises(FrozenInstanceError):
            nr.x = 1

    def test_raises_runtime_error(self):
        nr = NoRelease()
        node = Storage(id="dam", capacity=100.0, release_policy=nr, loss_rule=NoLoss())
        t = Timestep(index=0, frequency=Frequency.MONTHLY)
        with pytest.raises(RuntimeError, match="NoRelease is a placeholder"):
            nr.release(node, 50.0, t)

    def test_error_message_includes_node_id(self):
        nr = NoRelease()
        node = Storage(id="my_reservoir", capacity=100.0, release_policy=nr, loss_rule=NoLoss())
        t = Timestep(index=0, frequency=Frequency.MONTHLY)
        with pytest.raises(RuntimeError, match="my_reservoir"):
            nr.release(node, 50.0, t)


class TestNoSplit:
    def test_satisfies_split_policy_protocol(self):
        assert isinstance(NoSplit(), SplitPolicy)

    def test_is_frozen_dataclass(self):
        ns = NoSplit()
        with pytest.raises(FrozenInstanceError):
            ns.x = 1

    def test_raises_runtime_error(self):
        ns = NoSplit()
        node = Splitter(id="junction", split_policy=ns)
        t = Timestep(index=0, frequency=Frequency.MONTHLY)
        with pytest.raises(RuntimeError, match="NoSplit is a placeholder"):
            ns.split(node, 100.0, t)

    def test_error_message_includes_node_id(self):
        ns = NoSplit()
        node = Splitter(id="my_junction", split_policy=ns)
        t = Timestep(index=0, frequency=Frequency.MONTHLY)
        with pytest.raises(RuntimeError, match="my_junction"):
            ns.split(node, 100.0, t)
