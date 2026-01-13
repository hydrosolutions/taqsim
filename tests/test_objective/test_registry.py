import pytest

from taqsim.objective.objective import Objective
from taqsim.objective.registry import ObjectiveRegistry


class TestObjectiveRegistry:
    def test_creates_with_direction(self) -> None:
        registry = ObjectiveRegistry(direction="minimize")
        assert registry.direction == "minimize"

        registry2 = ObjectiveRegistry(direction="maximize")
        assert registry2.direction == "maximize"

    def test_register_adds_factory(self) -> None:
        registry = ObjectiveRegistry(direction="minimize")

        def my_factory(x: str) -> Objective:
            return Objective(name=x, direction="minimize", evaluate=lambda s: 0.0)

        registry.register("my_obj", my_factory)

        assert "my_obj" in registry.list_available()

    def test_register_raises_on_duplicate(self) -> None:
        registry = ObjectiveRegistry(direction="minimize")

        def factory1(x: str) -> Objective:
            return Objective(name=x, direction="minimize", evaluate=lambda s: 0.0)

        def factory2(x: str) -> Objective:
            return Objective(name=x, direction="minimize", evaluate=lambda s: 1.0)

        registry.register("duplicate", factory1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("duplicate", factory2)

    def test_getattr_returns_factory(self) -> None:
        registry = ObjectiveRegistry(direction="maximize")

        def spill_factory(node_id: str) -> Objective:
            return Objective(
                name=f"{node_id}.spill",
                direction="minimize",
                evaluate=lambda s: 0.0,
            )

        registry.register("spill", spill_factory)

        factory = registry.spill
        assert factory is spill_factory

        obj = registry.spill("dam")
        assert obj.name == "dam.spill"

    def test_getattr_raises_for_unknown_with_helpful_message(self) -> None:
        registry = ObjectiveRegistry(direction="minimize")
        registry.register("spill", lambda: None)
        registry.register("deficit", lambda: None)

        with pytest.raises(AttributeError) as exc_info:
            _ = registry.unknown_objective

        error_msg = str(exc_info.value)
        assert "Unknown objective 'unknown_objective'" in error_msg
        assert "deficit" in error_msg
        assert "spill" in error_msg

    def test_list_available_returns_names(self) -> None:
        registry = ObjectiveRegistry(direction="minimize")

        assert registry.list_available() == []

        registry.register("alpha", lambda: None)
        registry.register("beta", lambda: None)
        registry.register("gamma", lambda: None)

        available = registry.list_available()
        assert "alpha" in available
        assert "beta" in available
        assert "gamma" in available
        assert len(available) == 3
