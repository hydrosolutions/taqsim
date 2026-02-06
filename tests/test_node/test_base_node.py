import pytest

from taqsim.node.base import BaseNode
from taqsim.node.events import WaterGenerated, WaterReceived, WaterStored
from taqsim.time import Frequency, Timestep


class TestBaseNodeInit:
    def test_creates_with_id(self):
        node = BaseNode(id="test_node")
        assert node.id == "test_node"

    def test_starts_with_empty_events(self):
        node = BaseNode(id="test")
        assert node.events == []

    def test_id_is_stored_correctly(self):
        node = BaseNode(id="unique_identifier_123")
        assert node.id == "unique_identifier_123"

    def test_starts_with_empty_targets(self):
        node = BaseNode(id="test")
        assert node.targets == []


class TestTargetManagement:
    def test_targets_property_returns_list(self):
        node = BaseNode(id="test")
        assert isinstance(node.targets, list)

    def test_set_targets_populates_targets(self):
        node = BaseNode(id="test")
        node._set_targets(["target_a", "target_b"])
        assert node.targets == ["target_a", "target_b"]

    def test_set_targets_replaces_existing_targets(self):
        node = BaseNode(id="test")
        node._set_targets(["old_target"])
        node._set_targets(["new_target_1", "new_target_2"])
        assert node.targets == ["new_target_1", "new_target_2"]

    def test_set_targets_with_empty_list(self):
        node = BaseNode(id="test")
        node._set_targets(["target_a"])
        node._set_targets([])
        assert node.targets == []

    def test_targets_not_included_in_init(self):
        node = BaseNode(id="test")
        assert not hasattr(node, "targets") or node.targets == []

    def test_internal_targets_field_is_private(self):
        node = BaseNode(id="test")
        node._set_targets(["t1"])
        assert node._targets == ["t1"]


class TestEventRecording:
    def test_record_appends_event(self):
        node = BaseNode(id="test")
        event = WaterGenerated(amount=100.0, t=0)
        node.record(event)
        assert len(node.events) == 1
        assert node.events[0] is event

    def test_record_multiple_events(self):
        node = BaseNode(id="test")
        event1 = WaterGenerated(amount=100.0, t=0)
        event2 = WaterReceived(amount=50.0, source_id="up", t=1)
        event3 = WaterStored(amount=75.0, t=2)

        node.record(event1)
        node.record(event2)
        node.record(event3)

        assert len(node.events) == 3
        assert node.events[0] is event1
        assert node.events[1] is event2
        assert node.events[2] is event3

    def test_events_at_filters_by_timestep(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=0))
        node.record(WaterGenerated(amount=50.0, t=1))
        node.record(WaterReceived(amount=25.0, source_id="up", t=0))

        events_t0 = node.events_at(0)
        assert len(events_t0) == 2

        events_t1 = node.events_at(1)
        assert len(events_t1) == 1

    def test_events_at_returns_empty_for_nonexistent_timestep(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=0))

        events_t5 = node.events_at(5)
        assert events_t5 == []

    def test_events_at_returns_all_events_at_timestep(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=2))
        node.record(WaterReceived(amount=50.0, source_id="a", t=2))
        node.record(WaterStored(amount=75.0, t=2))
        node.record(WaterGenerated(amount=25.0, t=3))

        events_t2 = node.events_at(2)
        assert len(events_t2) == 3

    def test_events_of_type_filters_by_class(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=0))
        node.record(WaterReceived(amount=50.0, source_id="up", t=0))
        node.record(WaterGenerated(amount=75.0, t=1))

        generated = node.events_of_type(WaterGenerated)
        assert len(generated) == 2
        assert all(isinstance(e, WaterGenerated) for e in generated)

    def test_events_of_type_returns_empty_for_absent_type(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=0))

        stored = node.events_of_type(WaterStored)
        assert stored == []

    def test_events_of_type_returns_all_matching_events(self):
        node = BaseNode(id="test")
        node.record(WaterReceived(amount=10.0, source_id="a", t=0))
        node.record(WaterReceived(amount=20.0, source_id="b", t=1))
        node.record(WaterReceived(amount=30.0, source_id="c", t=2))
        node.record(WaterGenerated(amount=100.0, t=0))

        received = node.events_of_type(WaterReceived)
        assert len(received) == 3
        assert all(isinstance(e, WaterReceived) for e in received)


class TestClearEvents:
    def test_removes_all_events(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=0))
        node.record(WaterGenerated(amount=50.0, t=1))

        node.clear_events()
        assert node.events == []

    def test_clear_allows_new_events_after(self):
        node = BaseNode(id="test")
        node.record(WaterGenerated(amount=100.0, t=0))
        node.clear_events()

        node.record(WaterGenerated(amount=200.0, t=5))
        assert len(node.events) == 1
        assert node.events[0].amount == 200.0

    def test_clear_on_empty_events_is_safe(self):
        node = BaseNode(id="test")
        node.clear_events()
        assert node.events == []


class TestUpdateNotImplemented:
    def test_raises_not_implemented(self):
        node = BaseNode(id="test")
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            node.update(t=Timestep(0, Frequency.MONTHLY))

    def test_raises_not_implemented_with_different_params(self):
        node = BaseNode(id="test")
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            node.update(t=Timestep(10, Frequency.MONTHLY))


class TestBaseNodeInheritance:
    def test_subclass_can_override_update(self):
        class ConcreteNode(BaseNode):
            def __init__(self, id: str):
                super().__init__(id=id)
                self.updated = False

            def update(self, t: Timestep) -> None:
                self.updated = True

        node = ConcreteNode(id="concrete")
        node.update(t=Timestep(0, Frequency.MONTHLY))
        assert node.updated is True

    def test_subclass_inherits_event_methods(self):
        class ConcreteNode(BaseNode):
            def update(self, t: Timestep) -> None:
                self.record(WaterGenerated(amount=50.0, t=t))

        node = ConcreteNode(id="concrete")
        node.update(t=Timestep(3, Frequency.MONTHLY))

        assert len(node.events) == 1
        assert node.events[0].t == 3


class TestBaseNodeTags:
    def test_default_tags_is_empty_frozenset(self):
        node = BaseNode(id="test")
        assert node.tags == frozenset()

    def test_custom_tags_accepted(self):
        node = BaseNode(id="test", tags=frozenset({"irrigation", "primary"}))
        assert node.tags == frozenset({"irrigation", "primary"})

    def test_tags_type_is_frozenset(self):
        node = BaseNode(id="test", tags=frozenset({"tag1"}))
        assert isinstance(node.tags, frozenset)

    def test_tags_is_immutable(self):
        node = BaseNode(id="test", tags=frozenset({"original"}))
        with pytest.raises(AttributeError):
            node.tags.add("new")  # frozenset has no add method


class TestBaseNodeMetadata:
    def test_default_metadata_is_empty_dict(self):
        node = BaseNode(id="test")
        assert node.metadata == {}

    def test_custom_metadata_accepted(self):
        node = BaseNode(id="test", metadata={"priority": 1, "color": "blue"})
        assert node.metadata == {"priority": 1, "color": "blue"}

    def test_metadata_type_is_dict(self):
        node = BaseNode(id="test", metadata={"key": "value"})
        assert isinstance(node.metadata, dict)

    def test_metadata_not_shared_between_instances(self):
        n1 = BaseNode(id="n1")
        n2 = BaseNode(id="n2")
        n1.metadata["key"] = "value"
        assert "key" not in n2.metadata


class TestBaseNodeInheritanceTags:
    """Verify all node subclasses inherit tags and metadata."""

    def test_source_inherits_tags(self):
        from taqsim.node import Source
        from taqsim.node.timeseries import TimeSeries

        source = Source(
            id="test",
            inflow=TimeSeries([100.0]),
            tags=frozenset({"upstream"}),
            metadata={"capacity": 1000},
        )
        assert source.tags == frozenset({"upstream"})
        assert source.metadata == {"capacity": 1000}

    def test_storage_inherits_tags(self):
        from taqsim.node import Storage

        # Storage requires release_rule and loss_rule - create minimal fakes
        class FakeReleaseRule:
            def release(self, node, inflow, t, dt):
                return 0.0

        class FakeLossRule:
            def calculate(self, node, t, dt):
                return {}

        storage = Storage(
            id="test",
            capacity=1000.0,
            release_rule=FakeReleaseRule(),
            loss_rule=FakeLossRule(),
            tags=frozenset({"reservoir"}),
            metadata={"built_year": 2020},
        )
        assert storage.tags == frozenset({"reservoir"})
        assert storage.metadata == {"built_year": 2020}

    def test_sink_inherits_tags(self):
        from taqsim.node import Sink

        sink = Sink(id="test", tags=frozenset({"terminal"}), metadata={"type": "ocean"})
        assert sink.tags == frozenset({"terminal"})
        assert sink.metadata == {"type": "ocean"}
