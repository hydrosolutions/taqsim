from typing import Any

import pytest

from taqsim.common import CAPACITY_EXCEEDED, EVAPORATION, SEEPAGE, LossReason
from taqsim.edge.edge import Edge
from taqsim.edge.events import (
    WaterDelivered,
    WaterLost,
    WaterReceived,
)
from taqsim.time import Frequency, Timestep

from .conftest import FakeEdgeLossRule


def make_edge(
    id: str = "test_edge",
    source: str = "source_node",
    target: str = "target_node",
    capacity: float = 100.0,
    losses: dict[LossReason, float] | None = None,
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Edge:
    return Edge(
        id=id,
        source=source,
        target=target,
        capacity=capacity,
        loss_rule=FakeEdgeLossRule(losses),
        tags=tags or frozenset(),
        metadata=metadata or {},
    )


class TestEdgeInit:
    def test_edge_requires_id(self):
        with pytest.raises(ValueError, match="id cannot be empty"):
            make_edge(id="")

    def test_edge_requires_source(self):
        with pytest.raises(ValueError, match="source cannot be empty"):
            make_edge(source="")

    def test_edge_requires_target(self):
        with pytest.raises(ValueError, match="target cannot be empty"):
            make_edge(target="")

    def test_edge_requires_positive_capacity(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            make_edge(capacity=0.0)

    def test_edge_requires_positive_capacity_negative(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            make_edge(capacity=-10.0)

    def test_edge_requires_loss_rule(self):
        with pytest.raises(ValueError, match="loss_rule is required"):
            Edge(
                id="test",
                source="source",
                target="target",
                capacity=100.0,
                loss_rule=None,
            )

    def test_edge_creates_with_valid_params(self):
        edge = make_edge(
            id="my_edge",
            source="node_a",
            target="node_b",
            capacity=500.0,
        )
        assert edge.id == "my_edge"
        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.capacity == 500.0


class TestEdgeReceive:
    def test_receive_records_water_received_event(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))

        events = edge.events_of_type(WaterReceived)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].t == 0

    def test_receive_accumulates_water(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.receive(30.0, t=Timestep(0, Frequency.MONTHLY))

        events = edge.events_of_type(WaterReceived)
        assert len(events) == 2
        assert sum(e.amount for e in events) == 80.0

    def test_receive_returns_amount(self):
        edge = make_edge()
        result = edge.receive(75.0, t=Timestep(0, Frequency.MONTHLY))
        assert result == 75.0


class TestEdgeUpdate:
    def test_update_records_water_delivered(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        events = edge.events_of_type(WaterDelivered)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].t == 0

    def test_update_returns_delivered_amount(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        delivered = edge.update(t=Timestep(0, Frequency.MONTHLY))
        assert delivered == 50.0

    def test_update_resets_received(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        # Second update should deliver 0
        delivered = edge.update(t=Timestep(1, Frequency.MONTHLY))
        assert delivered == 0.0

        events = edge.events_of_type(WaterDelivered)
        t1_events = [e for e in events if e.t == 1]
        assert len(t1_events) == 1
        assert t1_events[0].amount == 0.0

    def test_update_capacity_exceeded_recorded(self):
        edge = make_edge(capacity=100.0)
        edge.receive(150.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        events = edge.events_of_type(WaterLost)
        capacity_exceeded_events = [e for e in events if e.reason == CAPACITY_EXCEEDED]
        assert len(capacity_exceeded_events) == 1
        assert capacity_exceeded_events[0].amount == 50.0
        assert capacity_exceeded_events[0].t == 0

    def test_update_clamps_to_capacity(self):
        edge = make_edge(capacity=100.0)
        edge.receive(150.0, t=Timestep(0, Frequency.MONTHLY))
        delivered = edge.update(t=Timestep(0, Frequency.MONTHLY))

        # Delivered should be clamped to capacity (no losses in this test)
        assert delivered == 100.0

    def test_update_calculates_losses(self):
        edge = make_edge(losses={EVAPORATION: 10.0})
        edge.receive(100.0, t=Timestep(0, Frequency.MONTHLY))
        delivered = edge.update(t=Timestep(0, Frequency.MONTHLY))

        assert delivered == 90.0

    def test_update_records_water_lost_events(self):
        edge = make_edge(losses={EVAPORATION: 10.0, SEEPAGE: 5.0})
        edge.receive(100.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        events = edge.events_of_type(WaterLost)
        assert len(events) == 2
        reasons = {e.reason: e.amount for e in events}
        assert reasons[EVAPORATION] == 10.0
        assert reasons[SEEPAGE] == 5.0

    def test_update_scales_losses_if_exceed_water(self):
        # Losses (80 + 20 = 100) exceed water (10)
        edge = make_edge(losses={EVAPORATION: 80.0, SEEPAGE: 20.0})
        edge.receive(10.0, t=Timestep(0, Frequency.MONTHLY))
        delivered = edge.update(t=Timestep(0, Frequency.MONTHLY))

        # All water lost, nothing delivered
        assert delivered == 0.0

        events = edge.events_of_type(WaterLost)
        evap = next(e for e in events if e.reason == EVAPORATION)
        seep = next(e for e in events if e.reason == SEEPAGE)

        # Scaled proportionally: 80% evap, 20% seepage
        assert evap.amount == pytest.approx(8.0)
        assert seep.amount == pytest.approx(2.0)

    def test_update_mass_balance(self):
        # Test: delivered + losses (including capacity exceeded) = received
        losses = {EVAPORATION: 10.0, SEEPAGE: 5.0}
        edge = make_edge(capacity=100.0, losses=losses)

        received = 120.0
        edge.receive(received, t=Timestep(0, Frequency.MONTHLY))
        delivered = edge.update(t=Timestep(0, Frequency.MONTHLY))

        # excess = 120 - 100 = 20 (recorded as WaterLost with CAPACITY_EXCEEDED)
        # received after clamping = 100
        # total_loss from rules = 10 + 5 = 15
        # delivered = 100 - 15 = 85

        loss_events = edge.events_of_type(WaterLost)
        total_loss = sum(e.amount for e in loss_events)
        capacity_exceeded_loss = sum(e.amount for e in loss_events if e.reason == CAPACITY_EXCEEDED)

        # Mass balance: delivered + all losses = received
        assert delivered + total_loss == received
        assert capacity_exceeded_loss == 20.0
        assert delivered == 85.0


class TestEdgeEventRecording:
    def test_events_of_type_returns_correct_events(self):
        edge = make_edge(losses={EVAPORATION: 5.0})
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        received_events = edge.events_of_type(WaterReceived)
        lost_events = edge.events_of_type(WaterLost)
        delivered_events = edge.events_of_type(WaterDelivered)

        assert len(received_events) == 1
        assert len(lost_events) == 1
        assert len(delivered_events) == 1

    def test_clear_events_removes_all(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        assert len(edge.events) > 0
        edge.clear_events()
        assert len(edge.events) == 0

    def test_record_appends_event(self):
        edge = make_edge()
        event = WaterReceived(amount=42.0, t=99)
        edge.record(event)

        assert event in edge.events


class TestEdgeTrace:
    def test_trace_returns_trace_object(self):
        from taqsim.objective import Trace

        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        trace = edge.trace(WaterDelivered)
        assert isinstance(trace, Trace)

    def test_trace_extracts_delivered_events(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))
        edge.receive(30.0, t=Timestep(1, Frequency.MONTHLY))
        edge.update(t=Timestep(1, Frequency.MONTHLY))

        trace = edge.trace(WaterDelivered)
        assert trace.to_dict() == {0: 50.0, 1: 30.0}

    def test_trace_extracts_received_events(self):
        edge = make_edge()
        edge.receive(50.0, t=Timestep(0, Frequency.MONTHLY))
        edge.receive(30.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        trace = edge.trace(WaterReceived)
        assert trace[0] == 80.0

    def test_trace_extracts_lost_events(self):
        edge = make_edge(losses={EVAPORATION: 10.0})
        edge.receive(100.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        trace = edge.trace(WaterLost)
        assert trace[0] == 10.0

    def test_trace_with_custom_field(self):
        edge = make_edge(losses={EVAPORATION: 10.0})
        edge.receive(100.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        trace = edge.trace(WaterDelivered, field="amount")
        assert trace[0] == 90.0

    def test_trace_empty_when_no_events(self):
        edge = make_edge()

        trace = edge.trace(WaterDelivered)
        assert len(trace) == 0

    def test_trace_aggregates_events_at_same_timestep(self):
        edge = make_edge(losses={EVAPORATION: 5.0, SEEPAGE: 3.0})
        edge.receive(100.0, t=Timestep(0, Frequency.MONTHLY))
        edge.update(t=Timestep(0, Frequency.MONTHLY))

        trace = edge.trace(WaterLost)
        assert trace[0] == 8.0


class TestEdgeProtocolCompliance:
    def test_edge_has_record_method(self):
        edge = make_edge()
        assert callable(getattr(edge, "record", None))

    def test_edge_has_events_of_type_method(self):
        edge = make_edge()
        assert callable(getattr(edge, "events_of_type", None))

    def test_edge_has_trace_method(self):
        edge = make_edge()
        assert callable(getattr(edge, "trace", None))


class TestEdgeTags:
    def test_default_tags_is_empty_frozenset(self):
        edge = make_edge()
        assert edge.tags == frozenset()

    def test_custom_tags_accepted(self):
        edge = make_edge(tags=frozenset({"canal", "primary"}))
        assert edge.tags == frozenset({"canal", "primary"})

    def test_tags_type_is_frozenset(self):
        edge = make_edge(tags=frozenset({"river"}))
        assert isinstance(edge.tags, frozenset)

    def test_tags_is_immutable(self):
        edge = make_edge(tags=frozenset({"original"}))
        with pytest.raises(AttributeError):
            edge.tags.add("new")  # frozenset has no add method


class TestEdgeMetadata:
    def test_default_metadata_is_empty_dict(self):
        edge = make_edge()
        assert edge.metadata == {}

    def test_custom_metadata_accepted(self):
        edge = make_edge(metadata={"length_km": 50.5, "material": "concrete"})
        assert edge.metadata == {"length_km": 50.5, "material": "concrete"}

    def test_metadata_type_is_dict(self):
        edge = make_edge(metadata={"key": "value"})
        assert isinstance(edge.metadata, dict)

    def test_metadata_not_shared_between_instances(self):
        e1 = make_edge(id="e1")
        e2 = make_edge(id="e2")
        e1.metadata["key"] = "value"
        assert "key" not in e2.metadata
