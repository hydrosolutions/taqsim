import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.edge.edge import Edge
from taqsim.edge.events import (
    CapacityExceeded,
    FlowDelivered,
    FlowLost,
    FlowReceived,
    RequirementUnmet,
)
from taqsim.node.timeseries import TimeSeries

from .conftest import FakeEdgeLossRule


def make_edge(
    id: str = "test_edge",
    source: str = "source_node",
    target: str = "target_node",
    capacity: float = 100.0,
    requirement: TimeSeries | None = None,
    losses: dict[LossReason, float] | None = None,
) -> Edge:
    return Edge(
        id=id,
        source=source,
        target=target,
        capacity=capacity,
        requirement=requirement,
        loss_rule=FakeEdgeLossRule(losses),
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

    def test_edge_requirement_optional(self):
        edge = make_edge()
        assert edge.requirement is None


class TestEdgeReceive:
    def test_receive_records_flow_received_event(self):
        edge = make_edge()
        edge.receive(50.0, t=0)

        events = edge.events_of_type(FlowReceived)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].t == 0

    def test_receive_accumulates_flow(self):
        edge = make_edge()
        edge.receive(50.0, t=0)
        edge.receive(30.0, t=0)

        events = edge.events_of_type(FlowReceived)
        assert len(events) == 2
        assert sum(e.amount for e in events) == 80.0

    def test_receive_returns_amount(self):
        edge = make_edge()
        result = edge.receive(75.0, t=0)
        assert result == 75.0


class TestEdgeUpdate:
    def test_update_records_flow_delivered(self):
        edge = make_edge()
        edge.receive(50.0, t=0)
        edge.update(t=0, dt=1.0)

        events = edge.events_of_type(FlowDelivered)
        assert len(events) == 1
        assert events[0].amount == 50.0
        assert events[0].t == 0

    def test_update_returns_delivered_amount(self):
        edge = make_edge()
        edge.receive(50.0, t=0)
        delivered = edge.update(t=0, dt=1.0)
        assert delivered == 50.0

    def test_update_resets_received(self):
        edge = make_edge()
        edge.receive(50.0, t=0)
        edge.update(t=0, dt=1.0)

        # Second update should deliver 0
        delivered = edge.update(t=1, dt=1.0)
        assert delivered == 0.0

        events = edge.events_of_type(FlowDelivered)
        t1_events = [e for e in events if e.t == 1]
        assert len(t1_events) == 1
        assert t1_events[0].amount == 0.0

    def test_update_capacity_exceeded_recorded(self):
        edge = make_edge(capacity=100.0)
        edge.receive(150.0, t=0)
        edge.update(t=0, dt=1.0)

        events = edge.events_of_type(CapacityExceeded)
        assert len(events) == 1
        assert events[0].excess == 50.0
        assert events[0].t == 0

    def test_update_clamps_to_capacity(self):
        edge = make_edge(capacity=100.0)
        edge.receive(150.0, t=0)
        delivered = edge.update(t=0, dt=1.0)

        # Delivered should be clamped to capacity (no losses in this test)
        assert delivered == 100.0

    def test_update_calculates_losses(self):
        edge = make_edge(losses={EVAPORATION: 10.0})
        edge.receive(100.0, t=0)
        delivered = edge.update(t=0, dt=1.0)

        assert delivered == 90.0

    def test_update_records_flow_lost_events(self):
        edge = make_edge(losses={EVAPORATION: 10.0, SEEPAGE: 5.0})
        edge.receive(100.0, t=0)
        edge.update(t=0, dt=1.0)

        events = edge.events_of_type(FlowLost)
        assert len(events) == 2
        reasons = {e.reason: e.amount for e in events}
        assert reasons[EVAPORATION] == 10.0
        assert reasons[SEEPAGE] == 5.0

    def test_update_scales_losses_if_exceed_flow(self):
        # Losses (80 + 20 = 100) exceed flow (10)
        edge = make_edge(losses={EVAPORATION: 80.0, SEEPAGE: 20.0})
        edge.receive(10.0, t=0)
        delivered = edge.update(t=0, dt=1.0)

        # All flow lost, nothing delivered
        assert delivered == 0.0

        events = edge.events_of_type(FlowLost)
        evap = next(e for e in events if e.reason == EVAPORATION)
        seep = next(e for e in events if e.reason == SEEPAGE)

        # Scaled proportionally: 80% evap, 20% seepage
        assert evap.amount == pytest.approx(8.0)
        assert seep.amount == pytest.approx(2.0)

    def test_update_requirement_unmet_recorded(self):
        requirement = TimeSeries([100.0] * 12)
        edge = make_edge(requirement=requirement)
        edge.receive(50.0, t=0)
        edge.update(t=0, dt=1.0)

        events = edge.events_of_type(RequirementUnmet)
        assert len(events) == 1
        assert events[0].required == 100.0
        assert events[0].actual == 50.0
        assert events[0].deficit == 50.0
        assert events[0].t == 0

    def test_update_no_requirement_unmet_when_met(self):
        requirement = TimeSeries([50.0] * 12)
        edge = make_edge(requirement=requirement)
        edge.receive(100.0, t=0)
        edge.update(t=0, dt=1.0)

        events = edge.events_of_type(RequirementUnmet)
        assert len(events) == 0

    def test_update_mass_balance(self):
        # Test: delivered + losses + excess = received
        losses = {EVAPORATION: 10.0, SEEPAGE: 5.0}
        edge = make_edge(capacity=100.0, losses=losses)

        received = 120.0
        edge.receive(received, t=0)
        delivered = edge.update(t=0, dt=1.0)

        # excess = 120 - 100 = 20
        # received after clamping = 100
        # total_loss = 10 + 5 = 15
        # delivered = 100 - 15 = 85

        excess_events = edge.events_of_type(CapacityExceeded)
        excess = excess_events[0].excess if excess_events else 0.0

        loss_events = edge.events_of_type(FlowLost)
        total_loss = sum(e.amount for e in loss_events)

        # Mass balance: after_clamping = capacity, delivered + losses = after_clamping
        assert delivered + total_loss == 100.0
        assert excess == 20.0
        assert delivered == 85.0


class TestEdgeEventRecording:
    def test_events_of_type_returns_correct_events(self):
        edge = make_edge(losses={EVAPORATION: 5.0})
        edge.receive(50.0, t=0)
        edge.update(t=0, dt=1.0)

        received_events = edge.events_of_type(FlowReceived)
        lost_events = edge.events_of_type(FlowLost)
        delivered_events = edge.events_of_type(FlowDelivered)

        assert len(received_events) == 1
        assert len(lost_events) == 1
        assert len(delivered_events) == 1

    def test_clear_events_removes_all(self):
        edge = make_edge()
        edge.receive(50.0, t=0)
        edge.update(t=0, dt=1.0)

        assert len(edge.events) > 0
        edge.clear_events()
        assert len(edge.events) == 0

    def test_record_appends_event(self):
        edge = make_edge()
        event = FlowReceived(amount=42.0, t=99)
        edge.record(event)

        assert event in edge.events


class TestEdgeProtocolCompliance:
    def test_edge_has_record_method(self):
        edge = make_edge()
        assert callable(getattr(edge, "record", None))

    def test_edge_has_events_of_type_method(self):
        edge = make_edge()
        assert callable(getattr(edge, "events_of_type", None))
