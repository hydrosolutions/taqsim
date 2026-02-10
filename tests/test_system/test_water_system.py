from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from taqsim.common import EVAPORATION, SEEPAGE, LossReason
from taqsim.node import WaterReceived
from taqsim.node.events import WaterGenerated, WaterLost
from taqsim.system import MissingAuxiliaryDataError, ValidationError, WaterSystem
from taqsim.testing import NoRouting, ProportionalReachLoss, make_reach
from taqsim.time import Frequency, Timestep

if TYPE_CHECKING:
    from taqsim.node import Reach, Storage

from .conftest import (
    make_edge,
    make_sink,
    make_source,
    make_splitter,
    make_storage,
)


class TestWaterSystemInit:
    def test_creates_with_frequency(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.frequency == Frequency.MONTHLY

    def test_creates_with_custom_frequency(self):
        system = WaterSystem(frequency=Frequency.DAILY)
        assert system.frequency == Frequency.DAILY

    def test_starts_with_empty_nodes(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.nodes == {}

    def test_starts_with_empty_edges(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.edges == {}

    def test_start_date_defaults_to_none(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        assert system.start_date is None

    def test_start_date_accepts_date(self):
        system = WaterSystem(frequency=Frequency.MONTHLY, start_date=date(2024, 1, 1))
        assert system.start_date == date(2024, 1, 1)


class TestAddNode:
    def test_adds_source_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source(id="src1")
        system.add_node(source)

        assert "src1" in system.nodes
        assert system.nodes["src1"] is source

    def test_adds_sink_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        sink = make_sink(id="snk1")
        system.add_node(sink)

        assert "snk1" in system.nodes
        assert system.nodes["snk1"] is sink

    def test_raises_on_duplicate_node_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source(id="node1")
        system.add_node(source)

        with pytest.raises(ValueError, match="Node 'node1' already exists"):
            system.add_node(make_sink(id="node1"))

    def test_invalidates_after_add(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.validate()

        assert system._validated is True

        # Adding node should invalidate
        system.add_node(make_sink(id="sink2"))
        assert system._validated is False


class TestAddEdge:
    def test_adds_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        edge = make_edge(id="e1", source="a", target="b")
        system.add_edge(edge)

        assert "e1" in system.edges
        assert system.edges["e1"] is edge

    def test_raises_on_duplicate_edge_id(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        edge1 = make_edge(id="edge1", source="a", target="b")
        edge2 = make_edge(id="edge1", source="c", target="d")

        system.add_edge(edge1)

        with pytest.raises(ValueError, match="Edge 'edge1' already exists"):
            system.add_edge(edge2)

    def test_invalidates_after_add(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.validate()

        assert system._validated is True

        # Adding edge should invalidate
        system.add_edge(make_edge(id="edge2", source="source", target="sink"))
        assert system._validated is False


class TestValidation:
    def test_validates_simple_network(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        system.add_edge(make_edge(source="source", target="sink"))

        system.validate()
        assert system._validated is True

    def test_fails_on_missing_source_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_sink())
        system.add_edge(make_edge(source="nonexistent", target="sink"))

        with pytest.raises(ValidationError, match="source node 'nonexistent' does not exist"):
            system.validate()

    def test_fails_on_missing_target_node(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_edge(make_edge(source="source", target="nonexistent"))

        with pytest.raises(ValidationError, match="target node 'nonexistent' does not exist"):
            system.validate()

    def test_fails_on_cycle(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        splitter1 = make_splitter(id="s1")
        splitter2 = make_splitter(id="s2")
        sink = make_sink()

        system.add_node(splitter1)
        system.add_node(splitter2)
        system.add_node(sink)

        # Create cycle: s1 -> s2 -> s1 (and s2 -> sink)
        system.add_edge(make_edge(id="e1", source="s1", target="s2"))
        system.add_edge(make_edge(id="e2", source="s2", target="s1"))
        system.add_edge(make_edge(id="e3", source="s2", target="sink"))

        with pytest.raises(ValidationError, match="contains cycles"):
            system.validate()

    def test_fails_on_disconnected_nodes(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source(id="src1"))
        system.add_node(make_sink(id="snk1"))
        system.add_node(make_source(id="src2"))  # disconnected
        system.add_node(make_sink(id="snk2"))  # disconnected

        system.add_edge(make_edge(id="e1", source="src1", target="snk1"))
        system.add_edge(make_edge(id="e2", source="src2", target="snk2"))

        with pytest.raises(ValidationError, match="not connected"):
            system.validate()

    def test_fails_on_source_with_incoming_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink = make_sink()

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink)

        # Edge going into source (invalid)
        system.add_edge(make_edge(id="e1", source="splitter", target="source"))
        system.add_edge(make_edge(id="e2", source="source", target="sink"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink"))

        with pytest.raises(ValidationError, match="Source 'source' must have in_degree=0"):
            system.validate()

    def test_fails_on_sink_with_outgoing_edge(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        other_sink = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(sink)
        system.add_node(other_sink)

        system.add_edge(make_edge(id="e1", source="source", target="sink"))
        system.add_edge(make_edge(id="e2", source="sink", target="sink2"))

        with pytest.raises(ValidationError, match="Sink 'sink' must have out_degree=0"):
            system.validate()

    def test_fails_on_non_splitter_with_multiple_outputs(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(sink1)
        system.add_node(sink2)

        # Source (non-splitter) with two outputs
        system.add_edge(make_edge(id="e1", source="source", target="sink1"))
        system.add_edge(make_edge(id="e2", source="source", target="sink2"))

        with pytest.raises(ValidationError, match="must have exactly 1 outgoing edge"):
            system.validate()

    def test_allows_splitter_with_multiple_outputs(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.validate()
        assert system._validated is True

    def test_fails_on_node_without_path_to_sink(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        storage = make_storage(id="storage")
        sink = make_sink()

        system.add_node(source)
        system.add_node(storage)
        system.add_node(sink)

        # Source -> sink, but storage is orphaned (no edges from/to it)
        # This will fail at "not connected" first
        system.add_edge(make_edge(id="e1", source="source", target="sink"))

        with pytest.raises(ValidationError, match="not connected"):
            system.validate()

    def test_populates_targets_on_validate(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(id="edge1", source="source", target="sink"))

        system.validate()

        # Source should have sink (target node ID) in its targets
        assert "sink" in source.targets


class TestSimulation:
    def test_runs_for_given_timesteps(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        system.simulate(timesteps=3)

        # Check source generated water at each timestep
        gen_events = source.events_of_type(WaterGenerated)
        assert len(gen_events) == 3
        assert [e.t for e in gen_events] == [0, 1, 2]

    def test_validates_if_not_validated(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        assert system._validated is False
        system.simulate(timesteps=1)
        assert system._validated is True

    def test_routes_water_through_network(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)

        system.simulate(timesteps=1)

        # Sink should have received water
        received_events = sink.events_of_type(WaterReceived)
        assert len(received_events) == 1
        assert received_events[0].t == 0

    def test_processes_in_topological_order(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        storage = make_storage()
        sink = make_sink()

        system.add_node(source)
        system.add_node(storage)
        system.add_node(sink)

        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))

        system.simulate(timesteps=1)

        # All nodes should have been processed
        source_gen = source.events_of_type(WaterGenerated)
        storage_recv = storage.events_of_type(WaterReceived)
        sink_recv = sink.events_of_type(WaterReceived)

        assert len(source_gen) == 1
        assert len(storage_recv) == 1
        assert len(sink_recv) == 1

    def test_splitter_distributes_to_multiple_targets(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.simulate(timesteps=1)

        # Both sinks should have received water
        sink1_recv = sink1.events_of_type(WaterReceived)
        sink2_recv = sink2.events_of_type(WaterReceived)

        assert len(sink1_recv) == 1
        assert len(sink2_recv) == 1


class TestSplitterRouting:
    """Regression tests for splitter routing with node IDs."""

    def test_splitter_routes_water_to_downstream_nodes(self):
        """SplitPolicy returning node IDs routes water correctly."""

        class ProportionalSplit:
            def split(self, node, amount: float, t: Timestep) -> dict[str, float]:
                return {"sink1": amount * 0.6, "sink2": amount * 0.4}

        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter(split_policy=ProportionalSplit())
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.simulate(timesteps=1)

        # sink1 should receive 60% (60.0)
        sink1_recv = sink1.events_of_type(WaterReceived)
        assert len(sink1_recv) == 1
        assert sink1_recv[0].amount == 60.0

        # sink2 should receive 40% (40.0)
        sink2_recv = sink2.events_of_type(WaterReceived)
        assert len(sink2_recv) == 1
        assert sink2_recv[0].amount == 40.0

    def test_splitter_node_targets_are_node_ids(self):
        """After validation, node.targets contains downstream node IDs."""
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter()
        sink1 = make_sink(id="sink1")
        sink2 = make_sink(id="sink2")

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink1)
        system.add_node(sink2)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink1"))
        system.add_edge(make_edge(id="e3", source="splitter", target="sink2"))

        system.validate()

        # Splitter targets should be node IDs, not edge IDs
        assert set(splitter.targets) == {"sink1", "sink2"}

    def test_splitter_raises_on_invalid_target_node(self):
        """SplitPolicy returning invalid node ID raises ValueError."""

        class BadSplit:
            def split(self, node, amount: float, t: Timestep) -> dict[str, float]:
                return {"nonexistent_node": amount}

        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        splitter = make_splitter(split_policy=BadSplit())
        sink = make_sink()

        system.add_node(source)
        system.add_node(splitter)
        system.add_node(sink)

        system.add_edge(make_edge(id="e1", source="source", target="splitter"))
        system.add_edge(make_edge(id="e2", source="splitter", target="sink"))

        with pytest.raises(ValueError, match="nonexistent_node"):
            system.simulate(timesteps=1)

    def test_single_output_node_routes_via_node_id(self):
        """Single-output nodes route correctly with node ID targets."""
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()

        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="sink"))

        system.simulate(timesteps=1)

        # Verify water flows correctly from source to sink
        received_events = sink.events_of_type(WaterReceived)
        assert len(received_events) == 1
        assert received_events[0].amount == 100.0  # Default source inflow


class TestCloneIsolation:
    """Tests for _clone_with_updates (Bug #2: lightweight cloning)."""

    def test_clone_does_not_share_mutable_state(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(source="source", target="sink"))
        system.validate()

        clone = system.with_vector(system.to_vector())
        clone.simulate(timesteps=3)

        # Original should have no events
        assert len(system.nodes["source"].events) == 0
        assert len(system.nodes["sink"].events) == 0

    def test_clone_shares_immutable_fields_by_identity(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(source="source", target="sink"))
        system.validate()

        clone = system.with_vector(system.to_vector())
        assert clone.nodes["source"].inflow is system.nodes["source"].inflow


class TestCloneSkipsValidation:
    """Tests for Bug #5: cloned systems should be pre-validated."""

    def test_cloned_system_is_validated(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(source="source", target="sink"))
        system.validate()

        clone = system.with_vector(system.to_vector())
        assert clone._validated is True

    def test_cloned_system_simulates_without_revalidation(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(make_edge(source="source", target="sink"))
        system.validate()

        clone = system.with_vector(system.to_vector())
        # Should run without calling validate() internally
        clone.simulate(timesteps=3)
        gen_events = clone.nodes["source"].events_of_type(WaterGenerated)
        assert len(gen_events) == 3


class TestWaterSystemTimeIndex:
    def test_raises_when_start_date_is_none(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        with pytest.raises(ValueError, match="start_date"):
            system.time_index(12)

    def test_monthly_returns_monthly_dates(self):
        system = WaterSystem(frequency=Frequency.MONTHLY, start_date=date(2024, 1, 1))
        result = system.time_index(3)
        assert result == (date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1))

    def test_daily_returns_daily_dates(self):
        system = WaterSystem(frequency=Frequency.DAILY, start_date=date(2024, 6, 1))
        result = system.time_index(3)
        assert result == (date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 3))

    def test_length_matches_n(self):
        system = WaterSystem(frequency=Frequency.WEEKLY, start_date=date(2024, 1, 1))
        result = system.time_index(10)
        assert len(result) == 10

    def test_first_date_is_start_date(self):
        system = WaterSystem(frequency=Frequency.YEARLY, start_date=date(2020, 5, 20))
        result = system.time_index(3)
        assert result[0] == date(2020, 5, 20)

    def test_returns_tuple_of_dates(self):
        system = WaterSystem(frequency=Frequency.MONTHLY, start_date=date(2024, 1, 1))
        result = system.time_index(3)
        assert isinstance(result, tuple)
        assert all(isinstance(d, date) for d in result)


class TestStartDatePreservation:
    def test_with_vector_preserves_start_date(self):
        system = WaterSystem(frequency=Frequency.MONTHLY, start_date=date(2024, 1, 1))
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.validate()

        vector = system.to_vector()
        new_system = system.with_vector(vector)
        assert new_system.start_date == date(2024, 1, 1)

    def test_reset_preserves_start_date(self):
        system = WaterSystem(frequency=Frequency.MONTHLY, start_date=date(2024, 6, 15))
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.simulate(timesteps=3)
        system.reset()
        assert system.start_date == date(2024, 6, 15)

    def test_system_simulates_without_start_date(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        sink = make_sink()
        edge = make_edge(source="source", target="sink")
        system.add_node(source)
        system.add_node(sink)
        system.add_edge(edge)
        system.simulate(timesteps=3)
        assert system.start_date is None


class TestMakeSystemStartDate:
    def test_make_system_passes_start_date(self):
        from taqsim.testing import make_system as testing_make_system

        system = testing_make_system(
            make_source(),
            make_sink(),
            make_edge(source="source", target="sink"),
            start_date=date(2024, 1, 1),
        )
        assert system.start_date == date(2024, 1, 1)

    def test_make_system_defaults_start_date_to_none(self):
        from taqsim.testing import make_system as testing_make_system

        system = testing_make_system(
            make_source(),
            make_sink(),
            make_edge(source="source", target="sink"),
        )
        assert system.start_date is None


class TestReachIntegration:
    def test_reach_passthrough_delivers_full_amount(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        reach = make_reach(id="reach")
        sink = make_sink()

        system.add_node(source)
        system.add_node(reach)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="reach"))
        system.add_edge(make_edge(id="e2", source="reach", target="sink"))

        system.simulate(timesteps=1)

        received = sink.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 100.0

    def test_reach_losses_reduce_downstream_delivery(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        reach = make_reach(id="reach", loss_rule=ProportionalReachLoss(loss_fraction=0.1))
        sink = make_sink()

        system.add_node(source)
        system.add_node(reach)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="reach"))
        system.add_edge(make_edge(id="e2", source="reach", target="sink"))

        system.simulate(timesteps=1)

        received = sink.events_of_type(WaterReceived)
        assert len(received) == 1
        assert received[0].amount == 90.0

        loss_events = reach.events_of_type(WaterLost)
        assert len(loss_events) == 1
        assert loss_events[0].amount == 10.0

    def test_reach_buffering_delays_downstream_delivery(self):
        class BufferingRouting:
            def initial_state(self, reach):
                return 0.0

            def route(self, reach, inflow, state, t):
                outflow = state
                new_state = inflow
                return (outflow, new_state)

            def storage(self, state):
                return state

        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        reach = make_reach(id="reach", routing_model=BufferingRouting())
        sink = make_sink()

        system.add_node(source)
        system.add_node(reach)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="reach"))
        system.add_edge(make_edge(id="e2", source="reach", target="sink"))

        system.simulate(timesteps=3)

        received = sink.events_of_type(WaterReceived)
        amounts = [e.amount for e in received]
        # t=0: nothing exits (buffered), t=1: 100 released, t=2: 100 released
        assert amounts == [100.0, 100.0]


# --- Fake models for auxiliary_data tests ---


class FakeLossRuleWithAux:
    required_auxiliary: frozenset[str] = frozenset({"surface_area", "pan_coefficient"})

    def calculate(self, node: Storage, t: Timestep) -> dict:
        return {}


class FakeLossRuleNoAux:
    def calculate(self, node: Storage, t: Timestep) -> dict:
        return {}


class FakeLossRuleEmptyAux:
    required_auxiliary: frozenset[str] = frozenset()

    def calculate(self, node: Storage, t: Timestep) -> dict:
        return {}


class FakeLossRuleMultipleKeys:
    required_auxiliary: frozenset[str] = frozenset({"area_curve", "pan_coefficient", "wind_speed"})

    def calculate(self, node: Storage, t: Timestep) -> dict:
        return {}


class AuxReadingLossRule:
    required_auxiliary: frozenset[str] = frozenset({"evap_rate"})

    def calculate(self, node: Storage, t: Timestep) -> dict[LossReason, float]:
        rate = node.auxiliary_data["evap_rate"]
        return {EVAPORATION: node.storage * rate}


class AuxReadingReachLoss:
    required_auxiliary: frozenset[str] = frozenset({"loss_factor"})

    def calculate(self, reach: Reach, flow: float, t: Timestep) -> dict[LossReason, float]:
        factor = reach.auxiliary_data["loss_factor"]
        return {SEEPAGE: flow * factor}


class TestAuxiliaryDataValidation:
    def test_passes_when_no_model_declares_required_auxiliary(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        system.add_node(make_sink())
        system.add_edge(make_edge(source="source", target="sink"))
        system.validate()  # Should not raise

    def test_passes_when_required_auxiliary_is_satisfied(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        storage = make_storage(
            loss_rule=FakeLossRuleWithAux(),
            auxiliary_data={"surface_area": 100.0, "pan_coefficient": 0.7},
        )
        system.add_node(storage)
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))
        system.validate()  # Should not raise

    def test_fails_when_required_auxiliary_key_is_missing(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        storage = make_storage(loss_rule=FakeLossRuleWithAux())
        system.add_node(storage)
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))
        with pytest.raises(MissingAuxiliaryDataError):
            system.validate()

    def test_error_message_includes_node_id_and_field_and_missing_keys(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        storage = make_storage(loss_rule=FakeLossRuleWithAux())
        system.add_node(storage)
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))
        with pytest.raises(MissingAuxiliaryDataError, match="storage") as exc_info:
            system.validate()
        msg = str(exc_info.value)
        assert "loss_rule" in msg
        assert "FakeLossRuleWithAux" in msg
        assert "pan_coefficient" in msg
        assert "surface_area" in msg

    def test_reports_multiple_missing_keys_together(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        storage = make_storage(
            loss_rule=FakeLossRuleMultipleKeys(),
            auxiliary_data={"pan_coefficient": 0.7},
        )
        system.add_node(storage)
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))
        with pytest.raises(MissingAuxiliaryDataError) as exc_info:
            system.validate()
        assert exc_info.value.missing_keys == frozenset({"area_curve", "wind_speed"})

    def test_passes_with_empty_required_auxiliary_frozenset(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        storage = make_storage(loss_rule=FakeLossRuleEmptyAux())
        system.add_node(storage)
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))
        system.validate()  # Should not raise

    def test_passes_when_model_lacks_required_auxiliary_attribute(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        system.add_node(make_source())
        storage = make_storage(loss_rule=FakeLossRuleNoAux())
        system.add_node(storage)
        system.add_node(make_sink())
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))
        system.validate()  # Should not raise


class TestAuxiliaryDataIntegration:
    def test_storage_loss_rule_reads_from_auxiliary_data(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        storage = make_storage(
            loss_rule=AuxReadingLossRule(),
            auxiliary_data={"evap_rate": 0.1},
        )
        sink = make_sink()
        system.add_node(source)
        system.add_node(storage)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="storage"))
        system.add_edge(make_edge(id="e2", source="storage", target="sink"))

        system.simulate(timesteps=1)

        loss_events = storage.events_of_type(WaterLost)
        assert len(loss_events) > 0
        assert any(e.reason == EVAPORATION for e in loss_events)

    def test_reach_loss_rule_reads_from_auxiliary_data(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        source = make_source()
        from taqsim.node import Reach

        reach = Reach(
            id="reach",
            routing_model=NoRouting(),
            loss_rule=AuxReadingReachLoss(),
            auxiliary_data={"loss_factor": 0.2},
        )
        sink = make_sink()
        system.add_node(source)
        system.add_node(reach)
        system.add_node(sink)
        system.add_edge(make_edge(id="e1", source="source", target="reach"))
        system.add_edge(make_edge(id="e2", source="reach", target="sink"))

        system.simulate(timesteps=1)

        loss_events = reach.events_of_type(WaterLost)
        assert len(loss_events) > 0
        assert any(e.reason == SEEPAGE for e in loss_events)
