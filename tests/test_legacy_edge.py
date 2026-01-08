import math

import pytest

from taqsim.edge import Edge
from taqsim.nodes import DemandNode, HydroWorks, RunoffNode, SinkNode, StorageNode, SupplyNode


@pytest.fixture
def simple_supply_node():
    return SupplyNode(id="supply1", easting=1000.0, northing=2000.0, constant_supply_rate=10.0, num_time_steps=12)


@pytest.fixture
def simple_demand_node():
    return DemandNode(id="demand1", easting=2000.0, northing=3000.0, constant_demand_rate=5.0, num_time_steps=12)


@pytest.fixture
def simple_sink_node():
    return SinkNode(id="sink1", easting=3000.0, northing=4000.0, constant_min_flow=2.0, num_time_steps=12)


@pytest.fixture
def simple_storage_node(simple_hv_csv):
    return StorageNode(
        id="storage1",
        easting=1500.0,
        northing=2500.0,
        hv_file=str(simple_hv_csv),
        initial_storage=5000000.0,
        dead_storage=1000000.0,
        num_time_steps=12,
    )


@pytest.fixture
def simple_hydroworks_node():
    return HydroWorks(id="hydro1", easting=2500.0, northing=3500.0)


@pytest.fixture
def simple_runoff_node(simple_rainfall_csv):
    return RunoffNode(
        id="runoff1",
        area=100.0,
        runoff_coefficient=0.5,
        easting=500.0,
        northing=1500.0,
        rainfall_csv=str(simple_rainfall_csv),
        start_year=2020,
        start_month=1,
        num_time_steps=12,
    )


class TestEdgeConstruction:
    def test_creates_edge_with_minimal_parameters(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        assert edge.source == simple_supply_node
        assert edge.target == simple_demand_node
        assert edge.capacity == 20.0
        assert edge.ecological_flow == 0
        assert edge.loss_factor == 0
        assert edge.flow_after_losses == []
        assert edge.flow_before_losses == []
        assert edge.losses == []
        assert edge.unmet_ecological_flow == []

    def test_creates_edge_with_all_parameters(self, simple_supply_node, simple_demand_node):
        edge = Edge(
            source=simple_supply_node,
            target=simple_demand_node,
            capacity=20.0,
            ecological_flow=2.0,
            length=50.0,
            loss_factor=0.01,
        )

        assert edge.capacity == 20.0
        assert edge.ecological_flow == 2.0
        assert edge.length == 50.0
        assert edge.loss_factor == 0.01

    def test_calculates_length_from_coordinates_when_not_provided(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        delta_easting = 1000.0
        delta_northing = 1000.0
        expected_length = math.sqrt(delta_easting**2 + delta_northing**2) / 1000

        assert abs(edge.length - expected_length) < 0.001

    def test_uses_provided_length_over_calculated(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=100.0)

        assert edge.length == 100.0

    def test_fails_with_negative_capacity(self, simple_supply_node, simple_demand_node):
        with pytest.raises(ValueError, match="Edge capacity"):
            Edge(source=simple_supply_node, target=simple_demand_node, capacity=-10.0)

    def test_fails_with_loss_factor_below_zero(self, simple_supply_node, simple_demand_node):
        with pytest.raises(ValueError, match="Edge loss factor"):
            Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, loss_factor=-0.1)

    def test_fails_with_loss_factor_above_one(self, simple_supply_node, simple_demand_node):
        with pytest.raises(ValueError, match="Edge loss factor"):
            Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, loss_factor=1.5)

    def test_fails_with_negative_ecological_flow(self, simple_supply_node, simple_demand_node):
        with pytest.raises(ValueError, match="Edge min flow"):
            Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, ecological_flow=-5.0)

    def test_fails_when_ecological_flow_exceeds_capacity(self, simple_supply_node, simple_demand_node):
        with pytest.raises(ValueError, match="Ecological flow.*cannot exceed capacity"):
            Edge(source=simple_supply_node, target=simple_demand_node, capacity=10.0, ecological_flow=15.0)

    def test_fails_with_negative_length(self, simple_supply_node, simple_demand_node):
        with pytest.raises(ValueError, match="Edge length"):
            Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=-10.0)

    def test_stores_edge_with_ecological_flow_in_class_dict(self, simple_supply_node, simple_demand_node):
        Edge.edges_with_min_flow.clear()

        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, ecological_flow=3.0)

        key = (simple_supply_node.id, simple_demand_node.id)
        assert key in Edge.edges_with_min_flow
        assert Edge.edges_with_min_flow[key] == 3.0

    def test_does_not_store_edge_without_ecological_flow(self, simple_supply_node, simple_demand_node):
        Edge.edges_with_min_flow.clear()

        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        key = (simple_supply_node.id, simple_demand_node.id)
        assert key not in Edge.edges_with_min_flow


class TestEdgeNodeConnections:
    def test_connects_supply_node_to_demand_node(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        assert simple_supply_node.outflow_edge == edge
        assert simple_demand_node.inflow_edges[simple_supply_node.id] == edge

    def test_connects_supply_node_to_storage_node(self, simple_supply_node, simple_storage_node):
        edge = Edge(source=simple_supply_node, target=simple_storage_node, capacity=20.0)

        assert simple_supply_node.outflow_edge == edge
        assert simple_storage_node.inflow_edges[simple_supply_node.id] == edge

    def test_connects_supply_node_to_sink_node(self, simple_supply_node, simple_sink_node):
        edge = Edge(source=simple_supply_node, target=simple_sink_node, capacity=20.0)

        assert simple_supply_node.outflow_edge == edge
        assert simple_sink_node.inflow_edges[simple_supply_node.id] == edge

    def test_connects_runoff_node_to_storage_node(self, simple_runoff_node, simple_storage_node):
        edge = Edge(source=simple_runoff_node, target=simple_storage_node, capacity=20.0)

        assert simple_runoff_node.outflow_edge == edge
        assert simple_storage_node.inflow_edges[simple_runoff_node.id] == edge

    def test_connects_storage_node_to_demand_node(self, simple_storage_node, simple_demand_node):
        edge = Edge(source=simple_storage_node, target=simple_demand_node, capacity=20.0)

        assert simple_storage_node.outflow_edge == edge
        assert simple_demand_node.inflow_edges[simple_storage_node.id] == edge

    def test_connects_demand_node_to_hydroworks_node(self, simple_demand_node, simple_hydroworks_node):
        edge = Edge(source=simple_demand_node, target=simple_hydroworks_node, capacity=20.0)

        assert simple_demand_node.outflow_edge == edge
        assert simple_hydroworks_node.inflow_edges[simple_demand_node.id] == edge

    def test_connects_hydroworks_node_to_sink_node(self, simple_hydroworks_node, simple_sink_node):
        edge = Edge(source=simple_hydroworks_node, target=simple_sink_node, capacity=20.0)

        assert simple_hydroworks_node.outflow_edges[simple_sink_node.id] == edge
        assert simple_sink_node.inflow_edges[simple_hydroworks_node.id] == edge

    def test_fails_when_connecting_invalid_source_type(self, simple_sink_node, simple_demand_node):
        with pytest.raises(AttributeError, match="Invalid source node type"):
            Edge(source=simple_sink_node, target=simple_demand_node, capacity=20.0)

    def test_fails_when_connecting_invalid_target_type(self, simple_supply_node, simple_runoff_node):
        with pytest.raises(AttributeError, match="Invalid target node type"):
            Edge(source=simple_supply_node, target=simple_runoff_node, capacity=20.0)


class TestEdgeUpdate:
    def test_sets_flow_to_zero_when_none(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        edge.update(None)

        assert edge.flow_before_losses[0] == 0
        assert edge.flow_after_losses[0] == 0

    def test_sets_flow_to_zero_when_negative(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        edge.update(-10.0)

        assert edge.flow_before_losses[0] == 0
        assert edge.flow_after_losses[0] == 0

    def test_caps_flow_by_capacity(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        edge.update(30.0)

        assert edge.flow_before_losses[0] == 20.0

    def test_records_flow_before_losses(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        edge.update(15.0)

        assert edge.flow_before_losses[0] == 15.0

    def test_records_flow_after_losses_with_no_loss_factor(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, loss_factor=0.0)

        edge.update(15.0)

        assert edge.flow_after_losses[0] == 15.0
        assert edge.losses[0] == 0

    def test_records_flow_after_losses_with_loss_factor(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=10.0, loss_factor=0.1)

        edge.update(15.0)

        total_loss_fraction = 1 - (1 - 0.1) ** 10
        expected_losses = 15.0 * total_loss_fraction
        expected_remaining = 15.0 - expected_losses

        assert abs(edge.flow_after_losses[0] - expected_remaining) < 0.001
        assert abs(edge.losses[0] - expected_losses) < 0.001

    def test_tracks_ecological_flow_deficit_when_below_requirement(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, ecological_flow=10.0)

        edge.update(7.0)

        assert edge.unmet_ecological_flow[0] == 3.0

    def test_does_not_track_deficit_when_meeting_ecological_flow(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, ecological_flow=10.0)

        edge.update(12.0)

        assert len(edge.unmet_ecological_flow) == 0

    def test_appends_multiple_time_steps(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        edge.update(10.0)
        edge.update(15.0)
        edge.update(5.0)

        assert edge.flow_before_losses == [10.0, 15.0, 5.0]
        assert edge.flow_after_losses == [10.0, 15.0, 5.0]
        assert len(edge.losses) == 3


class TestCalculateEdgeLosses:
    def test_returns_no_losses_with_zero_loss_factor(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=50.0, loss_factor=0.0)

        remaining_flow, losses = edge.calculate_edge_losses(10.0)

        assert remaining_flow == 10.0
        assert losses == 0.0

    def test_calculates_exponential_losses_with_length_and_factor(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=10.0, loss_factor=0.05)

        remaining_flow, losses = edge.calculate_edge_losses(20.0)

        total_loss_fraction = 1 - (1 - 0.05) ** 10
        expected_losses = 20.0 * total_loss_fraction
        expected_remaining = 20.0 - expected_losses

        assert abs(remaining_flow - expected_remaining) < 0.001
        assert abs(losses - expected_losses) < 0.001

    def test_caps_losses_at_one_hundred_percent(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=100.0, loss_factor=0.99)

        remaining_flow, losses = edge.calculate_edge_losses(10.0)

        assert losses <= 10.0
        assert remaining_flow >= 0
        assert abs(remaining_flow + losses - 10.0) < 0.001

    def test_handles_zero_flow(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=10.0, loss_factor=0.1)

        remaining_flow, losses = edge.calculate_edge_losses(0.0)

        assert remaining_flow == 0.0
        assert losses == 0.0

    def test_handles_very_small_loss_factor(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0, length=1.0, loss_factor=0.001)

        remaining_flow, losses = edge.calculate_edge_losses(100.0)

        expected_loss_fraction = 1 - (1 - 0.001) ** 1
        expected_losses = 100.0 * expected_loss_fraction

        assert abs(losses - expected_losses) < 0.001
        assert remaining_flow > 99.8


class TestGetEdgeLength:
    def test_calculates_euclidean_distance_in_km(self, simple_supply_node, simple_demand_node):
        edge = Edge(source=simple_supply_node, target=simple_demand_node, capacity=20.0)

        delta_easting = 1000.0
        delta_northing = 1000.0
        expected_length_km = math.sqrt(delta_easting**2 + delta_northing**2) / 1000

        length = edge.get_edge_length()

        assert abs(length - expected_length_km) < 0.001

    def test_handles_nodes_at_same_location(self):
        source = SupplyNode(id="supply1", easting=1000.0, northing=2000.0, constant_supply_rate=10.0, num_time_steps=12)
        target = DemandNode(id="demand1", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=12)

        edge = Edge(source=source, target=target, capacity=20.0)

        assert edge.length == 0.0

    def test_fails_with_missing_source_coordinates(self):
        source = SupplyNode(id="supply1", easting=1000.0, northing=2000.0, constant_supply_rate=10.0, num_time_steps=12)
        source.easting = None

        target = DemandNode(id="demand1", easting=2000.0, northing=3000.0, constant_demand_rate=5.0, num_time_steps=12)

        with pytest.raises(ValueError, match="Missing coordinate values"):
            source.easting = None
            edge = Edge(source=source, target=target, capacity=20.0)

    def test_fails_with_missing_target_coordinates(self):
        source = SupplyNode(id="supply1", easting=1000.0, northing=2000.0, constant_supply_rate=10.0, num_time_steps=12)

        target = DemandNode(id="demand1", easting=2000.0, northing=3000.0, constant_demand_rate=5.0, num_time_steps=12)
        target.northing = None

        with pytest.raises(ValueError, match="Missing coordinate values"):
            edge = Edge(source=source, target=target, capacity=20.0)

    def test_calculates_correct_distance_for_various_coordinates(self):
        source = SupplyNode(id="supply1", easting=0.0, northing=0.0, constant_supply_rate=10.0, num_time_steps=12)
        target = DemandNode(id="demand1", easting=3000.0, northing=4000.0, constant_demand_rate=5.0, num_time_steps=12)

        edge = Edge(source=source, target=target, capacity=20.0)

        expected_length = math.sqrt(3000**2 + 4000**2) / 1000
        assert abs(edge.length - expected_length) < 0.001
