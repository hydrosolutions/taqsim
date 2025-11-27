
import pandas as pd
import pytest

from taqsim.nodes import DemandNode


class MockEdge:
    def __init__(
        self, source_id: str | None = None, capacity: float = float("inf"), flow_after_losses: list[float] | None = None
    ):
        if source_id:
            self.source = type("obj", (object,), {"id": source_id})
        self.capacity = capacity
        self.flow_after_losses = flow_after_losses if flow_after_losses else []
        self.flow = 0.0

    def set_flow_at_timestep(self, timestep: int, flow: float):
        while len(self.flow_after_losses) <= timestep:
            self.flow_after_losses.append(0.0)
        self.flow_after_losses[timestep] = flow

    def update(self, flow: float):
        self.flow = flow


class TestDemandNodeConstructor:
    def setup_method(self):
        DemandNode.all_ids.clear()
        DemandNode.high_priority_demand_ids.clear()
        DemandNode.low_priority_demand_ids.clear()

    def test_valid_initialization_with_csv_file(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=12, freq="MS"), "demand1": [10.0] * 12}, "demand.csv"
        )

        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            csv_file=str(csv_path),
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        assert node.id == "demand1"
        assert node.easting == 1000.0
        assert node.northing == 2000.0
        assert len(node.demand_rates) == 12
        assert all(rate == 10.0 for rate in node.demand_rates)
        assert node.field_efficiency == 1.0
        assert node.conveyance_efficiency == 1.0
        assert node.non_consumptive_rate == 0.0
        assert node.priority == 2
        assert "demand1" in DemandNode.all_ids
        assert "demand1" in DemandNode.low_priority_demand_ids
        assert "demand1" not in DemandNode.high_priority_demand_ids

    def test_fallback_to_constant_demand_rate(self):
        node = DemandNode(id="demand2", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=10)

        assert len(node.demand_rates) == 10
        assert all(rate == 5.0 for rate in node.demand_rates)

    def test_fallback_to_zeros(self):
        node = DemandNode(id="demand3", easting=1000.0, northing=2000.0, num_time_steps=8)

        assert len(node.demand_rates) == 8
        assert all(rate == 0 for rate in node.demand_rates)

    def test_csv_priority_over_constant(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=12, freq="MS"), "demand4": [10.0] * 12},
            "demand_priority.csv",
        )

        node = DemandNode(
            id="demand4",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=99.0,
            csv_file=str(csv_path),
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        assert all(rate == 10.0 for rate in node.demand_rates)

    def test_priority_validation_high_priority(self):
        node = DemandNode(
            id="demand5", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=5, priority=1
        )

        assert node.priority == 1
        assert "demand5" in DemandNode.high_priority_demand_ids
        assert "demand5" not in DemandNode.low_priority_demand_ids

    def test_priority_validation_low_priority(self):
        node = DemandNode(
            id="demand6", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=5, priority=2
        )

        assert node.priority == 2
        assert "demand6" in DemandNode.low_priority_demand_ids
        assert "demand6" not in DemandNode.high_priority_demand_ids

    def test_priority_validation_invalid_value(self):
        with pytest.raises(ValueError, match="priority must be 1 \\(high\\) or 2 \\(low\\)"):
            DemandNode(
                id="demand7", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=5, priority=3
            )

    def test_priority_validation_zero_is_invalid(self):
        with pytest.raises(ValueError, match="priority must be 1 \\(high\\) or 2 \\(low\\)"):
            DemandNode(
                id="demand8", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=5, priority=0
            )

    def test_field_efficiency_validation_valid_range(self):
        node = DemandNode(
            id="demand9",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=5.0,
            num_time_steps=5,
            field_efficiency=0.8,
        )

        assert node.field_efficiency == 0.8

    def test_field_efficiency_validation_upper_bound(self):
        node = DemandNode(
            id="demand10",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=5.0,
            num_time_steps=5,
            field_efficiency=1.0,
        )

        assert node.field_efficiency == 1.0

    def test_field_efficiency_zero_with_constant_demand_raises_error(self):
        with pytest.raises(ZeroDivisionError):
            DemandNode(
                id="demand11",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                field_efficiency=0.0,
            )

    def test_field_efficiency_validation_negative(self):
        with pytest.raises(ValueError):
            DemandNode(
                id="demand12",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                field_efficiency=-0.1,
            )

    def test_field_efficiency_validation_above_one(self):
        with pytest.raises(ValueError):
            DemandNode(
                id="demand13",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                field_efficiency=1.1,
            )

    def test_conveyance_efficiency_validation_valid_range(self):
        node = DemandNode(
            id="demand14",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=5.0,
            num_time_steps=5,
            conveyance_efficiency=0.75,
        )

        assert node.conveyance_efficiency == 0.75

    def test_conveyance_efficiency_validation_negative(self):
        with pytest.raises(ValueError):
            DemandNode(
                id="demand15",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                conveyance_efficiency=-0.1,
            )

    def test_conveyance_efficiency_validation_above_one(self):
        with pytest.raises(ValueError):
            DemandNode(
                id="demand16",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                conveyance_efficiency=1.5,
            )

    def test_non_consumptive_rate_validation_valid(self):
        node = DemandNode(
            id="demand17",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=3.0,
        )

        assert node.non_consumptive_rate == 3.0

    def test_non_consumptive_rate_validation_zero(self):
        node = DemandNode(
            id="demand18",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=5.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )

        assert node.non_consumptive_rate == 0.0

    def test_non_consumptive_rate_validation_negative(self):
        with pytest.raises(ValueError):
            DemandNode(
                id="demand19",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                non_consumptive_rate=-1.0,
            )

    def test_non_consumptive_rate_greater_than_demand_raises_error(self):
        with pytest.raises(ValueError, match="cannot be less than non-consumptive rate"):
            DemandNode(
                id="demand20",
                easting=1000.0,
                northing=2000.0,
                constant_demand_rate=5.0,
                num_time_steps=5,
                non_consumptive_rate=10.0,
            )

    def test_efficiency_factor_application_constant_demand(self):
        node = DemandNode(
            id="demand21",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=8.0,
            num_time_steps=5,
            field_efficiency=0.8,
            conveyance_efficiency=0.5,
        )

        expected_rate = 8.0 / (0.8 * 0.5)
        assert all(abs(rate - expected_rate) < 0.0001 for rate in node.demand_rates)

    def test_efficiency_factor_application_csv_demand(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=5, freq="MS"), "demand22": [10.0] * 5}, "demand_eff.csv"
        )

        node = DemandNode(
            id="demand22",
            easting=1000.0,
            northing=2000.0,
            csv_file=str(csv_path),
            start_year=2020,
            start_month=1,
            num_time_steps=5,
            field_efficiency=0.9,
            conveyance_efficiency=0.8,
        )

        expected_rate = 10.0 / (0.9 * 0.8)
        assert all(abs(rate - expected_rate) < 0.0001 for rate in node.demand_rates)

    def test_invalid_node_id_empty_string(self):
        with pytest.raises(ValueError, match="ID cannot be empty"):
            DemandNode(id="", easting=1000.0, northing=2000.0, num_time_steps=5)

    def test_invalid_node_id_not_string(self):
        with pytest.raises(ValueError, match="ID must be a string"):
            DemandNode(id=123, easting=1000.0, northing=2000.0, num_time_steps=5)

    def test_invalid_coordinates_none_easting(self):
        with pytest.raises(ValueError, match="Missing coordinate value"):
            DemandNode(id="demand23", easting=None, northing=2000.0, num_time_steps=5)

    def test_invalid_coordinates_none_northing(self):
        with pytest.raises(ValueError, match="Missing coordinate value"):
            DemandNode(id="demand24", easting=1000.0, northing=None, num_time_steps=5)

    def test_negative_demand_rate_in_csv_raises_error(self, tmp_csv):
        csv_path = tmp_csv(
            {
                "Date": pd.date_range(start="2020-01-01", periods=5, freq="MS"),
                "demand25": [10.0, -5.0, 10.0, 10.0, 10.0],
            },
            "bad_demand.csv",
        )

        with pytest.raises(ValueError, match="Demand rates cannot be negative"):
            DemandNode(
                id="demand25",
                easting=1000.0,
                northing=2000.0,
                csv_file=str(csv_path),
                start_year=2020,
                start_month=1,
                num_time_steps=5,
            )

    def test_csv_demand_less_than_non_consumptive_raises_error(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=5, freq="MS"), "demand26": [5.0, 5.0, 5.0, 5.0, 5.0]},
            "demand_nonconsumptive.csv",
        )

        with pytest.raises(ValueError, match="cannot be less than non-consumptive rate"):
            DemandNode(
                id="demand26",
                easting=1000.0,
                northing=2000.0,
                csv_file=str(csv_path),
                start_year=2020,
                start_month=1,
                num_time_steps=5,
                non_consumptive_rate=10.0,
            )

    def test_registry_tracks_multiple_nodes_by_priority(self):
        node1 = DemandNode(
            id="high1", easting=1000.0, northing=2000.0, constant_demand_rate=5.0, num_time_steps=5, priority=1
        )
        node2 = DemandNode(
            id="high2", easting=1100.0, northing=2100.0, constant_demand_rate=5.0, num_time_steps=5, priority=1
        )
        node3 = DemandNode(
            id="low1", easting=1200.0, northing=2200.0, constant_demand_rate=5.0, num_time_steps=5, priority=2
        )

        assert len(DemandNode.all_ids) == 3
        assert len(DemandNode.high_priority_demand_ids) == 2
        assert len(DemandNode.low_priority_demand_ids) == 1
        assert "high1" in DemandNode.high_priority_demand_ids
        assert "high2" in DemandNode.high_priority_demand_ids
        assert "low1" in DemandNode.low_priority_demand_ids


class TestDemandNodeAddInflowEdge:
    def setup_method(self):
        DemandNode.all_ids.clear()
        DemandNode.high_priority_demand_ids.clear()
        DemandNode.low_priority_demand_ids.clear()

    def test_stores_edge_keyed_by_source_id(self):
        node = DemandNode(id="demand1", easting=1000.0, northing=2000.0, num_time_steps=5)
        edge = MockEdge(source_id="source1")

        node.add_inflow_edge(edge)

        assert "source1" in node.inflow_edges
        assert node.inflow_edges["source1"] is edge

    def test_handles_multiple_inflows(self):
        node = DemandNode(id="demand1", easting=1000.0, northing=2000.0, num_time_steps=5)
        edge1 = MockEdge(source_id="source1")
        edge2 = MockEdge(source_id="source2")
        edge3 = MockEdge(source_id="source3")

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)
        node.add_inflow_edge(edge3)

        assert len(node.inflow_edges) == 3
        assert "source1" in node.inflow_edges
        assert "source2" in node.inflow_edges
        assert "source3" in node.inflow_edges


class TestDemandNodeAddOutflowEdge:
    def setup_method(self):
        DemandNode.all_ids.clear()
        DemandNode.high_priority_demand_ids.clear()
        DemandNode.low_priority_demand_ids.clear()

    def test_stores_single_outflow_edge(self):
        node = DemandNode(id="demand1", easting=1000.0, northing=2000.0, num_time_steps=5)
        edge = MockEdge(capacity=10.0)

        node.add_outflow_edge(edge)

        assert node.outflow_edge is edge

    def test_raises_error_if_outflow_already_set(self):
        node = DemandNode(id="demand1", easting=1000.0, northing=2000.0, num_time_steps=5)
        edge1 = MockEdge(capacity=10.0)
        edge2 = MockEdge(capacity=15.0)

        node.add_outflow_edge(edge1)

        with pytest.raises(ValueError, match="already has an outflow edge"):
            node.add_outflow_edge(edge2)


class TestDemandNodeUpdate:
    def setup_method(self):
        DemandNode.all_ids.clear()
        DemandNode.high_priority_demand_ids.clear()
        DemandNode.low_priority_demand_ids.clear()

    def test_inflow_exceeds_demand_no_unmet(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 15.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [10.0]
        assert node.unmet_demand == [0.0]

    def test_inflow_less_than_consumptive_partial_satisfaction(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 5.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [5.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [5.0]
        assert node.unmet_demand == [5.0]

    def test_inflow_exactly_equals_demand(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [10.0]
        assert node.unmet_demand == [0.0]

    def test_multiple_inflows_sum_correctly(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge1 = MockEdge(source_id="source1")
        edge1.set_flow_at_timestep(0, 3.0)
        edge2 = MockEdge(source_id="source2")
        edge2.set_flow_at_timestep(0, 4.0)
        edge3 = MockEdge(source_id="source3")
        edge3.set_flow_at_timestep(0, 5.0)

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)
        node.add_inflow_edge(edge3)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [10.0]
        assert node.unmet_demand == [0.0]

    def test_consumptive_satisfied_first_then_non_consumptive(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=15.0,
            num_time_steps=5,
            non_consumptive_rate=5.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 20.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [5.0]
        assert node.satisfied_demand_total == [15.0]
        assert node.unmet_demand == [0.0]

    def test_consumptive_partially_satisfied_no_non_consumptive(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=15.0,
            num_time_steps=5,
            non_consumptive_rate=5.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 7.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [7.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [7.0]
        assert node.unmet_demand == [8.0]

    def test_consumptive_fully_satisfied_non_consumptive_partial(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=15.0,
            num_time_steps=5,
            non_consumptive_rate=5.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 12.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [2.0]
        assert node.satisfied_demand_total == [12.0]
        assert node.unmet_demand == [3.0]

    def test_zero_non_consumptive_rate(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [10.0]
        assert node.unmet_demand == [0.0]

    def test_zero_demand_zero_unmet(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=0.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [0.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [0.0]
        assert node.unmet_demand == [0.0]

    def test_forwards_remaining_flow_to_outflow_edge(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        inflow_edge = MockEdge(source_id="source1")
        inflow_edge.set_flow_at_timestep(0, 15.0)
        outflow_edge = MockEdge(capacity=20.0)

        node.add_inflow_edge(inflow_edge)
        node.add_outflow_edge(outflow_edge)

        node.update(time_step=0, dt=3600.0)

        assert outflow_edge.flow == 5.0

    def test_forwards_non_consumptive_flow_to_outflow_edge(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=15.0,
            num_time_steps=5,
            non_consumptive_rate=5.0,
        )
        inflow_edge = MockEdge(source_id="source1")
        inflow_edge.set_flow_at_timestep(0, 20.0)
        outflow_edge = MockEdge(capacity=20.0)

        node.add_inflow_edge(inflow_edge)
        node.add_outflow_edge(outflow_edge)

        node.update(time_step=0, dt=3600.0)

        assert outflow_edge.flow == 10.0

    def test_outflow_limited_by_edge_capacity(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        inflow_edge = MockEdge(source_id="source1")
        inflow_edge.set_flow_at_timestep(0, 20.0)
        outflow_edge = MockEdge(capacity=5.0)

        node.add_inflow_edge(inflow_edge)
        node.add_outflow_edge(outflow_edge)

        node.update(time_step=0, dt=3600.0)

        assert outflow_edge.flow == 5.0

    def test_no_outflow_edge_no_error(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 15.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [0.0]

    def test_no_inflow_edges_results_in_full_unmet_demand(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=5,
            non_consumptive_rate=0.0,
        )

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [0.0]
        assert node.satisfied_non_consumptive_demand == [0.0]
        assert node.satisfied_demand_total == [0.0]
        assert node.unmet_demand == [10.0]

    def test_multiple_timesteps_accumulate_history(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=10.0,
            num_time_steps=3,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 12.0)
        edge.set_flow_at_timestep(1, 5.0)
        edge.set_flow_at_timestep(2, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)
        node.update(time_step=1, dt=3600.0)
        node.update(time_step=2, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0, 5.0, 10.0]
        assert node.satisfied_non_consumptive_demand == [0.0, 0.0, 0.0]
        assert node.satisfied_demand_total == [10.0, 5.0, 10.0]
        assert node.unmet_demand == [0.0, 5.0, 0.0]

    def test_varying_demand_with_csv(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=3, freq="MS"), "demand1": [5.0, 10.0, 15.0]},
            "varying_demand.csv",
        )

        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            csv_file=str(csv_path),
            start_year=2020,
            start_month=1,
            num_time_steps=3,
            non_consumptive_rate=0.0,
        )
        edge = MockEdge(source_id="source1")
        edge.set_flow_at_timestep(0, 10.0)
        edge.set_flow_at_timestep(1, 10.0)
        edge.set_flow_at_timestep(2, 10.0)
        node.add_inflow_edge(edge)

        node.update(time_step=0, dt=3600.0)
        node.update(time_step=1, dt=3600.0)
        node.update(time_step=2, dt=3600.0)

        assert node.satisfied_consumptive_demand == [5.0, 10.0, 10.0]
        assert node.unmet_demand == [0.0, 0.0, 5.0]

    def test_excess_flow_after_full_satisfaction(self):
        node = DemandNode(
            id="demand1",
            easting=1000.0,
            northing=2000.0,
            constant_demand_rate=15.0,
            num_time_steps=5,
            non_consumptive_rate=5.0,
        )
        inflow_edge = MockEdge(source_id="source1")
        inflow_edge.set_flow_at_timestep(0, 25.0)
        outflow_edge = MockEdge(capacity=50.0)

        node.add_inflow_edge(inflow_edge)
        node.add_outflow_edge(outflow_edge)

        node.update(time_step=0, dt=3600.0)

        assert node.satisfied_consumptive_demand == [10.0]
        assert node.satisfied_non_consumptive_demand == [5.0]
        assert node.satisfied_demand_total == [15.0]
        assert node.unmet_demand == [0.0]
        assert outflow_edge.flow == 15.0
