
import pandas as pd
import pytest

from taqsim.nodes import SupplyNode


class MockEdge:
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.flow = 0.0

    def update(self, flow: float):
        self.flow = flow


class TestSupplyNode:
    def test_init_with_csv_file(self, simple_supply_csv):
        node = SupplyNode(
            id="supply1",
            easting=100.0,
            northing=200.0,
            csv_file=str(simple_supply_csv),
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        assert node.id == "supply1"
        assert node.easting == 100.0
        assert node.northing == 200.0
        assert len(node.supply_rates) == 12
        assert all(rate == 10.0 for rate in node.supply_rates)
        assert node.supply_history == []
        assert node.outflow_edge is None

    def test_init_with_constant_supply_rate(self):
        node = SupplyNode(id="supply2", easting=150.0, northing=250.0, constant_supply_rate=5.0, num_time_steps=10)

        assert node.id == "supply2"
        assert node.easting == 150.0
        assert node.northing == 250.0
        assert len(node.supply_rates) == 10
        assert all(rate == 5.0 for rate in node.supply_rates)
        assert node.supply_history == []
        assert node.outflow_edge is None

    def test_init_fallback_to_zeros(self):
        node = SupplyNode(id="supply3", easting=300.0, northing=400.0, num_time_steps=8)

        assert node.id == "supply3"
        assert len(node.supply_rates) == 8
        assert all(rate == 0 for rate in node.supply_rates)

    def test_init_csv_priority_over_constant(self, simple_supply_csv):
        node = SupplyNode(
            id="supply4",
            easting=100.0,
            northing=200.0,
            constant_supply_rate=99.0,
            csv_file=str(simple_supply_csv),
            start_year=2020,
            start_month=1,
            num_time_steps=12,
        )

        assert all(rate == 10.0 for rate in node.supply_rates)

    def test_init_invalid_node_id_empty_string(self):
        with pytest.raises(ValueError, match="ID cannot be empty"):
            SupplyNode(id="", easting=100.0, northing=200.0, num_time_steps=5)

    def test_init_invalid_node_id_not_string(self):
        with pytest.raises(ValueError, match="ID must be a string"):
            SupplyNode(id=123, easting=100.0, northing=200.0, num_time_steps=5)

    def test_init_invalid_coordinates_none_easting(self):
        with pytest.raises(ValueError, match="Missing coordinate value"):
            SupplyNode(id="supply5", easting=None, northing=200.0, num_time_steps=5)

    def test_init_invalid_coordinates_none_northing(self):
        with pytest.raises(ValueError, match="Missing coordinate value"):
            SupplyNode(id="supply6", easting=100.0, northing=None, num_time_steps=5)

    def test_init_invalid_coordinates_wrong_type(self):
        with pytest.raises(ValueError, match="must be a number"):
            SupplyNode(id="supply7", easting="not_a_number", northing=200.0, num_time_steps=5)

    def test_init_csv_file_not_found(self):
        with pytest.raises(ValueError, match="Data file not found"):
            SupplyNode(
                id="supply8",
                easting=100.0,
                northing=200.0,
                csv_file="/nonexistent/file.csv",
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_init_csv_missing_column(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=12, freq="MS"), "Flow": [10.0] * 12}, "bad_supply.csv"
        )

        with pytest.raises(ValueError, match="Failed to import time series data"):
            SupplyNode(
                id="supply9",
                easting=100.0,
                northing=200.0,
                csv_file=str(csv_path),
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_init_csv_insufficient_data(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=5, freq="MS"), "Q": [10.0] * 5}, "short_supply.csv"
        )

        with pytest.raises(ValueError, match="Failed to import time series data"):
            SupplyNode(
                id="supply10",
                easting=100.0,
                northing=200.0,
                csv_file=str(csv_path),
                start_year=2020,
                start_month=1,
                num_time_steps=12,
            )

    def test_add_outflow_edge_accepts_single_edge(self):
        node = SupplyNode(id="supply11", easting=100.0, northing=200.0, constant_supply_rate=5.0, num_time_steps=5)

        edge = MockEdge(capacity=10.0)
        node.add_outflow_edge(edge)

        assert node.outflow_edge is edge

    def test_add_outflow_edge_raises_if_already_set(self):
        node = SupplyNode(id="supply12", easting=100.0, northing=200.0, constant_supply_rate=5.0, num_time_steps=5)

        edge1 = MockEdge(capacity=10.0)
        edge2 = MockEdge(capacity=15.0)

        node.add_outflow_edge(edge1)

        with pytest.raises(ValueError, match="already has an outflow edge"):
            node.add_outflow_edge(edge2)

    def test_update_gets_supply_rate_at_time_step(self):
        node = SupplyNode(id="supply13", easting=100.0, northing=200.0, constant_supply_rate=8.0, num_time_steps=5)

        edge = MockEdge(capacity=20.0)
        node.add_outflow_edge(edge)

        node.update(time_step=0, dt=3600)

        assert node.supply_history == [8.0]
        assert edge.flow == 8.0

    def test_update_appends_to_supply_history(self):
        node = SupplyNode(id="supply14", easting=100.0, northing=200.0, constant_supply_rate=3.0, num_time_steps=5)

        edge = MockEdge(capacity=10.0)
        node.add_outflow_edge(edge)

        for step in range(3):
            node.update(time_step=step, dt=3600)

        assert node.supply_history == [3.0, 3.0, 3.0]

    def test_update_flow_capped_by_capacity(self):
        node = SupplyNode(id="supply15", easting=100.0, northing=200.0, constant_supply_rate=15.0, num_time_steps=5)

        edge = MockEdge(capacity=10.0)
        node.add_outflow_edge(edge)

        node.update(time_step=0, dt=3600)

        assert edge.flow == 10.0
        assert node.supply_history == [15.0]

    def test_update_handles_missing_outflow_edge_gracefully(self):
        node = SupplyNode(id="supply16", easting=100.0, northing=200.0, constant_supply_rate=5.0, num_time_steps=5)

        node.update(time_step=0, dt=3600)

        assert node.supply_history == [5.0]

    def test_update_with_varying_supply_rates(self, tmp_csv):
        csv_path = tmp_csv(
            {"Date": pd.date_range(start="2020-01-01", periods=5, freq="MS"), "Q": [1.0, 2.0, 3.0, 4.0, 5.0]},
            "varying_supply.csv",
        )

        node = SupplyNode(
            id="supply17",
            easting=100.0,
            northing=200.0,
            csv_file=str(csv_path),
            start_year=2020,
            start_month=1,
            num_time_steps=5,
        )

        edge = MockEdge(capacity=10.0)
        node.add_outflow_edge(edge)

        for step in range(5):
            node.update(time_step=step, dt=3600)

        assert node.supply_history == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert edge.flow == 5.0

    def test_update_with_zero_supply_rate(self):
        node = SupplyNode(id="supply18", easting=100.0, northing=200.0, constant_supply_rate=0.0, num_time_steps=3)

        edge = MockEdge(capacity=10.0)
        node.add_outflow_edge(edge)

        node.update(time_step=0, dt=3600)

        assert node.supply_history == [0.0]
        assert edge.flow == 0.0

    def test_update_with_multiple_time_steps_edge_capacity_changes(self):
        node = SupplyNode(id="supply19", easting=100.0, northing=200.0, constant_supply_rate=12.0, num_time_steps=3)

        edge = MockEdge(capacity=8.0)
        node.add_outflow_edge(edge)

        node.update(time_step=0, dt=3600)
        assert edge.flow == 8.0

        edge.capacity = 15.0
        node.update(time_step=1, dt=3600)
        assert edge.flow == 12.0

        edge.capacity = 5.0
        node.update(time_step=2, dt=3600)
        assert edge.flow == 5.0
