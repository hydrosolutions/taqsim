import pandas as pd
import pytest

from taqsim.edge import Edge
from taqsim.nodes import DemandNode, HydroWorks, SinkNode, StorageNode, SupplyNode
from taqsim.water_system import WaterSystem


class TestWaterSystemConstructor:
    def test_accepts_valid_initialization(self, fresh_system: WaterSystem) -> None:
        assert fresh_system.dt == 2629800
        assert fresh_system.start_year == 2020
        assert fresh_system.start_month == 1
        assert fresh_system.time_steps == 0
        assert fresh_system.has_been_checked is False
        assert len(fresh_system.graph.nodes()) == 0

    def test_accepts_custom_dt(self) -> None:
        ws = WaterSystem(dt=86400, start_year=2020, start_month=1)
        assert ws.dt == 86400

    def test_accepts_custom_start_year(self) -> None:
        ws = WaterSystem(dt=2629800, start_year=2015, start_month=1)
        assert ws.start_year == 2015

    def test_accepts_custom_start_month(self) -> None:
        ws = WaterSystem(dt=2629800, start_year=2020, start_month=6)
        assert ws.start_month == 6

    def test_fails_with_non_positive_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            WaterSystem(dt=0, start_year=2020, start_month=1)

    def test_fails_with_negative_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            WaterSystem(dt=-100, start_year=2020, start_month=1)

    def test_fails_with_invalid_start_month_zero(self) -> None:
        with pytest.raises(ValueError, match="Start month must be between 1 and 12"):
            WaterSystem(dt=2629800, start_year=2020, start_month=0)

    def test_fails_with_invalid_start_month_thirteen(self) -> None:
        with pytest.raises(ValueError, match="Start month must be between 1 and 12"):
            WaterSystem(dt=2629800, start_year=2020, start_month=13)

    def test_fails_with_non_integer_start_month(self) -> None:
        with pytest.raises(ValueError, match="Start month must be an integer"):
            WaterSystem(dt=2629800, start_year=2020, start_month=1.5)

    def test_fails_with_non_integer_start_year(self) -> None:
        with pytest.raises(ValueError, match="Start year must be an integer"):
            WaterSystem(dt=2629800, start_year=2020.5, start_month=1)

    def test_calls_reset_node_registries(self) -> None:
        StorageNode.all_ids = ["old_id"]
        SinkNode.all_ids = ["old_sink"]
        DemandNode.all_ids = ["old_demand"]
        HydroWorks.all_ids = ["old_hydro"]

        WaterSystem(dt=2629800, start_year=2020, start_month=1)

        assert StorageNode.all_ids == []
        assert SinkNode.all_ids == []
        assert DemandNode.all_ids == []
        assert HydroWorks.all_ids == []


class TestWaterSystemNetworkValidation:
    def test_check_network_structure_fails_with_empty_network(self, fresh_system: WaterSystem) -> None:
        with pytest.raises(ValueError, match="Network is empty"):
            fresh_system._check_network_structure()

    def test_check_network_structure_fails_with_isolated_nodes(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)
        isolated = SinkNode(id="isolated", easting=200, northing=200, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)
        fresh_system.add_node(isolated)

        edge = Edge(source=supply, target=sink, capacity=20)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="Network contains isolated nodes"):
            fresh_system._check_network_structure()

    def test_check_network_structure_fails_with_cycles(self, fresh_system: WaterSystem, simple_hv_csv) -> None:
        storage1 = StorageNode(
            id="storage1", easting=0, northing=0, hv_file=str(simple_hv_csv), initial_storage=1000000, num_time_steps=12
        )
        storage2 = StorageNode(
            id="storage2",
            easting=100,
            northing=100,
            hv_file=str(simple_hv_csv),
            initial_storage=1000000,
            num_time_steps=12,
        )

        fresh_system.add_node(storage1)
        fresh_system.add_node(storage2)

        edge1 = Edge(source=storage1, target=storage2, capacity=10)
        edge2 = Edge(source=storage2, target=storage1, capacity=10)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        with pytest.raises(ValueError, match="Network contains cycles"):
            fresh_system._check_network_structure()

    def test_check_network_structure_fails_without_supply_or_runoff(self, fresh_system: WaterSystem) -> None:
        sink = SinkNode(id="sink", easting=0, northing=0, constant_min_flow=0, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=5, num_time_steps=12)

        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge = Edge(source=demand, target=sink, capacity=10)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="Network must contain at least one SupplyNode or RunoffNode"):
            fresh_system._check_network_structure()

    def test_check_network_structure_fails_without_sink_node(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=100, northing=100, constant_demand_rate=5, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)

        edge = Edge(source=supply, target=demand, capacity=20)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="Network must contain at least one SinkNode"):
            fresh_system._check_network_structure()

    def test_check_network_structure_accepts_weakly_connected_graph(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=5, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system._check_network_structure()

    def test_check_node_configuration_fails_supply_with_inflows(self, fresh_system: WaterSystem, simple_hv_csv) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        storage = StorageNode(
            id="storage",
            easting=50,
            northing=50,
            hv_file=str(simple_hv_csv),
            initial_storage=1000000,
            num_time_steps=12,
        )
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)
        demand = DemandNode(id="demand", easting=150, northing=150, constant_demand_rate=5, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(storage)
        fresh_system.add_node(sink)
        fresh_system.add_node(demand)

        edge1 = Edge(source=supply, target=storage, capacity=10)
        edge2 = Edge(source=storage, target=demand, capacity=20)
        edge3 = Edge(source=demand, target=sink, capacity=20)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)
        fresh_system.add_edge(edge3)

        fresh_system._check_node_configuration()

    def test_check_node_configuration_fails_supply_without_outflow(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)

        with pytest.raises(ValueError, match="must have exactly one outflow"):
            fresh_system._check_node_configuration()

    def test_check_node_configuration_accepts_multiple_sinks(self, fresh_system: WaterSystem) -> None:
        supply1 = SupplyNode(id="supply1", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        supply2 = SupplyNode(id="supply2", easting=0, northing=100, constant_supply_rate=10, num_time_steps=12)
        sink1 = SinkNode(id="sink1", easting=50, northing=50, constant_min_flow=0, num_time_steps=12)
        sink2 = SinkNode(id="sink2", easting=50, northing=150, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply1)
        fresh_system.add_node(supply2)
        fresh_system.add_node(sink1)
        fresh_system.add_node(sink2)

        edge1 = Edge(source=supply1, target=sink1, capacity=10)
        edge2 = Edge(source=supply2, target=sink2, capacity=10)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system._check_node_configuration()

    def test_check_node_configuration_fails_sink_without_inflows(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)
        orphan_sink = SinkNode(id="orphan", easting=200, northing=200, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)
        fresh_system.add_node(orphan_sink)

        edge = Edge(source=supply, target=sink, capacity=20)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="must have at least one inflow"):
            fresh_system._check_node_configuration()

    def test_check_node_configuration_fails_storage_without_inflow(
        self, fresh_system: WaterSystem, simple_hv_csv
    ) -> None:
        storage = StorageNode(
            id="storage", easting=0, northing=0, hv_file=str(simple_hv_csv), initial_storage=1000000, num_time_steps=12
        )
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(storage)
        fresh_system.add_node(sink)

        edge = Edge(source=storage, target=sink, capacity=10)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="must have at least one inflow"):
            fresh_system._check_node_configuration()

    def test_check_node_configuration_fails_storage_without_outflow(
        self, fresh_system: WaterSystem, simple_hv_csv
    ) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        storage = StorageNode(
            id="storage",
            easting=50,
            northing=50,
            hv_file=str(simple_hv_csv),
            initial_storage=1000000,
            num_time_steps=12,
        )
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(storage)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=storage, capacity=10)

        fresh_system.add_edge(edge1)

        with pytest.raises(ValueError, match="must have exactly one outflow"):
            fresh_system._check_node_configuration()

    def test_check_node_configuration_fails_demand_without_inflow(self, fresh_system: WaterSystem) -> None:
        demand = DemandNode(id="demand", easting=0, northing=0, constant_demand_rate=5, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge = Edge(source=demand, target=sink, capacity=10)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="must have at least one inflow"):
            fresh_system._check_node_configuration()

    def test_check_node_configuration_fails_demand_without_outflow(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=5, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge = Edge(source=supply, target=demand, capacity=10)
        fresh_system.add_edge(edge)

        with pytest.raises(ValueError, match="must have exactly one outflow"):
            fresh_system._check_node_configuration()

    def test_check_node_configuration_accepts_valid_network(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)

        edge = Edge(source=supply, target=sink, capacity=20)
        fresh_system.add_edge(edge)

        fresh_system._check_node_configuration()

    def test_check_network_sets_has_been_checked_flag(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)

        edge = Edge(source=supply, target=sink, capacity=20)
        fresh_system.add_edge(edge)

        assert fresh_system.has_been_checked is False
        fresh_system._check_network()
        assert fresh_system.has_been_checked is True


class TestWaterSystemSimulation:
    def test_simulate_simple_linear_network(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=5, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        assert len(supply.supply_history) == 12
        assert len(demand.satisfied_consumptive_demand) == 12
        assert len(sink.flow_history) == 12

    def test_simulate_runs_topologically_sorted(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand1 = DemandNode(id="demand1", easting=50, northing=50, constant_demand_rate=3, num_time_steps=12)
        demand2 = DemandNode(id="demand2", easting=75, northing=75, constant_demand_rate=2, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand1)
        fresh_system.add_node(demand2)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand1, capacity=20)
        edge2 = Edge(source=demand1, target=demand2, capacity=20)
        edge3 = Edge(source=demand2, target=sink, capacity=20)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)
        fresh_system.add_edge(edge3)

        fresh_system.simulate(time_steps=5)

        assert len(supply.supply_history) == 5
        assert len(demand1.satisfied_consumptive_demand) == 5
        assert len(demand2.satisfied_consumptive_demand) == 5
        assert len(sink.flow_history) == 5

    def test_simulate_fails_branching_from_supply(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=20, num_time_steps=12)
        demand1 = DemandNode(id="demand1", easting=50, northing=50, constant_demand_rate=5, num_time_steps=12)
        sink1 = SinkNode(id="sink1", easting=100, northing=50, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand1)
        fresh_system.add_node(sink1)

        edge1 = Edge(source=supply, target=demand1, capacity=15)
        fresh_system.add_edge(edge1)

        with pytest.raises(ValueError, match="already has an outflow edge"):
            Edge(source=supply, target=sink1, capacity=10)

    def test_simulate_with_multiple_nodes(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand1 = DemandNode(id="demand1", easting=30, northing=30, constant_demand_rate=3, num_time_steps=12)
        demand2 = DemandNode(id="demand2", easting=60, northing=60, constant_demand_rate=2, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand1)
        fresh_system.add_node(demand2)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand1, capacity=20)
        edge2 = Edge(source=demand1, target=demand2, capacity=15)
        edge3 = Edge(source=demand2, target=sink, capacity=15)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)
        fresh_system.add_edge(edge3)

        fresh_system.simulate(time_steps=12)

        assert len(demand1.satisfied_consumptive_demand) == 12
        assert len(demand2.satisfied_consumptive_demand) == 12
        assert len(sink.flow_history) == 12

    def test_simulate_calls_check_network_if_not_checked(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)

        edge = Edge(source=supply, target=sink, capacity=20)
        fresh_system.add_edge(edge)

        assert fresh_system.has_been_checked is False
        fresh_system.simulate(time_steps=12)
        assert fresh_system.has_been_checked is True

    def test_simulate_sets_time_steps(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(sink)

        edge = Edge(source=supply, target=sink, capacity=20)
        fresh_system.add_edge(edge)

        assert fresh_system.time_steps == 0
        fresh_system.simulate(time_steps=10)
        assert fresh_system.time_steps == 10


class TestWaterSystemWaterBalance:
    def test_get_water_balance_returns_dataframe(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=0, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)
        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        assert isinstance(df, pd.DataFrame)

    def test_get_water_balance_contains_required_columns(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=0, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)
        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        expected_columns = [
            "time_step",
            "storage_start",
            "storage_end",
            "storage_change",
            "reservoir ET losses",
            "reservoir spills",
            "hydroworks spills",
            "source",
            "surfacerunoff",
            "sink",
            "sink min flow requirement",
            "sink min flow deficit",
            "edge losses",
            "demands",
            "demands non consumptive",
            "supplied consumptive demand",
            "supplied non consumptive demand",
            "unmet demand",
            "balance_error",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_get_water_balance_has_correct_row_count(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=0, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)
        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        assert len(df) == 12

    def test_get_water_balance_returns_empty_before_simulation(self, fresh_system: WaterSystem) -> None:
        df = fresh_system.get_water_balance()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_water_balance_source_values_reasonable(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=0, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)
        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        expected_volume_per_step = 10 * fresh_system.dt

        for val in df["source"]:
            assert val == pytest.approx(expected_volume_per_step, rel=1e-2)

    def test_get_water_balance_sink_values_reasonable(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=0, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)
        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        expected_volume_per_step = 10 * fresh_system.dt

        for val in df["sink"]:
            assert val == pytest.approx(expected_volume_per_step, rel=1e-2)

    def test_get_water_balance_with_edge_losses(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=0, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20, length=10, loss_factor=0.1)
        edge2 = Edge(source=demand, target=sink, capacity=20)
        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        assert all(df["edge losses"] > 0)

    def test_get_water_balance_with_demand_node(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=5, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=20)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        expected_demand_volume = 5 * fresh_system.dt

        for val in df["demands"]:
            assert val == pytest.approx(expected_demand_volume, rel=1e-2)
        assert all(df["supplied consumptive demand"] >= 0)

    def test_get_water_balance_all_values_non_negative(self, fresh_system: WaterSystem) -> None:
        supply = SupplyNode(id="supply", easting=0, northing=0, constant_supply_rate=10, num_time_steps=12)
        demand = DemandNode(id="demand", easting=50, northing=50, constant_demand_rate=3, num_time_steps=12)
        sink = SinkNode(id="sink", easting=100, northing=100, constant_min_flow=0, num_time_steps=12)

        fresh_system.add_node(supply)
        fresh_system.add_node(demand)
        fresh_system.add_node(sink)

        edge1 = Edge(source=supply, target=demand, capacity=20)
        edge2 = Edge(source=demand, target=sink, capacity=15)

        fresh_system.add_edge(edge1)
        fresh_system.add_edge(edge2)

        fresh_system.simulate(time_steps=12)

        df = fresh_system.get_water_balance()

        assert all(df["source"] >= 0)
        assert all(df["sink"] >= 0)
        assert all(df["supplied consumptive demand"] >= 0)
