from unittest.mock import Mock

import numpy as np
import pytest

from taqsim.optimization.objectives import (
    priority_demand_deficit,
    regular_demand_deficit,
    sink_node_min_flow_deficit,
    total_spillage,
    total_unmet_ecological_flow,
)


class TestRegularDemandDeficit:
    def test_correct_volume_calculation_single_node(self):
        unmet_demand = [10.0, 20.0, 30.0]
        dt = 86400
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"demand1": {"node": demand_node}}

        result = regular_demand_deficit(system, ["demand1"], dt, num_years)

        expected_volume = sum(unmet_demand) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_volume_calculation_multiple_nodes(self):
        unmet_demand1 = [10.0, 20.0]
        unmet_demand2 = [5.0, 15.0]
        dt = 86400
        num_years = 1

        demand_node1 = Mock()
        demand_node1.unmet_demand = unmet_demand1

        demand_node2 = Mock()
        demand_node2.unmet_demand = unmet_demand2

        system = Mock()
        system.graph.nodes = {
            "demand1": {"node": demand_node1},
            "demand2": {"node": demand_node2},
        }

        result = regular_demand_deficit(system, ["demand1", "demand2"], dt, num_years)

        expected_volume = (sum(unmet_demand1) + sum(unmet_demand2)) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_unit_conversion_m3_to_km3(self):
        unmet_demand = [1e6]
        dt = 1
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"demand1": {"node": demand_node}}

        result = regular_demand_deficit(system, ["demand1"], dt, num_years)

        expected_m3 = 1e6
        expected_km3 = expected_m3 / 1e9

        assert result == pytest.approx(expected_km3)

    def test_correct_annualization(self):
        unmet_demand = [100.0] * 730
        dt = 86400
        num_years = 2

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"demand1": {"node": demand_node}}

        result = regular_demand_deficit(system, ["demand1"], dt, num_years)

        expected_volume = sum(unmet_demand) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_returns_zero_when_no_unmet_demand(self):
        unmet_demand = [0.0, 0.0, 0.0]
        dt = 86400
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"demand1": {"node": demand_node}}

        result = regular_demand_deficit(system, ["demand1"], dt, num_years)

        assert result == pytest.approx(0.0)

    def test_empty_demand_ids_returns_zero(self):
        system = Mock()
        system.graph.nodes = {}

        result = regular_demand_deficit(system, [], 86400, 1)

        assert result == pytest.approx(0.0)

    def test_handles_numpy_arrays(self):
        unmet_demand = np.array([10.0, 20.0, 30.0])
        dt = 86400
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"demand1": {"node": demand_node}}

        result = regular_demand_deficit(system, ["demand1"], dt, num_years)

        expected_volume = np.sum(unmet_demand) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)


class TestPriorityDemandDeficit:
    def test_correct_volume_calculation_single_node(self):
        unmet_demand = [15.0, 25.0, 35.0]
        dt = 86400
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"priority1": {"node": demand_node}}

        result = priority_demand_deficit(system, ["priority1"], dt, num_years)

        expected_volume = sum(unmet_demand) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_volume_calculation_multiple_nodes(self):
        unmet_demand1 = [12.0, 18.0]
        unmet_demand2 = [8.0, 22.0]
        dt = 86400
        num_years = 1

        demand_node1 = Mock()
        demand_node1.unmet_demand = unmet_demand1

        demand_node2 = Mock()
        demand_node2.unmet_demand = unmet_demand2

        system = Mock()
        system.graph.nodes = {
            "priority1": {"node": demand_node1},
            "priority2": {"node": demand_node2},
        }

        result = priority_demand_deficit(system, ["priority1", "priority2"], dt, num_years)

        expected_volume = (sum(unmet_demand1) + sum(unmet_demand2)) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_unit_conversion_m3_to_km3(self):
        unmet_demand = [2e6]
        dt = 1
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"priority1": {"node": demand_node}}

        result = priority_demand_deficit(system, ["priority1"], dt, num_years)

        expected_m3 = 2e6
        expected_km3 = expected_m3 / 1e9

        assert result == pytest.approx(expected_km3)

    def test_correct_annualization(self):
        unmet_demand = [50.0] * 1460
        dt = 86400
        num_years = 4

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"priority1": {"node": demand_node}}

        result = priority_demand_deficit(system, ["priority1"], dt, num_years)

        expected_volume = sum(unmet_demand) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_returns_zero_when_no_unmet_demand(self):
        unmet_demand = [0.0, 0.0]
        dt = 86400
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"priority1": {"node": demand_node}}

        result = priority_demand_deficit(system, ["priority1"], dt, num_years)

        assert result == pytest.approx(0.0)

    def test_empty_demand_ids_returns_zero(self):
        system = Mock()
        system.graph.nodes = {}

        result = priority_demand_deficit(system, [], 86400, 1)

        assert result == pytest.approx(0.0)

    def test_handles_numpy_arrays(self):
        unmet_demand = np.array([15.0, 25.0, 35.0])
        dt = 86400
        num_years = 1

        demand_node = Mock()
        demand_node.unmet_demand = unmet_demand

        system = Mock()
        system.graph.nodes = {"priority1": {"node": demand_node}}

        result = priority_demand_deficit(system, ["priority1"], dt, num_years)

        expected_volume = np.sum(unmet_demand) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)


class TestSinkNodeMinFlowDeficit:
    def test_correct_calculation_from_flow_deficits_single_sink(self):
        flow_deficits = [5.0, 10.0, 15.0]
        dt = 86400
        num_years = 1

        sink_node = Mock()
        sink_node.flow_deficits = flow_deficits

        system = Mock()
        system.graph.nodes = {"sink1": {"node": sink_node}}

        result = sink_node_min_flow_deficit(system, ["sink1"], dt, num_years)

        expected_volume = sum(flow_deficits) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_works_with_multiple_sinks(self):
        flow_deficits1 = [8.0, 12.0]
        flow_deficits2 = [3.0, 7.0]
        dt = 86400
        num_years = 1

        sink_node1 = Mock()
        sink_node1.flow_deficits = flow_deficits1

        sink_node2 = Mock()
        sink_node2.flow_deficits = flow_deficits2

        system = Mock()
        system.graph.nodes = {
            "sink1": {"node": sink_node1},
            "sink2": {"node": sink_node2},
        }

        result = sink_node_min_flow_deficit(system, ["sink1", "sink2"], dt, num_years)

        expected_volume = (sum(flow_deficits1) + sum(flow_deficits2)) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_unit_conversion(self):
        flow_deficits = [5e5]
        dt = 1
        num_years = 1

        sink_node = Mock()
        sink_node.flow_deficits = flow_deficits

        system = Mock()
        system.graph.nodes = {"sink1": {"node": sink_node}}

        result = sink_node_min_flow_deficit(system, ["sink1"], dt, num_years)

        expected_m3 = 5e5
        expected_km3 = expected_m3 / 1e9

        assert result == pytest.approx(expected_km3)

    def test_correct_annualization(self):
        flow_deficits = [20.0] * 1095
        dt = 86400
        num_years = 3

        sink_node = Mock()
        sink_node.flow_deficits = flow_deficits

        system = Mock()
        system.graph.nodes = {"sink1": {"node": sink_node}}

        result = sink_node_min_flow_deficit(system, ["sink1"], dt, num_years)

        expected_volume = sum(flow_deficits) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_returns_zero_when_no_deficit(self):
        flow_deficits = [0.0, 0.0, 0.0]
        dt = 86400
        num_years = 1

        sink_node = Mock()
        sink_node.flow_deficits = flow_deficits

        system = Mock()
        system.graph.nodes = {"sink1": {"node": sink_node}}

        result = sink_node_min_flow_deficit(system, ["sink1"], dt, num_years)

        assert result == pytest.approx(0.0)

    def test_empty_sink_ids_returns_zero(self):
        system = Mock()
        system.graph.nodes = {}

        result = sink_node_min_flow_deficit(system, [], 86400, 1)

        assert result == pytest.approx(0.0)

    def test_handles_numpy_arrays(self):
        flow_deficits = np.array([5.0, 10.0, 15.0])
        dt = 86400
        num_years = 1

        sink_node = Mock()
        sink_node.flow_deficits = flow_deficits

        system = Mock()
        system.graph.nodes = {"sink1": {"node": sink_node}}

        result = sink_node_min_flow_deficit(system, ["sink1"], dt, num_years)

        expected_volume = np.sum(flow_deficits) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)


class TestTotalSpillage:
    def test_combines_hydroworks_spills(self):
        spill_register1 = [100.0, 200.0]
        spill_register2 = [50.0, 150.0]
        num_years = 1

        hydroworks1 = Mock()
        hydroworks1.spill_register = spill_register1

        hydroworks2 = Mock()
        hydroworks2.spill_register = spill_register2

        system = Mock()
        system.graph.nodes = {
            "hydro1": {"node": hydroworks1},
            "hydro2": {"node": hydroworks2},
        }

        result = total_spillage(system, ["hydro1", "hydro2"], [], num_years)

        expected_volume = sum(spill_register1) + sum(spill_register2)
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_combines_reservoir_spillways(self):
        spillway_register1 = [300.0, 400.0]
        spillway_register2 = [100.0, 200.0]
        num_years = 1

        reservoir1 = Mock()
        reservoir1.spillway_register = spillway_register1

        reservoir2 = Mock()
        reservoir2.spillway_register = spillway_register2

        system = Mock()
        system.graph.nodes = {
            "res1": {"node": reservoir1},
            "res2": {"node": reservoir2},
        }

        result = total_spillage(system, [], ["res1", "res2"], num_years)

        expected_volume = sum(spillway_register1) + sum(spillway_register2)
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_combines_both_hydroworks_and_reservoirs(self):
        spill_register = [100.0, 200.0]
        spillway_register = [50.0, 150.0]
        num_years = 1

        hydroworks = Mock()
        hydroworks.spill_register = spill_register

        reservoir = Mock()
        reservoir.spillway_register = spillway_register

        system = Mock()
        system.graph.nodes = {
            "hydro1": {"node": hydroworks},
            "res1": {"node": reservoir},
        }

        result = total_spillage(system, ["hydro1"], ["res1"], num_years)

        expected_volume = sum(spill_register) + sum(spillway_register)
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_unit_conversion(self):
        spill_register = [1e9]
        num_years = 1

        hydroworks = Mock()
        hydroworks.spill_register = spill_register

        system = Mock()
        system.graph.nodes = {"hydro1": {"node": hydroworks}}

        result = total_spillage(system, ["hydro1"], [], num_years)

        expected_km3 = 1.0

        assert result == pytest.approx(expected_km3)

    def test_correct_annualization(self):
        spill_register = [1e9] * 5
        num_years = 5

        hydroworks = Mock()
        hydroworks.spill_register = spill_register

        system = Mock()
        system.graph.nodes = {"hydro1": {"node": hydroworks}}

        result = total_spillage(system, ["hydro1"], [], num_years)

        expected_km3 = 1.0

        assert result == pytest.approx(expected_km3)

    def test_returns_zero_when_no_spillage(self):
        spill_register = [0.0, 0.0]
        spillway_register = [0.0, 0.0]
        num_years = 1

        hydroworks = Mock()
        hydroworks.spill_register = spill_register

        reservoir = Mock()
        reservoir.spillway_register = spillway_register

        system = Mock()
        system.graph.nodes = {
            "hydro1": {"node": hydroworks},
            "res1": {"node": reservoir},
        }

        result = total_spillage(system, ["hydro1"], ["res1"], num_years)

        assert result == pytest.approx(0.0)

    def test_empty_node_ids_returns_zero(self):
        system = Mock()
        system.graph.nodes = {}

        result = total_spillage(system, [], [], 1)

        assert result == pytest.approx(0.0)

    def test_handles_numpy_arrays(self):
        spill_register = np.array([100.0, 200.0])
        spillway_register = np.array([50.0, 150.0])
        num_years = 1

        hydroworks = Mock()
        hydroworks.spill_register = spill_register

        reservoir = Mock()
        reservoir.spillway_register = spillway_register

        system = Mock()
        system.graph.nodes = {
            "hydro1": {"node": hydroworks},
            "res1": {"node": reservoir},
        }

        result = total_spillage(system, ["hydro1"], ["res1"], num_years)

        expected_volume = np.sum(spill_register) + np.sum(spillway_register)
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)


class TestTotalUnmetEcologicalFlow:
    def test_only_processes_edges_with_ecological_flow(self):
        dt = 86400
        num_years = 1

        edge_with_eco = Mock()
        edge_with_eco.ecological_flow = 10.0
        edge_with_eco.unmet_ecological_flow = [5.0, 10.0]

        edge_without_eco = Mock()
        edge_without_eco.ecological_flow = 0.0
        edge_without_eco.unmet_ecological_flow = [100.0, 200.0]

        system = Mock()
        system.graph.edges.return_value = [
            (1, 2, {"edge": edge_with_eco}),
            (2, 3, {"edge": edge_without_eco}),
        ]

        result = total_unmet_ecological_flow(system, dt, num_years)

        expected_volume = sum(edge_with_eco.unmet_ecological_flow) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_calculation_multiple_edges(self):
        dt = 86400
        num_years = 1

        edge1 = Mock()
        edge1.ecological_flow = 5.0
        edge1.unmet_ecological_flow = [10.0, 20.0]

        edge2 = Mock()
        edge2.ecological_flow = 8.0
        edge2.unmet_ecological_flow = [15.0, 25.0]

        system = Mock()
        system.graph.edges.return_value = [
            (1, 2, {"edge": edge1}),
            (2, 3, {"edge": edge2}),
        ]

        result = total_unmet_ecological_flow(system, dt, num_years)

        expected_volume = (sum(edge1.unmet_ecological_flow) + sum(edge2.unmet_ecological_flow)) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_handles_missing_ecological_flow_attribute(self):
        dt = 86400
        num_years = 1

        edge_with_eco = Mock()
        edge_with_eco.ecological_flow = 10.0
        edge_with_eco.unmet_ecological_flow = [5.0, 10.0]

        edge_without_attr = Mock(spec=[])

        system = Mock()
        system.graph.edges.return_value = [
            (1, 2, {"edge": edge_with_eco}),
            (2, 3, {"edge": edge_without_attr}),
        ]

        result = total_unmet_ecological_flow(system, dt, num_years)

        expected_volume = sum(edge_with_eco.unmet_ecological_flow) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_correct_unit_conversion(self):
        dt = 1
        num_years = 1

        edge = Mock()
        edge.ecological_flow = 1.0
        edge.unmet_ecological_flow = [1e9]

        system = Mock()
        system.graph.edges.return_value = [(1, 2, {"edge": edge})]

        result = total_unmet_ecological_flow(system, dt, num_years)

        expected_km3 = 1.0

        assert result == pytest.approx(expected_km3)

    def test_correct_annualization(self):
        dt = 86400
        num_years = 2

        edge = Mock()
        edge.ecological_flow = 1.0
        edge.unmet_ecological_flow = [100.0] * 730

        system = Mock()
        system.graph.edges.return_value = [(1, 2, {"edge": edge})]

        result = total_unmet_ecological_flow(system, dt, num_years)

        expected_volume = sum(edge.unmet_ecological_flow) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)

    def test_returns_zero_when_no_unmet_flow(self):
        dt = 86400
        num_years = 1

        edge = Mock()
        edge.ecological_flow = 10.0
        edge.unmet_ecological_flow = [0.0, 0.0]

        system = Mock()
        system.graph.edges.return_value = [(1, 2, {"edge": edge})]

        result = total_unmet_ecological_flow(system, dt, num_years)

        assert result == pytest.approx(0.0)

    def test_returns_zero_when_no_edges_with_ecological_flow(self):
        dt = 86400
        num_years = 1

        edge1 = Mock()
        edge1.ecological_flow = 0.0
        edge1.unmet_ecological_flow = [100.0, 200.0]

        edge2 = Mock(spec=[])

        system = Mock()
        system.graph.edges.return_value = [
            (1, 2, {"edge": edge1}),
            (2, 3, {"edge": edge2}),
        ]

        result = total_unmet_ecological_flow(system, dt, num_years)

        assert result == pytest.approx(0.0)

    def test_empty_graph_returns_zero(self):
        dt = 86400
        num_years = 1

        system = Mock()
        system.graph.edges.return_value = []

        result = total_unmet_ecological_flow(system, dt, num_years)

        assert result == pytest.approx(0.0)

    def test_handles_numpy_arrays(self):
        dt = 86400
        num_years = 1

        edge = Mock()
        edge.ecological_flow = 10.0
        edge.unmet_ecological_flow = np.array([5.0, 10.0, 15.0])

        system = Mock()
        system.graph.edges.return_value = [(1, 2, {"edge": edge})]

        result = total_unmet_ecological_flow(system, dt, num_years)

        expected_volume = np.sum(edge.unmet_ecological_flow) * dt
        expected_km3 = expected_volume / 1e9 / num_years

        assert result == pytest.approx(expected_km3)
