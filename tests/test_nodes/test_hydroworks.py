import numpy as np
import pytest

from taqsim.nodes import HydroWorks


class MockEdge:
    def __init__(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        capacity: float = 100.0,
        flow_after_losses: list[float] | None = None,
    ):
        if source_id:
            self.source = type("obj", (object,), {"id": source_id})
        if target_id:
            self.target = type("obj", (object,), {"id": target_id})
        self.capacity = capacity
        self.flow_after_losses = flow_after_losses if flow_after_losses else []
        self.flow = 0.0

    def set_flow_at_timestep(self, timestep: int, flow: float):
        while len(self.flow_after_losses) <= timestep:
            self.flow_after_losses.append(0.0)
        self.flow_after_losses[timestep] = flow

    def update(self, flow: float):
        self.flow = flow


class TestHydroWorksConstructor:
    def setup_method(self):
        HydroWorks.all_ids.clear()

    def test_valid_initialization(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        assert node.id == "hw1"
        assert node.easting == 1000.0
        assert node.northing == 2000.0
        assert node.inflow_edges == {}
        assert node.outflow_edges == {}
        assert node.distribution_params == {}
        assert node.spill_register == []

    def test_registry_tracking(self):
        hw1 = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        hw2 = HydroWorks(id="hw2", easting=1100.0, northing=2100.0)

        assert len(HydroWorks.all_ids) == 2
        assert "hw1" in HydroWorks.all_ids
        assert "hw2" in HydroWorks.all_ids

    def test_invalid_node_id_raises_error(self):
        with pytest.raises(ValueError, match="HydroWorksID cannot be empty"):
            HydroWorks(id="", easting=1000.0, northing=2000.0)

    def test_invalid_coordinates_raise_error(self):
        with pytest.raises(ValueError):
            HydroWorks(id="hw1", easting=None, northing=2000.0)


class TestHydroWorksAddInflowEdge:
    def setup_method(self):
        HydroWorks.all_ids.clear()

    def test_stores_edge_keyed_by_source_id(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge = MockEdge(source_id="source1")

        node.add_inflow_edge(edge)

        assert "source1" in node.inflow_edges
        assert node.inflow_edges["source1"] is edge

    def test_handles_multiple_inflows(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(source_id="source1")
        edge2 = MockEdge(source_id="source2")

        node.add_inflow_edge(edge1)
        node.add_inflow_edge(edge2)

        assert len(node.inflow_edges) == 2
        assert "source1" in node.inflow_edges
        assert "source2" in node.inflow_edges


class TestHydroWorksAddOutflowEdge:
    def setup_method(self):
        HydroWorks.all_ids.clear()

    def test_stores_edge_keyed_by_target_id(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge = MockEdge(target_id="target1")

        node.add_outflow_edge(edge)

        assert "target1" in node.outflow_edges
        assert node.outflow_edges["target1"] is edge

    def test_handles_multiple_outflows(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        edge3 = MockEdge(target_id="target3")

        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)
        node.add_outflow_edge(edge3)

        assert len(node.outflow_edges) == 3
        assert "target1" in node.outflow_edges
        assert "target2" in node.outflow_edges
        assert "target3" in node.outflow_edges


class TestHydroWorksSetDistributionParameters:
    def setup_method(self):
        HydroWorks.all_ids.clear()

    def test_validates_all_node_ids_exist_in_outflow_edges(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        node.add_outflow_edge(edge1)

        with pytest.raises(KeyError, match="Edge to node target2 not found"):
            node.set_distribution_parameters({"target1": 0.5, "target2": 0.5})

    def test_converts_single_values_to_12_month_arrays(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)

        node.set_distribution_parameters({"target1": 0.6, "target2": 0.4})

        assert len(node.distribution_params["target1"]) == 12
        assert len(node.distribution_params["target2"]) == 12
        assert all(node.distribution_params["target1"] == 0.6)
        assert all(node.distribution_params["target2"] == 0.4)

    def test_accepts_12_month_arrays(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)

        params1 = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        params2 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

        node.set_distribution_parameters({"target1": params1, "target2": params2})

        assert np.allclose(node.distribution_params["target1"], params1)
        assert np.allclose(node.distribution_params["target2"], params2)

    def test_validates_all_params_in_range_0_1(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            node.set_distribution_parameters({"target1": 1.5, "target2": -0.5})

    def test_validates_params_sum_to_1_for_each_month(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)

        with pytest.raises(ValueError, match="must sum to 1"):
            node.set_distribution_parameters({"target1": 0.3, "target2": 0.5})

    def test_validates_sum_with_tolerance(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)

        node.set_distribution_parameters({"target1": 0.6 + 1e-11, "target2": 0.4 - 1e-11})

        assert "target1" in node.distribution_params
        assert "target2" in node.distribution_params

    def test_raises_error_for_wrong_array_length(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        node.add_outflow_edge(edge1)

        with pytest.raises(ValueError, match="must be either a single value or a list of 12"):
            node.set_distribution_parameters({"target1": [0.5, 0.5, 0.5]})

    def test_allows_updating_subset_of_edges(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)
        edge1 = MockEdge(target_id="target1")
        edge2 = MockEdge(target_id="target2")
        edge3 = MockEdge(target_id="target3")
        node.add_outflow_edge(edge1)
        node.add_outflow_edge(edge2)
        node.add_outflow_edge(edge3)

        node.set_distribution_parameters({"target1": 0.4, "target2": 0.3, "target3": 0.3})

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.2})

        assert all(node.distribution_params["target1"] == 0.5)
        assert all(node.distribution_params["target2"] == 0.2)
        assert all(node.distribution_params["target3"] == 0.3)


class TestHydroWorksUpdate:
    def setup_method(self):
        HydroWorks.all_ids.clear()

    def test_sums_inflows(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow1 = MockEdge(source_id="source1")
        inflow1.set_flow_at_timestep(0, 10.0)
        inflow2 = MockEdge(source_id="source2")
        inflow2.set_flow_at_timestep(0, 5.0)

        outflow1 = MockEdge(target_id="target1", capacity=100.0)

        node.add_inflow_edge(inflow1)
        node.add_inflow_edge(inflow2)
        node.add_outflow_edge(outflow1)

        node.set_distribution_parameters({"target1": 1.0})
        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 15.0

    def test_gets_current_month_from_time_step(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)
        inflow.set_flow_at_timestep(1, 100.0)
        inflow.set_flow_at_timestep(13, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=100.0)
        outflow2 = MockEdge(target_id="target2", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        params1 = [0.8 if i == 0 else 0.5 if i == 1 else 0.3 for i in range(12)]
        params2 = [0.2 if i == 0 else 0.5 if i == 1 else 0.7 for i in range(12)]

        node.set_distribution_parameters({"target1": params1, "target2": params2})

        node.update(time_step=0, dt=3600.0)
        assert outflow1.flow == 80.0
        assert outflow2.flow == 20.0

        node.update(time_step=1, dt=3600.0)
        assert outflow1.flow == 50.0
        assert outflow2.flow == 50.0

        node.update(time_step=13, dt=3600.0)
        assert outflow1.flow == 50.0
        assert outflow2.flow == 50.0

    def test_two_outflows_50_50_distribution(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=100.0)
        outflow2 = MockEdge(target_id="target2", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.5})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 50.0
        assert outflow2.flow == 50.0
        assert node.spill_register == [0.0]

    def test_three_outflows_varying_distribution(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=100.0)
        outflow2 = MockEdge(target_id="target2", capacity=100.0)
        outflow3 = MockEdge(target_id="target3", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)
        node.add_outflow_edge(outflow3)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.3, "target3": 0.2})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 50.0
        assert outflow2.flow == 30.0
        assert outflow3.flow == 20.0
        assert node.spill_register == [0.0]

    def test_overflow_redistribution_when_one_edge_at_capacity(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=30.0)
        outflow2 = MockEdge(target_id="target2", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.5})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 30.0
        assert outflow2.flow == 70.0
        assert node.spill_register == [0.0]

    def test_overflow_redistribution_proportional_to_remaining_capacity(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=20.0)
        outflow2 = MockEdge(target_id="target2", capacity=60.0)
        outflow3 = MockEdge(target_id="target3", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)
        node.add_outflow_edge(outflow3)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.25, "target3": 0.25})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 20.0
        overflow = 50.0 - 20.0

        remaining_capacity_2 = 60.0 - 25.0
        remaining_capacity_3 = 100.0 - 25.0
        total_remaining = remaining_capacity_2 + remaining_capacity_3

        expected_flow_2 = 25.0 + (overflow * remaining_capacity_2 / total_remaining)
        expected_flow_3 = 25.0 + (overflow * remaining_capacity_3 / total_remaining)

        assert np.isclose(outflow2.flow, expected_flow_2)
        assert np.isclose(outflow3.flow, expected_flow_3)
        assert node.spill_register == [0.0]

    def test_spill_when_all_edges_at_capacity(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=30.0)
        outflow2 = MockEdge(target_id="target2", capacity=20.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        node.set_distribution_parameters({"target1": 0.6, "target2": 0.4})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 30.0
        assert outflow2.flow == 20.0
        expected_spill = (100.0 - 30.0 - 20.0) * 3600.0
        assert np.isclose(node.spill_register[0], expected_spill)

    def test_partial_spill_after_redistribution(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=40.0)
        outflow2 = MockEdge(target_id="target2", capacity=45.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.5})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 40.0
        assert outflow2.flow == 45.0
        expected_spill = (100.0 - 40.0 - 45.0) * 3600.0
        assert np.isclose(node.spill_register[0], expected_spill)

    def test_monthly_parameter_variation(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        for i in range(24):
            inflow.set_flow_at_timestep(i, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=100.0)
        outflow2 = MockEdge(target_id="target2", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        params1 = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        params2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

        node.set_distribution_parameters({"target1": params1, "target2": params2})

        for month in range(12):
            node.update(time_step=month, dt=3600.0)
            assert np.isclose(outflow1.flow, 100.0 * params1[month])
            assert np.isclose(outflow2.flow, 100.0 * params2[month])

        for month in range(12, 24):
            node.update(time_step=month, dt=3600.0)
            assert np.isclose(outflow1.flow, 100.0 * params1[month % 12])
            assert np.isclose(outflow2.flow, 100.0 * params2[month % 12])

    def test_raises_error_when_distribution_params_not_set(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)
        outflow1 = MockEdge(target_id="target1", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)

        with pytest.raises(ValueError, match="Distribution parameters not set"):
            node.update(time_step=0, dt=3600.0)

    def test_zero_inflow_results_in_zero_outflows(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 0.0)

        outflow1 = MockEdge(target_id="target1", capacity=100.0)
        outflow2 = MockEdge(target_id="target2", capacity=100.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.5})

        node.update(time_step=0, dt=3600.0)

        assert outflow1.flow == 0.0
        assert outflow2.flow == 0.0
        assert node.spill_register == [0.0]

    def test_multiple_timesteps_accumulate_spill_register(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)
        inflow.set_flow_at_timestep(1, 80.0)
        inflow.set_flow_at_timestep(2, 60.0)

        outflow1 = MockEdge(target_id="target1", capacity=30.0)
        outflow2 = MockEdge(target_id="target2", capacity=30.0)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)
        node.add_outflow_edge(outflow2)

        node.set_distribution_parameters({"target1": 0.5, "target2": 0.5})

        node.update(time_step=0, dt=3600.0)
        node.update(time_step=1, dt=3600.0)
        node.update(time_step=2, dt=3600.0)

        assert len(node.spill_register) == 3
        assert np.isclose(node.spill_register[0], (100.0 - 60.0) * 3600.0)
        assert np.isclose(node.spill_register[1], (80.0 - 60.0) * 3600.0)
        assert node.spill_register[2] == 0.0

    def test_very_small_spills_treated_as_zero(self):
        node = HydroWorks(id="hw1", easting=1000.0, northing=2000.0)

        inflow = MockEdge(source_id="source1")
        inflow.set_flow_at_timestep(0, 100.0)

        outflow1 = MockEdge(target_id="target1", capacity=99.999999999)

        node.add_inflow_edge(inflow)
        node.add_outflow_edge(outflow1)

        node.set_distribution_parameters({"target1": 1.0})

        node.update(time_step=0, dt=3600.0)

        assert node.spill_register[0] == 0.0
