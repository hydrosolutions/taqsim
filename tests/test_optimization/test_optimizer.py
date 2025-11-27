from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from taqsim.nodes import DemandNode, HydroWorks, SinkNode, StorageNode
from taqsim.optimization.optimizer import DeapOptimizer, decode_individual, normalize_distribution
from taqsim.water_system import WaterSystem


class TestNormalizeDistribution:
    def test_normalizes_positive_values_correctly(self):
        values = np.array([1.0, 2.0, 3.0])
        result = normalize_distribution(values)
        expected = np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_zero_sum_returns_equal_distribution(self):
        values = np.array([0.0, 0.0, 0.0])
        result = normalize_distribution(values)
        expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_result_sums_to_one(self):
        values = np.array([5.0, 10.0, 15.0, 20.0])
        result = normalize_distribution(values)
        assert np.isclose(np.sum(result), 1.0)

    def test_works_with_lists(self):
        values = [1.0, 2.0, 3.0]
        result = normalize_distribution(values)
        assert np.isclose(np.sum(result), 1.0)
        assert len(result) == 3

    def test_works_with_numpy_arrays(self):
        values = np.array([1.0, 2.0, 3.0])
        result = normalize_distribution(values)
        assert isinstance(result, np.ndarray)
        assert np.isclose(np.sum(result), 1.0)


class TestDecodeIndividual:
    def test_correct_gene_extraction_for_single_reservoir(self):
        reservoir_ids = ["res1"]
        hydroworks_ids = []
        hydroworks_targets = {}
        individual = []
        for month in range(12):
            individual.extend([10.0 + month, 100.0 + month, 200.0 + month])

        reservoir_params, hydroworks_params = decode_individual(
            reservoir_ids, hydroworks_ids, hydroworks_targets, individual
        )

        assert "res1" in reservoir_params
        assert len(reservoir_params["res1"]["Vr"]) == 12
        assert len(reservoir_params["res1"]["V1"]) == 12
        assert len(reservoir_params["res1"]["V2"]) == 12
        assert reservoir_params["res1"]["Vr"][0] == 10.0
        assert reservoir_params["res1"]["V1"][0] == 100.0
        assert reservoir_params["res1"]["V2"][0] == 200.0
        assert reservoir_params["res1"]["Vr"][11] == 21.0
        assert reservoir_params["res1"]["V1"][11] == 111.0
        assert reservoir_params["res1"]["V2"][11] == 211.0

    def test_correct_gene_extraction_for_multiple_reservoirs(self):
        reservoir_ids = ["res1", "res2"]
        hydroworks_ids = []
        hydroworks_targets = {}
        individual = []
        for month in range(12):
            individual.extend([10.0 + month, 100.0 + month, 200.0 + month])
        for month in range(12):
            individual.extend([20.0 + month, 300.0 + month, 400.0 + month])

        reservoir_params, _ = decode_individual(reservoir_ids, hydroworks_ids, hydroworks_targets, individual)

        assert "res1" in reservoir_params
        assert "res2" in reservoir_params
        assert reservoir_params["res1"]["Vr"][0] == 10.0
        assert reservoir_params["res2"]["Vr"][0] == 20.0

    def test_correct_gene_extraction_for_single_hydroworks(self):
        reservoir_ids = []
        hydroworks_ids = ["hw1"]
        hydroworks_targets = {"hw1": ["target1", "target2"]}
        individual = []
        for month in range(12):
            individual.extend([1.0 + month * 0.1, 2.0 + month * 0.1])

        _, hydroworks_params = decode_individual(reservoir_ids, hydroworks_ids, hydroworks_targets, individual)

        assert "hw1" in hydroworks_params
        assert "target1" in hydroworks_params["hw1"]
        assert "target2" in hydroworks_params["hw1"]
        assert len(hydroworks_params["hw1"]["target1"]) == 12
        assert len(hydroworks_params["hw1"]["target2"]) == 12

    def test_distribution_normalization_per_month(self):
        reservoir_ids = []
        hydroworks_ids = ["hw1"]
        hydroworks_targets = {"hw1": ["target1", "target2", "target3"]}
        individual = []
        for month in range(12):
            individual.extend([1.0, 2.0, 3.0])

        _, hydroworks_params = decode_individual(reservoir_ids, hydroworks_ids, hydroworks_targets, individual)

        for month in range(12):
            total = (
                hydroworks_params["hw1"]["target1"][month]
                + hydroworks_params["hw1"]["target2"][month]
                + hydroworks_params["hw1"]["target3"][month]
            )
            assert np.isclose(total, 1.0)

    def test_handles_zero_distribution(self):
        reservoir_ids = []
        hydroworks_ids = ["hw1"]
        hydroworks_targets = {"hw1": ["target1", "target2"]}
        individual = []
        for month in range(12):
            individual.extend([0.0, 0.0])

        _, hydroworks_params = decode_individual(reservoir_ids, hydroworks_ids, hydroworks_targets, individual)

        for month in range(12):
            assert np.isclose(hydroworks_params["hw1"]["target1"][month], 0.5)
            assert np.isclose(hydroworks_params["hw1"]["target2"][month], 0.5)

    def test_returns_properly_structured_dicts(self):
        reservoir_ids = ["res1"]
        hydroworks_ids = ["hw1"]
        hydroworks_targets = {"hw1": ["target1"]}
        individual = []
        for month in range(12):
            individual.extend([10.0, 100.0, 200.0])
        for month in range(12):
            individual.append(1.0)

        reservoir_params, hydroworks_params = decode_individual(
            reservoir_ids, hydroworks_ids, hydroworks_targets, individual
        )

        assert isinstance(reservoir_params, dict)
        assert isinstance(hydroworks_params, dict)
        assert isinstance(reservoir_params["res1"], dict)
        assert "Vr" in reservoir_params["res1"]
        assert "V1" in reservoir_params["res1"]
        assert "V2" in reservoir_params["res1"]


@pytest.fixture
def mock_water_system():
    system = Mock(spec=WaterSystem)
    system.dt = 2629800
    system.graph = MagicMock()

    mock_reservoir = Mock()
    mock_reservoir.capacity = 1000000
    mock_reservoir.dead_storage = 100000
    mock_reservoir.outflow_edge.capacity = 10.0

    mock_hydrowork = Mock()
    mock_hydrowork.outflow_edges = {"demand1": None}

    system.graph.nodes = {"res1": {"node": mock_reservoir}, "hw1": {"node": mock_hydrowork}}

    return system


@pytest.fixture
def simple_objective_weights():
    return {"objective_1": [1, 0, 0, 0, 0], "objective_2": [0, 1, 0, 0, 0]}


class TestDeapOptimizerInit:
    def test_extracts_node_ids_from_system(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = ["demand1"]
        DemandNode.low_priority_demand_ids = ["demand2"]
        SinkNode.all_ids = ["sink1"]

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        assert optimizer.reservoir_ids == ["res1"]
        assert optimizer.hydroworks_ids == ["hw1"]
        assert optimizer.priority_demand_ids == ["demand1"]
        assert optimizer.regular_demand_ids == ["demand2"]
        assert optimizer.sink_ids == ["sink1"]

    def test_calculates_reservoir_bounds(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        assert "res1" in optimizer.reservoir_bounds
        bounds = optimizer.reservoir_bounds["res1"]
        assert "Vr" in bounds
        assert "V1" in bounds
        assert "V2" in bounds
        assert bounds["Vr"] == (0, 10.0 * 2629800)
        assert bounds["V1"] == (100000, 1000000)
        assert bounds["V2"] == (100000, 1000000)

    def test_extracts_hydroworks_targets(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        assert "hw1" in optimizer.hydroworks_targets
        assert optimizer.hydroworks_targets["hw1"] == ["demand1"]

    def test_initializes_toolbox(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        assert optimizer.toolbox is not None
        assert hasattr(optimizer.toolbox, "individual")
        assert hasattr(optimizer.toolbox, "population")
        assert hasattr(optimizer.toolbox, "mate")
        assert hasattr(optimizer.toolbox, "mutate")
        assert hasattr(optimizer.toolbox, "select")


class TestDeapOptimizerCreateIndividual:
    def test_creates_valid_individual_with_correct_length(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()

        expected_length = (1 * 12 * 3) + (1 * 12 * 1)
        assert len(individual) == expected_length

    def test_values_within_bounds(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()

        reservoir_params, hydroworks_params = decode_individual(
            optimizer.reservoir_ids, optimizer.hydroworks_ids, optimizer.hydroworks_targets, individual
        )

        bounds = optimizer.reservoir_bounds["res1"]
        for month in range(12):
            Vr = reservoir_params["res1"]["Vr"][month]
            V1 = reservoir_params["res1"]["V1"][month]
            V2 = reservoir_params["res1"]["V2"][month]

            assert bounds["Vr"][0] <= Vr <= bounds["Vr"][1]
            assert bounds["V1"][0] <= V1 <= bounds["V1"][1]
            assert bounds["V2"][0] <= V2 <= bounds["V2"][1]
            assert V1 <= V2

    def test_hydroworks_distributions_sum_to_one(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        mock_water_system.graph.nodes["hw1"]["node"].outflow_edges = {"target1": None, "target2": None, "target3": None}

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()

        _, hydroworks_params = decode_individual(
            optimizer.reservoir_ids, optimizer.hydroworks_ids, optimizer.hydroworks_targets, individual
        )

        for month in range(12):
            total = sum(hydroworks_params["hw1"][target][month] for target in optimizer.hydroworks_targets["hw1"])
            assert np.isclose(total, 1.0)


class TestDeapOptimizerCrossoverMutation:
    def test_crossover_produces_valid_individuals(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        ind1 = optimizer.create_individual()
        ind2 = optimizer.create_individual()

        child1, child2 = optimizer.crossover(ind1, ind2)

        assert len(child1) == len(ind1)
        assert len(child2) == len(ind2)
        assert isinstance(child1, list)
        assert isinstance(child2, list)

    def test_crossover_preserves_hydroworks_normalization(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        mock_water_system.graph.nodes["hw1"]["node"].outflow_edges = {"target1": None, "target2": None}

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        ind1 = optimizer.create_individual()
        ind2 = optimizer.create_individual()

        child1, child2 = optimizer.crossover(ind1, ind2)

        _, hw_params1 = decode_individual(
            optimizer.reservoir_ids, optimizer.hydroworks_ids, optimizer.hydroworks_targets, child1
        )

        _, hw_params2 = decode_individual(
            optimizer.reservoir_ids, optimizer.hydroworks_ids, optimizer.hydroworks_targets, child2
        )

        for month in range(12):
            total1 = sum(hw_params1["hw1"][target][month] for target in optimizer.hydroworks_targets["hw1"])
            total2 = sum(hw_params2["hw1"][target][month] for target in optimizer.hydroworks_targets["hw1"])
            assert np.isclose(total1, 1.0)
            assert np.isclose(total2, 1.0)

    def test_mutation_produces_valid_individuals(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()

        (mutated,) = optimizer.mutate_individual(individual, indpb=1.0)

        assert len(mutated) == len(individual)
        assert isinstance(mutated, list)

    def test_mutation_maintains_bounds(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()

        (mutated,) = optimizer.mutate_individual(individual, indpb=1.0)

        reservoir_params, _ = decode_individual(
            optimizer.reservoir_ids, optimizer.hydroworks_ids, optimizer.hydroworks_targets, mutated
        )

        bounds = optimizer.reservoir_bounds["res1"]
        for month in range(12):
            Vr = reservoir_params["res1"]["Vr"][month]
            V1 = reservoir_params["res1"]["V1"][month]
            V2 = reservoir_params["res1"]["V2"][month]

            assert bounds["Vr"][0] <= Vr <= bounds["Vr"][1]
            assert bounds["V1"][0] <= V1 <= bounds["V1"][1]
            assert bounds["V2"][0] <= V2 <= bounds["V2"][1]
            assert V1 <= V2

    def test_mutation_maintains_hydroworks_normalization(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        mock_water_system.graph.nodes["hw1"]["node"].outflow_edges = {"target1": None, "target2": None}

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()

        (mutated,) = optimizer.mutate_individual(individual, indpb=1.0)

        _, hw_params = decode_individual(
            optimizer.reservoir_ids, optimizer.hydroworks_ids, optimizer.hydroworks_targets, mutated
        )

        for month in range(12):
            total = sum(hw_params["hw1"][target][month] for target in optimizer.hydroworks_targets["hw1"])
            assert np.isclose(total, 1.0)

    def test_mutation_with_zero_indpb_returns_unchanged(self, mock_water_system, simple_objective_weights):
        StorageNode.all_ids = ["res1"]
        HydroWorks.all_ids = ["hw1"]
        DemandNode.high_priority_demand_ids = []
        DemandNode.low_priority_demand_ids = []
        SinkNode.all_ids = []

        optimizer = DeapOptimizer(
            base_system=mock_water_system,
            num_time_steps=12,
            population_size=10,
            ngen=5,
            cxpb=0.65,
            mutpb=0.32,
            objective_weights=simple_objective_weights,
        )

        np.random.seed(42)
        individual = optimizer.create_individual()
        original = individual.copy()

        (mutated,) = optimizer.mutate_individual(individual, indpb=0.0)

        np.testing.assert_array_almost_equal(mutated, original)
