import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from taqsim.io_utils import (
    load_optimized_parameters,
    load_parameters_from_file,
    save_optimized_parameters,
)


class TestSaveOptimizedParameters:
    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested_path = tmp_path / "results" / "optimization"
        filename = str(nested_path / "params.json")

        optimization_results = {
            "population_size": 50,
            "generations": 100,
            "crossover_probability": 0.8,
            "mutation_probability": 0.2,
            "pareto_front": [],
        }

        save_optimized_parameters(optimization_results, filename)

        assert nested_path.exists()
        assert (nested_path / "params.json").exists()

    def test_saves_valid_json_file(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        mock_individual = Mock()
        mock_individual.fitness.values = (0.5, 0.8)

        optimization_results = {
            "population_size": 50,
            "generations": 100,
            "crossover_probability": 0.8,
            "mutation_probability": 0.2,
            "pareto_front": [mock_individual],
            "optimizer": None,
        }

        save_optimized_parameters(optimization_results, filename)

        with open(filename) as f:
            data = json.load(f)

        assert data["population_size"] == 50
        assert data["generations"] == 100
        assert data["crossover_probability"] == 0.8
        assert data["mutation_probability"] == 0.2
        assert data["num_pareto_solutions"] == 1

    def test_handles_numpy_arrays_in_conversion(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        mock_individual = Mock()
        mock_individual.fitness.values = (np.float64(0.5), np.float64(0.8))

        mock_optimizer = Mock()
        mock_optimizer.reservoir_ids = ["res1"]
        mock_optimizer.hydroworks_ids = ["hw1"]
        mock_optimizer.hydroworks_targets = {"hw1": 100.0}

        optimization_results = {
            "population_size": 50,
            "generations": 100,
            "crossover_probability": 0.8,
            "mutation_probability": 0.2,
            "pareto_front": [mock_individual],
            "optimizer": mock_optimizer,
        }

        save_optimized_parameters(optimization_results, filename)

        with open(filename) as f:
            data = json.load(f)

        assert isinstance(data["pareto_solutions"][0]["objective_1"], float)
        assert isinstance(data["pareto_solutions"][0]["objective_2"], float)

    def test_handles_tuples_in_conversion(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        mock_individual = Mock()
        mock_individual.fitness.values = (0.5, 0.8, 0.3)

        optimization_results = {
            "population_size": 50,
            "generations": 100,
            "crossover_probability": 0.8,
            "mutation_probability": 0.2,
            "pareto_front": [mock_individual],
            "optimizer": None,
        }

        save_optimized_parameters(optimization_results, filename)

        with open(filename) as f:
            data = json.load(f)

        assert isinstance(data["pareto_solutions"][0]["objective_values"], list)
        assert len(data["pareto_solutions"][0]["objective_values"]) == 3
        assert data["pareto_solutions"][0]["objective_1"] == 0.5
        assert data["pareto_solutions"][0]["objective_2"] == 0.8
        assert data["pareto_solutions"][0]["objective_3"] == 0.3

    def test_metadata_stored_correctly(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        mock_individual_1 = Mock()
        mock_individual_1.fitness.values = (0.5, 0.8)

        mock_individual_2 = Mock()
        mock_individual_2.fitness.values = (0.6, 0.7)

        optimization_results = {
            "population_size": 100,
            "generations": 200,
            "crossover_probability": 0.9,
            "mutation_probability": 0.1,
            "pareto_front": [mock_individual_1, mock_individual_2],
            "optimizer": None,
        }

        save_optimized_parameters(optimization_results, filename)

        with open(filename) as f:
            data = json.load(f)

        assert data["population_size"] == 100
        assert data["generations"] == 200
        assert data["crossover_probability"] == 0.9
        assert data["mutation_probability"] == 0.1
        assert data["num_pareto_solutions"] == 2
        assert len(data["pareto_solutions"]) == 2
        assert data["pareto_solutions"][0]["id"] == 0
        assert data["pareto_solutions"][1]["id"] == 1

    def test_saves_empty_pareto_front(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        optimization_results = {
            "population_size": 50,
            "generations": 100,
            "crossover_probability": 0.8,
            "mutation_probability": 0.2,
            "pareto_front": [],
        }

        save_optimized_parameters(optimization_results, filename)

        with open(filename) as f:
            data = json.load(f)

        assert data["num_pareto_solutions"] == 0
        assert "pareto_solutions" not in data

    def test_handles_missing_optimizer_gracefully(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        mock_individual = Mock()
        mock_individual.fitness.values = (0.5, 0.8)

        optimization_results = {
            "population_size": 50,
            "generations": 100,
            "crossover_probability": 0.8,
            "mutation_probability": 0.2,
            "pareto_front": [mock_individual],
        }

        save_optimized_parameters(optimization_results, filename)

        with open(filename) as f:
            data = json.load(f)

        assert data["pareto_solutions"][0]["reservoir_parameters"] is None
        assert data["pareto_solutions"][0]["hydroworks_parameters"] is None


class TestLoadOptimizedParameters:
    def test_finds_solution_by_id(self) -> None:
        pareto_solutions = [
            {"id": 0, "reservoir_parameters": {}, "hydroworks_parameters": {}},
            {"id": 1, "reservoir_parameters": {}, "hydroworks_parameters": {}},
            {"id": 2, "reservoir_parameters": {}, "hydroworks_parameters": {}},
        ]

        mock_system = Mock()
        mock_system.graph.nodes = {}

        result = load_optimized_parameters(mock_system, pareto_solutions, 1)

        assert result == mock_system

    def test_raises_valueerror_for_invalid_solution_id(self) -> None:
        pareto_solutions = [
            {"id": 0, "reservoir_parameters": {}, "hydroworks_parameters": {}},
            {"id": 1, "reservoir_parameters": {}, "hydroworks_parameters": {}},
        ]

        mock_system = Mock()

        with pytest.raises(ValueError, match="Solution with id 5 not found"):
            load_optimized_parameters(mock_system, pareto_solutions, 5)

    def test_updates_reservoir_nodes_with_parameters(self) -> None:
        reservoir_params = {
            "res1": [1.0, 2.0, 3.0],
            "res2": [4.0, 5.0, 6.0],
        }

        pareto_solutions = [
            {
                "id": 0,
                "reservoir_parameters": reservoir_params,
                "hydroworks_parameters": {},
            }
        ]

        mock_res1_node = Mock()
        mock_res2_node = Mock()

        mock_system = Mock()
        mock_system.graph.nodes = {
            "res1": {"node": mock_res1_node},
            "res2": {"node": mock_res2_node},
        }

        load_optimized_parameters(mock_system, pareto_solutions, 0)

        mock_res1_node.set_release_params.assert_called_once_with([1.0, 2.0, 3.0])
        mock_res2_node.set_release_params.assert_called_once_with([4.0, 5.0, 6.0])

    def test_updates_hydroworks_nodes_with_parameters(self) -> None:
        hydroworks_params = {
            "hw1": [0.5, 0.3, 0.2],
            "hw2": [0.4, 0.4, 0.2],
        }

        pareto_solutions = [
            {
                "id": 0,
                "reservoir_parameters": {},
                "hydroworks_parameters": hydroworks_params,
            }
        ]

        mock_hw1_node = Mock()
        mock_hw2_node = Mock()

        mock_system = Mock()
        mock_system.graph.nodes = {
            "hw1": {"node": mock_hw1_node},
            "hw2": {"node": mock_hw2_node},
        }

        load_optimized_parameters(mock_system, pareto_solutions, 0)

        mock_hw1_node.set_distribution_parameters.assert_called_once_with([0.5, 0.3, 0.2])
        mock_hw2_node.set_distribution_parameters.assert_called_once_with([0.4, 0.4, 0.2])

    def test_raises_valueerror_for_non_reservoir_node(self) -> None:
        reservoir_params = {"res1": [1.0, 2.0, 3.0]}

        pareto_solutions = [
            {
                "id": 0,
                "reservoir_parameters": reservoir_params,
                "hydroworks_parameters": {},
            }
        ]

        mock_node = Mock()
        del mock_node.set_release_params

        mock_system = Mock()
        mock_system.graph.nodes = {"res1": {"node": mock_node}}

        with pytest.raises(ValueError, match="Node res1 does not appear to be a StorageNode"):
            load_optimized_parameters(mock_system, pareto_solutions, 0)

    def test_raises_valueerror_for_non_hydroworks_node(self) -> None:
        hydroworks_params = {"hw1": [0.5, 0.3, 0.2]}

        pareto_solutions = [
            {
                "id": 0,
                "reservoir_parameters": {},
                "hydroworks_parameters": hydroworks_params,
            }
        ]

        mock_node = Mock()
        del mock_node.set_distribution_parameters

        mock_system = Mock()
        mock_system.graph.nodes = {"hw1": {"node": mock_node}}

        with pytest.raises(ValueError, match="Node hw1 does not appear to be a HydroWorks"):
            load_optimized_parameters(mock_system, pareto_solutions, 0)

    def test_handles_empty_parameters(self) -> None:
        pareto_solutions = [
            {
                "id": 0,
                "reservoir_parameters": {},
                "hydroworks_parameters": {},
            }
        ]

        mock_system = Mock()
        mock_system.graph.nodes = {}

        result = load_optimized_parameters(mock_system, pareto_solutions, 0)

        assert result == mock_system


class TestLoadParametersFromFile:
    def test_reads_valid_json_file(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        data = {
            "population_size": 50,
            "generations": 100,
            "pareto_solutions": [
                {"id": 0, "objective_values": [0.5, 0.8]},
                {"id": 1, "objective_values": [0.6, 0.7]},
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f)

        result = load_parameters_from_file(filename)

        assert len(result) == 2
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1

    def test_raises_filenotfounderror_for_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="Parameter file /nonexistent/file.json does not exist"):
            load_parameters_from_file("/nonexistent/file.json")

    def test_raises_valueerror_if_no_pareto_solutions_key(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        data = {
            "population_size": 50,
            "generations": 100,
        }

        with open(filename, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="does not contain Pareto solutions"):
            load_parameters_from_file(filename)

    def test_returns_empty_list_for_empty_pareto_solutions(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        data = {"population_size": 50, "generations": 100, "pareto_solutions": []}

        with open(filename, "w") as f:
            json.dump(data, f)

        result = load_parameters_from_file(filename)

        assert len(result) == 0
        assert result == []

    def test_preserves_all_solution_fields(self, tmp_path: Path) -> None:
        filename = str(tmp_path / "params.json")

        data = {
            "pareto_solutions": [
                {
                    "id": 0,
                    "objective_values": [0.5, 0.8],
                    "objective_1": 0.5,
                    "objective_2": 0.8,
                    "reservoir_parameters": {"res1": [1.0, 2.0]},
                    "hydroworks_parameters": {"hw1": [0.3, 0.7]},
                }
            ]
        }

        with open(filename, "w") as f:
            json.dump(data, f)

        result = load_parameters_from_file(filename)

        assert result[0]["id"] == 0
        assert result[0]["objective_values"] == [0.5, 0.8]
        assert result[0]["objective_1"] == 0.5
        assert result[0]["objective_2"] == 0.8
        assert result[0]["reservoir_parameters"] == {"res1": [1.0, 2.0]}
        assert result[0]["hydroworks_parameters"] == {"hw1": [0.3, 0.7]}
