from __future__ import annotations

import json
from pathlib import Path

from taqsim.system import WaterSystem
from taqsim.time import Frequency


def minimal_system() -> WaterSystem:
    return WaterSystem.from_json(
        json.dumps(
            {
                "frequency": "monthly",
                "nodes": [
                    {"type": "source", "id": "src"},
                    {"type": "sink", "id": "snk"},
                ],
                "edges": [{"id": "e1", "source": "src", "target": "snk"}],
            }
        )
    )


def full_system() -> WaterSystem:
    return WaterSystem.from_json(
        json.dumps(
            {
                "frequency": "daily",
                "start_date": "2024-01-15",
                "nodes": [
                    {"type": "source", "id": "river"},
                    {
                        "type": "storage",
                        "id": "dam",
                        "capacity": 500.0,
                        "initial_storage": 200.0,
                        "dead_storage": 10.0,
                    },
                    {
                        "type": "demand",
                        "id": "city",
                        "consumption_fraction": 0.8,
                        "efficiency": 0.9,
                    },
                    {"type": "splitter", "id": "junction"},
                    {"type": "reach", "id": "canal", "capacity": 2000.0},
                    {"type": "passthrough", "id": "gauge"},
                    {"type": "sink", "id": "ocean"},
                ],
                "edges": [
                    {"id": "e1", "source": "river", "target": "dam"},
                    {"id": "e2", "source": "dam", "target": "junction"},
                    {"id": "e3", "source": "junction", "target": "canal"},
                    {"id": "e4", "source": "canal", "target": "gauge"},
                    {"id": "e5", "source": "gauge", "target": "city"},
                    {"id": "e6", "source": "city", "target": "ocean"},
                    {"id": "e7", "source": "junction", "target": "ocean"},
                ],
            }
        )
    )


def find_node(result: dict, node_id: str) -> dict:
    return next(n for n in result["nodes"] if n["id"] == node_id)


def find_edge(result: dict, edge_id: str) -> dict:
    return next(e for e in result["edges"] if e["id"] == edge_id)


class TestToJsonBasic:
    def test_returns_dict(self):
        result = minimal_system().to_json()
        assert isinstance(result, dict)

    def test_has_frequency_key(self):
        result = minimal_system().to_json()
        assert "frequency" in result

    def test_has_nodes_key(self):
        result = minimal_system().to_json()
        assert "nodes" in result

    def test_has_edges_key(self):
        result = minimal_system().to_json()
        assert "edges" in result

    def test_frequency_is_lowercase_string(self):
        result = minimal_system().to_json()
        assert result["frequency"] == "monthly"

    def test_start_date_as_iso_string(self):
        result = full_system().to_json()
        assert result["start_date"] == "2024-01-15"

    def test_start_date_omitted_when_none(self):
        result = minimal_system().to_json()
        assert "start_date" not in result

    def test_nodes_is_list(self):
        result = minimal_system().to_json()
        assert isinstance(result["nodes"], list)

    def test_edges_is_list(self):
        result = minimal_system().to_json()
        assert isinstance(result["edges"], list)

    def test_empty_system(self):
        system = WaterSystem(frequency=Frequency.MONTHLY)
        result = system.to_json()
        assert result["nodes"] == []
        assert result["edges"] == []


class TestToJsonNodes:
    def test_source_type(self):
        result = full_system().to_json()
        node = find_node(result, "river")
        assert node["type"] == "source"

    def test_storage_type(self):
        result = full_system().to_json()
        node = find_node(result, "dam")
        assert node["type"] == "storage"

    def test_demand_type(self):
        result = full_system().to_json()
        node = find_node(result, "city")
        assert node["type"] == "demand"

    def test_splitter_type(self):
        result = full_system().to_json()
        node = find_node(result, "junction")
        assert node["type"] == "splitter"

    def test_reach_type(self):
        result = full_system().to_json()
        node = find_node(result, "canal")
        assert node["type"] == "reach"

    def test_passthrough_type(self):
        result = full_system().to_json()
        node = find_node(result, "gauge")
        assert node["type"] == "passthrough"

    def test_sink_type(self):
        result = full_system().to_json()
        node = find_node(result, "ocean")
        assert node["type"] == "sink"

    def test_node_id(self):
        result = full_system().to_json()
        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"river", "dam", "city", "junction", "canal", "gauge", "ocean"}

    def test_location_as_list(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "location": [31.5, 35.2]},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "src")
        assert node["location"] == [31.5, 35.2]

    def test_tags_as_sorted_list(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "tags": ["critical", "main", "alpha"]},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "src")
        assert node["tags"] == ["alpha", "critical", "main"]

    def test_metadata_preserved(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "metadata": {"region": "north", "priority": 1}},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "src")
        assert node["metadata"] == {"region": "north", "priority": 1}

    def test_auxiliary_data_preserved(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "auxiliary_data": {"slope": 0.01}},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "src")
        assert node["auxiliary_data"] == {"slope": 0.01}

    def test_storage_capacity(self):
        result = full_system().to_json()
        node = find_node(result, "dam")
        assert node["capacity"] == 500.0

    def test_storage_initial_storage(self):
        result = full_system().to_json()
        node = find_node(result, "dam")
        assert node["initial_storage"] == 200.0

    def test_storage_dead_storage(self):
        result = full_system().to_json()
        node = find_node(result, "dam")
        assert node["dead_storage"] == 10.0

    def test_storage_defaults_always_included(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "storage", "id": "dam", "capacity": 100.0},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "dam"},
                        {"id": "e2", "source": "dam", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "dam")
        assert node["initial_storage"] == 0.0
        assert node["dead_storage"] == 0.0

    def test_demand_consumption_fraction(self):
        result = full_system().to_json()
        node = find_node(result, "city")
        assert node["consumption_fraction"] == 0.8

    def test_demand_efficiency(self):
        result = full_system().to_json()
        node = find_node(result, "city")
        assert node["efficiency"] == 0.9

    def test_demand_defaults_always_included(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "demand", "id": "city"},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "city"},
                        {"id": "e2", "source": "city", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "city")
        assert node["consumption_fraction"] == 1.0
        assert node["efficiency"] == 1.0

    def test_reach_capacity_when_set(self):
        result = full_system().to_json()
        node = find_node(result, "canal")
        assert node["capacity"] == 2000.0

    def test_reach_capacity_omitted_when_none(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "reach", "id": "canal"},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "canal"},
                        {"id": "e2", "source": "canal", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "canal")
        assert "capacity" not in node

    def test_passthrough_capacity_when_set(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "passthrough", "id": "gauge", "capacity": 750.0},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "gauge"},
                        {"id": "e2", "source": "gauge", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "gauge")
        assert node["capacity"] == 750.0

    def test_passthrough_capacity_omitted_when_none(self):
        result = full_system().to_json()
        node = find_node(result, "gauge")
        assert "capacity" not in node

    def test_strategies_not_serialized(self):
        result = full_system().to_json()
        strategy_keys = {
            "routing_model",
            "loss_rule",
            "release_policy",
            "split_policy",
            "inflow",
            "requirement",
        }
        for node in result["nodes"]:
            for key in strategy_keys:
                assert key not in node, f"Strategy key '{key}' found in node '{node['id']}'"


class TestToJsonEdges:
    def test_edge_id(self):
        result = minimal_system().to_json()
        edge = find_edge(result, "e1")
        assert edge["id"] == "e1"

    def test_edge_source_target(self):
        result = minimal_system().to_json()
        edge = find_edge(result, "e1")
        assert edge["source"] == "src"
        assert edge["target"] == "snk"

    def test_edge_tags_when_present(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [{"type": "source", "id": "s"}, {"type": "sink", "id": "t"}],
                    "edges": [{"id": "e1", "source": "s", "target": "t", "tags": ["main"]}],
                }
            )
        )
        result = system.to_json()
        edge = find_edge(result, "e1")
        assert edge["tags"] == ["main"]

    def test_edge_metadata_when_present(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [{"type": "source", "id": "s"}, {"type": "sink", "id": "t"}],
                    "edges": [{"id": "e1", "source": "s", "target": "t", "metadata": {"flow": "high"}}],
                }
            )
        )
        result = system.to_json()
        edge = find_edge(result, "e1")
        assert edge["metadata"] == {"flow": "high"}


class TestToJsonDefaults:
    def test_location_omitted_when_none(self):
        result = minimal_system().to_json()
        node = find_node(result, "src")
        assert "location" not in node

    def test_empty_tags_omitted(self):
        result = minimal_system().to_json()
        node = find_node(result, "src")
        assert "tags" not in node

    def test_empty_metadata_omitted(self):
        result = minimal_system().to_json()
        node = find_node(result, "src")
        assert "metadata" not in node

    def test_empty_auxiliary_data_omitted(self):
        result = minimal_system().to_json()
        node = find_node(result, "src")
        assert "auxiliary_data" not in node

    def test_edge_empty_tags_omitted(self):
        result = minimal_system().to_json()
        edge = find_edge(result, "e1")
        assert "tags" not in edge

    def test_edge_empty_metadata_omitted(self):
        result = minimal_system().to_json()
        edge = find_edge(result, "e1")
        assert "metadata" not in edge

    def test_storage_type_specific_defaults_always_included(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "storage", "id": "dam", "capacity": 100.0},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "dam"},
                        {"id": "e2", "source": "dam", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "dam")
        assert "capacity" in node
        assert node["initial_storage"] == 0.0
        assert node["dead_storage"] == 0.0

    def test_demand_type_specific_defaults_always_included(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "demand", "id": "city"},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "city"},
                        {"id": "e2", "source": "city", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "city")
        assert node["consumption_fraction"] == 1.0
        assert node["efficiency"] == 1.0

    def test_reach_no_capacity_by_default(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "reach", "id": "canal"},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "canal"},
                        {"id": "e2", "source": "canal", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "canal")
        assert "capacity" not in node

    def test_passthrough_no_capacity_by_default(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "passthrough", "id": "gauge"},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "gauge"},
                        {"id": "e2", "source": "gauge", "target": "snk"},
                    ],
                }
            )
        )
        result = system.to_json()
        node = find_node(result, "gauge")
        assert "capacity" not in node


class TestToJsonRoundTrip:
    def test_roundtrip_frequency(self):
        original = minimal_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.frequency == Frequency.MONTHLY

    def test_roundtrip_start_date(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        from datetime import date

        assert rebuilt.start_date == date(2024, 1, 15)

    def test_roundtrip_start_date_none(self):
        original = minimal_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.start_date is None

    def test_roundtrip_node_ids(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert set(rebuilt.nodes.keys()) == {"river", "dam", "city", "junction", "canal", "gauge", "ocean"}

    def test_roundtrip_node_types(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert type(rebuilt.nodes["river"]).__name__ == "Source"
        assert type(rebuilt.nodes["dam"]).__name__ == "Storage"
        assert type(rebuilt.nodes["city"]).__name__ == "Demand"
        assert type(rebuilt.nodes["junction"]).__name__ == "Splitter"
        assert type(rebuilt.nodes["canal"]).__name__ == "Reach"
        assert type(rebuilt.nodes["gauge"]).__name__ == "PassThrough"
        assert type(rebuilt.nodes["ocean"]).__name__ == "Sink"

    def test_roundtrip_edge_ids(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert set(rebuilt.edges.keys()) == {"e1", "e2", "e3", "e4", "e5", "e6", "e7"}

    def test_roundtrip_edge_source_target(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.edges["e1"].source == "river"
        assert rebuilt.edges["e1"].target == "dam"
        assert rebuilt.edges["e3"].source == "junction"
        assert rebuilt.edges["e3"].target == "canal"

    def test_roundtrip_storage_fields(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        dam = rebuilt.nodes["dam"]
        assert dam.capacity == 500.0
        assert dam.initial_storage == 200.0
        assert dam.dead_storage == 10.0

    def test_roundtrip_demand_fields(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        city = rebuilt.nodes["city"]
        assert city.consumption_fraction == 0.8
        assert city.efficiency == 0.9

    def test_roundtrip_reach_capacity(self):
        original = full_system().to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.nodes["canal"].capacity == 2000.0

    def test_roundtrip_passthrough_capacity(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src"},
                        {"type": "passthrough", "id": "gauge", "capacity": 750.0},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [
                        {"id": "e1", "source": "src", "target": "gauge"},
                        {"id": "e2", "source": "gauge", "target": "snk"},
                    ],
                }
            )
        )
        original = system.to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.nodes["gauge"].capacity == 750.0

    def test_roundtrip_location(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "location": [31.5, 35.2]},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        original = system.to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.nodes["src"].location == (31.5, 35.2)

    def test_roundtrip_tags(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "tags": ["critical", "main"]},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        original = system.to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.nodes["src"].tags == frozenset({"critical", "main"})

    def test_roundtrip_metadata(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "metadata": {"region": "north"}},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        original = system.to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.nodes["src"].metadata == {"region": "north"}

    def test_roundtrip_auxiliary_data(self):
        system = WaterSystem.from_json(
            json.dumps(
                {
                    "frequency": "monthly",
                    "nodes": [
                        {"type": "source", "id": "src", "auxiliary_data": {"slope": 0.01}},
                        {"type": "sink", "id": "snk"},
                    ],
                    "edges": [{"id": "e1", "source": "src", "target": "snk"}],
                }
            )
        )
        original = system.to_json()
        rebuilt = WaterSystem.from_json(json.dumps(original))
        assert rebuilt.nodes["src"].auxiliary_data == {"slope": 0.01}

    def test_roundtrip_full_system(self):
        system = full_system()
        first = system.to_json()
        rebuilt = WaterSystem.from_json(json.dumps(first))
        second = rebuilt.to_json()
        assert first == second


class TestToJsonSaveTo:
    def test_writes_file(self, tmp_path: Path):
        out = tmp_path / "out.json"
        minimal_system().to_json(save_to=out)
        assert out.exists()

    def test_written_file_is_valid_json(self, tmp_path: Path):
        out = tmp_path / "out.json"
        minimal_system().to_json(save_to=out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_returns_dict_when_saving(self, tmp_path: Path):
        out = tmp_path / "out.json"
        result = minimal_system().to_json(save_to=out)
        assert isinstance(result, dict)

    def test_accepts_str_path(self, tmp_path: Path):
        out = tmp_path / "out.json"
        minimal_system().to_json(save_to=str(out))
        assert out.exists()

    def test_accepts_path_object(self, tmp_path: Path):
        out = tmp_path / "out.json"
        minimal_system().to_json(save_to=Path(out))
        assert out.exists()

    def test_saved_file_roundtrips(self, tmp_path: Path):
        out = tmp_path / "system.json"
        original = full_system()
        original_json = original.to_json(save_to=out)
        rebuilt = WaterSystem.from_json(out)
        rebuilt_json = rebuilt.to_json()
        assert original_json == rebuilt_json
