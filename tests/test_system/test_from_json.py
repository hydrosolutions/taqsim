from __future__ import annotations

import json

import pytest

from taqsim.node import (
    Demand,
    NoLoss,
    NoReachLoss,
    NoRelease,
    NoRouting,
    NoSplit,
    PassThrough,
    Reach,
    Sink,
    Source,
    Splitter,
    Storage,
    TimeSeries,
)
from taqsim.system import WaterSystem
from taqsim.time import Frequency


def minimal_json_dict() -> dict:
    return {
        "frequency": "monthly",
        "nodes": [
            {"type": "source", "id": "src"},
            {"type": "sink", "id": "snk"},
        ],
        "edges": [
            {"id": "e1", "source": "src", "target": "snk"},
        ],
    }


def full_json_dict() -> dict:
    return {
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
            {"type": "passthrough", "id": "gauge", "capacity": 1000.0},
            {"type": "sink", "id": "ocean"},
        ],
        "edges": [
            {"id": "e1", "source": "river", "target": "dam"},
            {"id": "e2", "source": "dam", "target": "junction"},
            {"id": "e3", "source": "junction", "target": "canal"},
            {"id": "e4", "source": "junction", "target": "city"},
            {"id": "e5", "source": "canal", "target": "gauge"},
            {"id": "e6", "source": "gauge", "target": "ocean"},
            {"id": "e7", "source": "city", "target": "ocean"},
        ],
    }


def as_json(d: dict) -> str:
    return json.dumps(d)


class TestFromJsonBasic:
    def test_parses_frequency(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert system.frequency == Frequency.MONTHLY

    def test_parses_start_date(self):
        d = minimal_json_dict()
        d["start_date"] = "2024-06-01"
        system = WaterSystem.from_json(as_json(d))
        from datetime import date

        assert system.start_date == date(2024, 6, 1)

    def test_start_date_defaults_to_none(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert system.start_date is None

    def test_frequency_is_case_insensitive(self):
        d = minimal_json_dict()
        d["frequency"] = "Monthly"
        system = WaterSystem.from_json(as_json(d))
        assert system.frequency == Frequency.MONTHLY

        d["frequency"] = "DAILY"
        system = WaterSystem.from_json(as_json(d))
        assert system.frequency == Frequency.DAILY


class TestFromJsonNodes:
    def test_parses_source_node(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert "src" in system.nodes
        assert isinstance(system.nodes["src"], Source)

    def test_parses_sink_node(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert "snk" in system.nodes
        assert isinstance(system.nodes["snk"], Sink)

    def test_parses_storage_node(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "storage", "id": "dam", "capacity": 500.0, "initial_storage": 100.0})
        d["edges"] = [
            {"id": "e1", "source": "src", "target": "dam"},
            {"id": "e2", "source": "dam", "target": "snk"},
        ]
        system = WaterSystem.from_json(as_json(d))
        dam = system.nodes["dam"]
        assert isinstance(dam, Storage)
        assert dam.capacity == 500.0
        assert dam.initial_storage == 100.0

    def test_parses_demand_node(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "demand", "id": "city", "consumption_fraction": 0.8, "efficiency": 0.95})
        d["edges"] = [
            {"id": "e1", "source": "src", "target": "city"},
            {"id": "e2", "source": "city", "target": "snk"},
        ]
        system = WaterSystem.from_json(as_json(d))
        city = system.nodes["city"]
        assert isinstance(city, Demand)
        assert city.consumption_fraction == 0.8
        assert city.efficiency == 0.95

    def test_parses_splitter_node(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "splitter", "id": "junc"})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["junc"], Splitter)

    def test_parses_reach_node(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "reach", "id": "canal"})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["canal"], Reach)

    def test_parses_passthrough_node(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "passthrough", "id": "gauge"})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["gauge"], PassThrough)

    def test_parses_reach_with_capacity(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "reach", "id": "canal", "capacity": 500.0})
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["canal"].capacity == 500.0

    def test_reach_capacity_defaults_to_none(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "reach", "id": "canal"})
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["canal"].capacity is None

    def test_parses_passthrough_with_capacity(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "passthrough", "id": "gauge", "capacity": 500.0})
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["gauge"].capacity == 500.0

    def test_parses_location(self):
        d = minimal_json_dict()
        d["nodes"][0]["location"] = [35.5, -120.3]
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["src"].location == (35.5, -120.3)

    def test_parses_tags(self):
        d = minimal_json_dict()
        d["nodes"][0]["tags"] = ["upstream", "monitored"]
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["src"].tags == frozenset({"upstream", "monitored"})

    def test_parses_metadata(self):
        d = minimal_json_dict()
        d["nodes"][0]["metadata"] = {"owner": "agency_a"}
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["src"].metadata == {"owner": "agency_a"}

    def test_parses_auxiliary_data(self):
        d = minimal_json_dict()
        d["nodes"][0]["auxiliary_data"] = {"temperature": [20, 22, 25]}
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["src"].auxiliary_data == {"temperature": [20, 22, 25]}

    def test_defaults_for_optional_fields(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        src = system.nodes["src"]
        assert src.location is None
        assert src.tags == frozenset()
        assert src.metadata == {}
        assert src.auxiliary_data == {}

    def test_node_type_is_case_insensitive(self):
        d = minimal_json_dict()
        d["nodes"][0]["type"] = "Source"
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["src"], Source)


class TestFromJsonPlaceholders:
    def test_source_inflow_is_none(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert system.nodes["src"].inflow is None

    def test_storage_has_no_release(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "storage", "id": "dam", "capacity": 100.0})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["dam"].release_policy, NoRelease)

    def test_storage_has_no_loss(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "storage", "id": "dam", "capacity": 100.0})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["dam"].loss_rule, NoLoss)

    def test_demand_requirement_is_none(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "demand", "id": "city"})
        system = WaterSystem.from_json(as_json(d))
        assert system.nodes["city"].requirement is None

    def test_splitter_has_no_split(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "splitter", "id": "junc"})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["junc"].split_policy, NoSplit)

    def test_reach_has_no_routing(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "reach", "id": "canal"})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["canal"].routing_model, NoRouting)

    def test_reach_has_no_reach_loss(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "reach", "id": "canal"})
        system = WaterSystem.from_json(as_json(d))
        assert isinstance(system.nodes["canal"].loss_rule, NoReachLoss)


class TestFromJsonEdges:
    def test_parses_edge_id_source_target(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert "e1" in system.edges
        e = system.edges["e1"]
        assert e.source == "src"
        assert e.target == "snk"

    def test_parses_edge_tags(self):
        d = minimal_json_dict()
        d["edges"][0]["tags"] = ["main", "natural"]
        system = WaterSystem.from_json(as_json(d))
        assert system.edges["e1"].tags == frozenset({"main", "natural"})

    def test_parses_edge_metadata(self):
        d = minimal_json_dict()
        d["edges"][0]["metadata"] = {"length_km": 42.5}
        system = WaterSystem.from_json(as_json(d))
        assert system.edges["e1"].metadata == {"length_km": 42.5}

    def test_edge_defaults_for_optional_fields(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        e = system.edges["e1"]
        assert e.tags == frozenset()
        assert e.metadata == {}


class TestFromJsonInput:
    def test_reads_from_file_path_string(self, tmp_path):
        f = tmp_path / "net.json"
        f.write_text(as_json(minimal_json_dict()))
        system = WaterSystem.from_json(str(f))
        assert system.frequency == Frequency.MONTHLY

    def test_reads_from_path_object(self, tmp_path):
        f = tmp_path / "net.json"
        f.write_text(as_json(minimal_json_dict()))
        system = WaterSystem.from_json(f)
        assert system.frequency == Frequency.MONTHLY

    def test_reads_from_json_string(self):
        system = WaterSystem.from_json(as_json(minimal_json_dict()))
        assert system.frequency == Frequency.MONTHLY

    def test_file_not_found_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            WaterSystem.from_json("/nonexistent/path/net.json")

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            WaterSystem.from_json("{not valid json!!")


class TestFromJsonErrors:
    def test_missing_frequency_raises(self):
        d = {"nodes": [], "edges": []}
        with pytest.raises(ValueError, match="frequency"):
            WaterSystem.from_json(as_json(d))

    def test_invalid_frequency_raises(self):
        d = {"frequency": "biweekly", "nodes": [], "edges": []}
        with pytest.raises(ValueError, match="biweekly"):
            WaterSystem.from_json(as_json(d))

    def test_missing_node_type_raises(self):
        d = {"frequency": "monthly", "nodes": [{"id": "x"}], "edges": []}
        with pytest.raises(ValueError, match="type"):
            WaterSystem.from_json(as_json(d))

    def test_missing_node_id_raises(self):
        d = {"frequency": "monthly", "nodes": [{"type": "source"}], "edges": []}
        with pytest.raises(ValueError, match="id"):
            WaterSystem.from_json(as_json(d))

    def test_unknown_node_type_raises(self):
        d = {"frequency": "monthly", "nodes": [{"type": "unknown", "id": "x"}], "edges": []}
        with pytest.raises(ValueError, match="Unknown node type"):
            WaterSystem.from_json(as_json(d))

    def test_storage_missing_capacity_raises(self):
        d = {"frequency": "monthly", "nodes": [{"type": "storage", "id": "dam"}], "edges": []}
        with pytest.raises(ValueError, match="capacity"):
            WaterSystem.from_json(as_json(d))

    def test_duplicate_node_id_raises(self):
        d = {
            "frequency": "monthly",
            "nodes": [
                {"type": "source", "id": "x"},
                {"type": "sink", "id": "x"},
            ],
            "edges": [],
        }
        with pytest.raises(ValueError, match="already exists"):
            WaterSystem.from_json(as_json(d))

    def test_duplicate_edge_id_raises(self):
        d = {
            "frequency": "monthly",
            "nodes": [{"type": "source", "id": "a"}, {"type": "sink", "id": "b"}],
            "edges": [
                {"id": "e1", "source": "a", "target": "b"},
                {"id": "e1", "source": "a", "target": "b"},
            ],
        }
        with pytest.raises(ValueError, match="already exists"):
            WaterSystem.from_json(as_json(d))

    def test_reach_negative_capacity_raises(self):
        d = minimal_json_dict()
        d["nodes"].insert(1, {"type": "reach", "id": "canal", "capacity": -10.0})
        with pytest.raises(ValueError, match="capacity must be positive"):
            WaterSystem.from_json(as_json(d))

    def test_invalid_start_date_raises(self):
        d = minimal_json_dict()
        d["start_date"] = "not-a-date"
        with pytest.raises(ValueError, match="start_date"):
            WaterSystem.from_json(as_json(d))


class TestFromJsonIntegration:
    def test_topology_matches_json(self):
        system = WaterSystem.from_json(as_json(full_json_dict()))
        assert len(system.nodes) == 7
        assert len(system.edges) == 7
        assert set(system.nodes.keys()) == {"river", "dam", "city", "junction", "canal", "gauge", "ocean"}
        assert set(system.edges.keys()) == {"e1", "e2", "e3", "e4", "e5", "e6", "e7"}

    def test_full_round_trip_load_inject_simulate(self):
        system = WaterSystem.from_json(as_json(full_json_dict()))

        # Inject strategies and time series
        system.nodes["river"].inflow = TimeSeries([100.0] * 12)
        system.nodes["dam"].release_policy = _FullRelease()
        system.nodes["dam"].loss_rule = NoLoss()
        system.nodes["city"].requirement = TimeSeries([30.0] * 12)
        system.nodes["junction"].split_policy = _EvenSplit()

        system.simulate(timesteps=3)

        # Verify simulation ran (source generated events)
        from taqsim.node.events import WaterGenerated

        gen_events = system.nodes["river"].events_of_type(WaterGenerated)
        assert len(gen_events) == 3


class _FullRelease:
    def release(self, node, inflow: float, t) -> float:
        return node.storage


class _EvenSplit:
    def split(self, node, amount: float, t) -> dict[str, float]:
        targets = node.targets
        if not targets:
            return {}
        share = amount / len(targets)
        return dict.fromkeys(targets, share)
