from typing import Any

import pytest

from taqsim.edge.edge import Edge


def make_edge(
    id: str = "test_edge",
    source: str = "source_node",
    target: str = "target_node",
    tags: frozenset[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Edge:
    return Edge(
        id=id,
        source=source,
        target=target,
        tags=tags or frozenset(),
        metadata=metadata or {},
    )


class TestEdgeInit:
    def test_edge_requires_id(self):
        with pytest.raises(ValueError, match="id cannot be empty"):
            make_edge(id="")

    def test_edge_requires_source(self):
        with pytest.raises(ValueError, match="source cannot be empty"):
            make_edge(source="")

    def test_edge_requires_target(self):
        with pytest.raises(ValueError, match="target cannot be empty"):
            make_edge(target="")

    def test_edge_creates_with_valid_params(self):
        edge = make_edge(
            id="my_edge",
            source="node_a",
            target="node_b",
        )
        assert edge.id == "my_edge"
        assert edge.source == "node_a"
        assert edge.target == "node_b"


class TestEdgeTags:
    def test_default_tags_is_empty_frozenset(self):
        edge = make_edge()
        assert edge.tags == frozenset()

    def test_custom_tags_accepted(self):
        edge = make_edge(tags=frozenset({"canal", "primary"}))
        assert edge.tags == frozenset({"canal", "primary"})

    def test_tags_type_is_frozenset(self):
        edge = make_edge(tags=frozenset({"river"}))
        assert isinstance(edge.tags, frozenset)

    def test_tags_is_immutable(self):
        edge = make_edge(tags=frozenset({"original"}))
        with pytest.raises(AttributeError):
            edge.tags.add("new")


class TestEdgeMetadata:
    def test_default_metadata_is_empty_dict(self):
        edge = make_edge()
        assert edge.metadata == {}

    def test_custom_metadata_accepted(self):
        edge = make_edge(metadata={"length_km": 50.5, "material": "concrete"})
        assert edge.metadata == {"length_km": 50.5, "material": "concrete"}

    def test_metadata_type_is_dict(self):
        edge = make_edge(metadata={"key": "value"})
        assert isinstance(edge.metadata, dict)

    def test_metadata_not_shared_between_instances(self):
        e1 = make_edge(id="e1")
        e2 = make_edge(id="e2")
        e1.metadata["key"] = "value"
        assert "key" not in e2.metadata


class TestEdgeFreshCopy:
    def test_creates_new_instance(self):
        edge = make_edge()
        copy = edge._fresh_copy()
        assert copy is not edge
        assert type(copy) is Edge

    def test_shares_immutable_config(self):
        edge = make_edge()
        copy = edge._fresh_copy()
        assert copy.id == edge.id
        assert copy.source == edge.source
        assert copy.target == edge.target
        assert copy.tags is edge.tags

    def test_fresh_copy_preserves_metadata(self):
        edge = make_edge(metadata={"key": "value"})
        copy = edge._fresh_copy()
        assert copy.metadata == edge.metadata
