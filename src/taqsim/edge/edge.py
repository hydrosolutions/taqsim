from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    id: str
    source: str
    target: str
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.source:
            raise ValueError("source cannot be empty")
        if not self.target:
            raise ValueError("target cannot be empty")

    def _fresh_copy(self) -> "Edge":
        return Edge(
            id=self.id,
            source=self.source,
            target=self.target,
            tags=self.tags,
            metadata=self.metadata,
        )
