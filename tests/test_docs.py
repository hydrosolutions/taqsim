import re
from pathlib import Path

from taqsim import get_docs_path

EXPECTED_FILES = [
    "00_philosophy.md",
    "testing.md",
    "common/01_loss_reasons.md",
    "common/02_constraints.md",
    "common/03_tags_metadata.md",
    "common/04_frequency.md",
    "edges/01_architecture.md",
    "edges/04_edge_class.md",
    "nodes/01_architecture.md",
    "nodes/02_events.md",
    "nodes/03_capabilities.md",
    "nodes/04_timeseries.md",
    "nodes/05_strategies.md",
    "nodes/06_node_types.md",
    "nodes/07_parameter_introspection.md",
    "nodes/08_reach.md",
    "objective/01_overview.md",
    "objective/02_trace.md",
    "objective/03_lift.md",
    "objective/04_builtins.md",
    "objective/05_custom.md",
    "optimization/01_overview.md",
    "optimization/02_optimize_api.md",
    "optimization/03_results.md",
    "optimization/04_pareto_concepts.md",
    "optimization/05_examples.md",
    "system/01_architecture.md",
    "system/02_validation.md",
    "system/03_parameter_exposure.md",
]

EXPECTED_SUBDIRECTORIES = ["common", "edges", "nodes", "objective", "optimization", "system"]

CROSS_REF_PATTERN = re.compile(r"\]\(([^)]*\.md)\)")


class TestGetDocsPath:
    def test_returns_path_object(self) -> None:
        assert isinstance(get_docs_path(), Path)

    def test_path_exists_and_is_directory(self) -> None:
        docs = get_docs_path()
        assert docs.exists()
        assert docs.is_dir()

    def test_contains_all_expected_files(self) -> None:
        docs = get_docs_path()
        missing = [f for f in EXPECTED_FILES if not (docs / f).exists()]
        assert missing == [], f"Missing documentation files: {missing}"

    def test_contains_expected_subdirectories(self) -> None:
        docs = get_docs_path()
        missing = [d for d in EXPECTED_SUBDIRECTORIES if not (docs / d).is_dir()]
        assert missing == [], f"Missing subdirectories: {missing}"

    def test_cross_references_valid(self) -> None:
        docs = get_docs_path()
        broken: list[tuple[str, str]] = []
        for md_file in docs.rglob("*.md"):
            text = md_file.read_text()
            for match in CROSS_REF_PATTERN.finditer(text):
                ref = match.group(1)
                target = (md_file.parent / ref).resolve()
                if not target.exists():
                    relative_source = md_file.relative_to(docs)
                    broken.append((str(relative_source), ref))
        assert broken == [], f"Broken cross-references: {broken}"
