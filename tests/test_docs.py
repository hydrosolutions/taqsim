from pathlib import Path

from taqsim import get_docs_path


def test_documentation_describes_the_modelling_layer() -> None:
    docs = get_docs_path()
    assert isinstance(docs, Path)
    overview = docs / "00_modelling_layer.md"
    assert overview.is_file()
    assert "incidence engine" in overview.read_text()
