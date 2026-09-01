import re
from pathlib import Path

from taqsim import get_docs_path

README_EXAMPLE = re.compile(
    r"<!-- readme-example:start -->\s*```python\n(?P<source>.*?)```\s*<!-- readme-example:end -->",
    re.DOTALL,
)
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\((?P<target>[^)]+)\)")


def test_documentation_describes_the_modelling_layer() -> None:
    docs = get_docs_path()
    assert isinstance(docs, Path)
    overview = docs / "00_modelling_layer.md"
    assert overview.is_file()
    assert "incidence engine" in overview.read_text()


def test_readme_example_executes() -> None:
    readme = Path("README.md").read_text()
    match = README_EXAMPLE.search(readme)

    assert match is not None
    exec(compile(match.group("source"), "README.md", "exec"), {})


def test_readme_internal_links_exist() -> None:
    readme = Path("README.md").read_text()
    internal_targets = [
        match.group("target").split("#", maxsplit=1)[0]
        for match in MARKDOWN_LINK.finditer(readme)
        if "://" not in match.group("target") and not match.group("target").startswith("#")
    ]

    assert internal_targets
    assert all(Path(target).exists() for target in internal_targets)
