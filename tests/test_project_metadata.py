"""The distribution metadata describes Taqsim's current role."""

import tomllib
from pathlib import Path


def test_distribution_describes_modelling_layer() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["description"] == (
        "Water-modelling and rule-authoring layer over the incidence engine"
    )
