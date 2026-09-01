from __future__ import annotations

import pytest

from taqsim import Reach, WaterVolume
from tests import make_water_system


def test_capacity_requires_a_named_overflow_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = False

    def compile_model(_document: object) -> None:
        nonlocal compiled
        compiled = True

    monkeypatch.setattr("taqsim.water_system.incidence.compile_model", compile_model)
    water_system = make_water_system(3, "1 mL")
    water_system.add_reach(Reach("capacity-limited-canal", "river", "farm", capacity=WaterVolume(10.0, "m3")))

    with pytest.raises(ValueError, match="capacity-limited-canal.*overflow destination"):
        water_system.build()

    assert not compiled
