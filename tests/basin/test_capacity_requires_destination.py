from __future__ import annotations

import pytest

from taqsim import Basin, Reach


def test_capacity_requires_a_named_overflow_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = False

    def compile_model(_document: object) -> None:
        nonlocal compiled
        compiled = True

    monkeypatch.setattr("taqsim.basin.incidence.compile_model", compile_model)
    basin = Basin(start_date="2020-01-01", timesteps=3, resolution="1 mL")
    basin.add_reach(Reach("capacity-limited-canal", "river", "farm", capacity=10.0))

    with pytest.raises(ValueError, match="capacity-limited-canal.*overflow destination"):
        basin.build()

    assert not compiled
