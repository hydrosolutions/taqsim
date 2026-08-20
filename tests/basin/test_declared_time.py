from __future__ import annotations

import pytest

from taqsim import Basin


def test_build_refuses_a_missing_start_date_before_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = False

    def compile_model(_document: object) -> None:
        nonlocal compiled
        compiled = True

    monkeypatch.setattr("taqsim.basin.incidence.compile_model", compile_model)
    basin = Basin(timesteps=3)
    basin.add_reach("canal", "river", "farm")

    with pytest.raises(ValueError, match="start date"):
        basin.build()

    assert not compiled
