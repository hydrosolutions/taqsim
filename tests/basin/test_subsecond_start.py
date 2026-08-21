from __future__ import annotations

from datetime import UTC, datetime

import pytest

from taqsim import Basin


def test_build_refuses_a_subsecond_start_before_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = False

    def compile_model(_document: object) -> None:
        nonlocal compiled
        compiled = True

    monkeypatch.setattr("taqsim.basin.incidence.compile_model", compile_model)
    basin = Basin(start_date=datetime(2020, 1, 1, microsecond=1, tzinfo=UTC), timesteps=3)

    with pytest.raises(ValueError, match="start.*whole second"):
        basin.build()

    assert not compiled
