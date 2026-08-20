from __future__ import annotations

import json
from pathlib import Path

import pytest

from taqsim import Basin, IncidenceVersionMismatchError, load_run


def test_saved_run_refuses_a_different_incidence_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    basin = Basin(start_date="2020-01-01", timesteps=2)
    basin.source("river", flow=[4.0, 0.0])
    basin.sink("farm")
    basin.add_reach("canal", "river", "farm")
    saved = tmp_path / "run.json"
    basin.build().run(bytes(16)).save(saved)

    document = json.loads(saved.read_text())
    document["incidence_version"] = "999.0.0-stranger"
    saved.write_text(json.dumps(document))

    reconstructed = False

    def reject_reconstruction(*args: object, **kwargs: object) -> None:
        nonlocal reconstructed
        reconstructed = True
        raise AssertionError("a foreign cache must be rejected before run reconstruction")

    monkeypatch.setattr("taqsim.basin.BasinRun._from_cache", reject_reconstruction)
    with pytest.raises(IncidenceVersionMismatchError, match=r"999\.0\.0-stranger.*installed version"):
        load_run(saved)
    assert not reconstructed
