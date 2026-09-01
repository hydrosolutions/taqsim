from __future__ import annotations

import json
from pathlib import Path

import pytest

from taqsim import IncidenceVersionMismatchError, load_run
from tests import interval_volume, make_water_system


def test_saved_run_refuses_a_different_incidence_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    water_system = make_water_system(2, "1 mL")
    water_system.source("river", interval_volume([4.0, 0.0]))
    water_system.sink("farm")
    water_system.add_reach("canal", "river", "farm")
    saved = tmp_path / "run.json"
    water_system.build().run(bytes(16)).save(saved)

    document = json.loads(saved.read_text())
    document["incidence_version"] = "999.0.0-stranger"
    saved.write_text(json.dumps(document))

    reconstructed = False

    def reject_reconstruction(*args: object, **kwargs: object) -> None:
        nonlocal reconstructed
        reconstructed = True
        raise AssertionError("a foreign cache must be rejected before run reconstruction")

    monkeypatch.setattr("taqsim.water_system.WaterSystemRun._from_cache", reject_reconstruction)
    with pytest.raises(IncidenceVersionMismatchError, match=r"999\.0\.0-stranger.*installed version"):
        load_run(saved)
    assert not reconstructed
