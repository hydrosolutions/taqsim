from __future__ import annotations

import json
from pathlib import Path

import pytest

from taqsim import SavedRunFormatError, load_run
from tests import interval_volume, make_water_system


def test_saved_run_refuses_tampered_cached_flow(tmp_path: Path) -> None:
    water_system = make_water_system(2, "1 mL")
    water_system.source("river", interval_volume([4.0, 2.0]))
    water_system.sink("farm")
    water_system.add_reach("canal", "river", "farm")
    saved = tmp_path / "run.json"
    water_system.build().run(bytes(16)).save(saved)

    document = json.loads(saved.read_text())
    document["flows"]["canal"]["values"][0] = 4000.0
    saved.write_text(json.dumps(document))

    with pytest.raises(SavedRunFormatError, match="artifact digest mismatch"):
        load_run(saved)
