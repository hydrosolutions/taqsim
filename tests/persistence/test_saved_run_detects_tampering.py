from __future__ import annotations

import json
from pathlib import Path

import pytest

from taqsim import Basin, SavedRunFormatError, load_run


def test_saved_run_refuses_tampered_cached_flow(tmp_path: Path) -> None:
    basin = Basin(start_date="2020-01-01", timesteps=2, resolution="1 mL")
    basin.source("river", flow=[4.0, 2.0])
    basin.sink("farm")
    basin.add_reach("canal", "river", "farm")
    saved = tmp_path / "run.json"
    basin.build().run(bytes(16)).save(saved)

    document = json.loads(saved.read_text())
    document["flows"]["canal"]["values"][0] = 4000.0
    saved.write_text(json.dumps(document))

    with pytest.raises(SavedRunFormatError, match="artifact digest mismatch"):
        load_run(saved)
