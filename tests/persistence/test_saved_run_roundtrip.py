from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from taqsim import Presence, load_run
from tests import interval_volume, make_water_system


def test_saved_run_reopens_in_a_fresh_process_without_running(tmp_path: Path) -> None:
    water_system = make_water_system(3, "1 mL")
    water_system.source("river", interval_volume([5.0, 0.0, 2.0]))
    water_system.sink("farm")
    water_system.add_reach("canal", "river", "farm")
    original = water_system.build().run(bytes(range(16)))
    saved = tmp_path / "run.json"
    original.save(saved)

    reopened = load_run(saved)
    assert reopened.flow("canal") == original.flow("canal")
    assert reopened.authoritative_log() == original.authoritative_log()
    assert reopened.model_digest == original.model_digest

    program = r"""
import json
import incidence
from taqsim import load_run

def simulation_was_called(*args, **kwargs):
    raise AssertionError("loading a cache must not compile or run a simulation")

incidence.compile_model = simulation_was_called
run = load_run(PATH)
series = run.flow("canal", start="2019-12-31", end="2020-01-04")
print(json.dumps({
    "values": list(series.values),
    "presence": [state.value for state in series.presence],
    "digest": run.authoritative_log_digest,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program.replace("PATH", repr(str(saved)))],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {
        "values": [None, 5.0, 0.0, 2.0, None],
        "presence": [
            Presence.NOT_MODELLED.value,
            Presence.PRESENT.value,
            Presence.PRESENT.value,
            Presence.PRESENT.value,
            Presence.NOT_MODELLED.value,
        ],
        "digest": original.authoritative_log_digest,
    }
