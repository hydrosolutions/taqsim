"""The incidence-backed modelling package does not expose the retired engine."""

from pathlib import Path

import taqsim


def test_retired_public_names_are_absent() -> None:
    retired = {"Edge", "Trace", "LossReason", "simulate", "reset"}
    assert retired.isdisjoint(vars(taqsim))
    assert not any(name.startswith("Water") and name.endswith(("Entered", "Exited")) for name in taqsim.__all__)


def test_retired_event_and_edge_modules_are_removed() -> None:
    package = Path(taqsim.__file__).parent
    assert not (package / "node" / "events.py").exists()
    assert not (package / "edge").exists()
