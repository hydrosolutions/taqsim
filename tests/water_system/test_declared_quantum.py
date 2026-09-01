"""WaterSystem declaration : conservation quantum → exact arithmetic or refusal."""

from typing import Any, cast

import pytest

from taqsim import TimeAxis, WaterSystem
from tests import interval_volume, make_water_system


def test_quantum_is_required_at_the_construction_boundary() -> None:
    with pytest.raises(TypeError, match="quantum"):
        cast(Any, WaterSystem)(time=TimeAxis("2020-01-01", periods=1, frequency="1d"))


def test_unknown_spelling_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        make_water_system(1, "1 acre-foot")
    message = str(caught.value)
    assert "unknown conservation quantum '1 acre-foot'" in message
    for accepted in ("1 m3", "1 L", "1 mL", "1 mm3"):
        assert repr(accepted) in message


def test_source_total_rounded_to_the_ceiling_is_refused() -> None:
    system = make_water_system(2, "1 m3")
    system.source("river", interval_volume([2**53, 1]))
    system.sink("sea")
    system.reach("channel", "river", "sea")
    with pytest.raises(ValueError) as caught:
        system.build()
    message = str(caught.value)
    assert "declared total 9007199254740993" in message
    assert "countable ceiling 9007199254740992" in message
