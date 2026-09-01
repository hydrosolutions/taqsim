"""Basin.build : resolution declaration → model quantum or refusal."""

import pytest

from taqsim import Basin


def test_undeclared_resolution_is_refused() -> None:
    basin = Basin(start_date="2020-01-01", timesteps=1)

    with pytest.raises(ValueError, match="missing required resolution declaration"):
        basin.build()


def test_unknown_spelling_is_refused() -> None:
    basin = Basin(start_date="2020-01-01", timesteps=1, resolution="1 acre-foot")

    with pytest.raises(ValueError) as caught:
        basin.build()

    message = str(caught.value)
    assert "unknown resolution '1 acre-foot'" in message
    for accepted in ("1 m3", "1 L", "1 mL", "1 mm3"):
        assert repr(accepted) in message


def test_source_total_rounded_to_the_ceiling_is_refused() -> None:
    basin = Basin(start_date="2020-01-01", timesteps=2, resolution="1 m3")
    basin.source("river", [2**53, 1])
    basin.sink("sea")
    basin.reach("channel", "river", "sea")

    with pytest.raises(ValueError) as caught:
        basin.build()

    message = str(caught.value)
    assert "declared total 9007199254740993" in message
    assert "countable ceiling 9007199254740992" in message
