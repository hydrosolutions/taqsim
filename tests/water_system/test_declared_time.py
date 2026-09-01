from datetime import UTC, datetime

import pytest

from taqsim import TimeAxis


def test_time_axis_requires_a_valid_start() -> None:
    with pytest.raises(ValueError, match="invalid start date"):
        TimeAxis("not-a-date", periods=3, frequency="1d")


def test_time_axis_normalizes_an_offset_to_utc() -> None:
    axis = TimeAxis("2020-01-01T06:00:00+06:00", periods=1, frequency="1d")
    assert axis.start == datetime(2020, 1, 1, tzinfo=UTC)
