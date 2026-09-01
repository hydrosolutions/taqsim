from datetime import UTC, datetime

import pytest

from taqsim import TimeAxis


def test_time_axis_refuses_a_subsecond_start() -> None:
    with pytest.raises(ValueError, match="start.*whole second"):
        TimeAxis(datetime(2020, 1, 1, microsecond=1, tzinfo=UTC), periods=3, frequency="1d")
