import math

import pytest

from taqsim.node.timeseries import TimeSeries


class TestTimeSeriesCreation:
    def test_creates_with_valid_values(self):
        ts = TimeSeries([1.0, 2.0, 3.0])
        assert ts.values == [1.0, 2.0, 3.0]

    def test_creates_with_zeros(self):
        ts = TimeSeries([0.0, 0.0, 0.0])
        assert ts.values == [0.0, 0.0, 0.0]


class TestTimeSeriesGetItem:
    def test_returns_correct_value_at_index(self):
        ts = TimeSeries([10.0, 20.0, 30.0])
        assert ts[0] == 10.0
        assert ts[1] == 20.0
        assert ts[2] == 30.0

    def test_negative_index_returns_from_end(self):
        ts = TimeSeries([10.0, 20.0, 30.0])
        assert ts[-1] == 30.0

    def test_index_out_of_bounds_raises_index_error(self):
        ts = TimeSeries([10.0, 20.0, 30.0])
        with pytest.raises(IndexError):
            _ = ts[5]


class TestTimeSeriesLen:
    def test_returns_correct_length(self):
        ts = TimeSeries([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(ts) == 5

    def test_single_element_has_length_one(self):
        ts = TimeSeries([42.0])
        assert len(ts) == 1


class TestTimeSeriesValidation:
    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            TimeSeries([])

    def test_negative_values_raise_value_error(self):
        with pytest.raises(ValueError, match="negative values"):
            TimeSeries([1.0, -2.0, 3.0])

    def test_nan_values_raise_value_error(self):
        with pytest.raises(ValueError, match="non-finite values"):
            TimeSeries([1.0, math.nan, 3.0])

    def test_positive_inf_raises_value_error(self):
        with pytest.raises(ValueError, match="non-finite values"):
            TimeSeries([1.0, math.inf, 3.0])

    def test_negative_inf_raises_value_error(self):
        with pytest.raises(ValueError, match="negative values"):
            TimeSeries([1.0, -math.inf, 3.0])


class TestTimeSeriesWithFixtures:
    def test_simple_timeseries_has_twelve_elements(self, simple_timeseries):
        assert len(simple_timeseries) == 12

    def test_simple_timeseries_all_values_are_ten(self, simple_timeseries):
        for i in range(12):
            assert simple_timeseries[i] == 10.0

    def test_varying_timeseries_has_twelve_elements(self, varying_timeseries):
        assert len(varying_timeseries) == 12

    def test_varying_timeseries_values_increase(self, varying_timeseries):
        for i in range(12):
            assert varying_timeseries[i] == float(i * 10)
