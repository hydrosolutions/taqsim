from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest

from taqsim.time import Frequency, Timestep, _add_months, time_index


class TestFrequency:
    def test_daily_value(self) -> None:
        assert Frequency.DAILY == 365

    def test_weekly_value(self) -> None:
        assert Frequency.WEEKLY == 52

    def test_monthly_value(self) -> None:
        assert Frequency.MONTHLY == 12

    def test_yearly_value(self) -> None:
        assert Frequency.YEARLY == 1

    def test_scale_monthly_to_daily(self) -> None:
        result = Frequency.scale(10.0, Frequency.MONTHLY, Frequency.DAILY)
        assert result == pytest.approx(10.0 * 12 / 365)

    def test_scale_daily_to_monthly(self) -> None:
        result = Frequency.scale(1.0, Frequency.DAILY, Frequency.MONTHLY)
        assert result == pytest.approx(365 / 12)

    def test_scale_same_frequency(self) -> None:
        result = Frequency.scale(42.0, Frequency.WEEKLY, Frequency.WEEKLY)
        assert result == pytest.approx(42.0)

    def test_scale_yearly_to_monthly(self) -> None:
        result = Frequency.scale(120.0, Frequency.YEARLY, Frequency.MONTHLY)
        assert result == pytest.approx(120.0 * 1 / 12)


class TestTimestepIndex:
    def test_index_used_for_sequence_lookup(self) -> None:
        items = [10, 20, 30]
        ts = Timestep(1, Frequency.MONTHLY)
        assert items[ts] == 20

    def test_index_zero(self) -> None:
        items = [10, 20, 30]
        ts = Timestep(0, Frequency.DAILY)
        assert items[ts] == 10


class TestTimestepInt:
    def test_int_returns_index(self) -> None:
        ts = Timestep(5, Frequency.DAILY)
        assert int(ts) == 5

    def test_int_returns_zero(self) -> None:
        ts = Timestep(0, Frequency.YEARLY)
        assert int(ts) == 0


class TestTimestepMod:
    def test_mod_returns_remainder(self) -> None:
        ts = Timestep(7, Frequency.MONTHLY)
        assert ts % 3 == 1

    def test_mod_evenly_divisible(self) -> None:
        ts = Timestep(12, Frequency.MONTHLY)
        assert ts % 4 == 0

    def test_mod_by_one(self) -> None:
        ts = Timestep(99, Frequency.DAILY)
        assert ts % 1 == 0


class TestTimestepEquality:
    def test_equal_timesteps(self) -> None:
        a = Timestep(3, Frequency.MONTHLY)
        b = Timestep(3, Frequency.MONTHLY)
        assert a == b

    def test_different_index(self) -> None:
        a = Timestep(3, Frequency.MONTHLY)
        b = Timestep(4, Frequency.MONTHLY)
        assert a != b

    def test_different_frequency(self) -> None:
        a = Timestep(3, Frequency.MONTHLY)
        b = Timestep(3, Frequency.DAILY)
        assert a != b

    def test_equal_to_int(self) -> None:
        ts = Timestep(5, Frequency.DAILY)
        assert ts == 5

    def test_not_equal_to_int(self) -> None:
        ts = Timestep(5, Frequency.DAILY)
        assert ts != 6

    def test_not_equal_to_string(self) -> None:
        ts = Timestep(5, Frequency.DAILY)
        assert ts != "5"


class TestTimestepHash:
    def test_hash_matches_index_hash(self) -> None:
        ts = Timestep(42, Frequency.WEEKLY)
        assert hash(ts) == hash(42)

    def test_usable_as_dict_key(self) -> None:
        ts = Timestep(1, Frequency.MONTHLY)
        d = {ts: "value"}
        assert d[ts] == "value"

    def test_retrievable_by_equal_timestep(self) -> None:
        ts_a = Timestep(1, Frequency.MONTHLY)
        ts_b = Timestep(1, Frequency.MONTHLY)
        d = {ts_a: "value"}
        assert d[ts_b] == "value"


class TestTimestepScale:
    def test_scale_with_explicit_to_freq(self) -> None:
        ts = Timestep(0, Frequency.MONTHLY)
        result = ts.scale(10.0, Frequency.DAILY, Frequency.YEARLY)
        assert result == pytest.approx(10.0 * 365 / 1)

    def test_scale_defaults_to_own_frequency(self) -> None:
        ts = Timestep(0, Frequency.MONTHLY)
        result = ts.scale(10.0, Frequency.DAILY)
        assert result == pytest.approx(10.0 * 365 / 12)

    def test_scale_same_frequency_is_identity(self) -> None:
        ts = Timestep(0, Frequency.WEEKLY)
        result = ts.scale(7.0, Frequency.WEEKLY)
        assert result == pytest.approx(7.0)


class TestTimestepFrozen:
    def test_cannot_assign_index(self) -> None:
        ts = Timestep(1, Frequency.DAILY)
        with pytest.raises(FrozenInstanceError):
            ts.index = 2  # type: ignore[misc]

    def test_cannot_assign_frequency(self) -> None:
        ts = Timestep(1, Frequency.DAILY)
        with pytest.raises(FrozenInstanceError):
            ts.frequency = Frequency.MONTHLY  # type: ignore[misc]

    def test_cannot_add_new_attribute(self) -> None:
        ts = Timestep(1, Frequency.DAILY)
        with pytest.raises((AttributeError, TypeError)):
            ts.new_attr = "nope"  # type: ignore[attr-defined]


class TestAddMonths:
    def test_zero_months_returns_same_date(self) -> None:
        assert _add_months(date(2024, 6, 15), 0) == date(2024, 6, 15)

    def test_forward_one_month(self) -> None:
        assert _add_months(date(2024, 1, 15), 1) == date(2024, 2, 15)

    def test_clamp_jan31_to_feb28_non_leap(self) -> None:
        assert _add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)

    def test_clamp_jan31_to_feb29_leap(self) -> None:
        assert _add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)

    def test_crosses_year_boundary(self) -> None:
        assert _add_months(date(2024, 11, 15), 3) == date(2025, 2, 15)

    def test_twelve_months_equals_one_year(self) -> None:
        assert _add_months(date(2024, 3, 10), 12) == date(2025, 3, 10)

    def test_feb29_plus_twelve_months_clamps(self) -> None:
        assert _add_months(date(2024, 2, 29), 12) == date(2025, 2, 28)


class TestTimeIndexDaily:
    def test_three_consecutive_days(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.DAILY, 3)
        assert result == (date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3))

    def test_crosses_month_boundary(self) -> None:
        result = time_index(date(2024, 1, 30), Frequency.DAILY, 3)
        assert result == (date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1))

    def test_crosses_year_boundary(self) -> None:
        result = time_index(date(2024, 12, 30), Frequency.DAILY, 4)
        assert result == (
            date(2024, 12, 30),
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2025, 1, 2),
        )

    def test_crosses_leap_day(self) -> None:
        result = time_index(date(2024, 2, 28), Frequency.DAILY, 3)
        assert result == (date(2024, 2, 28), date(2024, 2, 29), date(2024, 3, 1))

    def test_full_year_has_365_entries(self) -> None:
        result = time_index(date(2023, 1, 1), Frequency.DAILY, 365)
        assert len(result) == 365
        assert result[-1] == date(2023, 12, 31)

    def test_full_leap_year_has_366_entries(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.DAILY, 366)
        assert len(result) == 366
        assert result[-1] == date(2024, 12, 31)


class TestTimeIndexWeekly:
    def test_four_consecutive_weeks(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.WEEKLY, 4)
        assert result == (
            date(2024, 1, 1),
            date(2024, 1, 8),
            date(2024, 1, 15),
            date(2024, 1, 22),
        )

    def test_crosses_month_boundary(self) -> None:
        result = time_index(date(2024, 1, 22), Frequency.WEEKLY, 3)
        assert result == (date(2024, 1, 22), date(2024, 1, 29), date(2024, 2, 5))

    def test_spacing_is_seven_days(self) -> None:
        result = time_index(date(2024, 3, 1), Frequency.WEEKLY, 5)
        for i in range(1, len(result)):
            assert result[i] - result[i - 1] == timedelta(days=7)


class TestTimeIndexMonthly:
    def test_three_consecutive_months(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.MONTHLY, 3)
        assert result == (date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1))

    def test_preserves_start_day(self) -> None:
        result = time_index(date(2024, 1, 15), Frequency.MONTHLY, 3)
        assert result == (date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15))

    def test_clamps_day_for_shorter_month(self) -> None:
        result = time_index(date(2024, 1, 31), Frequency.MONTHLY, 4)
        assert result == (
            date(2024, 1, 31),
            date(2024, 2, 29),
            date(2024, 3, 31),
            date(2024, 4, 30),
        )

    def test_clamps_jan31_to_feb28(self) -> None:
        result = time_index(date(2023, 1, 31), Frequency.MONTHLY, 2)
        assert result == (date(2023, 1, 31), date(2023, 2, 28))

    def test_clamps_jan31_to_feb29_in_leap_year(self) -> None:
        result = time_index(date(2024, 1, 31), Frequency.MONTHLY, 2)
        assert result == (date(2024, 1, 31), date(2024, 2, 29))

    def test_crosses_year_boundary(self) -> None:
        result = time_index(date(2024, 11, 1), Frequency.MONTHLY, 4)
        assert result == (
            date(2024, 11, 1),
            date(2024, 12, 1),
            date(2025, 1, 1),
            date(2025, 2, 1),
        )

    def test_two_full_years(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.MONTHLY, 24)
        assert len(result) == 24
        assert result[0] == date(2024, 1, 1)
        assert result[12] == date(2025, 1, 1)
        assert result[23] == date(2025, 12, 1)


class TestTimeIndexYearly:
    def test_three_consecutive_years(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.YEARLY, 3)
        assert result == (date(2024, 1, 1), date(2025, 1, 1), date(2026, 1, 1))

    def test_preserves_month_and_day(self) -> None:
        result = time_index(date(2024, 6, 15), Frequency.YEARLY, 3)
        assert result == (date(2024, 6, 15), date(2025, 6, 15), date(2026, 6, 15))

    def test_leap_day_start_clamps_to_feb28(self) -> None:
        result = time_index(date(2024, 2, 29), Frequency.YEARLY, 2)
        assert result == (date(2024, 2, 29), date(2025, 2, 28))

    def test_leap_day_start_returns_to_feb29_on_next_leap_year(self) -> None:
        result = time_index(date(2024, 2, 29), Frequency.YEARLY, 5)
        assert result[0] == date(2024, 2, 29)
        assert result[1] == date(2025, 2, 28)
        assert result[2] == date(2026, 2, 28)
        assert result[3] == date(2027, 2, 28)
        assert result[4] == date(2028, 2, 29)


class TestTimeIndexBoundary:
    def test_n_zero_returns_empty_tuple(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.DAILY, 0)
        assert result == ()

    def test_n_one_returns_start_date_only(self) -> None:
        result = time_index(date(2024, 6, 15), Frequency.MONTHLY, 1)
        assert result == (date(2024, 6, 15),)

    def test_returns_tuple_not_list(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.DAILY, 3)
        assert isinstance(result, tuple)

    def test_all_elements_are_date_instances(self) -> None:
        result = time_index(date(2024, 1, 1), Frequency.WEEKLY, 4)
        assert all(isinstance(d, date) for d in result)

    def test_first_element_is_always_start_date(self) -> None:
        for freq in Frequency:
            result = time_index(date(2024, 3, 15), freq, 1)
            assert result[0] == date(2024, 3, 15)


class TestTimeIndexErrors:
    def test_negative_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            time_index(date(2024, 1, 1), Frequency.DAILY, -1)
