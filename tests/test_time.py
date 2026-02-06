from dataclasses import FrozenInstanceError

import pytest

from taqsim.time import Frequency, Timestep


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
