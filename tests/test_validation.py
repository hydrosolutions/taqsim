import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taqsim.validation import (
    validate_coordinates,
    validate_dataframe_period,
    validate_file_exists,
    validate_month,
    validate_node_id,
    validate_nonnegativity_int_or_float,
    validate_positive_float,
    validate_positive_integer,
    validate_probability,
    validate_year,
)


class TestValidateNodeId:
    def test_accepts_valid_string(self) -> None:
        validate_node_id("node_123")
        validate_node_id("A")

    def test_fails_with_empty_string(self) -> None:
        with pytest.raises(ValueError, match="ID cannot be empty"):
            validate_node_id("")

    def test_fails_with_non_string_integer(self) -> None:
        with pytest.raises(ValueError, match="ID must be a string, got int"):
            validate_node_id(123)

    def test_fails_with_non_string_none(self) -> None:
        with pytest.raises(ValueError, match="ID must be a string, got NoneType"):
            validate_node_id(None)

    def test_fails_with_non_string_list(self) -> None:
        with pytest.raises(ValueError, match="ID must be a string, got list"):
            validate_node_id(["id"])

    def test_includes_prefix_in_error_message(self) -> None:
        with pytest.raises(ValueError, match="Node: ID cannot be empty"):
            validate_node_id("", prefix="Node: ")


class TestValidateCoordinates:
    def test_accepts_valid_integers(self) -> None:
        validate_coordinates(100, 200)

    def test_accepts_valid_floats(self) -> None:
        validate_coordinates(100.5, 200.5)

    def test_accepts_numpy_numbers(self) -> None:
        validate_coordinates(np.int32(100), np.float64(200.5))

    def test_accepts_negative_coordinates(self) -> None:
        validate_coordinates(-100.5, -200.5)

    def test_accepts_zero_coordinates(self) -> None:
        validate_coordinates(0, 0)

    def test_fails_with_none_easting(self) -> None:
        with pytest.raises(ValueError, match="Missing coordinate value"):
            validate_coordinates(None, 200)

    def test_fails_with_none_northing(self) -> None:
        with pytest.raises(ValueError, match="Missing coordinate value"):
            validate_coordinates(100, None)

    def test_fails_with_both_none(self) -> None:
        with pytest.raises(ValueError, match="Missing coordinate value"):
            validate_coordinates(None, None)

    def test_fails_with_string_easting(self) -> None:
        with pytest.raises(ValueError, match="Easting must be a number, got str"):
            validate_coordinates("100", 200)

    def test_fails_with_string_northing(self) -> None:
        with pytest.raises(ValueError, match="Northing must be a number, got str"):
            validate_coordinates(100, "200")

    def test_fails_with_list_coordinates(self) -> None:
        with pytest.raises(ValueError, match="Easting must be a number, got list"):
            validate_coordinates([100], 200)

    def test_includes_prefix_in_error_message(self) -> None:
        with pytest.raises(ValueError, match="Node A: Easting must be a number"):
            validate_coordinates("invalid", 200, prefix="Node A")


class TestValidatePositiveInteger:
    def test_accepts_positive_integer(self) -> None:
        validate_positive_integer(1)
        validate_positive_integer(100)

    def test_accepts_numpy_integer(self) -> None:
        validate_positive_integer(np.int32(5))
        validate_positive_integer(np.int64(10))

    def test_fails_with_zero(self) -> None:
        with pytest.raises(ValueError, match="must be positive, got 0"):
            validate_positive_integer(0)

    def test_fails_with_negative_integer(self) -> None:
        with pytest.raises(ValueError, match="must be positive, got -5"):
            validate_positive_integer(-5)

    def test_fails_with_float(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got float"):
            validate_positive_integer(5.5)

    def test_fails_with_string(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got str"):
            validate_positive_integer("5")

    def test_fails_with_none(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got NoneType"):
            validate_positive_integer(None)

    def test_includes_custom_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="Count must be positive"):
            validate_positive_integer(0, name="Count")


class TestValidatePositiveFloat:
    def test_accepts_positive_float(self) -> None:
        result = validate_positive_float(1.5)
        assert result == 1.5

    def test_accepts_positive_integer_as_float(self) -> None:
        result = validate_positive_float(5)
        assert result == 5.0

    def test_accepts_numpy_number(self) -> None:
        result = validate_positive_float(np.float64(3.14))
        assert result == 3.14

    def test_fails_with_zero(self) -> None:
        with pytest.raises(ValueError, match="must be positive, got 0"):
            validate_positive_float(0)

    def test_fails_with_negative_float(self) -> None:
        with pytest.raises(ValueError, match="must be positive, got -1.5"):
            validate_positive_float(-1.5)

    def test_fails_with_string(self) -> None:
        with pytest.raises(ValueError, match="must be a number, got str"):
            validate_positive_float("3.14")

    def test_fails_with_none(self) -> None:
        with pytest.raises(ValueError, match="must be a number, got NoneType"):
            validate_positive_float(None)

    def test_includes_custom_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="Rate must be positive"):
            validate_positive_float(-1.0, name="Rate")


class TestValidateMonth:
    def test_accepts_valid_months(self) -> None:
        for month in range(1, 13):
            validate_month(month)

    def test_fails_with_zero(self) -> None:
        with pytest.raises(ValueError, match="must be between 1 and 12, got 0"):
            validate_month(0)

    def test_fails_with_thirteen(self) -> None:
        with pytest.raises(ValueError, match="must be between 1 and 12, got 13"):
            validate_month(13)

    def test_fails_with_negative(self) -> None:
        with pytest.raises(ValueError, match="must be between 1 and 12, got -1"):
            validate_month(-1)

    def test_fails_with_float(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got float"):
            validate_month(6.5)

    def test_fails_with_string(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got str"):
            validate_month("6")

    def test_fails_with_none(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got NoneType"):
            validate_month(None)


class TestValidateYear:
    def test_accepts_valid_year_1900(self) -> None:
        validate_year(1900)

    def test_accepts_valid_year_2000(self) -> None:
        validate_year(2000)

    def test_accepts_valid_year_2099(self) -> None:
        validate_year(2099)

    def test_fails_with_float(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got float"):
            validate_year(2000.5)

    def test_fails_with_string(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got str"):
            validate_year("2000")

    def test_fails_with_none(self) -> None:
        with pytest.raises(ValueError, match="must be an integer, got NoneType"):
            validate_year(None)

    def test_accepts_year_below_1900(self) -> None:
        validate_year(1899)

    def test_accepts_year_2100_and_above(self) -> None:
        validate_year(2100)


class TestValidateProbability:
    def test_accepts_zero(self) -> None:
        validate_probability(0)

    def test_accepts_one(self) -> None:
        validate_probability(1)

    def test_accepts_half(self) -> None:
        validate_probability(0.5)

    def test_accepts_small_positive(self) -> None:
        validate_probability(0.001)

    def test_accepts_near_one(self) -> None:
        validate_probability(0.999)

    def test_accepts_integer_zero(self) -> None:
        validate_probability(0)

    def test_accepts_integer_one(self) -> None:
        validate_probability(1)

    def test_accepts_numpy_float(self) -> None:
        validate_probability(np.float64(0.5))

    def test_fails_with_negative(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1, got -0.1"):
            validate_probability(-0.1)

    def test_fails_with_greater_than_one(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1, got 1.1"):
            validate_probability(1.1)

    def test_fails_with_string(self) -> None:
        with pytest.raises(ValueError, match="must be a number, got str"):
            validate_probability("0.5")

    def test_fails_with_none(self) -> None:
        with pytest.raises(ValueError, match="must be a number, got NoneType"):
            validate_probability(None)

    def test_includes_custom_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="Chance must be between 0 and 1"):
            validate_probability(1.5, name="Chance")


class TestValidateNonnegativityIntOrFloat:
    def test_accepts_zero(self) -> None:
        result = validate_nonnegativity_int_or_float(0)
        assert result == 0.0

    def test_accepts_positive_float(self) -> None:
        result = validate_nonnegativity_int_or_float(5.5)
        assert result == 5.5

    def test_accepts_positive_integer(self) -> None:
        result = validate_nonnegativity_int_or_float(10)
        assert result == 10.0

    def test_accepts_numpy_number(self) -> None:
        result = validate_nonnegativity_int_or_float(np.float64(3.14))
        assert result == 3.14

    def test_fails_with_negative_float(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative, got -1.5"):
            validate_nonnegativity_int_or_float(-1.5)

    def test_fails_with_negative_integer(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative, got -10"):
            validate_nonnegativity_int_or_float(-10)

    def test_fails_with_string(self) -> None:
        with pytest.raises(ValueError, match="must be a number, got str"):
            validate_nonnegativity_int_or_float("5")

    def test_fails_with_none(self) -> None:
        with pytest.raises(ValueError, match="must be a number, got NoneType"):
            validate_nonnegativity_int_or_float(None)

    def test_includes_custom_name_in_error(self) -> None:
        with pytest.raises(ValueError, match="Balance must be non-negative"):
            validate_nonnegativity_int_or_float(-5, name="Balance")


class TestValidateDataframePeriod:
    def test_accepts_valid_dataframe_with_sufficient_rows(self) -> None:
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        df = pd.DataFrame({"Date": dates, "flow": range(12)})
        result = validate_dataframe_period(df, 2020, 1, 12, "flow")
        assert len(result) == 12

    def test_accepts_dataframe_with_extra_rows(self) -> None:
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        df = pd.DataFrame({"Date": dates, "precip": range(24)})
        result = validate_dataframe_period(df, 2020, 1, 12, "precip")
        assert len(result) >= 12

    def test_fails_with_missing_date_column(self) -> None:
        df = pd.DataFrame({"flow": range(12)})
        with pytest.raises(ValueError, match="must contain 'Date' and 'flow' columns"):
            validate_dataframe_period(df, 2020, 1, 12, "flow")

    def test_fails_with_missing_data_column(self) -> None:
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        df = pd.DataFrame({"Date": dates})
        with pytest.raises(ValueError, match="must contain 'Date' and 'temp' columns"):
            validate_dataframe_period(df, 2020, 1, 12, "temp")

    def test_fails_with_insufficient_rows(self) -> None:
        dates = pd.date_range("2020-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"Date": dates, "flow": range(6)})
        with pytest.raises(ValueError, match="has 6 rows but expected 12"):
            validate_dataframe_period(df, 2020, 1, 12, "flow")

    def test_fails_with_empty_dataframe(self) -> None:
        df = pd.DataFrame({"Date": [], "flow": []})
        with pytest.raises(ValueError, match="has 0 rows but expected 12"):
            validate_dataframe_period(df, 2020, 1, 12, "flow")


class TestValidateFileExists:
    def test_accepts_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            validate_file_exists(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_fails_with_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found: /nonexistent/path.txt"):
            validate_file_exists("/nonexistent/path.txt")

    def test_fails_with_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(FileNotFoundError, match="File not found"):
            validate_file_exists(tmp_dir)

    def test_fails_with_empty_string_path(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found: "):
            validate_file_exists("")
