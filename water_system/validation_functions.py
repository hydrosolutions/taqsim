"""
This module provides validation utilities for the water system simulation framework.

It contains functions for validating various types of input parameters including:
- Numeric ranges and constraints
- Geographic coordinates
- Time series data
- File existence and formats
- Parameter relationships
"""
from typing import Union, List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import os
import math
from datetime import datetime

def validate_node_id(id: Any, prefix: str = ""):
    """
    Validate that an ID is a non-empty string.
    
    Args:
        id: The ID to validate
        prefix: Optional prefix for error messages
        
    Returns:
        str: The validated ID
        
    Raises:
        ValueError: If the ID is not a string or is empty
    """
    if not isinstance(id, str):
        raise ValueError(f"{prefix}ID must be a string, got {type(id).__name__}")
    if not id:
        raise ValueError(f"{prefix}ID cannot be empty")

def validate_coordinates(easting: Any, northing: Any, prefix: str = ""):
    """
    Validate that easting and northing are valid coordinates.
    
    Args:
        easting: The easting coordinate
        northing: The northing coordinate
        
    Returns:
        Tuple[float, float]: The validated easting and northing coordinates
        
    Raises:
        ValueError: If either coordinate is not a valid number
    """
    if easting is None or northing is None:
        raise ValueError(f"{prefix}: Missing coordinate value: easting={easting}, northing={northing}")
    
    if not isinstance(easting, (int, float, np.number)):
        raise ValueError(f"{prefix}: Easting must be a number, got {type(easting).__name__}")
    
    if not isinstance(northing, (int, float, np.number)):
        raise ValueError(f"{prefix}: Northing must be a number, got {type(northing).__name__}")

def validate_positive_integer(value: Any, name: str = "Value"):
    """
    Validate that a value is a positive integer (greater than zero).
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
    Returns:
        int: The validated positive integer
    Raises:
        ValueError: If the value is not a valid positive integer
    """
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def validate_positive_float(value: Any, name: str = "Value"):
    """
    Validate that a value is a positive float (greater than zero).
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
    Returns:
        float: The validated positive float
    Raises:
        ValueError: If the value is not a valid positive float
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    
    return value

def validate_probability(value: Any, name: str = "Probability"):
    """
    Validate that a value is a probability (float between 0 and 1, inclusive).
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
        
    Returns:
        float: The validated probability
        
    Raises:
        ValueError: If the value is not a valid probability
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")

def validate_nonnegativity_int_or_float(value: Any, name: str = "Value"):
    """
    Validate that a value is non-negative (float).
    
    Args:
        value: The value to validate
        name: Name of the value for error messages
    Returns:
        float: The validated non-negative value
    Raises:
        ValueError: If the value is not a valid non-negative float
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    
    return value

def validate_dataframe_period(df: pd.DataFrame, 
                            start_year: int,
                            start_month: int,
                            num_time_steps: int, 
                            column_name: str,) -> pd.DataFrame:
    """
    Validate that a DataFrame's dates match the requested simulation period.
    
    Args:
        df: DataFrame containing the time series data
        date_column: Name of the column containing dates
        start_year: Starting year
        start_month: Starting month (1-12)
        num_time_steps: Expected number of time steps
        
    Returns:
        pd.DataFrame: The validated DataFrame
        
    Raises:
        ValueError: If the DataFrame doesn't match the requested period
    """ 
    expected_start = pd.Timestamp(year=start_year, month=start_month, day=1)
    expected_dates = pd.date_range(expected_start, periods=num_time_steps, freq='MS')

    if 'Date' not in df.columns or column_name not in df.columns:
                raise ValueError(f"CSV file must contain 'Date' and '{column_name}' columns")
    
    if len(df) != num_time_steps:
        raise ValueError(f"DataFrame has {len(df)} rows but expected {num_time_steps}")
        
    if not df['Date'].equals(pd.Series(expected_dates)):
        raise ValueError(f"DataFrame dates do not match expected period starting {expected_start}")
    
    return df

# TODO: Add more specific validation functions for different types of parameters
def validate_file_exists(filepath: str, file_type: str = "File") -> str:
    """
    Validate that a file exists.
    
    Args:
        filepath: The path to the file
        file_type: Type of file for error messages
        
    Returns:
        str: The validated filepath
        
    Raises:
        ValueError: If the file does not exist
    """
    if not isinstance(filepath, str):
        raise ValueError(f"{file_type} path must be a string, got {type(filepath).__name__}")
    
    if not os.path.exists(filepath):
        raise ValueError(f"{file_type} not found: {filepath}")
    
    return filepath

def validate_csv_format(filepath: str, required_columns: List[str]) -> pd.DataFrame:
    """
    Validate that a CSV file exists and contains the required columns.
    
    Args:
        filepath: The path to the CSV file
        required_columns: List of column names that must be present
        
    Returns:
        pd.DataFrame: The loaded CSV data
        
    Raises:
        ValueError: If the file does not exist or lacks required columns
    """
    validate_file_exists(filepath, "CSV file")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file {filepath}: {str(e)}")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file {filepath} is missing required columns: {missing_columns}")
    
    return df

def validate_date_range(date_str: str, format: str = "%Y-%m-%d") -> datetime:
    """
    Validate that a string can be parsed as a date in the specified format.
    
    Args:
        date_str: The date string to validate
        format: The expected date format
        
    Returns:
        datetime: The parsed date
        
    Raises:
        ValueError: If the string cannot be parsed as a date
    """
    try:
        return datetime.strptime(date_str, format)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}, expected format: {format}")

def validate_month(month: Any) -> int:
    """
    Validate that a value is a valid month (1-12).
    
    Args:
        month: The month value to validate
        
    Returns:
        int: The validated month
        
    Raises:
        ValueError: If the value is not a valid month
    """
    if not isinstance(month, (int, np.integer)):
        raise ValueError(f"Month must be an integer, got {type(month).__name__}")
    
    month = int(month)
    if month < 1 or month > 12:
        raise ValueError(f"Month must be between 1 and 12, got {month}")
    
    return month

def validate_year(year: Any) -> int:
    """
    Validate that a value is a valid year (reasonably bounded).
    
    Args:
        year: The year value to validate
        
    Returns:
        int: The validated year
        
    Raises:
        ValueError: If the value is not a valid year
    """
    if not isinstance(year, (int, np.integer)):
        raise ValueError(f"Year must be an integer, got {type(year).__name__}")
    
    year = int(year)
    current_year = datetime.now().year
    
    # Reasonable bounds for simulation years (adjust as needed)
    if year < 1900 or year > current_year + 100:
        raise ValueError(f"Year {year} is outside reasonable bounds (1900-{current_year+100})")
    
    return year

def validate_array_length(array: Any, expected_length: int, name: str = "Array") -> np.ndarray:
    """
    Validate that an array has the expected length.
    
    Args:
        array: The array to validate
        expected_length: The expected length of the array
        name: Name of the array for error messages
        
    Returns:
        np.ndarray: The validated array
        
    Raises:
        ValueError: If the array does not have the expected length
    """
    if not hasattr(array, '__len__'):
        raise ValueError(f"{name} must be a sequence type, got {type(array).__name__}")
    
    if len(array) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(array)}")
    
    return np.array(array)

def validate_parameter_relationship(value1: float, value2: float, 
                                  relationship: str, 
                                  name1: str = "Value1", 
                                  name2: str = "Value2") -> None:
    """
    Validate that two parameters have the specified relationship.
    
    Args:
        value1: The first value
        value2: The second value
        relationship: The relationship to validate ('>', '<', '>=', '<=', '==')
        name1: Name of the first value for error messages
        name2: Name of the second value for error messages
        
    Raises:
        ValueError: If the relationship does not hold
    """
    if relationship == '>':
        if not value1 > value2:
            raise ValueError(f"{name1} ({value1}) must be greater than {name2} ({value2})")
    elif relationship == '<':
        if not value1 < value2:
            raise ValueError(f"{name1} ({value1}) must be less than {name2} ({value2})")
    elif relationship == '>=':
        if not value1 >= value2:
            raise ValueError(f"{name1} ({value1}) must be greater than or equal to {name2} ({value2})")
    elif relationship == '<=':
        if not value1 <= value2:
            raise ValueError(f"{name1} ({value1}) must be less than or equal to {name2} ({value2})")
    elif relationship == '==':
        if not math.isclose(value1, value2, rel_tol=1e-9):
            raise ValueError(f"{name1} ({value1}) must be equal to {name2} ({value2})")
    else:
        raise ValueError(f"Unknown relationship: {relationship}")

def validate_distribution_parameters(params: Dict[str, List[float]], prefix: str = "") -> Dict[str, List[float]]:
    """
    Validate that a set of distribution parameters sum to 1.0 for each time step.
    
    Args:
        params: Dictionary mapping target IDs to lists of distribution parameters
        prefix: Optional prefix for error messages
        
    Returns:
        Dict[str, List[float]]: The validated parameters
        
    Raises:
        ValueError: If parameters do not sum to 1.0 for any time step
    """
    if not isinstance(params, dict):
        raise ValueError(f"{prefix}Distribution parameters must be a dictionary")
    
    if not params:
        raise ValueError(f"{prefix}Distribution parameters dictionary cannot be empty")
    
    # Check that all parameter lists have the same length
    first_key = next(iter(params))
    expected_length = len(params[first_key])
    
    for target_id, values in params.items():
        if len(values) != expected_length:
            raise ValueError(
                f"{prefix}All distribution parameter lists must have the same length. "
                f"Expected {expected_length}, but {target_id} has {len(values)}"
            )
    
    # Check that parameters sum to 1.0 for each time step
    for t in range(expected_length):
        total = sum(params[target_id][t] for target_id in params)
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"{prefix}Distribution parameters for time step {t} sum to {total}, must sum to 1.0"
            )
    
    return params

def validate_reservoir_parameters(params: Dict[str, List[float]], 
                               dead_storage: float,
                               capacity: float,
                               prefix: str = "") -> Dict[str, List[float]]:
    """
    Validate that reservoir parameters satisfy the necessary constraints.
    
    Args:
        params: Dictionary with keys 'Vr', 'V1', 'V2', mapping to lists of monthly values
        dead_storage: Dead storage volume constraint
        capacity: Maximum storage capacity constraint
        prefix: Optional prefix for error messages
        
    Returns:
        Dict[str, List[float]]: The validated parameters
        
    Raises:
        ValueError: If parameters violate constraints
    """
    required_params = ['Vr', 'V1', 'V2']
    for param in required_params:
        if param not in params:
            raise ValueError(f"{prefix}Missing required parameter: {param}")
    
    for month in range(12):
        Vr = params['Vr'][month]
        V1 = params['V1'][month]
        V2 = params['V2'][month]
        
        # Check individual constraints
        if Vr < 0:
            raise ValueError(f"{prefix}Month {month+1}: Vr ({Vr}) cannot be negative")
            
        if V1 <= dead_storage:
            raise ValueError(
                f"{prefix}Month {month+1}: V1 ({V1}) must be greater than dead storage ({dead_storage})"
            )

        if V1 >= V2:
            raise ValueError(f"{prefix}Month {month+1}: V1 ({V1}) must be less than V2 ({V2})")

        if V2 <= V1:
            raise ValueError(f"{prefix}Month {month+1}: V2 ({V2}) must be greater than V1 ({V1})")
            
        if V2 > capacity:
            raise ValueError(
                f"{prefix}Month {month+1}: V2 ({V2}) cannot exceed reservoir capacity ({capacity})"
            )
    
    return params

def validate_csv_time_series(filepath: str, data_column: str, date_column: str = 'Date',
                          date_format: str = '%Y-%m-%d', min_value: float = None, 
                          max_value: float = None) -> pd.DataFrame:
    """
    Validate a CSV file contains a valid time series with dates and data values.
    
    Args:
        filepath: Path to the CSV file
        data_column: Name of the column containing data values
        date_column: Name of the column containing dates
        date_format: Expected format of date strings
        min_value: Minimum allowed value (if None, no lower bound)
        max_value: Maximum allowed value (if None, no upper bound)
        
    Returns:
        pd.DataFrame: The validated time series data
        
    Raises:
        ValueError: If the file doesn't contain valid time series data
    """
    # Validate file exists and can be loaded
    df = validate_csv_format(filepath, [date_column, data_column])
    
    # Validate date column
    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    except Exception as e:
        raise ValueError(f"Invalid date format in column {date_column}: {str(e)}")
    
    # Validate data column
    try:
        df[data_column] = pd.to_numeric(df[data_column])
    except Exception as e:
        raise ValueError(f"Non-numeric values in data column {data_column}: {str(e)}")
    
    # Check bounds if specified
    if min_value is not None and (df[data_column] < min_value).any():
        raise ValueError(f"Values in {data_column} must be >= {min_value}")
    
    if max_value is not None and (df[data_column] > max_value).any():
        raise ValueError(f"Values in {data_column} must be <= {max_value}")
    
    return df

def validate_simulation_period(start_year: Any, start_month: Any, num_time_steps: Any) -> Tuple[int, int, int]:
    """
    Validate that simulation period parameters are valid.
    
    Args:
        start_year: Starting year for the simulation
        start_month: Starting month for the simulation (1-12)
        num_time_steps: Number of time steps to simulate
        
    Returns:
        Tuple[int, int, int]: Validated start_year, start_month, and num_time_steps
        
    Raises:
        ValueError: If any parameter is invalid
    """
    validated_year = validate_year(start_year)
    validated_month = validate_month(start_month)
    validated_time_steps = validate_nonnegativity_integer(num_time_steps, "Number of time steps")
    
    if validated_time_steps == 0:
        raise ValueError("Number of time steps must be positive")
    
    return validated_year, validated_month, validated_time_steps
