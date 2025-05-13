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

def validate_file_exists(file_path: str):
    """
    Validate that a file exists at the given path.
    
    Args:
        file_path: Path to the file
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}") 