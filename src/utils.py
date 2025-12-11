"""
Utility functions for the E-Commerce CLV Prediction project.

This module provides helper functions for:
- Configuration loading
- Logging setup
- Data validation
- Common operations
"""

import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file. If None, logs to console only
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
        
    Returns
    -------
    bool
        True if all required columns present, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    dict
        Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    return info


def print_data_info(df: pd.DataFrame) -> None:
    """
    Print comprehensive information about a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    """
    info = get_data_info(df)
    print(f"\n{'='*60}")
    print(f"DataFrame Information")
    print(f"{'='*60}")
    print(f"Shape: {info['shape']}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    print(f"Duplicate Rows: {info['duplicates']}")
    print(f"\nColumns: {len(info['columns'])}")
    print(f"Data Types:\n{pd.Series(info['dtypes'])}")
    print(f"\nMissing Values:\n{pd.Series(info['missing_values'])}")
    print(f"{'='*60}\n")


def save_dataframe(df: pd.DataFrame, path: str, format: str = "parquet") -> None:
    """
    Save DataFrame to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    path : str
        Output file path
    format : str
        File format ('parquet', 'csv', 'excel')
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    elif format == "excel":
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"DataFrame saved to: {path}")


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Parameters
    ----------
    path : str
        Input file path
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    return df


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Parameters
    ----------
    old_value : float
        Original value
    new_value : float
        New value
        
    Returns
    -------
    float
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / abs(old_value)) * 100


def format_currency(value: float, currency: str = "$") -> str:
    """
    Format value as currency.
    
    Parameters
    ----------
    value : float
        Value to format
    currency : str
        Currency symbol
        
    Returns
    -------
    str
        Formatted currency string
    """
    return f"{currency}{value:,.2f}"


def create_directory_structure(base_path: str) -> None:
    """
    Create standard project directory structure.
    
    Parameters
    ----------
    base_path : str
        Base path for project
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src",
        "tests",
        "reports/figures",
        "config",
        "logs",
    ]
    
    for directory in directories:
        Path(base_path) / directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created at: {base_path}")
