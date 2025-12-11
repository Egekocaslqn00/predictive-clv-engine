"""
Data processing module for E-Commerce CLV Prediction.

This module provides functions for:
- Data loading and validation
- Data cleaning
- Missing value handling
- Outlier detection and treatment
- Data transformation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from file.
    
    Parameters
    ----------
    filepath : str
        Path to data file (CSV, Parquet, Excel)
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    logger.info(f"Loading data from: {filepath}")
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """
    Check for missing values in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    threshold : float
        Threshold for percentage of missing values (0-1)
        
    Returns
    -------
    dict
        Dictionary with columns and their missing value percentages
    """
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_dict = missing_percent[missing_percent > 0].to_dict()
    
    logger.info(f"Missing values detected: {len(missing_dict)} columns")
    for col, pct in missing_dict.items():
        logger.warning(f"  {col}: {pct:.2f}%")
    
    return missing_dict


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str
        Strategy for handling missing values:
        - 'drop': Remove rows with missing values
        - 'mean': Fill with mean (numeric columns)
        - 'median': Fill with median (numeric columns)
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == "drop":
        df_copy = df_copy.dropna()
        logger.info(f"Dropped rows with missing values. New shape: {df_copy.shape}")
    
    elif strategy == "mean":
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        logger.info("Filled numeric columns with mean")
    
    elif strategy == "median":
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_copy[col].isnull().any():
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
        logger.info("Filled numeric columns with median")
    
    elif strategy == "forward_fill":
        df_copy = df_copy.fillna(method='ffill')
        logger.info("Applied forward fill")
    
    elif strategy == "backward_fill":
        df_copy = df_copy.fillna(method='bfill')
        logger.info("Applied backward fill")
    
    return df_copy


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[list] = None, 
                        iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to check for outliers. If None, check all numeric columns
    iqr_multiplier : float
        IQR multiplier for outlier detection (default: 1.5)
        
    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    logger.info(f"Outliers detected: {outliers.sum().sum()} total outlier points")
    
    return outliers


def remove_outliers(df: pd.DataFrame, columns: Optional[list] = None,
                   iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from DataFrame using IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to check for outliers
    iqr_multiplier : float
        IQR multiplier for outlier detection
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed
    """
    outliers = detect_outliers_iqr(df, columns, iqr_multiplier)
    outlier_rows = outliers.any(axis=1)
    
    df_clean = df[~outlier_rows].copy()
    logger.info(f"Removed {outlier_rows.sum()} outlier rows. New shape: {df_clean.shape}")
    
    return df_clean


def convert_date_columns(df: pd.DataFrame, date_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Convert date columns to datetime format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_columns : list, optional
        List of column names to convert. If None, auto-detect
        
    Returns
    -------
    pd.DataFrame
        DataFrame with converted date columns
    """
    df_copy = df.copy()
    
    if date_columns is None:
        # Auto-detect date columns
        date_columns = []
        for col in df_copy.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
    
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            logger.info(f"Converted {col} to datetime")
    
    return df_copy


def remove_duplicates(df: pd.DataFrame, subset: Optional[list] = None,
                     keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    subset : list, optional
        Column names to consider for identifying duplicates
    keep : str
        Which duplicates to keep ('first', 'last', False)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed = initial_shape - df_clean.shape[0]
    
    logger.info(f"Removed {removed} duplicate rows. New shape: {df_clean.shape}")
    
    return df_clean


def normalize_numeric_columns(df: pd.DataFrame, columns: Optional[list] = None,
                             method: str = 'minmax') -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to normalize. If None, normalize all numeric columns
    method : str
        Normalization method ('minmax', 'zscore')
        
    Returns
    -------
    tuple
        (Normalized DataFrame, Normalization parameters)
    """
    df_copy = df.copy()
    norm_params = {}
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
        
        if method == 'minmax':
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
            norm_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        elif method == 'zscore':
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            df_copy[col] = (df_copy[col] - mean_val) / std_val
            norm_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
    
    logger.info(f"Normalized {len(norm_params)} columns using {method} method")
    
    return df_copy, norm_params


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summary = df.describe().T
    summary['missing_count'] = df.isnull().sum()
    summary['missing_percent'] = (df.isnull().sum() / len(df)) * 100
    summary['dtype'] = df.dtypes
    
    return summary


def prepare_data_for_analysis(df: pd.DataFrame, 
                             remove_outliers_flag: bool = True,
                             iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Complete data preparation pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    remove_outliers_flag : bool
        Whether to remove outliers
    iqr_multiplier : float
        IQR multiplier for outlier detection
        
    Returns
    -------
    pd.DataFrame
        Prepared DataFrame
    """
    logger.info("Starting data preparation pipeline...")
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Convert date columns
    df = convert_date_columns(df)
    
    # Remove outliers
    if remove_outliers_flag:
        df = remove_outliers(df, iqr_multiplier=iqr_multiplier)
    
    logger.info("Data preparation completed successfully")
    
    return df
