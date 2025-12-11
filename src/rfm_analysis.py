"""
RFM (Recency, Frequency, Monetary) Analysis module.

This module provides functions for:
- RFM metric calculation
- RFM scoring
- Customer segmentation based on RFM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_rfm_metrics(df: pd.DataFrame, 
                         customer_id_col: str = 'CustomerID',
                         transaction_date_col: str = 'InvoiceDate',
                         amount_col: str = 'Amount',
                         reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Calculate RFM metrics for each customer.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction data
    customer_id_col : str
        Name of customer ID column
    transaction_date_col : str
        Name of transaction date column
    amount_col : str
        Name of transaction amount column
    reference_date : pd.Timestamp, optional
        Reference date for recency calculation. If None, uses max date in data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with RFM metrics for each customer
    """
    logger.info("Calculating RFM metrics...")
    
    # Ensure date column is datetime
    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])
    
    # Set reference date
    if reference_date is None:
        reference_date = df[transaction_date_col].max()
    
    logger.info(f"Reference date for RFM: {reference_date}")
    
    # Calculate RFM
    rfm = df.groupby(customer_id_col).agg({
        transaction_date_col: lambda x: (reference_date - x.max()).days,  # Recency
        amount_col: 'sum'  # Monetary
    })
    
    # Add frequency
    rfm['Frequency'] = df.groupby(customer_id_col).size()
    
    # Reset index and rename columns
    rfm = rfm.reset_index()
    rfm.columns = [customer_id_col, 'Recency', 'Monetary', 'Frequency']
    
    # Reorder columns
    rfm = rfm[[customer_id_col, 'Recency', 'Frequency', 'Monetary']]
    
    logger.info(f"RFM calculated for {len(rfm)} customers")
    logger.info(f"Recency - Min: {rfm['Recency'].min()}, Max: {rfm['Recency'].max()}")
    logger.info(f"Frequency - Min: {rfm['Frequency'].min()}, Max: {rfm['Frequency'].max()}")
    logger.info(f"Monetary - Min: {rfm['Monetary'].min():.2f}, Max: {rfm['Monetary'].max():.2f}")
    
    return rfm


def calculate_rfm_scores(rfm: pd.DataFrame, 
                        score_range: int = 5) -> pd.DataFrame:
    """
    Calculate RFM scores using quantile-based approach.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with RFM metrics
    score_range : int
        Range for scores (1 to score_range)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with RFM scores
    """
    logger.info(f"Calculating RFM scores (1-{score_range})...")
    
    rfm_copy = rfm.copy()
    
    # Recency: Lower is better, so reverse the ranking
    rfm_copy['R_Score'] = pd.qcut(rfm_copy['Recency'], 
                                  q=score_range, 
                                  labels=range(score_range, 0, -1),
                                  duplicates='drop').astype(int)
    
    # Frequency: Higher is better
    rfm_copy['F_Score'] = pd.qcut(rfm_copy['Frequency'].rank(method='first'),
                                  q=score_range,
                                  labels=range(1, score_range + 1),
                                  duplicates='drop').astype(int)
    
    # Monetary: Higher is better
    rfm_copy['M_Score'] = pd.qcut(rfm_copy['Monetary'].rank(method='first'),
                                  q=score_range,
                                  labels=range(1, score_range + 1),
                                  duplicates='drop').astype(int)
    
    # Combined RFM Score
    rfm_copy['RFM_Score'] = (rfm_copy['R_Score'].astype(str) + 
                             rfm_copy['F_Score'].astype(str) + 
                             rfm_copy['M_Score'].astype(str))
    
    logger.info("RFM scores calculated successfully")
    
    return rfm_copy


def segment_customers_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers based on RFM scores.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with RFM scores
        
    Returns
    -------
    pd.DataFrame
        DataFrame with customer segments
    """
    logger.info("Segmenting customers based on RFM scores...")
    
    rfm_copy = rfm.copy()
    
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        
        # Champions: Best customers
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal Customers: Good frequency and monetary
        elif f >= 3 and m >= 3:
            return 'Loyal Customers'
        # Potential Loyalists: Recent, good frequency
        elif r >= 4 and f >= 3:
            return 'Potential Loyalists'
        # At Risk: High monetary but low recency
        elif r <= 2 and m >= 3:
            return 'At Risk'
        # Need Attention: Medium metrics
        elif r >= 3 and f >= 2:
            return 'Need Attention'
        # New Customers: High recency, low frequency
        elif r >= 4 and f <= 2:
            return 'New Customers'
        # Lost: Low recency
        elif r <= 1:
            return 'Lost'
        else:
            return 'Others'
    
    rfm_copy['Segment'] = rfm_copy.apply(assign_segment, axis=1)
    
    # Print segment distribution
    segment_dist = rfm_copy['Segment'].value_counts()
    logger.info("Customer Segment Distribution:")
    for segment, count in segment_dist.items():
        pct = (count / len(rfm_copy)) * 100
        logger.info(f"  {segment}: {count} ({pct:.1f}%)")
    
    return rfm_copy


def get_rfm_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for RFM metrics.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with RFM metrics
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summary = rfm[['Recency', 'Frequency', 'Monetary']].describe().T
    return summary


def get_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics by customer segment.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with segments
        
    Returns
    -------
    pd.DataFrame
        Segment summary statistics
    """
    segment_summary = rfm.groupby('Segment').agg({
        'Recency': ['mean', 'median', 'min', 'max'],
        'Frequency': ['mean', 'median', 'min', 'max'],
        'Monetary': ['mean', 'median', 'min', 'max'],
        'Segment': 'count'
    }).round(2)
    
    segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns.values]
    segment_summary = segment_summary.rename(columns={'Segment_count': 'Customer_Count'})
    
    return segment_summary


def calculate_clv_simple(rfm: pd.DataFrame,
                        avg_order_value: Optional[float] = None,
                        customer_lifespan_years: int = 3) -> pd.DataFrame:
    """
    Calculate simple CLV based on RFM metrics.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with RFM metrics
    avg_order_value : float, optional
        Average order value. If None, calculated from Monetary/Frequency
    customer_lifespan_years : int
        Expected customer lifespan in years
        
    Returns
    -------
    pd.DataFrame
        DataFrame with CLV estimates
    """
    logger.info("Calculating simple CLV...")
    
    rfm_copy = rfm.copy()
    
    if avg_order_value is None:
        avg_order_value = rfm_copy['Monetary'].sum() / rfm_copy['Frequency'].sum()
    
    # Annual frequency
    rfm_copy['Annual_Frequency'] = rfm_copy['Frequency'] / (rfm_copy['Recency'] / 365 + 1)
    
    # Simple CLV
    rfm_copy['Simple_CLV'] = rfm_copy['Annual_Frequency'] * avg_order_value * customer_lifespan_years
    
    logger.info(f"Simple CLV calculated. Average CLV: ${rfm_copy['Simple_CLV'].mean():.2f}")
    
    return rfm_copy


def prepare_rfm_analysis(df: pd.DataFrame,
                        customer_id_col: str = 'CustomerID',
                        transaction_date_col: str = 'InvoiceDate',
                        amount_col: str = 'Amount',
                        reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Complete RFM analysis pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction data
    customer_id_col : str
        Name of customer ID column
    transaction_date_col : str
        Name of transaction date column
    amount_col : str
        Name of transaction amount column
    reference_date : pd.Timestamp, optional
        Reference date for recency calculation
        
    Returns
    -------
    pd.DataFrame
        Complete RFM analysis with scores and segments
    """
    logger.info("Starting RFM analysis pipeline...")
    
    # Calculate RFM metrics
    rfm = calculate_rfm_metrics(df, customer_id_col, transaction_date_col, 
                               amount_col, reference_date)
    
    # Calculate RFM scores
    rfm = calculate_rfm_scores(rfm)
    
    # Segment customers
    rfm = segment_customers_rfm(rfm)
    
    # Calculate simple CLV
    rfm = calculate_clv_simple(rfm)
    
    logger.info("RFM analysis pipeline completed successfully")
    
    return rfm
