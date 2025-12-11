"""
Advanced CLV Modeling using Probabilistic Models (BG/NBD and Pareto/NBD).

This module provides functions for:
- BG/NBD (Beta-Geometric/Negative Binomial Distribution) model
- Pareto/NBD model
- Model fitting and evaluation
- CLV prediction
"""

import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, ParetoNBDFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def prepare_rfm_for_lifetimes(df: pd.DataFrame,
                             customer_id_col: str = 'CustomerID',
                             transaction_date_col: str = 'InvoiceDate',
                             amount_col: str = 'Amount',
                             observation_period_days: int = 365) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Prepare transaction data for lifetimes library (BG/NBD and Pareto/NBD).
    
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
    observation_period_days : int
        Length of observation period in days
        
    Returns
    -------
    tuple
        (RFM DataFrame for lifetimes, Reference date)
    """
    logger.info("Preparing data for lifetimes models...")
    
    # Ensure date column is datetime
    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])
    
    # Set reference date as max date in data
    reference_date = df[transaction_date_col].max()
    logger.info(f"Reference date: {reference_date}")
    
    # Get RFM summary using summary_data_from_transaction_data
    rfm = summary_data_from_transaction_data(
        df,
        customer_id_col=customer_id_col,
        datetime_col=transaction_date_col,
        monetary_value_col=amount_col,
        observation_period_end=reference_date
    )
    
    logger.info(f"RFM data prepared for {len(rfm)} customers")
    logger.info(f"Frequency - Min: {rfm['frequency'].min()}, Max: {rfm['frequency'].max()}")
    logger.info(f"Recency - Min: {rfm['recency'].min()}, Max: {rfm['recency'].max()}")
    logger.info(f"T - Min: {rfm['T'].min()}, Max: {rfm['T'].max()}")
    
    return rfm, reference_date


def fit_bgf_model(rfm: pd.DataFrame,
                 penalizer_coef: float = 0.0,
                 max_iterations: int = 10000) -> BetaGeoFitter:
    """
    Fit BG/NBD (Beta-Geometric/Negative Binomial Distribution) model.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with columns: frequency, recency, T, monetary_value
    penalizer_coef : float
        Regularization coefficient
    max_iterations : int
        Maximum iterations for fitting
        
    Returns
    -------
    BetaGeoFitter
        Fitted BG/NBD model
    """
    logger.info("Fitting BG/NBD model...")
    
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'], verbose=True)
    
    logger.info("BG/NBD model fitted successfully")
    logger.info(f"Model parameters:\n{bgf.params_}")
    
    return bgf


def fit_pareto_model(rfm: pd.DataFrame,
                    penalizer_coef: float = 0.0,
                    max_iterations: int = 10000) -> ParetoNBDFitter:
    """
    Fit Pareto/NBD model.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with columns: frequency, recency, T, monetary_value
    penalizer_coef : float
        Regularization coefficient
    max_iterations : int
        Maximum iterations for fitting
        
    Returns
    -------
    ParetoNBDFitter
        Fitted Pareto/NBD model
    """
    logger.info("Fitting Pareto/NBD model...")
    
    pnbd = ParetoNBDFitter(penalizer_coef=penalizer_coef)
    pnbd.fit(rfm['frequency'], rfm['recency'], rfm['T'], verbose=True)
    
    logger.info("Pareto/NBD model fitted successfully")
    logger.info(f"Model parameters:\n{pnbd.params_}")
    
    return pnbd


def predict_clv(model,
               rfm: pd.DataFrame,
               prediction_period_days: int = 365,
               discount_rate: float = 0.01,
               monetary_col: str = 'monetary_value') -> pd.DataFrame:
    """
    Predict Customer Lifetime Value using fitted model.
    
    Parameters
    ----------
    model : BetaGeoFitter or ParetoNBDFitter
        Fitted CLV model
    rfm : pd.DataFrame
        RFM data
    prediction_period_days : int
        Number of days for prediction
    discount_rate : float
        Annual discount rate for CLV calculation
    monetary_col : str
        Name of monetary value column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with CLV predictions
    """
    logger.info(f"Predicting CLV for {prediction_period_days} days...")
    
    # Predict expected transactions
    expected_transactions = model.predict(
        periods=prediction_period_days / 365,
        frequency=rfm['frequency'],
        recency=rfm['recency'],
        T=rfm['T']
    )
    
    # Calculate average monetary value
    avg_monetary = rfm[monetary_col].mean()
    
    # Calculate CLV
    clv = expected_transactions * avg_monetary
    
    # Create prediction dataframe
    predictions = pd.DataFrame({
        'Expected_Transactions': expected_transactions,
        'CLV': clv
    })
    
    logger.info(f"CLV predictions completed")
    logger.info(f"Average CLV: ${clv.mean():.2f}")
    logger.info(f"Median CLV: ${clv.median():.2f}")
    logger.info(f"Max CLV: ${clv.max():.2f}")
    
    return predictions


def evaluate_model(model,
                  rfm: pd.DataFrame,
                  test_data: Optional[pd.DataFrame] = None,
                  prediction_period_days: int = 365,
                  monetary_col: str = 'monetary_value') -> Dict[str, float]:
    """
    Evaluate CLV model performance.
    
    Parameters
    ----------
    model : BetaGeoFitter or ParetoNBDFitter
        Fitted CLV model
    rfm : pd.DataFrame
        Training RFM data
    test_data : pd.DataFrame, optional
        Test data for evaluation
    prediction_period_days : int
        Number of days for prediction
    monetary_col : str
        Name of monetary value column
        
    Returns
    -------
    dict
        Model evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Get predictions
    predictions = predict_clv(model, rfm, prediction_period_days, 
                             monetary_col=monetary_col)
    
    metrics = {
        'model_type': model.__class__.__name__,
        'n_customers': len(rfm),
        'avg_clv': predictions['CLV'].mean(),
        'median_clv': predictions['CLV'].median(),
        'std_clv': predictions['CLV'].std(),
        'min_clv': predictions['CLV'].min(),
        'max_clv': predictions['CLV'].max(),
    }
    
    # If test data available, calculate error metrics
    if test_data is not None:
        test_predictions = predict_clv(model, test_data, prediction_period_days,
                                      monetary_col=monetary_col)
        
        # Assume test_data has actual CLV (for comparison)
        if 'actual_clv' in test_data.columns:
            actual_clv = test_data['actual_clv']
            pred_clv = test_predictions['CLV']
            
            mae = mean_absolute_error(actual_clv, pred_clv)
            rmse = np.sqrt(mean_squared_error(actual_clv, pred_clv))
            mape = np.mean(np.abs((actual_clv - pred_clv) / actual_clv)) * 100
            r2 = r2_score(actual_clv, pred_clv)
            
            metrics.update({
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
            })
    
    logger.info(f"Model evaluation metrics:\n{metrics}")
    
    return metrics


def compare_models(rfm: pd.DataFrame,
                  test_data: Optional[pd.DataFrame] = None,
                  prediction_period_days: int = 365,
                  monetary_col: str = 'monetary_value') -> pd.DataFrame:
    """
    Compare BG/NBD and Pareto/NBD models.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        Training RFM data
    test_data : pd.DataFrame, optional
        Test data for evaluation
    prediction_period_days : int
        Number of days for prediction
    monetary_col : str
        Name of monetary value column
        
    Returns
    -------
    pd.DataFrame
        Comparison of model metrics
    """
    logger.info("Comparing BG/NBD and Pareto/NBD models...")
    
    # Fit both models
    bgf = fit_bgf_model(rfm)
    pnbd = fit_pareto_model(rfm)
    
    # Evaluate both models
    bgf_metrics = evaluate_model(bgf, rfm, test_data, prediction_period_days, monetary_col)
    pnbd_metrics = evaluate_model(pnbd, rfm, test_data, prediction_period_days, monetary_col)
    
    # Create comparison dataframe
    comparison = pd.DataFrame([bgf_metrics, pnbd_metrics]).T
    comparison.columns = ['BG/NBD', 'Pareto/NBD']
    
    logger.info(f"Model comparison:\n{comparison}")
    
    return comparison


def prepare_clv_analysis(df: pd.DataFrame,
                        customer_id_col: str = 'CustomerID',
                        transaction_date_col: str = 'InvoiceDate',
                        amount_col: str = 'Amount',
                        model_type: str = 'bgf',
                        prediction_period_days: int = 365) -> Tuple[pd.DataFrame, Any]:
    """
    Complete CLV modeling pipeline.
    
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
    model_type : str
        Type of model ('bgf' or 'pareto')
    prediction_period_days : int
        Number of days for prediction
        
    Returns
    -------
    tuple
        (CLV predictions with RFM, Fitted model)
    """
    logger.info("Starting CLV modeling pipeline...")
    
    # Prepare RFM data
    rfm, reference_date = prepare_rfm_for_lifetimes(
        df, customer_id_col, transaction_date_col, amount_col
    )
    
    # Fit model
    if model_type == 'bgf':
        model = fit_bgf_model(rfm)
    elif model_type == 'pareto':
        model = fit_pareto_model(rfm)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Predict CLV
    clv_predictions = predict_clv(model, rfm, prediction_period_days)
    
    # Combine with RFM
    result = rfm.copy()
    result['CLV'] = clv_predictions['CLV']
    result['Expected_Transactions'] = clv_predictions['Expected_Transactions']
    result['Model_Type'] = model_type
    result['Reference_Date'] = reference_date
    
    logger.info("CLV modeling pipeline completed successfully")
    
    return result, model
