"""
Customer Segmentation module using K-Means clustering.

This module provides functions for:
- K-Means clustering
- Optimal cluster determination
- Segment analysis and profiling
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def find_optimal_clusters(X: np.ndarray, 
                         max_clusters: int = 10,
                         method: str = 'elbow') -> Tuple[int, Dict[str, list]]:
    """
    Find optimal number of clusters using Elbow or Silhouette method.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled)
    max_clusters : int
        Maximum number of clusters to test
    method : str
        Method for finding optimal clusters ('elbow', 'silhouette')
        
    Returns
    -------
    tuple
        (Optimal number of clusters, Metrics dictionary)
    """
    logger.info(f"Finding optimal clusters using {method} method...")
    
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(X, kmeans.labels_))
    
    metrics = {
        'K': list(K_range),
        'Inertia': inertias,
        'Silhouette': silhouette_scores,
        'Davies_Bouldin': davies_bouldin_scores
    }
    
    if method == 'elbow':
        # Find elbow point using the "knee" detection
        optimal_k = 3  # Default
        max_silhouette = max(silhouette_scores)
        optimal_k = silhouette_scores.index(max_silhouette) + 2
    elif method == 'silhouette':
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Optimal number of clusters: {optimal_k}")
    logger.info(f"Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}")
    logger.info(f"Davies-Bouldin Score: {davies_bouldin_scores[optimal_k-2]:.4f}")
    
    return optimal_k, metrics


def perform_kmeans_clustering(X: np.ndarray,
                             n_clusters: int = 4,
                             random_state: int = 42,
                             n_init: int = 10) -> Tuple[KMeans, np.ndarray]:
    """
    Perform K-Means clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (scaled)
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
    n_init : int
        Number of initializations
        
    Returns
    -------
    tuple
        (Fitted KMeans model, Cluster labels)
    """
    logger.info(f"Performing K-Means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, 
                   random_state=random_state,
                   n_init=n_init,
                   max_iter=300)
    
    labels = kmeans.fit_predict(X)
    
    logger.info(f"K-Means clustering completed")
    logger.info(f"Cluster distribution: {np.bincount(labels)}")
    
    return kmeans, labels


def segment_customers_kmeans(df: pd.DataFrame,
                            features: list,
                            n_clusters: int = 4,
                            random_state: int = 42) -> Tuple[pd.DataFrame, StandardScaler, KMeans]:
    """
    Segment customers using K-Means clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer data with features
    features : list
        List of feature column names
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    tuple
        (DataFrame with segments, Scaler, KMeans model)
    """
    logger.info("Segmenting customers using K-Means...")
    
    # Prepare features
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans, labels = perform_kmeans_clustering(X_scaled, n_clusters, random_state)
    
    # Add cluster labels to dataframe
    df_segmented = df.copy()
    df_segmented['Cluster'] = labels
    
    logger.info("Customer segmentation completed")
    
    return df_segmented, scaler, kmeans


def analyze_segments(df: pd.DataFrame,
                    features: list,
                    cluster_col: str = 'Cluster') -> pd.DataFrame:
    """
    Analyze and profile customer segments.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments
    features : list
        List of feature column names
    cluster_col : str
        Name of cluster column
        
    Returns
    -------
    pd.DataFrame
        Segment profiles
    """
    logger.info("Analyzing segment profiles...")
    
    segment_profiles = df.groupby(cluster_col)[features].agg(['mean', 'median', 'std']).round(2)
    
    # Add cluster size
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    segment_profiles['Cluster_Size'] = cluster_sizes
    segment_profiles['Cluster_Percent'] = (cluster_sizes / len(df) * 100).round(1)
    
    logger.info("Segment profiles:")
    logger.info(f"\n{segment_profiles}")
    
    return segment_profiles


def assign_segment_names(df: pd.DataFrame,
                        cluster_col: str = 'Cluster',
                        segment_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    """
    Assign meaningful names to clusters.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments
    cluster_col : str
        Name of cluster column
    segment_names : dict, optional
        Dictionary mapping cluster numbers to names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with named segments
    """
    df_copy = df.copy()
    
    if segment_names is None:
        # Default names based on cluster size
        cluster_sizes = df_copy[cluster_col].value_counts().sort_values(ascending=False)
        segment_names = {
            cluster_sizes.index[0]: 'Large Segment',
            cluster_sizes.index[1]: 'Medium Segment' if len(cluster_sizes) > 1 else 'Small Segment',
            cluster_sizes.index[2]: 'Small Segment' if len(cluster_sizes) > 2 else 'Tiny Segment',
        }
        if len(cluster_sizes) > 3:
            for i in range(3, len(cluster_sizes)):
                segment_names[cluster_sizes.index[i]] = f'Segment_{i+1}'
    
    df_copy['Segment_Name'] = df_copy[cluster_col].map(segment_names)
    
    logger.info(f"Segment names assigned: {segment_names}")
    
    return df_copy


def get_segment_statistics(df: pd.DataFrame,
                          cluster_col: str = 'Cluster') -> Dict[int, Dict[str, Any]]:
    """
    Get detailed statistics for each segment.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cluster assignments
    cluster_col : str
        Name of cluster column
        
    Returns
    -------
    dict
        Detailed statistics for each cluster
    """
    statistics = {}
    
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster]
        
        statistics[cluster] = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(df)) * 100,
            'numeric_stats': cluster_data.select_dtypes(include=[np.number]).describe().to_dict()
        }
    
    return statistics


def prepare_segmentation(df: pd.DataFrame,
                        features: list,
                        n_clusters: Optional[int] = None,
                        find_optimal: bool = True,
                        random_state: int = 42) -> Tuple[pd.DataFrame, StandardScaler, KMeans, Dict]:
    """
    Complete segmentation pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer data
    features : list
        Feature column names
    n_clusters : int, optional
        Number of clusters. If None and find_optimal=True, finds optimal
    find_optimal : bool
        Whether to find optimal number of clusters
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    tuple
        (Segmented DataFrame, Scaler, KMeans model, Metrics)
    """
    logger.info("Starting segmentation pipeline...")
    
    # Prepare features
    X = df[features].copy()
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters if needed
    metrics = {}
    if find_optimal and n_clusters is None:
        n_clusters, metrics = find_optimal_clusters(X_scaled, max_clusters=10, method='silhouette')
    elif n_clusters is None:
        n_clusters = 4
    
    # Perform clustering
    df_segmented, scaler, kmeans = segment_customers_kmeans(
        df, features, n_clusters, random_state
    )
    
    # Analyze segments
    segment_profiles = analyze_segments(df_segmented, features)
    
    logger.info("Segmentation pipeline completed successfully")
    
    return df_segmented, scaler, kmeans, metrics
