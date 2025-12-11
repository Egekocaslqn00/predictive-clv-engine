"""
Visualization module for E-Commerce CLV Prediction.

This module provides functions for:
- RFM distribution visualizations
- Segmentation visualizations
- CLV prediction visualizations
- Model evaluation visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_rfm_distribution(rfm: pd.DataFrame, figsize: tuple = (15, 5)) -> None:
    """
    Plot RFM metric distributions.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data
    figsize : tuple
        Figure size
    """
    logger.info("Plotting RFM distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Recency
    axes[0].hist(rfm['Recency'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Recency Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Days Since Last Purchase')
    axes[0].set_ylabel('Number of Customers')
    
    # Frequency
    axes[1].hist(rfm['Frequency'], bins=50, color='lightcoral', edgecolor='black')
    axes[1].set_title('Frequency Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Purchases')
    axes[1].set_ylabel('Number of Customers')
    
    # Monetary
    axes[2].hist(rfm['Monetary'], bins=50, color='lightgreen', edgecolor='black')
    axes[2].set_title('Monetary Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Total Spending ($)')
    axes[2].set_ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.savefig('reports/figures/rfm_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("RFM distribution plot saved")
    plt.show()


def plot_rfm_scores(rfm: pd.DataFrame, figsize: tuple = (15, 5)) -> None:
    """
    Plot RFM score distributions.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with scores
    figsize : tuple
        Figure size
    """
    logger.info("Plotting RFM score distributions...")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # R Score
    rfm['R_Score'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Recency Score Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('R Score')
    axes[0].set_ylabel('Number of Customers')
    
    # F Score
    rfm['F_Score'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Frequency Score Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('F Score')
    axes[1].set_ylabel('Number of Customers')
    
    # M Score
    rfm['M_Score'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Monetary Score Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('M Score')
    axes[2].set_ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.savefig('reports/figures/rfm_scores_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("RFM score distribution plot saved")
    plt.show()


def plot_segment_distribution(rfm: pd.DataFrame, figsize: tuple = (12, 6)) -> None:
    """
    Plot customer segment distribution.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with segments
    figsize : tuple
        Figure size
    """
    logger.info("Plotting segment distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Segment counts
    segment_counts = rfm['Segment'].value_counts()
    segment_counts.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Customers')
    
    # Segment percentages
    segment_pct = rfm['Segment'].value_counts(normalize=True) * 100
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_pct)))
    axes[1].pie(segment_pct, labels=segment_pct.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
    axes[1].set_title('Customer Segment Percentage', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/figures/segment_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Segment distribution plot saved")
    plt.show()


def plot_segment_rfm(rfm: pd.DataFrame, figsize: tuple = (15, 5)) -> None:
    """
    Plot RFM metrics by segment.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with segments
    figsize : tuple
        Figure size
    """
    logger.info("Plotting RFM metrics by segment...")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Recency by segment
    rfm.boxplot(column='Recency', by='Segment', ax=axes[0])
    axes[0].set_title('Recency by Segment', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Days Since Last Purchase')
    
    # Frequency by segment
    rfm.boxplot(column='Frequency', by='Segment', ax=axes[1])
    axes[1].set_title('Frequency by Segment', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Segment')
    axes[1].set_ylabel('Number of Purchases')
    
    # Monetary by segment
    rfm.boxplot(column='Monetary', by='Segment', ax=axes[2])
    axes[2].set_title('Monetary by Segment', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Segment')
    axes[2].set_ylabel('Total Spending ($)')
    
    plt.suptitle('RFM Metrics by Customer Segment', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/segment_rfm_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Segment RFM analysis plot saved")
    plt.show()


def plot_clv_distribution(clv_data: pd.DataFrame, figsize: tuple = (12, 6)) -> None:
    """
    Plot CLV distribution.
    
    Parameters
    ----------
    clv_data : pd.DataFrame
        Data with CLV predictions
    figsize : tuple
        Figure size
    """
    logger.info("Plotting CLV distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(clv_data['CLV'], bins=50, color='steelblue', edgecolor='black')
    axes[0].set_title('CLV Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Customer Lifetime Value ($)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].axvline(clv_data['CLV'].mean(), color='red', linestyle='--', 
                   label=f"Mean: ${clv_data['CLV'].mean():.2f}")
    axes[0].axvline(clv_data['CLV'].median(), color='green', linestyle='--',
                   label=f"Median: ${clv_data['CLV'].median():.2f}")
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(clv_data['CLV'], vert=True)
    axes[1].set_title('CLV Box Plot', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Customer Lifetime Value ($)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/clv_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("CLV distribution plot saved")
    plt.show()


def plot_cluster_scatter(df: pd.DataFrame, 
                        x_col: str = 'Frequency',
                        y_col: str = 'Monetary',
                        cluster_col: str = 'Cluster',
                        figsize: tuple = (12, 8)) -> None:
    """
    Plot customer clusters in 2D space.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with cluster assignments
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    cluster_col : str
        Cluster column name
    figsize : tuple
        Figure size
    """
    logger.info("Plotting cluster scatter...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    clusters = df[cluster_col].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    for i, cluster in enumerate(sorted(clusters)):
        cluster_data = df[df[cluster_col] == cluster]
        ax.scatter(cluster_data[x_col], cluster_data[y_col], 
                  label=f'Cluster {cluster}', alpha=0.6, s=50, color=colors[i])
    
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax.set_title(f'Customer Clusters: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/cluster_scatter.png', dpi=300, bbox_inches='tight')
    logger.info("Cluster scatter plot saved")
    plt.show()


def plot_elbow_curve(metrics: dict, figsize: tuple = (12, 6)) -> None:
    """
    Plot elbow curve for optimal cluster determination.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from find_optimal_clusters
    figsize : tuple
        Figure size
    """
    logger.info("Plotting elbow curve...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Inertia
    axes[0].plot(metrics['K'], metrics['Inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('Elbow Method - Inertia', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[1].plot(metrics['K'], metrics['Silhouette'], 'go-', linewidth=2, markersize=8)
    axes[1].set_title('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/elbow_curve.png', dpi=300, bbox_inches='tight')
    logger.info("Elbow curve plot saved")
    plt.show()


def plot_model_comparison(comparison: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    """
    Plot model comparison metrics.
    
    Parameters
    ----------
    comparison : pd.DataFrame
        Model comparison dataframe
    figsize : tuple
        Figure size
    """
    logger.info("Plotting model comparison...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    comparison.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('BG/NBD vs Pareto/NBD Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.legend(title='Model', loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Model comparison plot saved")
    plt.show()


def create_summary_dashboard(rfm: pd.DataFrame, 
                            clv_data: Optional[pd.DataFrame] = None,
                            figsize: tuple = (16, 12)) -> None:
    """
    Create comprehensive summary dashboard.
    
    Parameters
    ----------
    rfm : pd.DataFrame
        RFM data with segments
    clv_data : pd.DataFrame, optional
        CLV predictions
    figsize : tuple
        Figure size
    """
    logger.info("Creating summary dashboard...")
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # RFM distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(rfm['Recency'], bins=30, color='skyblue', edgecolor='black')
    ax1.set_title('Recency Distribution', fontweight='bold')
    ax1.set_xlabel('Days')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rfm['Frequency'], bins=30, color='lightcoral', edgecolor='black')
    ax2.set_title('Frequency Distribution', fontweight='bold')
    ax2.set_xlabel('Purchases')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(rfm['Monetary'], bins=30, color='lightgreen', edgecolor='black')
    ax3.set_title('Monetary Distribution', fontweight='bold')
    ax3.set_xlabel('Spending ($)')
    
    # Segment distribution
    ax4 = fig.add_subplot(gs[1, 0])
    segment_counts = rfm['Segment'].value_counts()
    ax4.barh(segment_counts.index, segment_counts.values, color='steelblue')
    ax4.set_title('Segment Distribution', fontweight='bold')
    ax4.set_xlabel('Count')
    
    # Segment percentages
    ax5 = fig.add_subplot(gs[1, 1])
    segment_pct = rfm['Segment'].value_counts(normalize=True) * 100
    colors = plt.cm.Set3(np.linspace(0, 1, len(segment_pct)))
    ax5.pie(segment_pct, labels=segment_pct.index, autopct='%1.1f%%', colors=colors)
    ax5.set_title('Segment Percentage', fontweight='bold')
    
    # RFM by segment
    ax6 = fig.add_subplot(gs[1, 2])
    rfm.boxplot(column='Monetary', by='Segment', ax=ax6)
    ax6.set_title('Monetary by Segment', fontweight='bold')
    ax6.set_xlabel('Segment')
    ax6.set_ylabel('Spending ($)')
    
    # CLV distribution (if available)
    if clv_data is not None:
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.hist(clv_data['CLV'], bins=50, color='steelblue', edgecolor='black')
        ax7.set_title('CLV Distribution', fontweight='bold')
        ax7.set_xlabel('Customer Lifetime Value ($)')
        ax7.set_ylabel('Count')
        ax7.axvline(clv_data['CLV'].mean(), color='red', linestyle='--', 
                   label=f"Mean: ${clv_data['CLV'].mean():.2f}")
        ax7.legend()
    
    # Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary_text = f"""
    SUMMARY STATISTICS
    
    Total Customers: {len(rfm):,}
    
    Recency:
      Mean: {rfm['Recency'].mean():.1f} days
      Median: {rfm['Recency'].median():.1f} days
    
    Frequency:
      Mean: {rfm['Frequency'].mean():.1f}
      Median: {rfm['Frequency'].median():.1f}
    
    Monetary:
      Mean: ${rfm['Monetary'].mean():.2f}
      Median: ${rfm['Monetary'].median():.2f}
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.suptitle('E-Commerce CLV Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('reports/figures/summary_dashboard.png', dpi=300, bbox_inches='tight')
    logger.info("Summary dashboard saved")
    plt.show()
