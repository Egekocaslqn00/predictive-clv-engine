"""
Complete E-Commerce CLV Analysis Pipeline
Runs all analysis steps and generates results
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src import data_processing, rfm_analysis, segmentation, clv_modeling, visualization, utils

# Setup
logger = utils.setup_logging('INFO')
utils.set_random_seed(42)

print("\n" + "="*80)
print("E-COMMERCE CUSTOMER LIFETIME VALUE (CLV) PREDICTION AND SEGMENTATION")
print("="*80)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n[STEP 1] Loading and Exploring Data...")
print("-" * 80)

df = data_processing.load_raw_data('data/raw/ecommerce_transactions.csv')
utils.print_data_info(df)

print(f"\nDate Range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
print(f"Total Transactions: {len(df):,}")
print(f"Unique Customers: {df['CustomerID'].nunique():,}")
print(f"Total Revenue: ${df['Amount'].sum():,.2f}")

# ============================================================================
# 2. DATA CLEANING AND PREPARATION
# ============================================================================
print("\n[STEP 2] Data Cleaning and Preparation...")
print("-" * 80)

df_clean = data_processing.prepare_data_for_analysis(df, remove_outliers_flag=False)
print(f"Cleaned data shape: {df_clean.shape}")
print(f"Rows removed: {len(df) - len(df_clean):,}")

# ============================================================================
# 3. RFM ANALYSIS
# ============================================================================
print("\n[STEP 3] RFM Analysis...")
print("-" * 80)

rfm = rfm_analysis.prepare_rfm_analysis(
    df_clean,
    customer_id_col='CustomerID',
    transaction_date_col='TransactionDate',
    amount_col='Amount'
)

print(f"\nTotal Customers: {len(rfm):,}")
print("\nRFM Summary:")
print(rfm_analysis.get_rfm_summary(rfm))

print("\nSegment Distribution:")
segment_dist = rfm['Segment'].value_counts()
for segment, count in segment_dist.items():
    pct = (count / len(rfm)) * 100
    print(f"  {segment}: {count:,} ({pct:.1f}%)")

# ============================================================================
# 4. CUSTOMER SEGMENTATION (K-MEANS)
# ============================================================================
print("\n[STEP 4] Customer Segmentation (K-Means)...")
print("-" * 80)

features = ['Recency', 'Frequency', 'Monetary']

df_segmented, scaler, kmeans, metrics = segmentation.prepare_segmentation(
    rfm,
    features=features,
    find_optimal=True,
    random_state=42
)

print(f"\nOptimal number of clusters: {df_segmented['Cluster'].nunique()}")
print("\nCluster Distribution:")
print(df_segmented['Cluster'].value_counts().sort_index())

segment_profiles = segmentation.analyze_segments(df_segmented, features)
print("\nSegment Profiles:")
print(segment_profiles)

# ============================================================================
# 5. ADVANCED CLV MODELING (BG/NBD AND PARETO/NBD)
# ============================================================================
print("\n[STEP 5] Advanced CLV Modeling...")
print("-" * 80)

# Prepare data for lifetimes models
rfm_lifetimes, reference_date = clv_modeling.prepare_rfm_for_lifetimes(
    df_clean,
    customer_id_col='CustomerID',
    transaction_date_col='TransactionDate',
    amount_col='Amount'
)

print(f"\nRFM Data for Lifetimes Models:")
print(f"Shape: {rfm_lifetimes.shape}")
print(f"Columns: {rfm_lifetimes.columns.tolist()}")

# ============================================================================
# 5A. FIT BG/NBD MODEL
# ============================================================================
print("\n[STEP 5A] Fitting BG/NBD Model...")
print("-" * 80)

try:
    bgf = clv_modeling.fit_bgf_model(rfm_lifetimes)
    
    # Predict CLV using BG/NBD
    clv_bgf = clv_modeling.predict_clv(
        bgf, 
        rfm_lifetimes, 
        prediction_period_days=365,
        monetary_col='monetary_value'
    )
    
    print(f"\nCLV Distribution (BG/NBD):")
    print(clv_bgf['CLV'].describe())
    
except Exception as e:
    print(f"Error fitting BG/NBD model: {e}")
    clv_bgf = None

# ============================================================================
# 5B. FIT PARETO/NBD MODEL
# ============================================================================
print("\n[STEP 5B] Fitting Pareto/NBD Model...")
print("-" * 80)

try:
    pnbd = clv_modeling.fit_pareto_model(rfm_lifetimes)
    
    # Predict CLV using Pareto/NBD
    clv_pnbd = clv_modeling.predict_clv(
        pnbd, 
        rfm_lifetimes, 
        prediction_period_days=365,
        monetary_col='monetary_value'
    )
    
    print(f"\nCLV Distribution (Pareto/NBD):")
    print(clv_pnbd['CLV'].describe())
    
except Exception as e:
    print(f"Error fitting Pareto/NBD model: {e}")
    clv_pnbd = None

# ============================================================================
# 6. MODEL EVALUATION AND COMPARISON
# ============================================================================
print("\n[STEP 6] Model Evaluation and Comparison...")
print("-" * 80)

if clv_bgf is not None:
    print("\nBG/NBD Model Evaluation:")
    bgf_metrics = clv_modeling.evaluate_model(bgf, rfm_lifetimes)
    for key, value in bgf_metrics.items():
        print(f"  {key}: {value}")

if clv_pnbd is not None:
    print("\nPareto/NBD Model Evaluation:")
    pnbd_metrics = clv_modeling.evaluate_model(pnbd, rfm_lifetimes)
    for key, value in pnbd_metrics.items():
        print(f"  {key}: {value}")

# ============================================================================
# 7. BUSINESS INSIGHTS AND RECOMMENDATIONS
# ============================================================================
print("\n[STEP 7] Business Insights and Recommendations...")
print("-" * 80)

# Combine CLV with RFM and segments
rfm_with_clv = rfm.copy()
if clv_bgf is not None:
    rfm_with_clv['CLV_BGF'] = clv_bgf['CLV'].values
if clv_pnbd is not None:
    rfm_with_clv['CLV_PNBD'] = clv_pnbd['CLV'].values

# Use average CLV if both models available
if clv_bgf is not None and clv_pnbd is not None:
    rfm_with_clv['CLV'] = (rfm_with_clv['CLV_BGF'] + rfm_with_clv['CLV_PNBD']) / 2
elif clv_bgf is not None:
    rfm_with_clv['CLV'] = rfm_with_clv['CLV_BGF']
elif clv_pnbd is not None:
    rfm_with_clv['CLV'] = rfm_with_clv['CLV_PNBD']
else:
    rfm_with_clv['CLV'] = 0

rfm_with_clv['Cluster'] = df_segmented['Cluster'].values

print(f"\nTop 20 Customers by CLV:")
top_customers = rfm_with_clv.nlargest(20, 'CLV')[['Recency', 'Frequency', 'Monetary', 'Segment', 'CLV']]
print(top_customers.to_string())

print(f"\n\nCLV Analysis by Segment:")
clv_by_segment = rfm_with_clv.groupby('Segment').agg({
    'CLV': ['count', 'mean', 'median', 'sum'],
    'Monetary': 'mean',
    'Frequency': 'mean'
}).round(2)
print(clv_by_segment)

print(f"\n\nCLV Analysis by Cluster:")
clv_by_cluster = rfm_with_clv.groupby('Cluster').agg({
    'CLV': ['count', 'mean', 'median', 'sum'],
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)
print(clv_by_cluster)

# ============================================================================
# 8. BUSINESS RECOMMENDATIONS
# ============================================================================
print("\n[STEP 8] Strategic Recommendations...")
print("-" * 80)

# Champions
champions = rfm_with_clv[rfm_with_clv['Segment'] == 'Champions']
print(f"\n1. CHAMPIONS ({len(champions)} customers):")
print(f"   - Average CLV: ${champions['CLV'].mean():.2f}")
print(f"   - Total Value: ${champions['CLV'].sum():,.2f}")
print(f"   - Recommendation: VIP treatment, exclusive offers, loyalty programs")

# Loyal Customers
loyal = rfm_with_clv[rfm_with_clv['Segment'] == 'Loyal Customers']
print(f"\n2. LOYAL CUSTOMERS ({len(loyal)} customers):")
print(f"   - Average CLV: ${loyal['CLV'].mean():.2f}")
print(f"   - Total Value: ${loyal['CLV'].sum():,.2f}")
print(f"   - Recommendation: Retention programs, personalized recommendations")

# At Risk
at_risk = rfm_with_clv[rfm_with_clv['Segment'] == 'At Risk']
print(f"\n3. AT RISK ({len(at_risk)} customers):")
print(f"   - Average CLV: ${at_risk['CLV'].mean():.2f}")
print(f"   - Total Value: ${at_risk['CLV'].sum():,.2f}")
print(f"   - Recommendation: Win-back campaigns, special discounts, re-engagement")

# New Customers
new = rfm_with_clv[rfm_with_clv['Segment'] == 'New Customers']
print(f"\n4. NEW CUSTOMERS ({len(new)} customers):")
print(f"   - Average CLV: ${new['CLV'].mean():.2f}")
print(f"   - Total Value: ${new['CLV'].sum():,.2f}")
print(f"   - Recommendation: Onboarding programs, welcome offers, education")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n[STEP 9] Saving Results...")
print("-" * 80)

# Save RFM with CLV
output_path = 'data/processed/rfm_with_clv.parquet'
utils.save_dataframe(rfm_with_clv, output_path, format='parquet')

# Save segmented data
output_path = 'data/processed/segmented_customers.parquet'
utils.save_dataframe(df_segmented, output_path, format='parquet')

# Save RFM analysis
output_path = 'data/processed/rfm_analysis.parquet'
utils.save_dataframe(rfm, output_path, format='parquet')

print("\n✓ Results saved successfully!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print(f"""
Total Customers Analyzed: {len(rfm):,}
Total Transactions: {len(df_clean):,}
Total Revenue: ${df_clean['Amount'].sum():,.2f}

RFM Analysis:
  - Recency: {rfm['Recency'].mean():.1f} days (avg)
  - Frequency: {rfm['Frequency'].mean():.1f} purchases (avg)
  - Monetary: ${rfm['Monetary'].mean():.2f} (avg)

Segmentation:
  - Number of Clusters: {df_segmented['Cluster'].nunique()}
  - RFM Segments: {rfm['Segment'].nunique()}

CLV Predictions:
  - Average CLV: ${rfm_with_clv['CLV'].mean():.2f}
  - Median CLV: ${rfm_with_clv['CLV'].median():.2f}
  - Total CLV: ${rfm_with_clv['CLV'].sum():,.2f}

Models Fitted:
  - BG/NBD: {'✓' if clv_bgf is not None else '✗'}
  - Pareto/NBD: {'✓' if clv_pnbd is not None else '✗'}
""")

print("="*80)
print("Analysis completed successfully!")
print("="*80 + "\n")
