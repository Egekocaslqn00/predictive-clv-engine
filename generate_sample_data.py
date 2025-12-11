"""
Generate sample e-commerce transaction data for CLV analysis.

This script creates a synthetic dataset that mimics real e-commerce transactions
for demonstration and testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_ecommerce_data(n_customers=10000, n_transactions=100000, 
                           start_date='2018-01-01', end_date='2025-12-11'):
    """
    Generate synthetic e-commerce transaction data.
    
    Parameters
    ----------
    n_customers : int
        Number of unique customers
    n_transactions : int
        Number of transactions
    start_date : str
        Start date for transactions
    end_date : str
        End date for transactions
        
    Returns
    -------
    pd.DataFrame
        Transaction data
    """
    
    print(f"Generating {n_transactions:,} transactions for {n_customers:,} customers...")
    
    # Date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = (end - start).days
    
    # Generate transactions
    data = {
        'TransactionID': range(1, n_transactions + 1),
        'CustomerID': np.random.randint(1, n_customers + 1, n_transactions),
        'TransactionDate': [start + timedelta(days=random.randint(0, date_range)) 
                           for _ in range(n_transactions)],
        'Amount': np.random.gamma(shape=2, scale=50, size=n_transactions),  # Realistic spending
        'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 
                                            'Books', 'Sports', 'Beauty', 'Toys', 'Grocery'],
                                           n_transactions),
        'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 
                                          'UPI', 'Net Banking', 'Cash on Delivery'],
                                         n_transactions),
    }
    
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('TransactionDate').reset_index(drop=True)
    
    # Add customer demographics
    customer_data = {
        'CustomerID': range(1, n_customers + 1),
        'Age': np.random.randint(18, 70, n_customers),
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'Country': np.random.choice(['USA', 'UK', 'Canada', 'India', 'Germany', 
                                    'France', 'Australia', 'Brazil', 'Japan', 'Mexico'],
                                   n_customers),
    }
    
    customer_df = pd.DataFrame(customer_data)
    
    # Merge customer data
    df = df.merge(customer_df, on='CustomerID', how='left')
    
    # Reorder columns
    df = df[['TransactionID', 'CustomerID', 'TransactionDate', 'Amount', 
            'ProductCategory', 'PaymentMethod', 'Age', 'Gender', 'Country']]
    
    print(f"✓ Data generated successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
    print(f"  Total amount: ${df['Amount'].sum():,.2f}")
    
    return df


def main():
    """Main function to generate and save sample data."""
    
    # Generate data
    df = generate_ecommerce_data(n_customers=10000, n_transactions=100000)
    
    # Save to CSV
    output_path = 'data/raw/ecommerce_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    
    # Display sample
    print("\nSample data:")
    print(df.head(10))
    
    print("\nData info:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
