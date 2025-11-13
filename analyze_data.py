#!/usr/bin/env python3
"""
Script untuk menganalisis data dan memahami mengapa banyak produk di-skip
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_data():
    print("=== ANALISIS DATA ===")
    
    # Load data
    df = pd.read_excel('Data_Penjualan_Dengan_ID_Pelanggan.xlsx')
    print(f"Total rows: {len(df)}")
    print(f"Unique products: {df['Nama Produk'].nunique()}")
    
    # Check date column
    print(f"\nDate column sample:")
    print(df['Tanggal Transaksi'].head(10))
    print(f"Date column dtype: {df['Tanggal Transaksi'].dtype}")
    
    # Try to parse dates
    try:
        df['Tanggal Transaksi'] = pd.to_datetime(df['Tanggal Transaksi'], errors='coerce')
        print(f"After parsing - Date range: {df['Tanggal Transaksi'].min()} to {df['Tanggal Transaksi'].max()}")
        print(f"Invalid dates: {df['Tanggal Transaksi'].isna().sum()}")
    except Exception as e:
        print(f"Error parsing dates: {e}")
    
    # Check quantity column
    print(f"\nQuantity column sample:")
    print(df['Jumlah'].head(10))
    print(f"Quantity dtype: {df['Jumlah'].dtype}")
    print(f"Quantity stats: min={df['Jumlah'].min()}, max={df['Jumlah'].max()}, mean={df['Jumlah'].mean():.2f}")
    
    # Check for zero quantities
    zero_qty = (df['Jumlah'] == 0).sum()
    print(f"Zero quantities: {zero_qty}")
    
    # Group by product and analyze
    print(f"\n=== ANALISIS PER PRODUK ===")
    
    # Normalize product names
    def normalize_product_name(name):
        if pd.isna(name):
            return ""
        s = str(name).lower().strip()
        s = s.replace("-", " ").replace("_", " ")
        s = " ".join(s.split())
        return s
    
    df['product_norm'] = df['Nama Produk'].apply(normalize_product_name)
    
    # Monthly aggregation
    df['month'] = df['Tanggal Transaksi'].values.astype("datetime64[M]")
    monthly = df.groupby(['product_norm', 'month'])['Jumlah'].sum().reset_index()
    
    print(f"Monthly aggregated rows: {len(monthly)}")
    
    # Analyze each product
    product_stats = []
    for prod, group in monthly.groupby('product_norm'):
        months = len(group)
        nonzero = (group['Jumlah'] > 0).sum()
        total_qty = group['Jumlah'].sum()
        
        product_stats.append({
            'product': prod,
            'months': months,
            'nonzero': nonzero,
            'total_qty': total_qty,
            'avg_qty': total_qty / months if months > 0 else 0
        })
    
    product_stats_df = pd.DataFrame(product_stats)
    
    print(f"\nProduct statistics:")
    print(f"Products with >= 3 months: {(product_stats_df['months'] >= 3).sum()}")
    print(f"Products with >= 6 months: {(product_stats_df['months'] >= 6).sum()}")
    print(f"Products with >= 12 months: {(product_stats_df['months'] >= 12).sum()}")
    
    print(f"\nProducts with >= 1 nonzero: {(product_stats_df['nonzero'] >= 1).sum()}")
    print(f"Products with >= 2 nonzero: {(product_stats_df['nonzero'] >= 2).sum()}")
    print(f"Products with >= 3 nonzero: {(product_stats_df['nonzero'] >= 3).sum()}")
    
    # Show top products by months
    print(f"\nTop 10 products by months:")
    top_by_months = product_stats_df.nlargest(10, 'months')[['product', 'months', 'nonzero', 'total_qty']]
    print(top_by_months)
    
    # Show products that would be eligible with current criteria
    eligible = product_stats_df[
        (product_stats_df['months'] >= 3) & 
        (product_stats_df['nonzero'] >= 1)
    ]
    print(f"\nProducts eligible with current criteria (>=3 months, >=1 nonzero): {len(eligible)}")
    
    if len(eligible) > 0:
        print("Eligible products:")
        print(eligible[['product', 'months', 'nonzero', 'total_qty']].head(10))
    
    # Show why products are skipped
    print(f"\nProducts that would be skipped:")
    skipped = product_stats_df[
        (product_stats_df['months'] < 3) | 
        (product_stats_df['nonzero'] < 1)
    ]
    print(f"Total skipped: {len(skipped)}")
    
    if len(skipped) > 0:
        print("Sample skipped products:")
        print(skipped[['product', 'months', 'nonzero', 'total_qty']].head(10))

if __name__ == "__main__":
    analyze_data()
