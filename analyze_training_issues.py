import pandas as pd
import numpy as np
from datetime import datetime
import re

def normalize_product_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def safe_parse_date(date_value):
    """Safely parse date values with overflow protection"""
    if pd.isna(date_value):
        return pd.NaT
    
    date_str = str(date_value).strip()
    if not date_str or date_str.lower() == 'nan':
        return pd.NaT
    
    try:
        parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
        if pd.isna(parsed):
            return pd.NaT
            
        if parsed.year < 1900 or parsed.year > 2100:
            if len(date_str.split('/')) == 3 or len(date_str.split('-')) == 3:
                parts = date_str.replace('-', '/').split('/')
                if len(parts) == 3:
                    try:
                        year = int(parts[2])
                        if year < 50:
                            year += 2000
                        elif year < 100:
                            year += 1900
                        
                        fixed_date_str = f"{parts[0]}/{parts[1]}/{year}"
                        fixed_parsed = pd.to_datetime(fixed_date_str, errors='coerce')
                        
                        if not pd.isna(fixed_parsed) and 1900 <= fixed_parsed.year <= 2100:
                            return fixed_parsed
                    except (ValueError, IndexError):
                        pass
            
            return pd.NaT
        
        return parsed
    except (OverflowError, ValueError, TypeError):
        return pd.NaT

# Read the data
print("=" * 80)
print("ANALYSIS OF TRAINING ISSUES")
print("=" * 80)

df = pd.read_excel('Data_Penjualan_Dengan_ID_Pelanggan.xlsx')
print(f"\nTotal rows in dataset: {len(df)}")
print(f"Total unique products: {df['Nama Produk'].nunique()}")

# Apply column mapping
column_mapping = {
    "Tanggal Transaksi": "date",
    "Jumlah": "sales",
    "Total Harga": "total_price",
    "Kategori": "category",
    "Nama Produk": "product_name",
}

existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
df = df.rename(columns=existing_columns)

# Select required columns
required_cols = ["date", "sales", "product_name", "category"]
df = df[required_cols].copy()

# Parse dates
df["date"] = df["date"].apply(safe_parse_date)
df = df.dropna(subset=["date"])

# Normalize product names
df["product_norm"] = df["product_name"].apply(normalize_product_name)

# Aggregate by month
df["month"] = df["date"].values.astype("datetime64[M]")
monthly_df = (df.groupby(["product_norm", "category", "month"])['sales']
              .sum()
              .reset_index())
monthly_df.rename(columns={"sales": "qty"}, inplace=True)

print(f"\nAfter preprocessing:")
print(f"  Total monthly records: {len(monthly_df)}")
print(f"  Total unique products: {monthly_df['product_norm'].nunique()}")

# Analyze per product
MIN_DATA_POINTS_MONTHS = 2
MIN_NONZERO_TRANSACTIONS = 1
TIME_STEPS = 2

product_stats = []
for prod, g in monthly_df.groupby("product_norm"):
    months = len(g)
    nonzero = int((g["qty"] > 0).sum())
    mean_qty = g["qty"].mean()
    median_qty = g["qty"].median()
    total_qty = g["qty"].sum()
    
    # Check eligibility criteria
    passes_basic = months >= MIN_DATA_POINTS_MONTHS and nonzero >= MIN_NONZERO_TRANSACTIONS
    
    # Check LSTM sequence requirement
    # After creating sequences with TIME_STEPS=2, we need at least max(2, TIME_STEPS) sequences
    # Sequences are created from (months - TIME_STEPS) data points
    sequences_available = months - TIME_STEPS
    passes_lstm = sequences_available >= max(2, TIME_STEPS)
    
    status = "SUCCESS"
    reason = ""
    if not passes_basic:
        status = "SKIP: Insufficient Data"
        reason = f"months={months}, nonzero={nonzero}"
    elif not passes_lstm:
        status = "SKIP: Insufficient Sequences"
        reason = f"sequences={sequences_available}, need={max(2, TIME_STEPS)}"
    
    product_stats.append({
        "product": prod,
        "category": g["category"].iloc[0] if len(g) > 0 else "",
        "months": months,
        "nonzero": nonzero,
        "mean_qty": mean_qty,
        "median_qty": median_qty,
        "total_qty": total_qty,
        "sequences_available": sequences_available,
        "status": status,
        "reason": reason
    })

stats_df = pd.DataFrame(product_stats)

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nTotal products analyzed: {len(stats_df)}")
print(f"\nStatus breakdown:")
print(stats_df["status"].value_counts())

print("\n" + "=" * 80)
print("DETAILED BREAKDOWN")
print("=" * 80)

# Insufficient data products
insufficient_data = stats_df[stats_df["status"] == "SKIP: Insufficient Data"]
print(f"\nProducts skipped due to insufficient data: {len(insufficient_data)}")
if len(insufficient_data) > 0:
    print("\nMonths distribution:")
    print(insufficient_data["months"].value_counts().sort_index())
    print(f"\nMost common: {insufficient_data['months'].mode().values[0] if len(insufficient_data['months'].mode()) > 0 else 'N/A'} months")

# Insufficient sequences products
insufficient_seq = stats_df[stats_df["status"] == "SKIP: Insufficient Sequences"]
print(f"\nProducts skipped due to insufficient sequences: {len(insufficient_seq)}")
if len(insufficient_seq) > 0:
    print("\nMonths distribution:")
    print(insufficient_seq["months"].value_counts().sort_index())
    print("\nSequences available distribution:")
    print(insufficient_seq["sequences_available"].value_counts().sort_index())

# Successful products
successful = stats_df[stats_df["status"] == "SUCCESS"]
print(f"\nProducts that should succeed: {len(successful)}")
if len(successful) > 0:
    print(f"\nMonths range: {successful['months'].min()} - {successful['months'].max()}")
    print(f"Mean months: {successful['months'].mean():.2f}")
    print(f"Median months: {successful['months'].median():.2f}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1. CURRENT THRESHOLDS:")
print(f"   - MIN_DATA_POINTS_MONTHS = {MIN_DATA_POINTS_MONTHS}")
print(f"   - MIN_NONZERO_TRANSACTIONS = {MIN_NONZERO_TRANSACTIONS}")
print(f"   - TIME_STEPS = {TIME_STEPS}")
print(f"   - Required sequences = max(2, TIME_STEPS) = {max(2, TIME_STEPS)}")
print(f"   - Minimum months needed = TIME_STEPS + max(2, TIME_STEPS) = {TIME_STEPS + max(2, TIME_STEPS)}")

print("\n2. ISSUE IDENTIFIED:")
print(f"   - {len(insufficient_data)} products have only 1 month of data")
print(f"   - {len(insufficient_seq)} products have 2-3 months but not enough for LSTM sequences")
print(f"   - With TIME_STEPS=2, products need at least 4 months to create 2 sequences")

print("\n3. PROPOSED SOLUTIONS:")
print("\n   Option A: Reduce TIME_STEPS to 1")
print("   - TIME_STEPS = 1")
print("   - Required sequences = max(2, 1) = 2")
print("   - Minimum months = 1 + 2 = 3 months")
print(f"   - Would enable: {len(stats_df[stats_df['months'] >= 3])} products")

print("\n   Option B: Keep TIME_STEPS=2 but reduce sequence requirement")
print("   - TIME_STEPS = 2")
print("   - Required sequences = 1 (instead of 2)")
print("   - Minimum months = 2 + 1 = 3 months")
print(f"   - Would enable: {len(stats_df[stats_df['months'] >= 3])} products")

print("\n   Option C: Use simpler model for low-data products")
print("   - Products with < 4 months: Use simple moving average or exponential smoothing")
print("   - Products with >= 4 months: Use LSTM")
print(f"   - LSTM products: {len(stats_df[stats_df['months'] >= 4])}")
print(f"   - Simple model products: {len(stats_df[stats_df['months'] < 4])}")

# Save detailed analysis
stats_df.to_csv("product_analysis_detailed.csv", index=False)
print("\n" + "=" * 80)
print(f"Detailed analysis saved to: product_analysis_detailed.csv")
print("=" * 80)
