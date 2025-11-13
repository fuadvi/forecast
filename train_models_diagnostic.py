"""
DIAGNOSTIC LOGGING untuk train_models.py
File ini menambahkan logging detail untuk diagnosis masalah skip products
JANGAN mengubah logic training, hanya menambahkan logging
"""

import os
import json
import pandas as pd
from typing import Dict, List
from datetime import datetime

# Import dari train_models.py
from train_models import (
    read_and_preprocess,
    monthly_aggregate,
    filter_eligible_products,
    train_per_product,
    TIME_STEPS,
    MIN_DATA_POINTS_MONTHS,
    MIN_NONZERO_TRANSACTIONS,
    DEFAULT_EXCEL,
    OUTPUT_DIR
)

# Output file untuk diagnostic
DIAGNOSTIC_CSV = os.path.join(OUTPUT_DIR, "diagnostic_skip_analysis.csv")
DETAILED_LOG = os.path.join(OUTPUT_DIR, "detailed_skip_log.txt")


def analyze_product_data(prod: str, g: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    """Analisis detail data per produk untuk diagnosis"""
    stats = {
        "product": prod,
        "category": g["category"].iloc[0] if len(g) > 0 else "",
        
        # Data sebelum preprocessing
        "raw_months": len(g),
        "raw_nonzero": int((g["qty"] > 0).sum()),
        "raw_total_qty": float(g["qty"].sum()),
        "raw_mean_qty": float(g["qty"].mean()),
        
        # Eligibility check
        "meets_min_months": len(g) >= MIN_DATA_POINTS_MONTHS,
        "meets_min_nonzero": int((g["qty"] > 0).sum()) >= MIN_NONZERO_TRANSACTIONS,
        "is_eligible": (len(g) >= MIN_DATA_POINTS_MONTHS and 
                       int((g["qty"] > 0).sum()) >= MIN_NONZERO_TRANSACTIONS),
        
        # Sequence calculation
        "min_sequences_possible": max(0, len(g) - TIME_STEPS),
        "sequences_required": max(4, TIME_STEPS),
        "meets_sequence_requirement": max(0, len(g) - TIME_STEPS) >= max(4, TIME_STEPS),
        
        # Feature requirements
        "has_lag6": len(g) >= 6,
        "has_lag12": len(g) >= 12,
        "has_rolling6": len(g) >= 6,
        
        # Date range
        "date_min": str(g["month"].min()) if len(g) > 0 else "",
        "date_max": str(g["month"].max()) if len(g) > 0 else "",
        "date_span_months": len(g),
    }
    
    return stats


def diagnostic_train_all(excel_path: str = DEFAULT_EXCEL):
    """Versi diagnostic dari train_all dengan logging detail"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Log file
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("DIAGNOSTIC LOG - TRAINING ANALYSIS")
    log_lines.append("=" * 80)
    log_lines.append(f"Timestamp: {datetime.now()}")
    log_lines.append(f"Configuration:")
    log_lines.append(f"  TIME_STEPS: {TIME_STEPS}")
    log_lines.append(f"  MIN_DATA_POINTS_MONTHS: {MIN_DATA_POINTS_MONTHS}")
    log_lines.append(f"  MIN_NONZERO_TRANSACTIONS: {MIN_NONZERO_TRANSACTIONS}")
    log_lines.append(f"  MIN_SEQUENCES_REQUIRED: {max(4, TIME_STEPS)}")
    log_lines.append("")
    
    # Read and preprocess
    log_lines.append("Step 1: Reading and preprocessing data...")
    df = read_and_preprocess(excel_path)
    log_lines.append(f"  Raw data rows: {len(df)}")
    log_lines.append(f"  Unique products (raw): {df['product_norm'].nunique()}")
    log_lines.append("")
    
    # Monthly aggregate
    log_lines.append("Step 2: Monthly aggregation...")
    monthly_df = monthly_aggregate(df)
    log_lines.append(f"  Monthly aggregated rows: {len(monthly_df)}")
    log_lines.append(f"  Unique products (monthly): {monthly_df['product_norm'].nunique()}")
    log_lines.append("")
    
    # Analyze all products
    log_lines.append("Step 3: Analyzing all products...")
    all_product_stats = []
    
    for prod, g in monthly_df.groupby("product_norm"):
        stats = analyze_product_data(prod, g, monthly_df)
        all_product_stats.append(stats)
    
    stats_df = pd.DataFrame(all_product_stats)
    
    # Summary statistics
    log_lines.append("=" * 80)
    log_lines.append("SUMMARY STATISTICS")
    log_lines.append("=" * 80)
    log_lines.append(f"Total products: {len(stats_df)}")
    log_lines.append("")
    
    log_lines.append("Distribution of months per product:")
    log_lines.append(str(stats_df["raw_months"].describe()))
    log_lines.append("")
    
    log_lines.append("Distribution of nonzero transactions per product:")
    log_lines.append(str(stats_df["raw_nonzero"].describe()))
    log_lines.append("")
    
    log_lines.append("Eligibility breakdown:")
    log_lines.append(f"  Meets min months ({MIN_DATA_POINTS_MONTHS}+): {stats_df['meets_min_months'].sum()}")
    log_lines.append(f"  Meets min nonzero ({MIN_NONZERO_TRANSACTIONS}+): {stats_df['meets_min_nonzero'].sum()}")
    log_lines.append(f"  Is eligible: {stats_df['is_eligible'].sum()}")
    log_lines.append("")
    
    log_lines.append("Sequence requirement breakdown:")
    log_lines.append(f"  Meets sequence requirement ({max(4, TIME_STEPS)}+ sequences): {stats_df['meets_sequence_requirement'].sum()}")
    log_lines.append(f"  Average sequences possible: {stats_df['min_sequences_possible'].mean():.2f}")
    log_lines.append(f"  Products with 0 sequences: {(stats_df['min_sequences_possible'] == 0).sum()}")
    log_lines.append(f"  Products with 1-3 sequences: {((stats_df['min_sequences_possible'] >= 1) & (stats_df['min_sequences_possible'] < max(4, TIME_STEPS))).sum()}")
    log_lines.append("")
    
    # Detailed breakdown
    log_lines.append("=" * 80)
    log_lines.append("DETAILED BREAKDOWN BY SKIP REASON")
    log_lines.append("=" * 80)
    
    # Reason 1: Insufficient months
    insufficient_months = stats_df[~stats_df["meets_min_months"]]
    log_lines.append(f"1. Insufficient months (< {MIN_DATA_POINTS_MONTHS}): {len(insufficient_months)} products")
    if len(insufficient_months) > 0:
        log_lines.append(f"   Months distribution: {insufficient_months['raw_months'].value_counts().to_dict()}")
    log_lines.append("")
    
    # Reason 2: Insufficient nonzero
    insufficient_nonzero = stats_df[stats_df["meets_min_months"] & ~stats_df["meets_min_nonzero"]]
    log_lines.append(f"2. Insufficient nonzero (< {MIN_NONZERO_TRANSACTIONS}): {len(insufficient_nonzero)} products")
    if len(insufficient_nonzero) > 0:
        log_lines.append(f"   Nonzero distribution: {insufficient_nonzero['raw_nonzero'].value_counts().to_dict()}")
    log_lines.append("")
    
    # Reason 3: Insufficient sequences (for eligible products)
    eligible_but_insufficient_seq = stats_df[stats_df["is_eligible"] & ~stats_df["meets_sequence_requirement"]]
    log_lines.append(f"3. Eligible but insufficient sequences (< {max(4, TIME_STEPS)}): {len(eligible_but_insufficient_seq)} products")
    if len(eligible_but_insufficient_seq) > 0:
        log_lines.append(f"   Sequences distribution: {eligible_but_insufficient_seq['min_sequences_possible'].value_counts().to_dict()}")
        log_lines.append(f"   Months distribution: {eligible_but_insufficient_seq['raw_months'].value_counts().to_dict()}")
    log_lines.append("")
    
    # Products that should succeed
    should_succeed = stats_df[stats_df["is_eligible"] & stats_df["meets_sequence_requirement"]]
    log_lines.append(f"4. Products that should succeed: {len(should_succeed)} products")
    log_lines.append("")
    
    # Save detailed CSV
    stats_df.to_csv(DIAGNOSTIC_CSV, index=False)
    log_lines.append(f"Detailed statistics saved to: {DIAGNOSTIC_CSV}")
    log_lines.append("")
    
    # Top 20 products that will be skipped
    log_lines.append("=" * 80)
    log_lines.append("TOP 20 PRODUCTS THAT WILL BE SKIPPED")
    log_lines.append("=" * 80)
    
    will_be_skipped = stats_df[~stats_df["is_eligible"] | ~stats_df["meets_sequence_requirement"]]
    will_be_skipped = will_be_skipped.sort_values(["is_eligible", "meets_sequence_requirement", "raw_months"], 
                                                   ascending=[True, True, False])
    
    for idx, row in will_be_skipped.head(20).iterrows():
        reasons = []
        if not row["meets_min_months"]:
            reasons.append(f"months={row['raw_months']}<{MIN_DATA_POINTS_MONTHS}")
        if not row["meets_min_nonzero"]:
            reasons.append(f"nonzero={row['raw_nonzero']}<{MIN_NONZERO_TRANSACTIONS}")
        if row["is_eligible"] and not row["meets_sequence_requirement"]:
            reasons.append(f"sequences={row['min_sequences_possible']}<{max(4, TIME_STEPS)}")
        
        log_lines.append(f"  {row['product']}: {', '.join(reasons)}")
    
    log_lines.append("")
    
    # Recommendations
    log_lines.append("=" * 80)
    log_lines.append("RECOMMENDATIONS")
    log_lines.append("=" * 80)
    
    # Calculate impact of parameter changes
    if len(insufficient_months) > 0:
        months_needed = insufficient_months["raw_months"].max()
        log_lines.append(f"1. To include products with insufficient months:")
        log_lines.append(f"   Current MIN_DATA_POINTS_MONTHS: {MIN_DATA_POINTS_MONTHS}")
        log_lines.append(f"   Recommended: {max(1, months_needed)} (would include {len(insufficient_months)} more products)")
        log_lines.append("")
    
    if len(eligible_but_insufficient_seq) > 0:
        seq_needed = eligible_but_insufficient_seq["min_sequences_possible"].max()
        log_lines.append(f"2. To include eligible products with insufficient sequences:")
        log_lines.append(f"   Current MIN_SEQUENCES_REQUIRED: {max(4, TIME_STEPS)}")
        log_lines.append(f"   Recommended: {max(1, int(seq_needed))} (would include {len(eligible_but_insufficient_seq)} more products)")
        log_lines.append("")
    
    # Save log
    with open(DETAILED_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    
    print("\n".join(log_lines))
    print(f"\nDiagnostic analysis complete!")
    print(f"  - Detailed CSV: {DIAGNOSTIC_CSV}")
    print(f"  - Detailed log: {DETAILED_LOG}")
    
    return {
        "total_products": len(stats_df),
        "eligible": stats_df["is_eligible"].sum(),
        "meets_sequence_req": stats_df["meets_sequence_requirement"].sum(),
        "should_succeed": len(should_succeed),
        "diagnostic_csv": DIAGNOSTIC_CSV,
        "detailed_log": DETAILED_LOG
    }


if __name__ == "__main__":
    result = diagnostic_train_all(DEFAULT_EXCEL)
    print("\nResult:", result)

