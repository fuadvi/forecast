"""
Script untuk membandingkan hasil forecast SES vs LSTM
Fokus pada quarterly rankings untuk verifikasi konsistensi Opsi A

Usage:
    python compare_ses_lstm_results.py
"""

import pandas as pd
from pathlib import Path

def compare_quarterly_results(year: int = 2026, quarter: str = "Q1"):
    """Compare SES vs LSTM quarterly results for a specific year and quarter."""
    
    # File paths
    ses_file = f"quarterly_top5_ses_{year}.csv"
    lstm_file = f"quarterly_top5_{year}.csv"
    
    # Check if files exist
    if not Path(ses_file).exists():
        print(f"❌ File tidak ditemukan: {ses_file}")
        print("   Jalankan: python ses_monthly_product_forecast_24m.py")
        return False
    
    if not Path(lstm_file).exists():
        print(f"❌ File tidak ditemukan: {lstm_file}")
        print("   Jalankan: python forecast.py")
        return False
    
    # Load data
    print(f"\n{'='*80}")
    print(f"COMPARISON: {quarter} {year}")
    print(f"{'='*80}\n")
    
    ses_df = pd.read_csv(ses_file)
    lstm_df = pd.read_csv(lstm_file)
    
    # Filter for specific quarter
    ses_q = ses_df[ses_df['quarter'] == quarter].copy()
    lstm_q = lstm_df[lstm_df['quarter'] == quarter].copy()
    
    # Display SES results
    print(f"📊 SES Top 5 - {quarter} {year}")
    print(f"{'-'*80}")
    if not ses_q.empty:
        # Check if category column exists (Opsi A)
        if 'category' in ses_q.columns:
            print("✅ Category column: FOUND (Opsi A implemented)")
            display_cols = ['rank', 'product_name', 'category', 'quarterly_sum']
        else:
            print("⚠️ Category column: NOT FOUND (Opsi A not implemented)")
            display_cols = ['rank', 'product_name', 'quarterly_sum']
        
        print(ses_q[display_cols].to_string(index=False))
    else:
        print(f"❌ No data for {quarter} {year}")
    
    print(f"\n{'-'*80}\n")
    
    # Display LSTM results
    print(f"🧠 LSTM Top 5 - {quarter} {year}")
    print(f"{'-'*80}")
    if not lstm_q.empty:
        if 'category' in lstm_q.columns:
            display_cols = ['rank', 'product', 'category', 'quarterly_sum']
        else:
            display_cols = ['rank', 'product', 'quarterly_sum']
        
        print(lstm_q[display_cols].to_string(index=False))
    else:
        print(f"❌ No data for {quarter} {year}")
    
    print(f"\n{'-'*80}\n")
    
    # Column structure comparison
    print("📋 COLUMN STRUCTURE COMPARISON")
    print(f"{'-'*80}")
    print(f"SES columns:  {ses_df.columns.tolist()}")
    print(f"LSTM columns: {lstm_df.columns.tolist()}")
    
    # Check consistency
    ses_has_category = 'category' in ses_df.columns
    lstm_has_category = 'category' in lstm_df.columns
    
    print(f"\n{'='*80}")
    if ses_has_category and lstm_has_category:
        print("✅ STATUS: KONSISTEN - Kedua output memiliki kolom category")
        print("   Opsi A telah berhasil diimplementasi!")
    elif not ses_has_category and not lstm_has_category:
        print("⚠️ STATUS: KONSISTEN (tanpa category) - Kedua output tidak ada category")
    else:
        print("❌ STATUS: INKONSISTEN - Satu punya category, satu tidak")
        if not ses_has_category:
            print("   SES perlu regenerate setelah Opsi A diimplementasi")
        if not lstm_has_category:
            print("   LSTM perlu regenerate atau update")
    print(f"{'='*80}\n")
    
    return True


def compare_forecast_per_product():
    """Compare per-product forecast file structures."""
    
    print(f"\n{'='*80}")
    print("PER-PRODUCT FORECAST FILE COMPARISON")
    print(f"{'='*80}\n")
    
    ses_file = "forecast_per_product_ses_24m.csv"
    lstm_file = "forecast_per_product_24m.csv"
    
    if Path(ses_file).exists():
        ses_df = pd.read_csv(ses_file, nrows=5)
        print(f"📊 SES Preview ({ses_file}):")
        print(f"{'-'*80}")
        print(f"Columns: {ses_df.columns.tolist()}")
        print(f"Sample (first 5 rows):")
        print(ses_df.head().to_string(index=False))
    else:
        print(f"❌ {ses_file} not found")
    
    print()
    
    if Path(lstm_file).exists():
        lstm_df = pd.read_csv(lstm_file, nrows=5)
        print(f"🧠 LSTM Preview ({lstm_file}):")
        print(f"{'-'*80}")
        print(f"Columns: {lstm_df.columns.tolist()}")
        print(f"Sample (first 5 rows):")
        print(lstm_df.head().to_string(index=False))
    else:
        print(f"❌ {lstm_file} not found")
    
    print(f"\n{'='*80}\n")


def compare_borda_count_results(year: int = 2026):
    """Compare yearly Borda Count results."""
    
    print(f"\n{'='*80}")
    print(f"BORDA COUNT COMPARISON - Year {year}")
    print(f"{'='*80}\n")
    
    ses_file = f"yearly_top5_borda_ses_{year}.csv"
    lstm_file = f"yearly_top5_borda_{year}.csv"
    
    if not Path(ses_file).exists():
        print(f"❌ {ses_file} not found")
        return
    
    if not Path(lstm_file).exists():
        print(f"❌ {lstm_file} not found")
        return
    
    ses_df = pd.read_csv(ses_file)
    lstm_df = pd.read_csv(lstm_file)
    
    # SES Top 5
    print(f"📊 SES Top 5 Borda Count - {year}")
    print(f"{'-'*80}")
    if 'category' in ses_df.columns:
        print("✅ Category column: FOUND")
        display_cols = ['rank', 'product', 'category', 'total_score', 'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score']
    else:
        print("⚠️ Category column: NOT FOUND")
        display_cols = ['rank', 'product', 'total_score', 'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score']
    
    available_cols = [col for col in display_cols if col in ses_df.columns]
    print(ses_df.head(5)[available_cols].to_string(index=False))
    
    print(f"\n{'-'*80}\n")
    
    # LSTM Top 5
    print(f"🧠 LSTM Top 5 Borda Count - {year}")
    print(f"{'-'*80}")
    if 'category' in lstm_df.columns:
        display_cols = ['rank', 'product', 'category', 'total_score', 'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score']
    else:
        display_cols = ['rank', 'product', 'total_score', 'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score']
    
    available_cols = [col for col in display_cols if col in lstm_df.columns]
    print(lstm_df.head(5)[available_cols].to_string(index=False))
    
    print(f"\n{'='*80}\n")


def main():
    """Run all comparisons."""
    
    print("\n" + "="*80)
    print("🔍 LSTM vs SES FORECAST COMPARISON TOOL")
    print("="*80)
    
    # 1. Compare quarterly results for Q1 2026
    success = compare_quarterly_results(year=2026, quarter="Q1")
    
    if success:
        # 2. Compare per-product forecast files
        compare_forecast_per_product()
        
        # 3. Compare Borda Count results
        compare_borda_count_results(year=2026)
    
    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("""
Jika terlihat inkonsistensi:

1. ❌ SES tidak punya category tapi LSTM punya:
   → Regenerate SES: python ses_monthly_product_forecast_24m.py
   
2. ❌ Kedua tidak punya category tapi Opsi A sudah diimplementasi:
   → Regenerate keduanya:
     - python forecast.py
     - python ses_monthly_product_forecast_24m.py
   
3. ✅ Kedua punya category dengan format konsisten:
   → SUKSES! Opsi A berhasil diimplementasi
   
4. ⚠️ Ranking berbeda tapi struktur konsisten:
   → NORMAL - Metode forecast berbeda (expected behavior)
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

