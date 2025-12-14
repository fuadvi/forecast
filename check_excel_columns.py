"""
Script untuk check kolom di Excel source data
Membantu identify nama kolom kategori yang sebenarnya

Usage:
    python check_excel_columns.py
"""

import pandas as pd
from pathlib import Path

def check_excel_columns(file_path: str = "Data_Penjualan_Dengan_ID_Pelanggan.xlsx"):
    """Check kolom yang ada di Excel file"""
    
    if not Path(file_path).exists():
        print(f"❌ File tidak ditemukan: {file_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"CHECKING EXCEL FILE: {file_path}")
    print(f"{'='*80}\n")
    
    # Read Excel
    df = pd.read_excel(file_path)
    
    print(f"📊 Total rows: {len(df)}")
    print(f"📋 Total columns: {len(df.columns)}")
    print(f"\n{'='*80}")
    print("COLUMNS FOUND:")
    print(f"{'='*80}\n")
    
    for i, col in enumerate(df.columns, 1):
        # Sample data
        sample = df[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(s)[:30] for s in sample])
        
        # Check if this might be category column
        is_category = False
        if 'kategori' in str(col).lower() or 'category' in str(col).lower():
            is_category = True
        
        marker = "✅ [CATEGORY]" if is_category else "   "
        print(f"{marker} {i}. '{col}'")
        print(f"       Type: {df[col].dtype}")
        print(f"       Sample: {sample_str}")
        print(f"       Unique values: {df[col].nunique()}")
        print()
    
    # Check for potential category columns
    print(f"{'='*80}")
    print("CATEGORY COLUMN DETECTION:")
    print(f"{'='*80}\n")
    
    category_cols = [
        col for col in df.columns 
        if 'kategori' in str(col).lower() or 'category' in str(col).lower()
    ]
    
    if category_cols:
        print(f"✅ FOUND {len(category_cols)} potential category column(s):")
        for col in category_cols:
            unique_vals = df[col].dropna().unique()
            print(f"\n   Column: '{col}'")
            print(f"   Unique categories ({len(unique_vals)}):")
            for val in unique_vals[:10]:  # Show first 10
                count = (df[col] == val).sum()
                print(f"      - {val} ({count} rows)")
            if len(unique_vals) > 10:
                print(f"      ... and {len(unique_vals) - 10} more")
    else:
        print("❌ NO category column found!")
        print("\nSuggested actions:")
        print("1. Add 'Kategori Barang' column to your Excel file")
        print("2. Or use existing column with similar name")
        print("\nColumns that might work as category:")
        
        # Check for columns with low unique values (might be categorical)
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1 and df[col].dtype == 'object':  # Less than 10% unique
                print(f"   - '{col}' ({df[col].nunique()} unique values)")
    
    print(f"\n{'='*80}\n")
    
    # Recommendation
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    if category_cols:
        print(f"✅ Category data tersedia di kolom: '{category_cols[0]}'")
        print(f"   SES akan otomatis menggunakan kolom ini setelah update.")
    else:
        print(f"⚠️ Tidak ada kolom kategori di Excel Anda.")
        print(f"   SES akan menggunakan 'Unknown' untuk semua produk.")
        print(f"\n   Untuk mendapatkan kategori yang benar:")
        print(f"   1. Tambahkan kolom 'Kategori Barang' di Excel")
        print(f"   2. Isi dengan kategori produk (contoh: 'Meja Kursi', 'Alat Tulis', dll)")
        print(f"   3. Regenerate forecast SES")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Try default file
    check_excel_columns()

