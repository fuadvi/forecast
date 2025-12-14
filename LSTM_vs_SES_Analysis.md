# 📊 Analisis Perbedaan Forecast: LSTM vs SES

## Ringkasan Masalah

Hasil forecast LSTM dan SES menunjukkan **perbedaan signifikan** dalam:
- Ranking produk per quarter
- Nilai `quarterly_sum`
- Produk yang masuk top 5

---

## 🔍 Perbedaan Fundamental

### 1. Metode Forecasting

| Aspek | LSTM | SES |
|-------|------|-----|
| **Algoritma** | Deep Learning (LSTM Neural Network) | Simple Exponential Smoothing + Holt-Winters |
| **Kompleksitas** | Tinggi (sequence-based) | Rendah (smoothing-based) |
| **Features** | Lag, rolling stats, trend, seasonality | Alpha, beta, gamma parameters |
| **Time Steps** | 6 (default) | N/A |
| **Fallback** | Baseline forecast | Alpha=0.3 fixed |

### 2. Preprocessing Data

| Aspek | LSTM | SES |
|-------|------|-----|
| **Outlier Handling** | Quantile 99% clipping | IQR method (Q1-1.5*IQR, Q3+1.5*IQR) |
| **Normalization** | MinMaxScaler | None |
| **Missing Months** | Forward fill dari baseline | Reindex dengan 0 |
| **Stabilization** | Optional (default DISABLED) | N/A |
| **Noise Injection** | 15% random noise | N/A |

### 3. Minimal Data Requirements

| Metode | Minimal Points | Behavior |
|--------|----------------|----------|
| **LSTM** | Tidak ada hard limit | Fallback forecast untuk produk dengan data sedikit |
| **SES** | 6 data points (default) | **SKIP produk** jika < 6 points |

**⚠️ PERHATIAN**: Ini bisa menyebabkan **produk yang berbeda** di-forecast oleh kedua metode!

---

## 🎯 Masalah Kritis yang Ditemukan

### **INKONSISTENSI QUARTERLY AGGREGATION**

#### LSTM (`forecast.py`)
```python
quarterly_agg = (
    quarter_data
    .groupby(['product', 'category'])  # ← Group by product + category
    .agg(quarterly_sum=('mean', 'sum'))
    .reset_index()
)
```

**Output columns**: `product`, `category`, `quarterly_sum`, `rank`

#### SES (`ses_monthly_product_forecast_24m.py`)
```python
quarterly_agg = (
    quarter_data
    .groupby(['product_name'])  # ← Group by product_name only
    .agg(quarterly_sum=('forecast', 'sum'))
    .reset_index()
)
```

**Output columns**: `product_name`, `quarterly_sum`, `rank`

### ❌ Dampak Inkonsistensi:

1. **Nama kolom berbeda**
   - LSTM: `product` + `mean`
   - SES: `product_name` + `forecast`

2. **Level agregasi berbeda**
   - LSTM: Per product + category (lebih granular)
   - SES: Per product saja (agregat semua category)

3. **Normalisasi nama produk**
   - Kemungkinan hasil `product_norm` berbeda antara LSTM dan SES

4. **Produk yang di-skip berbeda**
   - LSTM: forecast semua produk (dengan fallback)
   - SES: skip produk dengan < 6 data points

---

## ✅ Rekomendasi Perbaikan

### Priority 1: **Standardisasi Kolom Output**

**Opsi A - Ubah SES untuk match LSTM** (Recommended jika data memiliki category)

Di `ses_monthly_product_forecast_24m.py`:

```python
# 1. Tambahkan category ke forecast output
def build_forecast_frames(...):
    # ...
    df_prod = pd.DataFrame({
        "date": future_dates,
        "product_name": prod,
        "category": category,  # ← Tambahkan ini
        "forecast": fc,
        "method": ["SES"] * horizon,
    })
    
# 2. Update quarterly aggregation
def aggregate_to_quarterly(...):
    quarterly_agg = (
        quarter_data
        .groupby(['product_name', 'category'])  # ← Tambahkan category
        .agg(quarterly_sum=('forecast', 'sum'))
        .reset_index()
    )
```

**Opsi B - Ubah LSTM untuk match SES** (Jika category tidak penting)

Di `forecast.py`:

```python
def aggregate_to_quarterly(...):
    quarterly_agg = (
        quarter_data
        .groupby(['product'])  # ← Hapus category
        .agg(quarterly_sum=('mean', 'sum'))
        .reset_index()
    )
```

### Priority 2: **Standardisasi Minimal Data Points**

**Pilihan:**

1. **Konsisten skip produk**: Set LSTM juga skip jika < 6 points
   ```python
   # Di forecast.py
   if len(hist_g) < 6:
       skipped_rows.append({"product": prod, "reason": "insufficient data (<6)"})
       continue
   ```

2. **Konsisten forecast semua**: Biarkan SES juga forecast dengan data minimal
   ```python
   # Di ses_monthly_product_forecast_24m.py
   MIN_DATA_POINTS = 1  # atau 3
   ```

### Priority 3: **Standardisasi Outlier Handling**

**Recommended**: Gunakan metode yang sama di keduanya

```python
# Contoh: Gunakan quantile-based untuk keduanya
upper = g["sales"].quantile(0.99)
lower = g["sales"].quantile(0.01)
g["sales"] = g["sales"].clip(lower=lower, upper=upper)
```

### Priority 4: **Standardisasi Normalisasi Nama Produk**

Gunakan fungsi yang sama:

```python
def normalize_product_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s
```

---

## 🔧 Implementasi yang Sudah Dilakukan

✅ **Update 1**: Menambahkan detail logging di SES quarterly aggregation untuk debugging

File: `ses_monthly_product_forecast_24m.py`
- Menambahkan print statement detail per produk (sama dengan LSTM)
- Memudahkan tracking perbedaan hasil

---

## 📋 Action Items

### Immediate (Harus Dilakukan)

1. ☐ **Pilih strategi standardisasi** (Opsi A atau B di atas)
2. ☐ **Implement standardisasi kolom** di salah satu file
3. ☐ **Verify normalisasi produk** konsisten
4. ☐ **Test dengan dataset yang sama**

### Recommended (Sebaiknya Dilakukan)

5. ☐ Standardisasi minimal data points requirement
6. ☐ Standardisasi outlier handling method
7. ☐ Dokumentasi perbedaan metodologi di README
8. ☐ Create comparison report otomatis

### Optional (Tambahan)

9. ☐ Add unit tests untuk konsistensi output format
10. ☐ Create unified config file untuk shared parameters
11. ☐ Implement data validation pipeline

---

## 📊 Cara Verify Hasil

Setelah perbaikan, jalankan:

```bash
# 1. Generate forecast LSTM
python forecast.py

# 2. Generate forecast SES  
python ses_monthly_product_forecast_24m.py

# 3. Compare quarterly outputs
python -c "
import pandas as pd
lstm_q = pd.read_csv('quarterly_top5_2026.csv')
ses_q = pd.read_csv('quarterly_top5_ses_2026.csv')

print('LSTM Top 5 Q1 2026:')
print(lstm_q[lstm_q['quarter']=='Q1'][['rank', 'product', 'quarterly_sum']])

print('\nSES Top 5 Q1 2026:')
print(ses_q[ses_q['quarter']=='Q1'][['rank', 'product_name', 'quarterly_sum']])
"
```

---

## 🎓 Kesimpulan

**Penyebab utama perbedaan:**

1. ✅ **Metode forecast berbeda** (ini **normal dan expected**)
2. ❌ **Format output tidak konsisten** (ini **masalah** yang harus diperbaiki)
3. ❌ **Aggregation level berbeda** (ini **masalah** yang harus diperbaiki)
4. ⚠️ **Preprocessing berbeda** (ini **trade-off** metodologi)

**Yang harus diperbaiki:**
- **#2 dan #3** adalah bugs yang harus diperbaiki segera
- **#4** adalah design decision, tapi harus didokumentasikan

**Yang normal (tidak perlu diperbaiki):**
- **#1** adalah karakteristik masing-masing metode

---

**Created**: 2025-12-14  
**Last Updated**: 2025-12-14  
**Status**: Analysis Complete, Awaiting Implementation

