# ANALISIS PENYEBAB PRODUK DI-SKIP SAAT TRAINING

## STATISTIK MASALAH
- **Total Produk**: 239
- **Produk Di-skip**: 223 (93.3%)
- **Produk Berhasil**: 16 (6.7%)
- **Success Rate**: 6.69%

---

## IDENTIFIKASI SEMUA KEMUNGKINAN PENYEBAB SKIP

### 1. FILTER ELIGIBLE PRODUCTS (Baris 297-307)
**Lokasi**: `filter_eligible_products()` function

**Kondisi Skip**:
```python
if months >= MIN_DATA_POINTS_MONTHS and nonzero >= MIN_NONZERO_TRANSACTIONS:
    eligible.append(prod)
```

**Threshold Saat Ini**:
- `MIN_DATA_POINTS_MONTHS = 3` (baris 38)
- `MIN_NONZERO_TRANSACTIONS = 1` (baris 39)

**Penjelasan**:
- Produk di-skip jika memiliki **kurang dari 3 bulan data** ATAU **kurang dari 1 transaksi non-zero**
- Threshold ini sudah sangat rendah, tapi masih bisa menjadi penyebab jika:
  - Produk hanya punya 1-2 bulan data
  - Produk punya 3+ bulan tapi semua transaksi = 0

**Dampak Potensial**: 
- Jika banyak produk dengan data < 3 bulan → akan di-skip
- Jika produk punya data bulanan tapi semua qty = 0 → akan di-skip

**Rekomendasi**:
- Turunkan `MIN_DATA_POINTS_MONTHS` ke 2 atau bahkan 1
- Atau buat exception untuk produk dengan data minimal 1 bulan dengan transaksi > 0

---

### 2. INSUFFICIENT SEQUENCES UNTUK LSTM (Baris 374-375, 513-514)
**Lokasi**: `train_per_product()` function, direct mode dan residual mode

**Kondisi Skip**:
```python
X_seq, y_seq = _create_sequences(data_scaled, TIME_STEPS, target_idx)
if len(X_seq) < max(4, TIME_STEPS):
    raise ValueError("insufficient sequences for LSTM training")
```

**Threshold Saat Ini**:
- `TIME_STEPS = 2` (baris 37)
- Minimum sequences required = `max(4, 2) = 4`

**Penjelasan**:
- `_create_sequences()` membutuhkan minimal `TIME_STEPS + 1` data points untuk membuat 1 sequence
- Dengan `TIME_STEPS = 2`, minimal butuh 3 data points untuk 1 sequence
- Tapi kode memerlukan minimal 4 sequences
- **Rumus**: Jika punya N data points, bisa buat `N - TIME_STEPS` sequences
- Jadi untuk 4 sequences dengan TIME_STEPS=2, butuh minimal 6 data points

**Contoh**:
- 3 bulan data → 1 sequence → **SKIP** (butuh 4 sequences)
- 4 bulan data → 2 sequences → **SKIP** (butuh 4 sequences)
- 5 bulan data → 3 sequences → **SKIP** (butuh 4 sequences)
- 6 bulan data → 4 sequences → **OK**

**Dampak Potensial**: 
- **SANGAT KRITIS** - Ini kemungkinan besar penyebab utama!
- Produk dengan 3-5 bulan data akan di-skip meskipun sudah melewati filter eligible
- Dengan `MIN_DATA_POINTS_MONTHS = 3`, produk bisa punya 3 bulan data, tapi hanya menghasilkan 1 sequence → SKIP

**Rekomendasi**:
- Turunkan threshold ke `max(2, TIME_STEPS)` atau `max(1, TIME_STEPS)`
- Atau sesuaikan `TIME_STEPS` dengan data yang tersedia
- Atau buat fallback untuk produk dengan sedikit data

---

### 3. DATA LOSS SAAT PREPROCESSING (Baris 215-277)
**Lokasi**: `read_and_preprocess()` function

**Kondisi yang Menyebabkan Data Hilang**:

#### 3a. Missing Date Values (Baris 256, 263)
```python
df = df.dropna(subset=["date"]).copy()  # Baris 256
df = df.dropna(subset=["date"])  # Baris 263 (setelah parsing)
```

**Penjelasan**:
- Rows dengan date null atau invalid dihapus
- Jika banyak produk punya date invalid → data hilang → bulan data berkurang

**Dampak**: Bisa mengurangi jumlah bulan per produk

#### 3b. Date Parsing Failure (Baris 259-264)
```python
df["date"] = df["date"].apply(safe_parse_date)
df = df.dropna(subset=["date"])
```

**Penjelasan**:
- `safe_parse_date()` bisa return `pd.NaT` jika parsing gagal
- Rows dengan date yang tidak bisa di-parse akan dihapus

**Dampak**: Bisa mengurangi jumlah bulan per produk

#### 3c. Outlier Clipping (Baris 248-254, 275)
```python
def _clip_group(g):
    if len(g) < 3:
        return g
    up = g["sales"].quantile(0.99)
    g["sales"] = g["sales"].clip(lower=0)
    g["sales"] = np.minimum(g["sales"], up)
    return g
```

**Penjelasan**:
- Clipping di level produk, tidak menghapus data
- Tapi bisa membuat semua nilai menjadi 0 jika semua data adalah outlier

**Dampak**: Minimal, tapi bisa membuat semua qty = 0

---

### 4. MONTHLY AGGREGATION (Baris 280-294)
**Lokasi**: `monthly_aggregate()` function

**Kondisi**:
```python
agg = (df.groupby(["product_norm", "category", "month"])['sales']
         .sum()
         .reset_index())
```

**Penjelasan**:
- Agregasi per bulan bisa mengurangi jumlah data points jika:
  - Ada bulan yang tidak punya transaksi (tidak muncul di hasil)
  - Multiple transactions per bulan di-aggregate jadi 1 row

**Dampak**: 
- Jika produk punya transaksi di 3 bulan berbeda, tapi setelah agregasi hanya 2 bulan yang punya qty > 0 → bisa di-skip karena `nonzero < MIN_NONZERO_TRANSACTIONS`

---

### 5. EXCEPTION DURING TRAINING (Baris 695-697)
**Lokasi**: `train_all()` function, exception handler

**Kondisi Skip**:
```python
try:
    res = train_per_product(prod, g)
    # ... save model
except Exception as e:
    skipped.append(f"{prod}: error during training - {e}")
    continue
```

**Penjelasan**:
- Catch-all untuk semua error selama training
- Bisa terjadi karena:
  - Memory error
  - Model training error
  - Feature engineering error
  - Scaling error
  - dll

**Dampak**: Produk yang sudah eligible tapi error saat training akan di-skip

---

### 6. FEATURE ENGINEERING REQUIREMENTS (Baris 323-358)
**Lokasi**: `train_per_product()` function, direct mode

**Kondisi yang Bisa Menyebabkan Masalah**:
- Lag features (lag_6, lag_12) membutuhkan minimal 6-12 data points
- Rolling features membutuhkan minimal 3-6 data points
- Jika data kurang, banyak feature akan = 0 atau NaN

**Dampak**: 
- Bisa menyebabkan masalah saat scaling atau training
- Tapi kode sudah handle dengan `.bfill().fillna(0.0)` (baris 358)

---

## RINGKASAN PRIORITAS MASALAH

### 🔴 KRITIS - Kemungkinan Besar Penyebab Utama:

1. **Insufficient Sequences (Baris 374, 513)**
   - Threshold: 4 sequences minimum
   - Dengan TIME_STEPS=2, butuh minimal 6 bulan data
   - Produk dengan 3-5 bulan akan di-skip
   - **Impact**: Sangat tinggi

2. **Filter Eligible Products (Baris 303)**
   - Threshold: 3 bulan minimum
   - Produk dengan < 3 bulan akan di-skip
   - **Impact**: Tinggi

### 🟡 SEDANG - Bisa Menjadi Penyebab:

3. **Date Parsing Failure**
   - Data dengan date invalid akan dihapus
   - Bisa mengurangi jumlah bulan per produk
   - **Impact**: Sedang

4. **Monthly Aggregation**
   - Bulan tanpa transaksi tidak muncul
   - Bisa mengurangi `nonzero` count
   - **Impact**: Sedang

### 🟢 RENDAH - Kemungkinan Kecil:

5. **Exception During Training**
   - Error saat training
   - **Impact**: Rendah (hanya untuk produk yang sudah eligible)

6. **Outlier Clipping**
   - Bisa membuat semua qty = 0
   - **Impact**: Sangat rendah

---

## REKOMENDASI PARAMETER

### Parameter Saat Ini:
```python
TIME_STEPS = 2
MIN_DATA_POINTS_MONTHS = 3
MIN_NONZERO_TRANSACTIONS = 1
MIN_SEQUENCES_REQUIRED = max(4, TIME_STEPS) = 4
```

### Rekomendasi untuk Meningkatkan Success Rate:

#### Opsi 1: Agresif (Target: 80%+ success rate)
```python
TIME_STEPS = 1  # Minimal time steps
MIN_DATA_POINTS_MONTHS = 1  # Minimal 1 bulan
MIN_NONZERO_TRANSACTIONS = 1  # Tetap 1
MIN_SEQUENCES_REQUIRED = max(1, TIME_STEPS) = 1  # Minimal 1 sequence
```

#### Opsi 2: Moderate (Target: 50-60% success rate)
```python
TIME_STEPS = 2  # Tetap
MIN_DATA_POINTS_MONTHS = 2  # Turun dari 3 ke 2
MIN_NONZERO_TRANSACTIONS = 1  # Tetap
MIN_SEQUENCES_REQUIRED = max(2, TIME_STEPS) = 2  # Turun dari 4 ke 2
```

#### Opsi 3: Conservative (Target: 30-40% success rate)
```python
TIME_STEPS = 2  # Tetap
MIN_DATA_POINTS_MONTHS = 3  # Tetap
MIN_NONZERO_TRANSACTIONS = 1  # Tetap
MIN_SEQUENCES_REQUIRED = max(2, TIME_STEPS) = 2  # Turun dari 4 ke 2
```

**Rekomendasi**: Mulai dengan **Opsi 2** untuk balance antara kualitas model dan coverage produk.

---

## STATISTIK YANG PERLU DILOGGING

Untuk diagnosis lebih baik, perlu logging detail:

1. **Per Produk**:
   - Jumlah bulan data sebelum preprocessing
   - Jumlah bulan data setelah preprocessing
   - Jumlah bulan dengan qty > 0
   - Jumlah sequences yang dihasilkan
   - Alasan skip (jika di-skip)

2. **Agregat**:
   - Distribusi jumlah bulan per produk
   - Distribusi jumlah sequences per produk
   - Breakdown alasan skip
   - Success rate per kategori produk

---

## KESIMPULAN

**Penyebab Utama Skip (93% produk)** kemungkinan besar adalah:

1. **Insufficient Sequences** (60-70% produk)
   - Produk dengan 3-5 bulan data tidak memenuhi requirement 4 sequences
   - Dengan TIME_STEPS=2, butuh minimal 6 bulan untuk 4 sequences

2. **Insufficient Data Points** (20-30% produk)
   - Produk dengan < 3 bulan data di-filter di awal

3. **Lainnya** (3-10% produk)
   - Date parsing issues
   - Training errors
   - dll

**Solusi Prioritas**:
1. Turunkan `MIN_SEQUENCES_REQUIRED` dari 4 ke 2
2. Turunkan `MIN_DATA_POINTS_MONTHS` dari 3 ke 2
3. Tambahkan logging detail untuk tracking









