# Dokumentasi Perbaikan Konsistensi Forecast SES vs LSTM

**Tanggal:** 14 Desember 2025  
**Tujuan:** Membuat hasil forecast SES dan LSTM lebih konsisten dan comparable

---

## 🎯 Masalah yang Teridentifikasi

### 1. **Skala Forecast Berbeda Drastis**
- LSTM menghasilkan nilai **5-6x lebih besar** dari SES
- Contoh: LSTM = 3013.47, SES = 552.47 untuk produk sejenis

### 2. **Perbedaan Preprocessing**
- **SES**: Menggunakan IQR-based outlier capping (Q1-1.5×IQR, Q3+1.5×IQR)
- **LSTM**: Menggunakan percentile-based clipping (99th percentile)

### 3. **Mode Experimental di LSTM**
- `DISABLE_STABILIZATION = True` → forecast tidak distabilkan
- `DISABLE_CLAMPING = True` → tidak ada batas atas/bawah
- `NOISE_INJECTION = 15%` → variasi terlalu besar

---

## ✅ Solusi yang Diimplementasikan

### **Perubahan 1: Aktifkan Stabilization & Clamping**

**File:** `forecast.py` (line 38-42)

**Sebelum:**
```python
DISABLE_STABILIZATION = True   # bypass stabilize_series
DISABLE_CLAMPING = True        # bypass clamp_with_historical_quantiles
NOISE_INJECTION = True         # add small noise
```

**Sesudah:**
```python
DISABLE_STABILIZATION = False  # ENABLED: apply stabilization for consistency with SES
DISABLE_CLAMPING = False       # ENABLED: apply clamping for consistency with SES
NOISE_INJECTION = True         # add small noise to residual forward loop
```

**Dampak:** Forecast LSTM sekarang akan melalui stabilization dan clamping seperti SES.

---

### **Perubahan 2: Kurangi Noise Injection**

**Direct Mode (line 401-404):**
```python
# Sebelum: noise_factor = rng.normal(0, 0.15)  # 15% noise
# Sesudah:
noise_factor = rng.normal(0, 0.05)  # 5% noise (reduced from 15%)
```

**Residual Mode (line 474-476):**
```python
# Sebelum: pred_resid = float(pred_resid + rng.normal(0.0, resid_std * 0.5))
# Sesudah:
pred_resid = float(pred_resid + rng.normal(0.0, resid_std * 0.15))
```

**Dampak:** Variasi forecast berkurang 3x lipat, lebih stabil dan comparable dengan SES.

---

### **Perubahan 3: Stabilization Function - Lebih Konservatif**

**File:** `forecast.py` (line 289-313)

**Parameter Lama:**
- `k = 5.0` → max monthly change 200%-500%
- `lower = prev * max(0.01, ...)` → bisa turun hingga 1% dari nilai sebelumnya

**Parameter Baru:**
- `k = 2.0` → max monthly change 50%-150% (lebih konservatif)
- `lower = prev * max(0.1, ...)` → minimal 10% dari nilai sebelumnya

**Dampak:** MoM change dibatasi lebih ketat, mengurangi spike ekstrem.

---

### **Perubahan 4: Clamping Function - Gunakan IQR Method**

**File:** `forecast.py` (line 316-333)

**Metode Lama:**
```python
lo = np.quantile(hist_values, 0.001)
hi = np.quantile(hist_values, 0.999)
lower_bound = max(0.0, lo * 0.01, mean_hist * 0.01)
upper_bound = max(hi * 100.0, mean_hist * 100.0)  # 100x historical!
```

**Metode Baru (IQR-based, sama dengan SES):**
```python
q1 = np.quantile(hist_values, 0.25)
q3 = np.quantile(hist_values, 0.75)
iqr = q3 - q1
lower_bound = max(0.0, q1 - 1.5 * iqr)
upper_bound = max(q3 + 3.0 * iqr, mean_hist * 3.0)  # Max 3x historical
```

**Dampak:** Forecast dibatasi dalam range yang realistis, konsisten dengan SES.

---

### **Perubahan 5: Outlier Treatment - Sinkronisasi dengan SES**

**File:** `forecast.py` (line 139-151)

**Metode Lama (percentile):**
```python
upper = g["sales"].quantile(0.99)
g["sales"] = np.minimum(g["sales"], upper)
```

**Metode Baru (IQR, sama dengan SES):**
```python
q1 = g["sales"].quantile(0.25)
q3 = g["sales"].quantile(0.75)
iqr = q3 - q1
if iqr > 0:
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    g["sales"] = g["sales"].clip(lower=lower, upper=upper)
```

**Dampak:** Preprocessing data historis sekarang identik antara SES dan LSTM.

---

### **Perubahan 6: Fallback Forecast - Lebih Konservatif**

**File:** `forecast.py` (line 552-570)

**Parameter yang Dikurangi:**
- Trend: 2% → 1% per bulan
- Random variation: 10% → 5%
- Month variation: 5% → 3%

**Dampak:** Fallback forecast (untuk produk tanpa model) lebih stabil.

---

### **Perubahan 7: Baseline Forecast - Lebih Moderat**

**File:** `forecast.py` (line 238-254)

**Parameter yang Dikurangi:**
- Trend multiplier: 0.5 → 0.3
- Random component: 15% → 5%
- Month variation: 10% → 5%
- Walk component: 5% → 2%

**Dampak:** Baseline forecast lebih smooth dan predictable.

---

## 📊 Perbandingan Parameter

| Aspek | SES | LSTM (Sebelum) | LSTM (Sesudah) |
|-------|-----|----------------|----------------|
| **Stabilization** | ✅ Aktif | ❌ Nonaktif | ✅ Aktif |
| **Clamping** | ✅ Aktif | ❌ Nonaktif | ✅ Aktif |
| **Noise Injection** | - | 15% | 5% |
| **Outlier Method** | IQR | Percentile | IQR |
| **Max Monthly Change** | - | 200-500% | 50-150% |
| **Upper Bound** | Q3+1.5×IQR | 100x mean | 3x mean |
| **Preprocessing** | IQR clipping | 99th percentile | IQR clipping |

---

## 🔍 Validasi yang Perlu Dilakukan

1. ✅ **Konsistensi Preprocessing**
   - Kedua metode sekarang menggunakan IQR-based outlier treatment

2. ✅ **Stabilization Aktif**
   - LSTM sekarang menerapkan stabilization dan clamping

3. ✅ **Noise Reduction**
   - Variasi random dikurangi dari 15% ke 5%

4. 🔄 **Test Forecast** (Pending)
   - Jalankan `python forecast.py` dan `python ses_monthly_product_forecast_24m.py`
   - Bandingkan hasil quarterly top 5 dari kedua metode
   - Verifikasi bahwa skala forecast lebih comparable

---

## 🎯 Expected Results

Setelah perbaikan ini:

1. **Skala forecast LSTM akan lebih mendekati SES** (tidak lagi 5-6x lebih besar)
2. **Ranking produk akan lebih konsisten** antar metode
3. **Variasi forecast lebih stabil** (tidak ada spike ekstrem)
4. **Preprocessing identik** untuk apple-to-apple comparison

---

## 📝 Catatan Penting

### Trade-offs yang Dipertimbangkan:

1. **SES Characteristics (Dipertahankan):**
   - ✅ Konservatif dan stabil
   - ✅ Cocok untuk perencanaan jangka pendek
   - ⚠️ Kurang responsif terhadap trend baru

2. **LSTM Characteristics (Disesuaikan):**
   - ✅ Masih bisa menangkap complex patterns
   - ✅ Lebih responsif terhadap trend (setelah moderation)
   - ✅ Sekarang lebih stabil dan comparable dengan SES

### Rekomendasi Penggunaan:

- **SES:** Gunakan untuk produk dengan demand stabil dan predictable
- **LSTM (Setelah perbaikan):** Gunakan untuk produk dengan trend growth yang terlihat
- **Ensemble (Future improvement):** Kombinasi 40% SES + 60% LSTM untuk hasil optimal

---

## 🚀 Langkah Selanjutnya

1. ✅ Implementasi perbaikan di `forecast.py` - **SELESAI**
2. 🔄 Jalankan forecast untuk validasi
3. 📊 Bandingkan hasil quarterly rankings
4. 📈 Monitor performance pada data baru
5. 💡 Pertimbangkan ensemble approach jika masih ada gap

---

**Status:** ✅ Implementasi selesai, ready for testing

