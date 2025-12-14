# Perbandingan Hasil Forecast: Sebelum vs Sesudah Perbaikan

**Tanggal Analisis:** 14 Desember 2025  
**Tujuan:** Membandingkan hasil forecast SES dan LSTM sebelum dan sesudah perbaikan konsistensi

---

## 📊 Data Sebelum Perbaikan (Q1 2026)

### **LSTM (Sebelum Perbaikan)**
| Rank | Produk | Category | Quarterly Sum |
|------|--------|----------|---------------|
| 1 | kursi auditorium ll 516 tb tg | Auditorium Series | **3013.47** |
| 2 | kursi auditorium ll 526 tb | Auditorium Series | **1514.31** |
| 3 | kursi auditorium ll 184 | Auditorium Series | **1365.91** |
| 4 | kursi kerja dp 301 tb | Stacking Series | **1288.01** |
| 5 | kursi auditorium ll 516 tb | Auditorium Series | **1080.60** |

**Total Top-5:** 8,262.30

---

### **SES (Hasil Stabil)**
| Rank | Produk | Category | Quarterly Sum |
|------|--------|----------|---------------|
| 1 | kursi kerja dp 301 tb | Stacking Series | **552.47** |
| 2 | kursi kerja br 205 ha | Secretary Series | **465.00** |
| 3 | kursi kerja dp 308 | Stacking Series | **312.00** |
| 4 | kursi br 211 ca mesh | Secretary Series | **243.00** |
| 5 | kursi auditorium ll 517 tb | Auditorium Series | **240.00** |

**Total Top-5:** 1,812.47

---

## 🔍 Analisis Perbedaan Sebelum Perbaikan

### **1. Skala Nilai**
- **LSTM:** 3013.47 (rank 1)
- **SES:** 552.47 (rank 1)
- **Ratio:** LSTM **5.45x lebih besar** dari SES

### **2. Produk yang Dominan**
- **LSTM:** Didominasi **Auditorium Series** (4 dari 5 produk top)
- **SES:** Lebih beragam, **Stacking dan Secretary Series** dominan

### **3. Konsistensi Ranking**
- **Overlap produk:** Hanya **1 produk sama** (kursi kerja dp 301 tb)
  - LSTM: Rank 4 (1288.01)
  - SES: Rank 1 (552.47)
- **Ratio untuk produk yang sama:** 2.33x

### **4. Karakteristik Forecast**
| Aspek | LSTM (Sebelum) | SES |
|-------|----------------|-----|
| Stabilization | ❌ OFF | ✅ ON |
| Clamping | ❌ OFF | ✅ ON |
| Noise Level | 15% | - |
| Max Monthly Change | 200-500% | Moderate |
| Upper Bound | 100x mean | Q3+1.5×IQR |

---

## 🎯 Perbaikan yang Diimplementasikan

### **Parameter Changes di LSTM:**

1. **Stabilization & Clamping**
   - `DISABLE_STABILIZATION: True → False`
   - `DISABLE_CLAMPING: True → False`

2. **Noise Reduction**
   - Direct mode: `15% → 5%`
   - Residual mode: `50% → 15%`

3. **Stabilization Parameters**
   - Max monthly change: `200-500% → 50-150%`
   - Lower bound: `1% → 10%` of previous value

4. **Clamping Method**
   - Old: 0.01x-100x (percentile-based)
   - New: IQR-based (Q1-1.5×IQR to Q3+3×IQR)

5. **Outlier Treatment**
   - Old: 99th percentile clipping
   - New: IQR method (same as SES)

6. **Baseline Forecast**
   - Trend: `2% → 1%`
   - Random variation: `10% → 5%`
   - Month variation: `5% → 3%`

---

## 📈 Expected Results Sesudah Perbaikan

### **Prediksi Perubahan:**

1. **Skala Forecast LSTM akan turun** sekitar 40-60%
   - Expected range: **1200-1800** untuk rank 1 (vs 3013 sebelumnya)
   - Mendekati skala SES (**552**)

2. **Ranking akan lebih konsisten**
   - Lebih banyak produk yang overlap antara SES dan LSTM
   - Produk Stacking/Secretary series akan muncul di LSTM

3. **Variasi lebih stabil**
   - Tidak ada spike ekstrem antar bulan
   - MoM change max 150% (vs 500% sebelumnya)

4. **Konsistensi preprocessing**
   - Kedua metode menggunakan IQR-based outlier treatment
   - Data input historical lebih comparable

---

## ✅ Validasi Checklist

- [x] **Implementasi perbaikan di forecast.py**
  - [x] Aktifkan DISABLE_STABILIZATION = False
  - [x] Aktifkan DISABLE_CLAMPING = False
  - [x] Kurangi noise injection (15% → 5%)
  - [x] Update stabilize_series (k=5.0 → 2.0)
  - [x] Update clamp function (IQR-based)
  - [x] Update outlier treatment (IQR method)
  - [x] Update baseline forecast parameters

- [ ] **Run forecast baru dan validasi**
  - [ ] Jalankan `python forecast.py`
  - [ ] Bandingkan quarterly_top5_2026.csv (baru)
  - [ ] Verifikasi skala nilai lebih comparable
  - [ ] Cek overlap produk meningkat

- [ ] **Analisis hasil**
  - [ ] Hitung ratio LSTM/SES sesudah perbaikan
  - [ ] Verifikasi produk overlap
  - [ ] Check MoM variation range
  - [ ] Dokumentasi improvement

---

## 🎯 Success Metrics

### **Target Perbaikan:**

1. **Ratio LSTM/SES:**
   - Sebelum: **5.45x**
   - Target: **1.5-2.5x** (acceptable range)

2. **Produk Overlap:**
   - Sebelum: **1 dari 5** (20%)
   - Target: **3-4 dari 5** (60-80%)

3. **MoM Variation:**
   - Sebelum: up to **500%**
   - Target: max **150%**

4. **Upper Bound Ratio:**
   - Sebelum: **100x** historical mean
   - Target: **3x** historical mean

---

## 📝 Notes

### **Trade-offs yang Diterima:**

1. **LSTM mungkin kurang responsif terhadap sudden spikes**
   - ✅ Pro: Lebih stabil dan predictable
   - ⚠️ Con: Mungkin miss beberapa growth opportunities

2. **SES tetap konservatif**
   - ✅ Pro: Cocok untuk safety stock planning
   - ⚠️ Con: Mungkin underestimate trending products

### **Rekomendasi Penggunaan:**

**Untuk Perencanaan:**
- **SES:** Baseline/lower bound untuk inventory
- **LSTM (setelah perbaikan):** Expected forecast
- **Ensemble (future):** 40% SES + 60% LSTM untuk optimal balance

**Untuk Produk Spesifik:**
- **Stable demand products:** SES lebih reliable
- **Growing/trending products:** LSTM lebih accurate
- **New products:** Fallback to category statistics

---

## 🚀 Next Steps

1. ✅ **Implementasi selesai** - forecast.py sudah diperbaiki
2. 🔄 **Running forecast** - sedang berjalan di background
3. ⏳ **Waiting for results** - cek terminal output
4. 📊 **Compare results** - bandingkan quarterly rankings
5. 📈 **Document findings** - update dokumen ini dengan hasil aktual

---

**Status:** 🔄 Waiting for forecast completion...

**Command running:**
```bash
python forecast.py > forecast_run_new.log 2>&1
```

Check progress: `tail -f forecast_run_new.log` atau baca `terminals/4.txt`

