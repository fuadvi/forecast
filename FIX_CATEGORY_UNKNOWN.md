# 🔧 Fix: Category "Unknown" di SES

## 🔍 Masalah

Hasil forecast SES menampilkan semua produk dengan `category = "Unknown"` meskipun implementasi Opsi A sudah benar.

## 🎯 Penyebab

**SES tidak menemukan kolom kategori di Excel** karena:

1. ❌ Kolom kategori tidak ada di file Excel
2. ❌ Nama kolom kategori berbeda dari "Kategori Barang"
3. ❌ SES tidak punya fuzzy matching (sudah diperbaiki sekarang)

## ✅ Solusi yang Sudah Dilakukan

### 1. **Update Fuzzy Matching di SES**

File: `ses_monthly_product_forecast_24m.py`

**Sekarang SES bisa mendeteksi berbagai nama kolom kategori:**
- "Kategori Barang"
- "Product Category"
- "Category"
- "Kategori"
- "kategori_barang"
- "product_category"
- Dan nama lain yang mengandung kata "kategori" atau "category"

**Console output akan menampilkan:**
```
✅ Kolom kategori ditemukan: 'Kategori Barang' → digunakan sebagai 'category'
```

Atau jika tidak ada:
```
⚠️ Kolom kategori TIDAK ditemukan. Menggunakan default 'Unknown'
```

### 2. **Script untuk Check Excel**

File baru: `check_excel_columns.py`

Script ini akan:
- ✅ List semua kolom di Excel
- ✅ Detect kolom kategori otomatis
- ✅ Show sample data per kolom
- ✅ Rekomendasi action

---

## 🚀 Langkah-Langkah Fix

### Step 1: Check Kolom Excel Anda

```bash
python check_excel_columns.py
```

**Output akan menunjukkan:**
- Semua kolom yang ada di Excel
- Kolom mana yang terdeteksi sebagai kategori
- Sample data dari setiap kolom

### Step 2: Analisis Hasil

#### Skenario A: ✅ **Kolom Kategori Ditemukan**

```
✅ FOUND 1 potential category column(s):
   Column: 'Kategori Barang'
   Unique categories (5):
      - Meja Kursi (150 rows)
      - Alat Tulis (80 rows)
      ...
```

**Action**: Lanjut ke Step 3 (regenerate)

---

#### Skenario B: ❌ **Kolom Kategori TIDAK Ada**

```
❌ NO category column found!
```

**Action**: Pilih salah satu opsi:

**Opsi 1 - Tambahkan Kolom di Excel** (Recommended)

1. Buka file Excel: `Data_Penjualan_Dengan_ID_Pelanggan.xlsx`
2. Tambahkan kolom baru: **"Kategori Barang"**
3. Isi dengan kategori yang sesuai untuk setiap produk:
   ```
   Nama Produk              | Kategori Barang
   ------------------------|------------------
   kursi kerja dp 301 tb   | Meja Kursi
   kursi kerja br 205 ha   | Meja Kursi
   pulpen standard hitam   | Alat Tulis
   ```
4. Save Excel
5. Lanjut ke Step 3

**Opsi 2 - Gunakan Kolom Existing**

Jika ada kolom lain yang bisa dijadikan kategori (misalnya "Jenis Produk", "Tipe", dll):

1. Rename kolom tersebut ke "Kategori Barang" di Excel
2. Atau update `COLUMN_MAPPING` di `ses_monthly_product_forecast_24m.py`:
   ```python
   COLUMN_MAPPING = {
       ...
       "Jenis Produk": "category",  # ← Ganti dengan nama kolom Anda
       ...
   }
   ```
3. Lanjut ke Step 3

**Opsi 3 - Terima "Unknown"** (Not Recommended)

Jika tidak ada data kategori dan tidak bisa ditambahkan:
- SES akan tetap menggunakan "Unknown" untuk semua produk
- Aggregation Opsi A tetap jalan, tapi semua produk di grup "Unknown"
- Hasil akan sama dengan Opsi B (aggregasi tanpa detail kategori)

---

### Step 3: Regenerate Forecast SES

```bash
python ses_monthly_product_forecast_24m.py
```

**Console akan menampilkan:**
```
📋 Kolom di Excel: ['Tanggal Transaksi', 'Nama Produk', 'Kategori Barang', 'Jumlah']
✅ Kolom kategori ditemukan: 'Kategori Barang' → digunakan sebagai 'category'

=== Aggregating Monthly Data to Quarterly (SES) ===
Strategy: Group by Product + Category (Opsi A - Detail per category)
...
```

### Step 4: Verify Hasil

```bash
python compare_ses_lstm_results.py
```

**Expected output:**
```
📊 SES Top 5 - Q1 2026
--------------------------------------------------------------------------------
✅ Category column: FOUND (Opsi A implemented)
 rank                 product_name     category  quarterly_sum
    1       kursi kerja dp 301 tb   Meja Kursi         552.47
    2       kursi kerja br 205 ha   Meja Kursi         465.00
    3            kursi kerja dp 308   Meja Kursi         312.00
```

---

## 📊 Contoh Excel Structure

### ❌ SEBELUM (Tanpa Kategori)

| Tanggal Transaksi | Nama Produk | Jumlah |
|-------------------|-------------|--------|
| 2024-01-15 | kursi kerja dp 301 tb | 10 |
| 2024-01-16 | pulpen standard hitam | 50 |

**Result**: Semua produk → category = "Unknown"

---

### ✅ SESUDAH (Dengan Kategori)

| Tanggal Transaksi | Nama Produk | Kategori Barang | Jumlah |
|-------------------|-------------|-----------------|--------|
| 2024-01-15 | kursi kerja dp 301 tb | Meja Kursi | 10 |
| 2024-01-16 | pulpen standard hitam | Alat Tulis | 50 |

**Result**: Kategori terdeteksi dengan benar ✅

---

## 🎓 FAQ

### Q: Kenapa LSTM punya kategori tapi SES tidak?

**A**: Kemungkinan:
1. File Excel sudah diupdate **setelah** LSTM dijalankan
2. LSTM menggunakan fuzzy matching lebih dulu
3. SES baru saja diupdate dengan fuzzy matching (sekarang sama)

### Q: Apakah harus regenerate LSTM juga?

**A**: Tidak perlu jika LSTM sudah punya kategori yang benar. Cek:
```bash
head -2 forecast_per_product_24m.csv
```
Jika sudah ada kolom `category` dengan nilai selain "Unknown", LSTM sudah OK.

### Q: Apa impact jika semua kategori = "Unknown"?

**A**: 
- ✅ Forecast tetap jalan
- ✅ Opsi A tetap implemented (struktur output benar)
- ❌ Tapi tidak ada detail per kategori (sama seperti Opsi B)
- ❌ Aggregation jadi flat (semua produk dalam satu grup)

### Q: Bisa pakai kategori dari database lain?

**A**: Ya! Bisa:
1. **Opsi 1**: Join data kategori ke Excel sebelum forecast
2. **Opsi 2**: Update Excel dengan VLOOKUP dari database kategori
3. **Opsi 3**: Hardcode mapping kategori per produk di kode (not recommended)

---

## 📝 Checklist

Setelah fix, pastikan:

- [ ] `check_excel_columns.py` mendeteksi kolom kategori
- [ ] Console output SES menampilkan: "✅ Kolom kategori ditemukan"
- [ ] File `quarterly_top5_ses_2026.csv` memiliki kategori selain "Unknown"
- [ ] Comparison tool menampilkan kategori yang benar
- [ ] Visualizations menampilkan kategori di y-axis

---

## 🔄 Rollback (Jika Perlu)

Jika ingin kembali ke versi sebelum fix:

```bash
git checkout ses_monthly_product_forecast_24m.py
```

Atau manual: hapus fuzzy matching code, kembalikan COLUMN_MAPPING yang simple.

---

**Next Step**: Run `check_excel_columns.py` untuk diagnose! 🚀

---

**Created**: 2025-12-14  
**Status**: Fix Ready  
**Estimated Time**: 5-10 minutes (tergantung perlu update Excel atau tidak)

