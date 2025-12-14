# 🔧 Troubleshooting: SES Forecast di Streamlit

## 🔍 Masalah yang Dilaporkan

**SES Forecast tidak bisa di-run dari Streamlit**, tapi berhasil kalau run langsung dari file Python.

---

## ✅ Solusi yang Sudah Diimplementasi

### 1. **Logging System** 📝

Sekarang semua proses SES **otomatis disave ke log file**.

**Log File Location**: `ses_forecast_run.log` (di root project)

**Isi Log**:
- ✅ Command yang dijalankan
- ✅ Working directory
- ✅ Python executable path
- ✅ Script path
- ✅ Data file path
- ✅ **Semua output dari proses** (termasuk error)
- ✅ Exit code
- ✅ File check status

### 2. **Error Detection & Display**

Streamlit sekarang akan:
- ✅ Detect error keywords (`ERROR`, `TRACEBACK`, `EXCEPTION`)
- ✅ Show error message jika proses gagal
- ✅ Display log file content di Streamlit
- ✅ Provide download button untuk log

---

## 🚀 Cara Menggunakan

### Step 1: Run SES dari Streamlit

```bash
streamlit run app.py
```

Navigate ke: **"SES Forecast"** page

### Step 2: Klik "Generate SES Forecast"

Streamlit akan:
1. Create log file: `ses_forecast_run.log`
2. Run subprocess dengan logging enabled
3. Stream output ke UI
4. Save semua output ke log file

### Step 3: Jika Ada Error

Streamlit akan menampilkan:
```
❌ Proses SES gagal!
Check log file: `ses_forecast_run.log` untuk detail
```

**Klik expander "📋 Full Process Log"** untuk lihat detail error.

### Step 4: Download Log (Optional)

Di dalam expander, ada button **"Download Log"** untuk save log ke file lokal.

---

## 📋 Analyzing Error Log

### Contoh 1: Module Not Found

```
ERROR: ModuleNotFoundError: No module named 'statsmodels'
```

**Solusi**:
```bash
pip install statsmodels
```

### Contoh 2: File Not Found

```
ERROR: File data Excel tidak ditemukan: D:\tesis\Data_Penjualan.xlsx
```

**Solusi**:
- Check file Excel ada di lokasi yang benar
- Check nama file spelling
- Upload ulang file jika hilang

### Contoh 3: Permission Error

```
ERROR: PermissionError: [Errno 13] Permission denied: 'forecast_per_product_ses_24m.csv'
```

**Solusi**:
- Close Excel jika file output sedang dibuka
- Check folder permissions
- Run Streamlit as administrator (Windows)

### Contoh 4: Pandas/Numpy Version Conflict

```
ERROR: ValueError: numpy.dtype size changed
```

**Solusi**:
```bash
pip install --upgrade numpy pandas statsmodels
```

### Contoh 5: Memory Error

```
ERROR: MemoryError: Unable to allocate array
```

**Solusi**:
- Reduce `forecast_months` (dari 24 ke 12)
- Reduce `top_k` (dari 10 ke 5)
- Filter dataset (hapus produk dengan data sedikit)

---

## 🔍 Common Issues & Solutions

### Issue 1: Streamlit Hang/Freeze

**Symptoms**:
- Progress bar stuck
- UI tidak responsive
- No output di log area

**Diagnosis**:
```bash
# Check log file manually
type ses_forecast_run.log  # Windows
cat ses_forecast_run.log   # Linux/Mac
```

**Possible Causes**:
1. **Dataset terlalu besar**
   - Check: Berapa banyak produk?
   - Fix: Filter Excel ke produk tertentu dulu

2. **Holt-Winters optimization stuck**
   - Check log: Apakah stuck di produk tertentu?
   - Fix: Skip produk tersebut atau gunakan SES saja

3. **Memory exhausted**
   - Check: Task Manager / Activity Monitor
   - Fix: Close aplikasi lain, restart Streamlit

**Emergency Fix**:
```python
# Edit ses_monthly_product_forecast_24m.py
# Line ~360-390: Force disable HW, use SES only
if n >= 24 and ExponentialSmoothing is not None:
    # TEMPORARILY DISABLE HW
    pass  # Skip HW, go straight to SES
```

### Issue 2: "Proses SES sudah berjalan"

**Symptoms**:
```
❌ Proses SES sudah berjalan
```

**Cause**: Previous process belum selesai

**Solution**:
1. Wait untuk proses sebelumnya selesai
2. Atau restart Streamlit:
   ```bash
   Ctrl+C  # Stop Streamlit
   streamlit run app.py  # Start again
   ```

### Issue 3: Output Files Not Generated

**Symptoms**:
```
[wrapper] Warning: Not all expected output files were found after completion.
```

**Diagnosis**:
Check log untuk:
- `File check: forecast_per_product_ses_24m.csv - MISSING`
- Apakah ada error sebelum file generation?

**Common Causes**:
1. **Script error sebelum save**
   - Check traceback di log
   - Fix error di script

2. **Permission denied**
   - Check folder write permissions
   - Run as administrator

3. **Disk full**
   - Check disk space
   - Clean temporary files

### Issue 4: Wrong Results / Category "Unknown"

**Check**:
```bash
# Check log untuk category detection
grep "kategori" ses_forecast_run.log
```

**Expected**:
```
✅ Kolom kategori ditemukan: 'Kategori Barang' → digunakan sebagai 'category'
```

**If NOT found**:
```
⚠️ Kolom kategori TIDAK ditemukan. Menggunakan default 'Unknown'
```

**Solution**: 
- Lihat dokumentasi: `FIX_CATEGORY_UNKNOWN.md`
- Run: `python check_excel_columns.py`

---

## 🧪 Testing Checklist

Sebelum report issue, test ini dulu:

### Test 1: Direct Python Run

```bash
python ses_monthly_product_forecast_24m.py
```

**Expected**: Berhasil tanpa error

**If FAILS**: 
- ❌ Masalah di script/data, bukan Streamlit
- Fix script dulu sebelum test Streamlit

**If SUCCESS**: 
- ✅ Script OK, masalah mungkin di Streamlit wrapper
- Lanjut ke Test 2

### Test 2: Check Log File

```bash
type ses_forecast_run.log
```

**Look for**:
- Exit code: `Process exited with code 0` ✅ (good) or non-zero ❌ (error)
- File checks: All `EXISTS` ✅ or some `MISSING` ❌
- Error keywords: `ERROR`, `TRACEBACK`, `EXCEPTION`

### Test 3: Manual Command from Log

Copy command dari log:
```
Command: C:\Python\python.exe -u ses_monthly_product_forecast_24m.py --file ...
```

Run manually di terminal:
```bash
# Copy-paste command dari log
```

**If SUCCESS**: Wrapper OK, masalah di Streamlit UI
**If FAILS**: Masalah di command/arguments

### Test 4: Minimal Config

Test dengan config minimal:
- `top_k = 3`
- `forecast_months = 6`
- `min_points = 4`

**If SUCCESS**: Dataset/config terlalu berat
**If FAILS**: Masalah fundamental di script

---

## 📞 Reporting Issues

Jika masih error setelah troubleshooting, siapkan info berikut:

### Required Information

1. **Log File Content**
   ```bash
   # Full log
   type ses_forecast_run.log
   ```

2. **Python Environment**
   ```bash
   python --version
   pip list | grep -E "(pandas|numpy|statsmodels|streamlit)"
   ```

3. **Dataset Info**
   ```python
   import pandas as pd
   df = pd.read_excel('Data_Penjualan_Dengan_ID_Pelanggan.xlsx')
   print(f"Rows: {len(df)}")
   print(f"Columns: {df.columns.tolist()}")
   print(f"Products: {df['Nama Produk'].nunique()}")
   ```

4. **System Info**
   - OS: Windows/Linux/Mac
   - RAM: How much?
   - Disk space: Free space available?

5. **Screenshots**
   - Streamlit UI saat error
   - Log expander content

### Where to Report

Include informasi di atas di:
- Issue tracker
- Support channel
- Documentation update request

---

## 🎓 Tips & Best Practices

### 1. Test Incrementally

```bash
# Step 1: Test script directly
python ses_monthly_product_forecast_24m.py

# Step 2: Test with small dataset
# (Filter Excel to 10 products)

# Step 3: Test in Streamlit with small dataset

# Step 4: Full dataset in Streamlit
```

### 2. Monitor Resources

- Open Task Manager / Activity Monitor
- Watch Memory usage
- Watch CPU usage
- If stuck >5 minutes → probably hung

### 3. Keep Logs

```bash
# Rename log before next run
copy ses_forecast_run.log ses_forecast_run_backup.log
```

### 4. Update Dependencies

```bash
# Once a month
pip install --upgrade pandas numpy statsmodels streamlit
```

---

## 📚 Related Documentation

- `FIX_STATSMODELS_WARNING.md` - Fix warning spam
- `FIX_CATEGORY_UNKNOWN.md` - Fix category detection
- `OPSI_A_IMPLEMENTED.md` - Opsi A documentation
- `NEXT_STEPS.md` - General next steps

---

## ✅ Success Indicators

Forecast berhasil jika:

- [ ] No errors in log file
- [ ] Exit code = 0
- [ ] All output files EXISTS
- [ ] Quarterly CSVs generated
- [ ] Plots generated (PNG files)
- [ ] Streamlit shows success message
- [ ] Can view results in UI

---

**Created**: 2025-12-14  
**Last Updated**: 2025-12-14  
**Status**: Logging System Active  
**Log Location**: `ses_forecast_run.log`

