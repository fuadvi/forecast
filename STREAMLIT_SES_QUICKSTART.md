# 🚀 Quick Start: SES Forecast di Streamlit (Dengan Logging)

## ⚡ TL;DR

**Problem**: SES tidak bisa run dari Streamlit  
**Solution**: Logging system added untuk capture error  
**Log File**: `ses_forecast_run.log` (auto-generated)

---

## 📝 Step-by-Step

### 1. Start Streamlit

```bash
streamlit run app.py
```

### 2. Navigate ke "SES Forecast" Page

Klik di sidebar: **📈 SES Forecast - 24 Bulan**

### 3. Configure (Optional)

- Top-K Produk: 5 (default)
- Horizon (bulan): 24 (default)  
- Minimal data points: 6 (default)
- Aktifkan outlier capping: ✅ (default)

### 4. Click "Generate SES Forecast"

Tunggu proses selesai...

---

## ✅ Jika Berhasil

**Tampilan Streamlit**:
```
✅ Proses selesai
[wrapper] Process exited with code 0
[wrapper] All output files generated successfully.
```

**Files Generated**:
- ✅ `forecast_per_product_ses_24m.csv`
- ✅ `forecast_total_ses_24m.csv`
- ✅ `topN_per_month_ses_24m.csv`
- ✅ `quarterly_top5_ses_2026.csv`
- ✅ `yearly_top5_borda_ses_2026.csv`
- ✅ `forecast_plots/top5_yearly_ses.png`

**Log File**: `ses_forecast_run.log` tersimpan untuk reference

---

## ❌ Jika Error

**Tampilan Streamlit**:
```
❌ Proses SES gagal!
Check log file: `ses_forecast_run.log` untuk detail
```

**Action**:

### Step 1: Open Log Expander

Klik **"📋 Full Process Log"** di Streamlit UI

### Step 2: Identify Error

Look for keywords:
- `ERROR:`
- `Traceback`
- `Exception`

### Step 3: Common Fixes

**Error: "ModuleNotFoundError"**
```bash
pip install statsmodels pandas numpy matplotlib
```

**Error: "File tidak ditemukan"**
- Check Excel file ada di: `Data_Penjualan_Dengan_ID_Pelanggan.xlsx`
- Check spelling nama file

**Error: "Permission denied"**
- Close Excel jika file output sedang dibuka
- Restart Streamlit

**Error: Category "Unknown"**
```bash
# Check Excel columns
python check_excel_columns.py

# Add "Kategori Barang" column jika tidak ada
```

### Step 4: Download Log

Klik **"Download Log"** button untuk save log file.

### Step 5: Check Full Troubleshooting Doc

```bash
# Read full guide
cat STREAMLIT_SES_TROUBLESHOOTING.md
```

---

## 🔍 Quick Diagnosis

### Test 1: Run Direct (Bypass Streamlit)

```bash
python ses_monthly_product_forecast_24m.py
```

**Berhasil?**
- ✅ YES → Masalah di Streamlit wrapper (rare)
- ❌ NO → Masalah di script/data (common)

### Test 2: Check Log

```bash
type ses_forecast_run.log  # Windows
cat ses_forecast_run.log   # Linux/Mac
```

**Last line should be**:
```
Process exited with code 0
```

**If NOT 0**: Ada error, cari di log

---

## 📊 Expected Timeline

**Small Dataset** (< 50 products):
- ⏱️ 1-3 minutes

**Medium Dataset** (50-200 products):
- ⏱️ 3-10 minutes

**Large Dataset** (> 200 products):
- ⏱️ 10-30 minutes

**If > 30 minutes**: Probably hung, check log

---

## 🎯 Success Checklist

After successful run:

- [ ] ✅ No error message in Streamlit
- [ ] ✅ "Proses selesai" toast notification
- [ ] ✅ Exit code = 0 in log
- [ ] ✅ All CSV files generated
- [ ] ✅ Plot files generated
- [ ] ✅ Can see results in "📦 Output Files" section
- [ ] ✅ Can see preview in tabs
- [ ] ✅ Category BUKAN "Unknown" (jika Excel punya kategori)

---

## 🆘 Emergency Fixes

### Fix 1: Restart Everything

```bash
# Stop Streamlit
Ctrl+C

# Clear process
# (Streamlit auto-cleanup)

# Start again
streamlit run app.py
```

### Fix 2: Clean Output Files

```bash
# Delete old output files
del *_ses_*.csv
del forecast_plots\*_ses.png

# Try generate again
```

### Fix 3: Test with Minimal Config

In Streamlit UI:
- Top-K: **3**
- Horizon: **6** months
- Min points: **3**

If this works → Dataset too heavy, need optimization

### Fix 4: Check Python Environment

```bash
python --version  # Should be 3.8+
pip list | grep statsmodels  # Should exist
pip install --upgrade statsmodels
```

---

## 💡 Pro Tips

### Tip 1: Always Check Log First

Before asking for help, **ALWAYS** check `ses_forecast_run.log`

### Tip 2: Keep Log Backups

```bash
# Before re-run, backup old log
copy ses_forecast_run.log logs\ses_log_backup_20251214.log
```

### Tip 3: Test Direct First

Before using Streamlit, always test:
```bash
python ses_monthly_product_forecast_24m.py
```

If this fails, fix script first before using Streamlit.

### Tip 4: Monitor Resources

Open Task Manager / Activity Monitor:
- Watch Memory usage
- Watch CPU usage
- If stuck → probably need to restart

### Tip 5: Download Log for Analysis

Use "Download Log" button untuk:
- Share dengan team
- Archive untuk reference
- Attach ke issue reports

---

## 📞 Need Help?

### Information to Provide

When asking for help, include:

1. **Log file content** (full `ses_forecast_run.log`)
2. **Dataset info**:
   - How many rows?
   - How many products?
   - Does it have "Kategori Barang" column?
3. **Python version**: `python --version`
4. **Installed packages**: `pip list`
5. **Error screenshot** from Streamlit

### Where to Get Help

- 📖 `STREAMLIT_SES_TROUBLESHOOTING.md` - Full troubleshooting guide
- 📋 `ses_forecast_run.log` - Auto-generated log file
- 🔧 `FIX_CATEGORY_UNKNOWN.md` - Category issues
- ⚠️ `FIX_STATSMODELS_WARNING.md` - Warning issues

---

## 📚 Related Files

| File | Purpose |
|------|---------|
| `ses_forecast_run.log` | **Auto-generated log** (CHECK THIS FIRST!) |
| `STREAMLIT_SES_TROUBLESHOOTING.md` | Full troubleshooting guide |
| `FIX_CATEGORY_UNKNOWN.md` | Fix category detection |
| `FIX_STATSMODELS_WARNING.md` | Fix warning spam |
| `OPSI_A_IMPLEMENTED.md` | Opsi A documentation |
| `check_excel_columns.py` | Check Excel structure |
| `compare_ses_lstm_results.py` | Compare SES vs LSTM |

---

## ✨ What's New

**Logging System** (2025-12-14):
- ✅ Auto-save all process output to `ses_forecast_run.log`
- ✅ Error detection in Streamlit UI
- ✅ Log viewer in Streamlit (expandable)
- ✅ Download log button
- ✅ File check status in log
- ✅ Timestamp untuk debugging

---

**Happy Forecasting! 🎉**

**Remember**: 
1. Check log file first
2. Test direct Python run
3. Read troubleshooting guide if stuck

---

**Created**: 2025-12-14  
**Status**: Active  
**Log File**: `ses_forecast_run.log` (auto-generated on each run)

