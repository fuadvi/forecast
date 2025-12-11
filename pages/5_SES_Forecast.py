from __future__ import annotations
from pathlib import Path
import io
import time
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st

from config.settings import (
    DEFAULT_EXCEL,
    SES_DEFAULTS,
    SES_FORECAST_PER_PRODUCT,
    SES_FORECAST_TOTAL,
    SES_TOPN_PER_MONTH,
    SES_SKIPPED_PRODUCTS,
    SES_MODEL_PARAMS,
    SES_GROUPED_TOP5_PNG,
)
from utils.ses_forecast_wrapper import (
    run_ses_forecast,
    check_ses_status,
    stream_ses_logs,
    load_ses_results,
    get_ses_summary,
)

st.set_page_config(page_title="SES Forecast - 24 Bulan", page_icon="📈", layout="wide")
st.title("📈 SES Forecast - 24 Bulan")
st.caption("Simple Exponential Smoothing (SES) dan fallback ke Holt-Winters untuk data dengan pola tren/musiman.")

# Info data/source
st.info(
    "Metode SES menggunakan file data yang sama dengan LSTM. Jika data cukup dan pola sesuai, skrip akan otomatis menggunakan Holt-Winters."
)

# Validations
script_path = Path("ses_monthly_product_forecast_24m.py")
data_path = Path(DEFAULT_EXCEL)
cols = st.columns(3)
with cols[0]:
    st.metric("Script SES", "Tersedia" if script_path.exists() else "Tidak Ada")
with cols[1]:
    st.metric("File Data Excel", data_path.name if data_path.exists() else "Tidak ditemukan")
with cols[2]:
    status = check_ses_status()
    st.metric("Status Proses", status.capitalize())

if not script_path.exists():
    st.error("❌ Script SES tidak ditemukan di root project: ses_monthly_product_forecast_24m.py")
if not data_path.exists():
    st.warning("⚠️ File data Excel default tidak ditemukan. Pastikan upload/sediakan data di root.")

# Configuration panel
with st.expander("Pengaturan (Opsional)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        top_k = st.slider("Top-K Produk", min_value=3, max_value=10, value=int(SES_DEFAULTS.get("top_k", 5)))
    with c2:
        horizon = st.number_input("Horizon (bulan)", min_value=12, max_value=36, value=int(SES_DEFAULTS.get("forecast_months", 24)))
    with c3:
        min_points = st.number_input("Minimal data points per produk", min_value=4, max_value=12, value=int(SES_DEFAULTS.get("min_points", 6)))
    with c4:
        cap_out = st.checkbox("Aktifkan outlier capping", value=bool(SES_DEFAULTS.get("outlier_capping", True)))
    st.caption("Tip: SES akan otomatis fallback ke Holt-Winters bila pola data mendukung.")

# Execution controls
run_col, stop_col, nav_col = st.columns([2,1,2])
with run_col:
    start_btn = st.button("Generate SES Forecast", type="primary")
with stop_col:
    stop_clicked = st.button("Stop Proses")
with nav_col:
    if st.button("➡️ Ke Results Analytics"):
        st.switch_page("pages/4_Results_Analytics.py")

log_area = st.empty()
progress = st.progress(0, text="Idle")
status_area = st.status(label="Status", state="complete")

if stop_clicked:
    st.warning("Fitur stop proses belum diimplementasikan penuh (subprocess terminate tidak diekspos). Tutup tab untuk menghentikan.")

if start_btn:
    ok, msg = run_ses_forecast(
        file_path=str(data_path) if data_path.exists() else None,
        top_k=top_k,
        forecast_months=horizon,
        min_points=min_points,
        outlier_capping=cap_out,
    )
    if not ok:
        st.error(msg)
    else:
        status_area.update(label="Sedang berjalan...", state="running")
        # naive estimate: 0.2s per product unknown here; we'll show indeterminate style
        progress.progress(5, text="Memulai...")
        logs = []
        for i, line in enumerate(stream_ses_logs() or []):
            logs.append(line)
            # Try parse product progress from logs if pattern present
            if "Processing" in line or "Memproses" in line:
                progress.progress(min(95, (i % 90) + 5), text=line[:100])
            if i % 5 == 0:
                log_area.code("\n".join(logs[-200:]))
        # finalize
        st.toast("Proses selesai", icon="✅")
        st.rerun()

st.subheader("📦 Output Files")
files = [
    ("Per-Product", Path(SES_FORECAST_PER_PRODUCT)),
    ("Total", Path(SES_FORECAST_TOTAL)),
    ("Top-N per Month", Path(SES_TOPN_PER_MONTH)),
    ("Skipped Products", Path(SES_SKIPPED_PRODUCTS)),
    ("Model Params", Path(SES_MODEL_PARAMS)),
]
fc1, fc2, fc3, fc4, fc5 = st.columns(5)
for (label, path), col in zip(files, [fc1, fc2, fc3, fc4, fc5]):
    with col:
        ok = path.exists()
        st.metric(label, "Tersedia" if ok else "-", help=str(path.name))

# Summary metrics
res = load_ses_results()
sumry = get_ses_summary()
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Produk berhasil diforecast", f"{sumry.get('n_products') or 0}")
with col2:
    st.metric("Produk diskip", f"{sumry.get('n_skipped') or 0}")
with col3:
    st.metric("Menggunakan Holt-Winters", f"{sumry.get('n_hw') or 0}")
with col4:
    st.metric("Menggunakan SES", f"{sumry.get('n_ses') or 0}")
with col5:
    per = sumry.get("period")
    st.metric("Periode Forecast", f"{per[0]} ➜ {per[1]}" if per else "-")

# Visualization
if Path(SES_GROUPED_TOP5_PNG).exists():
    st.subheader("📊 Top-5 Produk per Bulan (Grouped)")
    st.image(str(SES_GROUPED_TOP5_PNG), width="stretch", caption="Grouped bar chart Top-5 per bulan (SES)")

# Preview Tabs
T1, T2, T3, T4 = st.tabs([
    "Per-Product Forecast",
    "Total Forecast",
    "Top-N per Month",
    "Skipped Products",
])

with T1:
    df = res.get("per_product")
    if df is None or df.empty:
        st.info("Belum ada hasil per-product.")
    else:
        prod_col = next((c for c in df.columns if "product" in c.lower()), df.columns[0])
        products = ["All"] + sorted(df[prod_col].unique().tolist())
        sel = st.selectbox("Pilih Produk", options=products)
        view = df if sel == "All" else df[df[prod_col] == sel]
        st.dataframe(view.head(10), use_container_width=True)

with T2:
    df = res.get("total")
    if df is None or df.empty:
        st.info("Belum ada hasil total.")
    else:
        st.dataframe(df, use_container_width=True)
        # simple line chart
        dcol = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        ycol = next((c for c in df.columns if c not in (dcol,)), df.columns[-1])
        chart_df = df[[dcol, ycol]].copy()
        chart_df[dcol] = pd.to_datetime(chart_df[dcol])
        chart_df = chart_df.set_index(dcol)
        st.line_chart(chart_df)

with T3:
    df = res.get("topn")
    if df is None or df.empty:
        st.info("Belum ada hasil Top-N.")
    else:
        mcol = df.columns[0]
        months = ["All"] + sorted(df[mcol].astype(str).unique().tolist())
        selm = st.selectbox("Pilih Bulan", options=months)
        view = df if selm == "All" else df[df[mcol].astype(str) == selm]
        st.dataframe(view, use_container_width=True)

with T4:
    df = res.get("skipped")
    if df is None or df.empty:
        st.info("Tidak ada produk yang diskip.")
    else:
        st.dataframe(df, use_container_width=True)

# Download section
st.subheader("⬇️ Download")
colA, colB, colC, colD, colE = st.columns(5)
for (label, path), col in zip(files, [colA, colB, colC, colD, colE]):
    if path.exists():
        with col:
            st.download_button(
                f"Download {label}",
                data=Path(path).read_bytes(),
                file_name=Path(path).name,
            )

# Download all
all_exist = any(Path(p).exists() for _, p in files)
if all_exist:
    with st.expander("Download All Results (ZIP)"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for _, p in files:
                p = Path(p)
                if p.exists():
                    z.write(p, arcname=p.name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "Download All Results (ZIP)",
            data=buf.getvalue(),
            file_name=f"ses_results_{ts}.zip",
        )

st.markdown("Kembali ke halaman LSTM: [Generate Forecast LSTM](3_Generate_Forecast.py)")
