from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import (
    UPLOAD_FOLDER,
    MODELS_FOLDER,
    ASSETS_FOLDER,
    DEFAULT_EXCEL,
    MODELS_METADATA,
    FORECAST_TOTAL,
)
from utils.data_handler import load_excel_data

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

# Inject CSS
css_path = ASSETS_FOLDER / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Session state defaults
ss = st.session_state
for k, v in {
    "data_uploaded": False,
    "data_file": None,
    "models_trained": MODELS_METADATA.exists(),
    "training_in_progress": False,
    "training_complete": False,
    "forecast_generated": FORECAST_TOTAL.exists(),
    "current_page": "Home",
}.items():
    ss.setdefault(k, v)

st.title("📈 Sales Forecasting System")
st.caption("LSTM-based Multi-Product Sales Forecasting")

# Quick stats
col1, col2, col3, col4 = st.columns(4)

# Total produk di data
try:
    data_files = list(Path(UPLOAD_FOLDER).glob("*.xlsx"))
    df = None
    if data_files:
        df = load_excel_data(data_files[-1])
        n_products = int(df.iloc[:, 0:10].shape[0])  # rough fallback
        # Try better: count unique of likely product column
        prod_col = next((c for c in df.columns if "Produk" in c or "Product" in c or "product" in c.lower()), None)
        if prod_col:
            n_products = int(df[prod_col].nunique())
    else:
        n_products = 0
except Exception:
    n_products = 0

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>🧾 Total Produk</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{n_products}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Jumlah model dilatih
n_models = len(list(Path(MODELS_FOLDER).glob("*_model.pkl")))
with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>🧠 Models Terlatih</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{n_models}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Terakhir forecast
last_forecast = None
if Path(FORECAST_TOTAL).exists():
    last_forecast = datetime.fromtimestamp(Path(FORECAST_TOTAL).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>⏱️ Terakhir Forecast</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{last_forecast or '-'}" + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Total data poin (bulan)
# Try to compute from data
n_points = 0
try:
    if  df is not None:
        date_col = next((c for c in df.columns if 'Tanggal' in c or 'Date' in c or c.lower().startswith('date')), None)
        if date_col:
            dts = pd.to_datetime(df[date_col], errors='coerce').dropna().dt.to_period('M').unique()
            n_points = len(dts)
except Exception:
    pass
with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>📅 Total Data Poin (bulan)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{n_points}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# System Status
st.subheader("🧭 Status Sistem")
sc1, sc2, sc3 = st.columns(3)
# Data status
has_data = len(list(Path(UPLOAD_FOLDER).glob("*.xlsx"))) > 0 or Path(DEFAULT_EXCEL).exists()
with sc1:
    if has_data:
        st.success("✅ Data tersedia di folder uploads/ atau default file.")
    else:
        st.warning("⚠️ Belum ada data. Silakan upload dulu di halaman Upload Data.")
# Model status
has_models = any(Path(MODELS_FOLDER).glob("*_model.pkl"))
with sc2:
    if has_models:
        st.success("✅ Trained models tersedia.")
    else:
        st.info("ℹ️ Belum ada model. Silakan lakukan training.")
# Forecast status
with sc3:
    if Path(FORECAST_TOTAL).exists():
        st.success("✅ Hasil forecast tersedia.")
    else:
        st.info("ℹ️ Forecast belum digenerate.")

st.subheader("⚡ Aksi Cepat")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("📤 Upload Data Baru", use_container_width=True):
        st.switch_page("pages/1_Upload_Data.py")
with c2:
    if st.button("🧠 Train Models", type="primary", use_container_width=True):
        st.switch_page("pages/2_Train_Models.py")
with c3:
    if st.button("🔮 Generate Forecast (LSTM)", use_container_width=True):
        st.switch_page("pages/3_Generate_Forecast.py")
with c4:
    if st.button("📈 SES Forecast", use_container_width=True):
        st.switch_page("pages/5_SES_Forecast.py")

st.info("Gunakan sidebar untuk navigasi antar halaman. Kini tersedia alternatif metode SES selain LSTM.")

# Optional recent activity
activity = []
if has_data:
    for p in Path(UPLOAD_FOLDER).glob("*.xlsx"):
        activity.append(("Upload", p.name, datetime.fromtimestamp(p.stat().st_mtime)))
if has_models:
    meta = MODELS_METADATA
    if meta.exists():
        activity.append(("Training", "models_metadata.json", datetime.fromtimestamp(meta.stat().st_mtime)))
if Path(FORECAST_TOTAL).exists():
    activity.append(("Forecast", Path(FORECAST_TOTAL).name, datetime.fromtimestamp(Path(FORECAST_TOTAL).stat().st_mtime)))

if activity:
    st.subheader("🕒 Aktivitas Terbaru")
    for kind, name, ts in sorted(activity, key=lambda x: x[2], reverse=True)[:10]:
        st.write(f"• {kind}: {name} — {ts.strftime('%Y-%m-%d %H:%M')}")
