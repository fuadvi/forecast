from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import MODELS_METADATA, FORECAST_TOTAL, TOPN_PER_MONTH, FORECAST_PER_PRODUCT
from utils.forecast_wrapper import run_forecast, check_forecast_status, load_forecast_results, get_forecast_summary
from utils.chart_generator import generate_total_forecast_chart

st.set_page_config(page_title="Generate Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Generate Forecast (24 bulan)")
st.caption("Menjalankan prediksi penjualan menggunakan trained models.")

# Check models
if not Path(MODELS_METADATA).exists():
    st.error("❌ Models metadata tidak ditemukan. Mohon lakukan training terlebih dahulu.")
    st.button("🧠 Pergi ke Train Models", on_click=lambda: st.switch_page("pages/2_Train_Models.py"))
    st.stop()
else:
    st.success("✅ Models siap digunakan.")

# Model info
try:
    import json
    meta = json.loads(Path(MODELS_METADATA).read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Trained Models", len(meta.get("products", [])) if isinstance(meta.get("products"), list) else meta.get("n_models", 0))
    with c2:
        dt = meta.get("generated_at") or meta.get("created_at")
        st.metric("Model Generated Date", dt or "-")
    with c3:
        st.metric("Forecast Horizon", meta.get("forecast_horizon", 24))
    with c4:
        st.metric("Time Steps", meta.get("time_steps", 6))
except Exception:
    st.info("Tidak dapat membaca metadata model secara lengkap.")

st.subheader("⚙️ Konfigurasi Forecast")
N = st.slider("Top N Products untuk Ranking", 5, 20, 10, help="Menentukan berapa produk teratas per bulan untuk ditampilkan pada ranking.")
st.caption("Periode forecast akan otomatis dari bulan berikutnya setelah data terakhir hingga 24 bulan ke depan.")

if st.button("▶️ Generate Forecast", type="primary"):
    ok, msg = run_forecast()
    if ok:
        st.info("Forecast sedang diproses... Mohon tunggu.")
    else:
        st.warning(msg)

status = check_forecast_status()
if status == "running":
    with st.spinner("Generating forecasts..."):
        st.progress(50, text="Processing product forecasts...")
        st.write("Sistem sedang menghitung...")
elif status in ("finished", "idle"):
    if Path(FORECAST_TOTAL).exists():
        st.success("✅ Forecast selesai.")
        st.balloons()
        res = load_forecast_results()
        # Summary
        st.subheader("📦 Output Files")
        files = [FORECAST_PER_PRODUCT, FORECAST_TOTAL, TOPN_PER_MONTH]
        for f in files:
            if Path(f).exists():
                st.write(f"📄 {Path(f).name}")
        # Quick preview
        if res.get("total") is not None:
            st.subheader("👀 Preview Total Forecast")
            st.dataframe(res["total"].head(), use_container_width=True)
            fig = generate_total_forecast_chart(res["total"], show_ci=False)
            st.plotly_chart(fig, use_container_width=True)
        st.info("Langkah selanjutnya: Lihat hasil dan analitik detail")
        st.button("📊 Buka Hasil & Analytics", on_click=lambda: st.switch_page("pages/4_Results_Analytics.py"))
else:
    st.info("Klik tombol di atas untuk memulai proses forecast.")
