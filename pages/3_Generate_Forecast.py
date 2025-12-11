from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from config.settings import (
    MODELS_METADATA, 
    FORECAST_TOTAL, 
    TOPN_PER_MONTH, 
    FORECAST_PER_PRODUCT,
    ROOT_DIR,
    LSTM_TOP5_YEARLY_PNG,
    LSTM_TOP5_QUARTERLY_PNG,
    LSTM_BORDA_COUNT_PNG,
)
from utils.forecast_wrapper import (
    run_forecast, 
    check_forecast_status, 
    load_forecast_results, 
    get_forecast_summary,
    get_forecast_log,
)
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

# Session state untuk tracking
if 'forecast_started' not in st.session_state:
    st.session_state.forecast_started = False

if st.button("▶️ Generate Forecast", type="primary"):
    ok, msg = run_forecast()
    if ok:
        st.session_state.forecast_started = True
        st.info(msg)
        st.rerun()
    else:
        st.warning(msg)

status = check_forecast_status()

if status == "running":
    st.session_state.forecast_started = True
    
    # Show running status with auto-refresh
    st.warning("⏳ **Forecast sedang berjalan...**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()
    
    # Show log output
    with st.expander("📋 Log Output", expanded=True):
        log_text = get_forecast_log()
        if log_text:
            st.code(log_text, language="text")
        else:
            st.info("Menunggu output...")
    
    # Auto-refresh setiap 3 detik
    time.sleep(3)
    st.rerun()

elif status == "finished" or (status == "idle" and Path(FORECAST_TOTAL).exists()):
    st.session_state.forecast_started = False
    
    st.success("✅ **Forecast selesai!**")
    st.balloons()
    
    # Show final log
    with st.expander("📋 Log Output (Final)"):
        log_text = get_forecast_log()
        if log_text:
            st.code(log_text, language="text")
    
    res = load_forecast_results()
    
    # Summary - Output Files
    st.subheader("📦 Output Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 CSV Files:**")
        csv_files = [FORECAST_PER_PRODUCT, FORECAST_TOTAL, TOPN_PER_MONTH]
        for f in csv_files:
            if Path(f).exists():
                st.write(f"✅ {Path(f).name}")
            else:
                st.write(f"❌ {Path(f).name}")
    
    with col2:
        st.markdown("**🖼️ Quarterly Visualizations:**")
        plot_files = [
            (LSTM_TOP5_YEARLY_PNG, "top5_yearly.png"),
            (LSTM_TOP5_QUARTERLY_PNG, "top5_quarterly.png"),
            (LSTM_BORDA_COUNT_PNG, "borda_count_process.png"),
        ]
        for p, name in plot_files:
            if Path(p).exists():
                st.write(f"✅ {name}")
            else:
                st.write(f"❌ {name}")
    
    # Quick preview
    if res.get("total") is not None:
        st.subheader("👀 Preview Total Forecast")
        st.dataframe(res["total"].head(), use_container_width=True)
        fig = generate_total_forecast_chart(res["total"], show_ci=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show quarterly plots preview
    st.subheader("📊 Quarterly Analysis Preview")
    if Path(LSTM_TOP5_YEARLY_PNG).exists():
        st.image(str(LSTM_TOP5_YEARLY_PNG), width="stretch", caption="Top 5 Produk Tahunan (Borda Count)")
    else:
        st.info("Visualisasi kuartal belum tersedia.")
    
    st.info("Langkah selanjutnya: Lihat hasil dan analitik detail")
    st.button("📊 Buka Hasil & Analytics", on_click=lambda: st.switch_page("pages/4_Results_Analytics.py"))

elif status == "failed":
    st.session_state.forecast_started = False
    st.error("❌ **Forecast gagal!**")
    
    # Show error log
    with st.expander("📋 Error Log", expanded=True):
        log_text = get_forecast_log()
        if log_text:
            st.code(log_text, language="text")
        else:
            st.warning("Tidak ada log tersedia.")
    
    st.info("Periksa log di atas untuk detail error. Coba jalankan ulang forecast.")

else:
    # idle dan belum ada hasil
    st.info("Klik tombol di atas untuk memulai proses forecast.")
