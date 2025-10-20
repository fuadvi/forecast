import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import UPLOAD_FOLDER, MODELS_FOLDER, TRAINING_DIAGNOSTICS, MODELS_METADATA
from utils.training_wrapper import run_training, check_training_status, stream_training_logs, stop_training, get_training_results

st.set_page_config(page_title="Train Models", page_icon="🧠", layout="wide")
st.title("🧠 Train Models (LSTM)")
st.caption("Latih model LSTM per produk dengan monitoring real-time.")

# Check data availability
data_files = list(Path(UPLOAD_FOLDER).glob("*.xlsx"))
if not data_files and not Path("Data_Penjualan_Dengan_ID_Pelanggan.xlsx").exists():
    st.error("❌ Data tidak ditemukan. Mohon upload data terlebih dahulu.")
    if st.button("📤 Pergi ke Upload Data"):
        st.switch_page("pages/1_Upload_Data.py")
    st.stop()
else:
    st.success("✅ File data ditemukan. Siap untuk training.")

# Dataset info (lightweight)
with st.expander("Informasi Dataset", expanded=False):
    try:
        df = pd.read_excel(data_files[-1] if data_files else "Data_Penjualan_Dengan_ID_Pelanggan.xlsx", engine="openpyxl")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            # try find product column
            prod_col = next((c for c in df.columns if "Product" in c or "Produk" in c or "product" in c.lower()), df.columns[0])
            st.metric("Unique Products", int(df[prod_col].nunique()))
        with col3:
            date_col = next((c for c in df.columns if 'Tanggal' in c or 'Date' in c or c.lower().startswith('date')), df.columns[0])
            dts = pd.to_datetime(df[date_col], errors='coerce').dropna().dt.to_period('M')
            st.metric("Data Coverage (months)", int(dts.nunique()))
        with col4:
            cat_col = next((c for c in df.columns if 'Category' in c or 'Kategori' in c or 'category' in c.lower()), None)
            st.metric("Categories", int(df[cat_col].nunique()) if cat_col else 0)
    except Exception:
        st.info("Tidak dapat menghitung statistik detail dari data.")

st.subheader("⚙️ Konfigurasi Training")
left, right = st.columns(2)
with left:
    horizon = st.number_input("Forecast Horizon (bulan)", min_value=6, max_value=36, value=24, help="Lama periode forecast ke depan.")
    time_steps = st.number_input("Time Steps", min_value=3, max_value=12, value=6, help="Jumlah langkah waktu historis untuk input model.")
with right:
    min_points = st.number_input("Min Data Points (bulan)", min_value=4, max_value=24, value=8, help="Minimal jumlah bulan data historis.")
    min_nonzero = st.number_input("Min Non-Zero Transactions", min_value=1, max_value=10, value=3, help="Minimal bulan dengan transaksi > 0.")

with st.expander("Advanced Settings", expanded=False):
    epochs = st.slider("Epochs", 10, 200, 100)
    batch_size = st.slider("Batch Size", 4, 32, 8)
    st.info("Pengaturan ini mempengaruhi durasi training dan akurasi model.")

st.subheader("🚀 Kontrol Training")
colA, colB, colC = st.columns(3)
status = check_training_status()
with colA:
    start_disabled = status == "running"
    if st.button("▶️ Start Training", type="primary", disabled=start_disabled):
        ok, msg = run_training(
            forecast_horizon=int(horizon), time_steps=int(time_steps),
            min_points=int(min_points), min_nonzero=int(min_nonzero),
            epochs=int(epochs), batch_size=int(batch_size)
        )
        if ok:
            st.session_state["training_in_progress"] = True
            st.success(msg)
        else:
            st.warning(msg)
with colB:
    if st.button("⏹️ Stop"):
        if stop_training():
            st.info("Training dihentikan oleh pengguna.")
with colC:
    if st.button("🔄 Reset"):
        st.session_state["training_in_progress"] = False
        st.session_state["training_complete"] = False
        st.experimental_rerun()

# Progress and logs
ph_prog = st.empty()
ph_text = st.empty()
log_container = st.container()

if check_training_status() == "running":
    ph_prog.progress(5, text="Training berjalan...")
    ph_text.info("Sedang melatih models per produk...")
    with log_container:
        st.subheader("📜 Training Logs (live)")
        box = st.empty()
        logs = []
        for line in stream_training_logs(tail=50):
            logs.append(line)
            logs = logs[-50:]
            box.markdown("<div class='log-box'>" + "\n".join([st.session_state.get('_', '') + l for l in logs]) + "</div>", unsafe_allow_html=True)
            time.sleep(0.1)
        st.success("Training selesai.")
        st.session_state["training_in_progress"] = False
        st.session_state["training_complete"] = True
else:
    if TRAINING_DIAGNOSTICS.exists() or MODELS_METADATA.exists():
        st.info("Model sudah pernah dilatih. Menjalankan training ulang akan menimpa model lama.")

# Completion summary
if st.session_state.get("training_complete") and check_training_status() != "running":
    st.subheader("✅ Training Selesai")
    res = get_training_results()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Berhasil", res.get("n_models", 0))
    with col2:
        diag = res.get("diagnostics")
        st.metric("Total Baris Diagnostics", int(diag.shape[0]) if diag is not None else 0)
    with col3:
        st.metric("Produk di-skip", len(res.get("skipped", [])))
    if res.get("diagnostics") is not None:
        with st.expander("Lihat Diagnostics"):
            st.dataframe(res["diagnostics"].head(50), use_container_width=True)
            st.download_button("Unduh Diagnostics CSV", data=res["diagnostics"].to_csv(index=False).encode("utf-8"), file_name="training_diagnostics.csv")
    if res.get("skipped"):
        with st.expander("Produk yang di-skip"):
            st.code("\n".join(res["skipped"]))
    st.info("Langkah selanjutnya: Jalankan forecast")
    st.button("🔮 Ke Halaman Forecast", on_click=lambda: st.switch_page("pages/3_Generate_Forecast.py"))
