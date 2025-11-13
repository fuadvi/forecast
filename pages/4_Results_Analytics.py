from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from config.settings import (
    FORECAST_TOTAL,
    TOPN_PER_MONTH,
    FORECAST_PER_PRODUCT,
    FORECAST_DIAGNOSTICS,
    SKIPPED_PRODUCTS_LOG,
    SES_FORECAST_TOTAL,
    SES_TOPN_PER_MONTH,
    SES_FORECAST_PER_PRODUCT,
    SES_GROUPED_TOP5_PNG,
    SES_MODEL_PARAMS,
    TRAINING_DIAGNOSTICS,
)
from utils.chart_generator import (
    generate_total_forecast_chart,
    generate_top_products_chart,
    generate_product_detail_chart,
    generate_diagnostics_charts,
)

st.set_page_config(page_title="Results & Analytics", page_icon="📊", layout="wide")
st.title("📊 Results & Analytics")
st.caption("Visualisasi interaktif dan analitik mendalam untuk hasil forecast LSTM dan SES.")

# =============== Helper Functions for Evaluation Metrics ===============
def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    return float(np.mean(np.abs(actual - predicted)))

def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    return float(np.mean((actual - predicted) ** 2))

def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100))

def get_evaluation_metrics_from_training_diagnostics() -> dict:
    """Extract evaluation metrics from training diagnostics file"""
    metrics = {
        "mae": None,
        "mse": None,
        "rmse": None,
        "mape": None,
        "source": "training_diagnostics"
    }
    
    training_diag_path = Path(TRAINING_DIAGNOSTICS)
    if not training_diag_path.exists():
        return metrics
    
    try:
        diag_df = pd.read_csv(training_diag_path)
        
        # Check if required columns exist
        if "train_mae_residual" in diag_df.columns and "train_rmse_residual" in diag_df.columns:
            # Calculate average metrics across all products
            mae_values = diag_df["train_mae_residual"].dropna()
            rmse_values = diag_df["train_rmse_residual"].dropna()
            
            if len(mae_values) > 0:
                metrics["mae"] = float(mae_values.mean())
            if len(rmse_values) > 0:
                metrics["rmse"] = float(rmse_values.mean())
                # MSE = RMSE^2
                metrics["mse"] = float(metrics["rmse"] ** 2) if metrics["rmse"] is not None else None
        
        # MAPE might not be in diagnostics, so we'll calculate it if we have actual vs predicted
        # For now, we'll leave it as None if not available
    except Exception as e:
        st.warning(f"Error reading training diagnostics: {e}")
    
    return metrics

def calculate_metrics_from_data(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate all evaluation metrics from actual and predicted arrays"""
    return {
        "mae": calculate_mae(actual, predicted),
        "mse": calculate_mse(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "mape": calculate_mape(actual, predicted),
        "source": "calculated"
    }

# Paths
lstm_ft = Path(FORECAST_TOTAL)
lstm_topn = Path(TOPN_PER_MONTH)
lstm_per = Path(FORECAST_PER_PRODUCT)

g_has_lstm = lstm_ft.exists()

# Outer tabs: LSTM, SES, Comparison
TAB_LSTM, TAB_SES, TAB_CMP = st.tabs(["LSTM Forecast", "SES Forecast", "Perbandingan Metode"])

# =============== LSTM TAB (existing content grouped) ===============
with TAB_LSTM:
    if not g_has_lstm:
        st.error("❌ Hasil LSTM tidak ditemukan. Jalankan forecast terlebih dahulu.")
        if st.button("🔮 Ke Halaman Forecast LSTM"):
            st.switch_page("pages/3_Generate_Forecast.py")
        st.stop()
    diag_path = Path(FORECAST_DIAGNOSTICS)
    # T1, T2, T3, T4 = st.tabs(["Total Forecast", "Top Products", "Product Details", "Diagnostics"])
    T1, T2, T3 = st.tabs(["Total Forecast", "Top Products", "Evaluation Metrics"])
    with T1:
        st.subheader("Total Sales Forecast (24 Months)")
        total_df = pd.read_csv(lstm_ft)
        show_ci = st.checkbox("Tampilkan Confidence Interval", value=False)
        fig = generate_total_forecast_chart(total_df, show_ci=show_ci)
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        num_cols = total_df.select_dtypes(include="number").columns
        ssum = float(total_df[num_cols].sum(numeric_only=True).sum()) if len(num_cols) else 0.0
        avg = ssum / max(1, len(total_df))
        with col1:
            st.metric("Total Forecast Sum", f"{ssum:,.0f}")
        with col2:
            st.metric("Average Monthly Forecast", f"{avg:,.0f}")
        ycol = num_cols[0] if len(num_cols) else None
        if ycol:
            idx_max = int(total_df[ycol].idxmax())
            idx_min = int(total_df[ycol].idxmin())
            with col3:
                st.metric("Peak Month", str(total_df.iloc[idx_max, 0]))
            with col4:
                st.metric("Lowest Month", str(total_df.iloc[idx_min, 0]))
        st.download_button("Download CSV", data=total_df.to_csv(index=False).encode("utf-8"), file_name=lstm_ft.name)
    with T2:
        st.subheader("Top Performing Products by Month")
        if not lstm_topn.exists():
            st.info("File topN tidak ditemukan. Jalankan forecast untuk menghasilkan.")
        else:
            topn_df = pd.read_csv(lstm_topn)
            months = [str(m) for m in sorted(topn_df[topn_df.columns[0]].unique())]
            sel = st.selectbox("Pilih Bulan", options=["All Months"] + months)
            view = st.radio("Tampilan", ["Bar Chart", "Grouped Chart", "Table View"], index=0, horizontal=True)
            if view == "Bar Chart" and sel != "All Months":
                fig = generate_top_products_chart(topn_df, month=sel, grouped=False)
                st.plotly_chart(fig, use_container_width=True)
            elif view == "Grouped Chart":
                fig = generate_top_products_chart(topn_df, month=None, grouped=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(topn_df, use_container_width=True)
            toexcel = io.BytesIO()
            with pd.ExcelWriter(toexcel, engine='openpyxl') as writer:
                topn_df.to_excel(writer, index=False)
            st.download_button("Export to Excel", data=toexcel.getvalue(), file_name="topN_per_month_24m.xlsx")
    
    # =============== Evaluation Metrics Tab ===============
    with T3:
        st.subheader("📊 Evaluation Metrics")
        st.caption("Metrik evaluasi model LSTM: MAE, MSE, RMSE, dan MAPE")
        
        # Try to get metrics from training diagnostics
        metrics = get_evaluation_metrics_from_training_diagnostics()
        
        # If metrics are not available from diagnostics, show message
        if metrics["mae"] is None and metrics["rmse"] is None:
            st.info("ℹ️ Metrik evaluasi dari training diagnostics tidak tersedia. Menampilkan metrik dari data yang tersedia.")
            
            # Try to calculate from forecast diagnostics if available
            if diag_path.exists():
                try:
                    forecast_diag = pd.read_csv(diag_path)
                    # If there are columns that might contain actual vs predicted, use them
                    # This is a fallback - adjust based on your actual diagnostics structure
                    st.warning("⚠️ Perhitungan metrik memerlukan data aktual vs prediksi. Silakan pastikan model sudah ditraining dengan data validasi.")
                except Exception:
                    pass
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mae_value = metrics.get("mae", 0.0) if metrics.get("mae") is not None else 0.0
            st.metric(
                label="MAE (Mean Absolute Error)",
                value=f"{mae_value:,.4f}" if mae_value > 0 else "N/A",
                help="Rata-rata dari nilai absolut selisih antara nilai aktual dan prediksi"
            )
        
        with col2:
            mse_value = metrics.get("mse", 0.0) if metrics.get("mse") is not None else 0.0
            st.metric(
                label="MSE (Mean Squared Error)",
                value=f"{mse_value:,.4f}" if mse_value > 0 else "N/A",
                help="Rata-rata dari kuadrat selisih antara nilai aktual dan prediksi"
            )
        
        with col3:
            rmse_value = metrics.get("rmse", 0.0) if metrics.get("rmse") is not None else 0.0
            st.metric(
                label="RMSE (Root Mean Squared Error)",
                value=f"{rmse_value:,.4f}" if rmse_value > 0 else "N/A",
                help="Akar kuadrat dari MSE, memberikan bobot lebih pada error yang besar"
            )
        
        with col4:
            mape_value = metrics.get("mape", 0.0) if metrics.get("mape") is not None else 0.0
            st.metric(
                label="MAPE (Mean Absolute Percentage Error)",
                value=f"{mape_value:.2f}%" if mape_value > 0 else "N/A",
                help="Rata-rata persentase error absolut, berguna untuk membandingkan skala berbeda"
            )
        
        # Visualization of metrics
        st.divider()
        st.subheader("📈 Visualisasi Metrik")
        
        # Prepare data for visualization
        metric_names = ["MAE", "MSE", "RMSE", "MAPE"]
        metric_values = [
            mae_value if mae_value > 0 else None,
            mse_value if mse_value > 0 else None,
            rmse_value if rmse_value > 0 else None,
            mape_value if mape_value > 0 else None
        ]
        
        # Filter out None values for visualization
        valid_metrics = [(name, val) for name, val in zip(metric_names, metric_values) if val is not None and val > 0]
        
        if valid_metrics:
            names, values = zip(*valid_metrics)
            
            # Normalize MAPE for better visualization (since it's in percentage)
            display_values = []
            display_names = []
            for name, val in valid_metrics:
                if name == "MAPE":
                    # Keep MAPE as is for display
                    display_values.append(val)
                    display_names.append(f"{name} ({val:.2f}%)")
                else:
                    display_values.append(val)
                    display_names.append(name)
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(names),
                    y=list(values),
                    text=[f"{v:.4f}" if names[i] != "MAPE" else f"{v:.2f}%" for i, v in enumerate(values)],
                    textposition="auto",
                    marker=dict(
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(names)],
                        line=dict(color="rgb(8,48,107)", width=1.5)
                    )
                )
            ])
            
            fig.update_layout(
                title="Perbandingan Metrik Evaluasi Model",
                xaxis_title="Metrik",
                yaxis_title="Nilai",
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info section
            with st.expander("ℹ️ Penjelasan Metrik"):
                st.markdown("""
                **MAE (Mean Absolute Error)**
                - Mengukur rata-rata dari nilai absolut error
                - Tidak memberikan bobot lebih pada error besar
                - Satuan sama dengan data asli
                - Semakin kecil semakin baik
                
                **MSE (Mean Squared Error)**
                - Mengukur rata-rata dari kuadrat error
                - Memberikan bobot lebih pada error besar
                - Satuan adalah kuadrat dari data asli
                - Semakin kecil semakin baik
                
                **RMSE (Root Mean Squared Error)**
                - Akar kuadrat dari MSE
                - Memberikan bobot lebih pada error besar (seperti MSE)
                - Satuan sama dengan data asli
                - Semakin kecil semakin baik
                
                **MAPE (Mean Absolute Percentage Error)**
                - Mengukur error dalam bentuk persentase
                - Berguna untuk membandingkan model pada skala berbeda
                - Tidak terpengaruh oleh skala data
                - Semakin kecil semakin baik (biasanya < 10% dianggap baik)
                """)
        else:
            st.warning("⚠️ Tidak ada metrik yang tersedia untuk divisualisasikan. Pastikan model sudah ditraining dan file training diagnostics tersedia.")
        
        # Show source of metrics
        if metrics.get("source"):
            st.caption(f"📌 Sumber metrik: {metrics['source']}")
    
    # with T4:
    #     st.subheader("Product Details")
    #     if not lstm_per.exists():
    #         st.info("File forecast per product tidak ditemukan.")
    #     else:
    #         per_df = pd.read_csv(lstm_per)
    #         prod_col = next((c for c in per_df.columns if "product" in c.lower()), per_df.columns[0])
    #         products = sorted(per_df[prod_col].unique())
    #         prod = st.selectbox("Pilih Produk", options=products)
    #         p_df = per_df[per_df[prod_col] == prod]
    #         fig = generate_product_detail_chart(None, p_df, prod)
    #         st.plotly_chart(fig, use_container_width=True)
    #         st.download_button("Download Product Data (CSV)", data=p_df.to_csv(index=False).encode("utf-8"), file_name=f"{prod}_forecast.csv")
    # with T4:
    #     st.subheader("Training & Forecast Diagnostics")
    #     if diag_path.exists():
    #         diag = pd.read_csv(diag_path)
    #         st.dataframe(diag, use_container_width=True)
    #         figs = generate_diagnostics_charts(diag)
    #         for key in ["cv_hist", "per_cat", "pie"]:
    #             if key in figs:
    #                 st.plotly_chart(figs[key], use_container_width=True)
    #         st.download_button("Download Full Diagnostics (CSV)", data=diag.to_csv(index=False).encode("utf-8"), file_name=diag_path.name)
    #     else:
    #         st.info("Diagnostics belum tersedia.")
    #     if Path(SKIPPED_PRODUCTS_LOG).exists():
    #         with st.expander("View Skipped Products"):
    #             st.code(Path(SKIPPED_PRODUCTS_LOG).read_text(encoding="utf-8"))
    with st.expander("Download Semua Hasil (ZIP)"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for p in [lstm_ft, lstm_topn, lstm_per, Path(FORECAST_DIAGNOSTICS)]:
                if p.exists():
                    z.write(p, arcname=p.name)
        st.download_button("Download All Results", data=buf.getvalue(), file_name="forecast_results_lstm.zip")

# =============== SES TAB ===============
with TAB_SES:
    ses_ft = Path(SES_FORECAST_TOTAL)
    ses_topn = Path(SES_TOPN_PER_MONTH)
    ses_per = Path(SES_FORECAST_PER_PRODUCT)
    if not ses_ft.exists():
        st.warning("⚠️ Hasil SES belum tersedia. Jalankan terlebih dahulu di halaman SES Forecast.")
        if st.button("➡️ Ke Halaman SES Forecast"):
            st.switch_page("pages/5_SES_Forecast.py")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_df = pd.read_csv(ses_ft)
        num_cols = total_df.select_dtypes(include="number").columns
        avg = float(total_df[num_cols].mean(numeric_only=True).mean()) if len(num_cols) else 0.0
        with col1:
            st.metric("Total produk diforecast", f"{pd.read_csv(ses_per)[pd.read_csv(ses_per).columns[0]].nunique() if ses_per.exists() else 0}")
        with col2:
            st.metric("Rata-rata forecast/bln", f"{avg:,.0f}")
        with col3:
            if ses_per.exists():
                per_df = pd.read_csv(ses_per)
                vcol = next((c for c in per_df.columns if c not in per_df.columns[:2]), per_df.columns[-1])
                pcol = next((c for c in per_df.columns if 'product' in c.lower()), per_df.columns[0])
                grp = per_df.groupby(pcol)[vcol].sum().sort_values(ascending=False)
                st.metric("Produk tertinggi", grp.index[0] if len(grp) else "-")
            else:
                st.metric("Produk tertinggi", "-")
        with col4:
            # naive: show availability of HW vs SES via params file if available
            st.metric("Metode dominan", "Lihat params" if Path(SES_MODEL_PARAMS).exists() else "-" )
        # product selector
        # st.subheader("Visualisasi Interaktif")
        # if ses_per.exists():
        #     per_df = pd.read_csv(ses_per)
        #     pcol = next((c for c in per_df.columns if 'product' in c.lower()), per_df.columns[0])
        #     products = ["All Products"] + sorted(per_df[pcol].unique())
        #     prod = st.selectbox("Pilih Produk", options=products)
        #     if prod == "All Products":
        #         st.line_chart(total_df.set_index(total_df.columns[0]).select_dtypes(include='number'))
        #     else:
        #         p_df = per_df[per_df[pcol] == prod]
        #         fig = generate_product_detail_chart(None, p_df, prod)
        #         st.plotly_chart(fig, use_container_width=True)
        if ses_topn.exists():
            st.subheader("Top-5 Produk per Bulan (Gambar)")
            if Path(SES_GROUPED_TOP5_PNG).exists():
                st.image(str(SES_GROUPED_TOP5_PNG), use_column_width=True)
            else:
                st.info("Gambar grouped chart belum tersedia.")

# =============== COMPARISON TAB ===============
with TAB_CMP:
    st.subheader("Perbandingan LSTM vs SES")
    ses_ft = Path(SES_FORECAST_TOTAL)
    ses_per = Path(SES_FORECAST_PER_PRODUCT)
    if not (lstm_ft.exists() and ses_ft.exists()):
        st.error("❌ Hasil LSTM atau SES tidak lengkap untuk perbandingan.")
    else:
        # Product selector with intersection
        if lstm_per.exists() and ses_per.exists():
            l_df = pd.read_csv(lstm_per)
            s_df = pd.read_csv(ses_per)
            lp = next((c for c in l_df.columns if 'product' in c.lower()), l_df.columns[0])
            sp = next((c for c in s_df.columns if 'product' in c.lower()), s_df.columns[0])
            common = sorted(set(l_df[lp]).intersection(set(s_df[sp])))
            sel_products = st.multiselect("Pilih Produk untuk dibandingkan", options=common, default=common[:1])
            # Overlay for each product
            for prod in sel_products:
                lpf = l_df[l_df[lp] == prod]
                spf = s_df[s_df[sp] == prod]
                dcol_l = next((c for c in lpf.columns if 'date' in c.lower()), lpf.columns[1])
                dcol_s = next((c for c in spf.columns if 'date' in c.lower()), spf.columns[1])
                ycol_l = next((c for c in lpf.columns if c not in (dcol_l, lp)), lpf.columns[-1])
                ycol_s = next((c for c in spf.columns if c not in (dcol_s, sp)), spf.columns[-1])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pd.to_datetime(lpf[dcol_l]), y=lpf[ycol_l], mode='lines+markers', name='LSTM', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=pd.to_datetime(spf[dcol_s]), y=spf[ycol_s], mode='lines+markers', name='SES', line=dict(color='orange')))
                fig.update_layout(title=f"Perbandingan Forecast: {prod}")
                st.plotly_chart(fig, use_container_width=True)
        # Aggregate comparison
        lt = pd.read_csv(lstm_ft)
        st_df = pd.read_csv(ses_ft)
        dcol_l = lt.columns[0]
        dcol_s = st_df.columns[0]
        ycol_l = next((c for c in lt.columns if c != dcol_l), lt.columns[-1])
        ycol_s = next((c for c in st_df.columns if c != dcol_s), st_df.columns[-1])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pd.to_datetime(lt[dcol_l]), y=lt[ycol_l], mode='lines', name='Total LSTM', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=pd.to_datetime(st_df[dcol_s]), y=st_df[ycol_s], mode='lines', name='Total SES', line=dict(color='orange')))
        st.plotly_chart(fig2, use_container_width=True)
        # Simple insight
        mean_l = lt[ycol_l].mean()
        mean_s = st_df[ycol_s].mean()
        diff_pct = (mean_s - mean_l) / mean_l * 100 if mean_l else 0.0
        if diff_pct < 0:
            st.info(f"Metode SES cenderung lebih konservatif dengan rata-rata {abs(diff_pct):.1f}% lebih rendah dari LSTM.")
        else:
            st.info(f"Metode SES cenderung lebih tinggi dengan rata-rata {abs(diff_pct):.1f}% di atas LSTM.")
