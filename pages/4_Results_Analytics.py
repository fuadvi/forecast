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
    ROOT_DIR,
    DEFAULT_EXCEL,
    # LSTM combined visualizations
    LSTM_TOP5_YEARLY_PNG,
    LSTM_TOP5_QUARTERLY_PNG,
    LSTM_BORDA_COUNT_PNG,
    LSTM_GROUPED_TOP5_PNG,
    LSTM_QUARTERLY_TOP5_2025,
    LSTM_QUARTERLY_TOP5_2026,
    LSTM_YEARLY_TOP5_2025,
    LSTM_YEARLY_TOP5_2026,
    # SES combined visualizations
    SES_TOP5_YEARLY_PNG,
    SES_TOP5_QUARTERLY_PNG,
    SES_BORDA_COUNT_PNG,
    SES_QUARTERLY_TOP5_2025,
    SES_QUARTERLY_TOP5_2026,
    SES_YEARLY_TOP5_2025,
    SES_YEARLY_TOP5_2026,
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

def get_ses_evaluation_metrics() -> dict:
    """Get SES evaluation metrics from saved CSV file or calculate from data if available"""
    metrics = {
        "mae": None,
        "mse": None,
        "rmse": None,
        "mape": None,
        "source": "calculated"
    }
    
    try:
        # First, try to load from saved evaluation metrics CSV
        ses_metrics_path = ROOT_DIR / "ses_evaluation_metrics.csv"
        if ses_metrics_path.exists():
            try:
                metrics_df = pd.read_csv(ses_metrics_path)
                if not metrics_df.empty and "mae" in metrics_df.columns and "rmse" in metrics_df.columns:
                    # Calculate average metrics across all products
                    mae_values = metrics_df["mae"].dropna()
                    rmse_values = metrics_df["rmse"].dropna()
                    mape_values = metrics_df["mape"].dropna() if "mape" in metrics_df.columns else pd.Series()
                    
                    if len(mae_values) > 0:
                        metrics["mae"] = float(mae_values.mean())
                    if len(rmse_values) > 0:
                        metrics["rmse"] = float(rmse_values.mean())
                        # MSE = RMSE^2
                        metrics["mse"] = float(metrics["rmse"] ** 2) if metrics["rmse"] is not None else None
                    if len(mape_values) > 0:
                        metrics["mape"] = float(mape_values.mean())
                    
                    metrics["source"] = "ses_evaluation_metrics_csv"
                    return metrics
            except Exception:
                pass
        
        # Fallback: Try to calculate from forecast data and historical data
        ses_per_path = Path(SES_FORECAST_PER_PRODUCT)
        if not ses_per_path.exists():
            return metrics
        
        ses_df = pd.read_csv(ses_per_path)
        
        # Try to load historical data from Excel for comparison
        excel_path = Path(DEFAULT_EXCEL)
        if excel_path.exists():
            try:
                # Read Excel and aggregate monthly
                df = pd.read_excel(excel_path, engine='openpyxl')
                
                # Try to infer column names
                date_col = next((c for c in df.columns if 'tanggal' in c.lower() or 'date' in c.lower()), None)
                qty_col = next((c for c in df.columns if 'jumlah' in c.lower() or 'qty' in c.lower() or 'quantity' in c.lower()), None)
                prod_col = next((c for c in df.columns if 'produk' in c.lower() or 'product' in c.lower()), None)
                
                if date_col and qty_col and prod_col:
                    df_hist = df[[date_col, qty_col, prod_col]].copy()
                    df_hist.columns = ['date', 'sales', 'product_name']
                    df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
                    df_hist = df_hist.dropna(subset=['date'])
                    
                    # Normalize product names (simple normalization)
                    def normalize_name(name):
                        if pd.isna(name):
                            return ""
                        s = str(name).lower().strip()
                        s = s.replace("-", " ").replace("_", " ")
                        s = " ".join(s.split())
                        return s
                    
                    df_hist['product_name'] = df_hist['product_name'].apply(normalize_name)
                    
                    # Aggregate monthly
                    df_hist['month'] = df_hist['date'].dt.to_period('M').dt.to_timestamp()
                    monthly_hist = df_hist.groupby(['product_name', 'month'])['sales'].sum().reset_index()
                    monthly_hist.columns = ['product_name', 'date', 'actual']
                    
                    # Prepare SES forecast data
                    ses_df = ses_df.copy()
                    ses_df['date'] = pd.to_datetime(ses_df['date'], errors='coerce')
                    ses_df = ses_df.dropna(subset=['date'])
                    
                    # Merge historical and forecast data for common products and dates
                    merged = pd.merge(
                        monthly_hist,
                        ses_df[['date', 'product_name', 'forecast']],
                        on=['date', 'product_name'],
                        how='inner'
                    )
                    
                    if len(merged) > 0:
                        actual_vals = merged['actual'].values
                        pred_vals = merged['forecast'].values
                        
                        # Calculate metrics
                        metrics["mae"] = calculate_mae(actual_vals, pred_vals)
                        metrics["mse"] = calculate_mse(actual_vals, pred_vals)
                        metrics["rmse"] = calculate_rmse(actual_vals, pred_vals)
                        metrics["mape"] = calculate_mape(actual_vals, pred_vals)
                        metrics["source"] = "calculated_from_historical"
            except Exception:
                pass
        
    except Exception as e:
        st.warning(f"Error calculating SES metrics: {e}")
    
    return metrics

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
    
    # Tabs: Analisis Tahunan, Analisis Kuartal, Per Bulan, Evaluation Metrics
    T_YEARLY, T_QUARTERLY, T_MONTHLY, T_METRICS = st.tabs([
        "📊 Analisis Tahunan", 
        "📈 Analisis Kuartal", 
        "📅 Per Bulan",
        "📉 Evaluation Metrics"
    ])
    
    # =============== TAB: Analisis Tahunan (Borda Count) ===============
    with T_YEARLY:
        st.subheader("🏆 Top 5 Produk Tahunan (Borda Count Voting)")
        st.caption("Perbandingan peringkat produk terbaik tahun 2025 dan 2026 menggunakan metode Borda Count")
        
        # Display combined yearly chart
        if Path(LSTM_TOP5_YEARLY_PNG).exists():
            st.image(str(LSTM_TOP5_YEARLY_PNG), width="stretch")
        else:
            st.warning("⚠️ Visualisasi Top 5 Tahunan belum tersedia. Jalankan forecast terlebih dahulu.")
        
        # Display Borda Count Process Chart
        st.divider()
        st.subheader("📊 Analisis Borda Count (Kontribusi Skor Per Kuartal)")
        st.caption("Stacked bar chart menunjukkan kontribusi skor dari setiap kuartal")
        
        if Path(LSTM_BORDA_COUNT_PNG).exists():
            st.image(str(LSTM_BORDA_COUNT_PNG), width="stretch")
        else:
            st.info("ℹ️ Visualisasi Borda Count Process belum tersedia.")
        
        # Show data tables
        st.divider()
        col_2025, col_2026 = st.columns(2)
        
        with col_2025:
            st.subheader("📋 Data Borda Count 2025")
            if Path(LSTM_YEARLY_TOP5_2025).exists():
                df_2025 = pd.read_csv(LSTM_YEARLY_TOP5_2025)
                st.dataframe(df_2025, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download CSV 2025", 
                    data=df_2025.to_csv(index=False).encode("utf-8"), 
                    file_name="yearly_top5_borda_2025.csv",
                    key="dl_yearly_2025"
                )
            else:
                st.info("Data 2025 belum tersedia.")
        
        with col_2026:
            st.subheader("📋 Data Borda Count 2026")
            if Path(LSTM_YEARLY_TOP5_2026).exists():
                df_2026 = pd.read_csv(LSTM_YEARLY_TOP5_2026)
                st.dataframe(df_2026, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download CSV 2026", 
                    data=df_2026.to_csv(index=False).encode("utf-8"), 
                    file_name="yearly_top5_borda_2026.csv",
                    key="dl_yearly_2026"
                )
            else:
                st.info("Data 2026 belum tersedia.")
    
    # =============== TAB: Analisis Kuartal ===============
    with T_QUARTERLY:
        st.subheader("📈 Top 5 Produk Per Kuartal (2025-2026)")
        st.caption("Perbandingan produk terbaik di setiap kuartal untuk tahun 2025 dan 2026")
        
        # Display combined quarterly chart
        if Path(LSTM_TOP5_QUARTERLY_PNG).exists():
            st.image(str(LSTM_TOP5_QUARTERLY_PNG), width="stretch")
        else:
            st.warning("⚠️ Visualisasi Top 5 Kuartal belum tersedia. Jalankan forecast terlebih dahulu.")
        
        # Show quarterly data tables
        st.divider()
        col_q2025, col_q2026 = st.columns(2)
        
        with col_q2025:
            st.subheader("📋 Data Kuartal 2025")
            if Path(LSTM_QUARTERLY_TOP5_2025).exists():
                df_q2025 = pd.read_csv(LSTM_QUARTERLY_TOP5_2025)
                
                # Quarter filter
                quarters = df_q2025['quarter'].unique().tolist() if 'quarter' in df_q2025.columns else []
                sel_q = st.selectbox("Filter Kuartal 2025", options=["Semua"] + quarters, key="q2025_filter")
                
                if sel_q != "Semua" and 'quarter' in df_q2025.columns:
                    view_df = df_q2025[df_q2025['quarter'] == sel_q]
                else:
                    view_df = df_q2025
                
                st.dataframe(view_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download CSV Q 2025", 
                    data=df_q2025.to_csv(index=False).encode("utf-8"), 
                    file_name="quarterly_top5_2025.csv",
                    key="dl_q2025"
                )
            else:
                st.info("Data kuartal 2025 belum tersedia.")
        
        with col_q2026:
            st.subheader("📋 Data Kuartal 2026")
            if Path(LSTM_QUARTERLY_TOP5_2026).exists():
                df_q2026 = pd.read_csv(LSTM_QUARTERLY_TOP5_2026)
                
                # Quarter filter
                quarters = df_q2026['quarter'].unique().tolist() if 'quarter' in df_q2026.columns else []
                sel_q = st.selectbox("Filter Kuartal 2026", options=["Semua"] + quarters, key="q2026_filter")
                
                if sel_q != "Semua" and 'quarter' in df_q2026.columns:
                    view_df = df_q2026[df_q2026['quarter'] == sel_q]
                else:
                    view_df = df_q2026
                
                st.dataframe(view_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download CSV Q 2026", 
                    data=df_q2026.to_csv(index=False).encode("utf-8"), 
                    file_name="quarterly_top5_2026.csv",
                    key="dl_q2026"
                )
            else:
                st.info("Data kuartal 2026 belum tersedia.")
    
    # =============== TAB: Per Bulan ===============
    with T_MONTHLY:
        st.subheader("📅 Top 5 Produk per Bulan (24 Bulan)")
        st.caption("Grouped bar chart menampilkan Top-5 produk per bulan dalam periode 24 bulan forecast")
        
        # Display grouped monthly chart
        if Path(LSTM_GROUPED_TOP5_PNG).exists():
            st.image(str(LSTM_GROUPED_TOP5_PNG), width="stretch")
        else:
            st.warning("⚠️ Visualisasi Top 5 per Bulan belum tersedia.")
        
        # Interactive data table
        st.divider()
        st.subheader("📋 Data Top-N per Bulan")
        
        if lstm_topn.exists():
            topn_df = pd.read_csv(lstm_topn)
            months = [str(m) for m in sorted(topn_df[topn_df.columns[0]].unique())]
            sel = st.selectbox("Filter Bulan", options=["Semua Bulan"] + months, key="month_filter")
            
            if sel != "Semua Bulan":
                view_df = topn_df[topn_df[topn_df.columns[0]].astype(str) == sel]
            else:
                view_df = topn_df
            
            st.dataframe(view_df, use_container_width=True, hide_index=True)
            
            toexcel = io.BytesIO()
            with pd.ExcelWriter(toexcel, engine='openpyxl') as writer:
                topn_df.to_excel(writer, index=False)
            st.download_button(
                "📥 Export to Excel", 
                data=toexcel.getvalue(), 
                file_name="topN_per_month_24m.xlsx",
                key="dl_topn_excel"
            )
        else:
            st.info("File topN tidak ditemukan. Jalankan forecast untuk menghasilkan.")
    
    # =============== Evaluation Metrics Tab ===============
    with T_METRICS:
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
        # Summary metrics
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
            st.metric("Metode dominan", "Lihat params" if Path(SES_MODEL_PARAMS).exists() else "-")
        
        # Tabs for SES analysis
        SES_T_YEARLY, SES_T_QUARTERLY, SES_T_MONTHLY = st.tabs([
            "📊 Analisis Tahunan", 
            "📈 Analisis Kuartal", 
            "📅 Per Bulan"
        ])
        
        # =============== TAB: SES Analisis Tahunan ===============
        with SES_T_YEARLY:
            st.subheader("🏆 Top 5 Produk Tahunan (Borda Count Voting) - SES")
            st.caption("Perbandingan peringkat produk terbaik menggunakan metode SES + Borda Count")
            
            if Path(SES_TOP5_YEARLY_PNG).exists():
                st.image(str(SES_TOP5_YEARLY_PNG), width="stretch")
            else:
                st.warning("⚠️ Visualisasi Top 5 Tahunan SES belum tersedia.")
            
            st.divider()
            st.subheader("📊 Analisis Borda Count (Kontribusi Skor Per Kuartal) - SES")
            
            if Path(SES_BORDA_COUNT_PNG).exists():
                st.image(str(SES_BORDA_COUNT_PNG), width="stretch")
            else:
                st.info("ℹ️ Visualisasi Borda Count Process SES belum tersedia.")
            
            # Data tables
            st.divider()
            col_ses_2025, col_ses_2026 = st.columns(2)
            
            with col_ses_2025:
                st.subheader("📋 Data Borda Count 2025 (SES)")
                if Path(SES_YEARLY_TOP5_2025).exists():
                    df_ses_2025 = pd.read_csv(SES_YEARLY_TOP5_2025)
                    st.dataframe(df_ses_2025, use_container_width=True, hide_index=True)
                else:
                    st.info("Data 2025 belum tersedia.")
            
            with col_ses_2026:
                st.subheader("📋 Data Borda Count 2026 (SES)")
                if Path(SES_YEARLY_TOP5_2026).exists():
                    df_ses_2026 = pd.read_csv(SES_YEARLY_TOP5_2026)
                    st.dataframe(df_ses_2026, use_container_width=True, hide_index=True)
                else:
                    st.info("Data 2026 belum tersedia.")
        
        # =============== TAB: SES Analisis Kuartal ===============
        with SES_T_QUARTERLY:
            st.subheader("📈 Top 5 Produk Per Kuartal - SES")
            st.caption("Perbandingan produk terbaik di setiap kuartal menggunakan metode SES")
            
            if Path(SES_TOP5_QUARTERLY_PNG).exists():
                st.image(str(SES_TOP5_QUARTERLY_PNG), width="stretch")
            else:
                st.warning("⚠️ Visualisasi Top 5 Kuartal SES belum tersedia.")
            
            # Quarterly data tables
            st.divider()
            col_sq2025, col_sq2026 = st.columns(2)
            
            with col_sq2025:
                st.subheader("📋 Data Kuartal 2025 (SES)")
                if Path(SES_QUARTERLY_TOP5_2025).exists():
                    df_sq2025 = pd.read_csv(SES_QUARTERLY_TOP5_2025)
                    quarters = df_sq2025['quarter'].unique().tolist() if 'quarter' in df_sq2025.columns else []
                    sel_q = st.selectbox("Filter Kuartal 2025", options=["Semua"] + quarters, key="ses_q2025_filter")
                    if sel_q != "Semua" and 'quarter' in df_sq2025.columns:
                        view_df = df_sq2025[df_sq2025['quarter'] == sel_q]
                    else:
                        view_df = df_sq2025
                    st.dataframe(view_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Data kuartal 2025 belum tersedia.")
            
            with col_sq2026:
                st.subheader("📋 Data Kuartal 2026 (SES)")
                if Path(SES_QUARTERLY_TOP5_2026).exists():
                    df_sq2026 = pd.read_csv(SES_QUARTERLY_TOP5_2026)
                    quarters = df_sq2026['quarter'].unique().tolist() if 'quarter' in df_sq2026.columns else []
                    sel_q = st.selectbox("Filter Kuartal 2026", options=["Semua"] + quarters, key="ses_q2026_filter")
                    if sel_q != "Semua" and 'quarter' in df_sq2026.columns:
                        view_df = df_sq2026[df_sq2026['quarter'] == sel_q]
                    else:
                        view_df = df_sq2026
                    st.dataframe(view_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Data kuartal 2026 belum tersedia.")
        
        # =============== TAB: SES Per Bulan ===============
        with SES_T_MONTHLY:
            st.subheader("📅 Top 5 Produk per Bulan (24 Bulan) - SES")
            st.caption("Grouped bar chart menampilkan Top-5 produk per bulan")
            
            if Path(SES_GROUPED_TOP5_PNG).exists():
                st.image(str(SES_GROUPED_TOP5_PNG), width="stretch")
            else:
                st.warning("⚠️ Visualisasi Top 5 per Bulan SES belum tersedia.")
            
            # Data table
            st.divider()
            st.subheader("📋 Data Top-N per Bulan (SES)")
            
            if ses_topn.exists():
                topn_df = pd.read_csv(ses_topn)
                months = [str(m) for m in sorted(topn_df[topn_df.columns[0]].unique())]
                sel = st.selectbox("Filter Bulan", options=["Semua Bulan"] + months, key="ses_month_filter")
                
                if sel != "Semua Bulan":
                    view_df = topn_df[topn_df[topn_df.columns[0]].astype(str) == sel]
                else:
                    view_df = topn_df
                
                st.dataframe(view_df, use_container_width=True, hide_index=True)
            else:
                st.info("File topN SES tidak ditemukan.")

# =============== COMPARISON TAB ===============
with TAB_CMP:
    # st.subheader("Perbandingan LSTM vs SES")
    # ses_ft = Path(SES_FORECAST_TOTAL)
    # ses_per = Path(SES_FORECAST_PER_PRODUCT)
    #
    # # Use new paths from settings
    # lstm_grouped_png = Path(LSTM_GROUPED_TOP5_PNG)
    # ses_grouped_png = Path(SES_GROUPED_TOP5_PNG)
    #
    # if not (lstm_ft.exists() and ses_ft.exists()):
    #     st.error("❌ Hasil LSTM atau SES tidak lengkap untuk perbandingan.")
    # else:
    #     # ==== LSTM Combined Visualizations Section ====
    #     st.subheader("📊 LSTM: Analisis Top 5 Produk (2025-2026)")
    #
    #     # Tab untuk visualisasi LSTM gabungan
    #     lstm_tab1, lstm_tab2 = st.tabs(["📅 Analisis Tahunan", "📈 Analisis Kuartal"])
    #
    #     with lstm_tab1:
    #         if Path(LSTM_TOP5_YEARLY_PNG).exists():
    #             st.image(str(LSTM_TOP5_YEARLY_PNG), width="stretch",
    #                     caption="LSTM: Top 5 Produk Tahunan (Borda Count Voting 2025-2026)")
    #         else:
    #             st.info("Visualisasi LSTM tahunan belum tersedia.")
    #
    #         # Show Borda Count Process
    #         if Path(LSTM_BORDA_COUNT_PNG).exists():
    #             with st.expander("🔍 Lihat Detail Kontribusi Borda Count"):
    #                 st.image(str(LSTM_BORDA_COUNT_PNG), width="stretch",
    #                         caption="Kontribusi skor per kuartal untuk setiap produk")
    #
    #     with lstm_tab2:
    #         if Path(LSTM_TOP5_QUARTERLY_PNG).exists():
    #             st.image(str(LSTM_TOP5_QUARTERLY_PNG), width="stretch",
    #                     caption="LSTM: Top 5 Produk Per Kuartal (Q1-Q4 2025 & 2026)")
    #         else:
    #             st.info("Visualisasi LSTM kuartal belum tersedia.")
    #
    #     st.divider()
        
        # ==== Monthly Comparison Section ====
        st.subheader("📅 Perbandingan Top-5 Produk per Bulan")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**LSTM Forecast**")
            if Path(LSTM_TOP5_YEARLY_PNG).exists():
                st.image(str(LSTM_TOP5_YEARLY_PNG), width="stretch")
            else:
                st.info("Gambar LSTM grouped chart belum tersedia.")
        with col_img2:
            st.markdown("**SES Forecast**")
            if Path(SES_TOP5_YEARLY_PNG).exists():
                st.image(str(SES_TOP5_YEARLY_PNG), width="stretch")
            else:
                st.info("Gambar SES grouped chart belum tersedia.")
        
        # # Perbandingan Metrik Evaluasi
        # st.subheader("Perbandingan Metrik Evaluasi")
        #
        # # Get LSTM metrics
        # lstm_metrics = get_evaluation_metrics_from_training_diagnostics()
        # lstm_mae = lstm_metrics.get("mae", 0.0) if lstm_metrics.get("mae") is not None else 0.0
        # lstm_rmse = lstm_metrics.get("rmse", 0.0) if lstm_metrics.get("rmse") is not None else 0.0
        # lstm_mape = lstm_metrics.get("mape", 0.0) if lstm_metrics.get("mape") is not None else 0.0
        #
        # # Get SES metrics
        # ses_metrics = get_ses_evaluation_metrics()
        # ses_mae = ses_metrics.get("mae", 0.0) if ses_metrics.get("mae") is not None else 0.0
        # ses_rmse = ses_metrics.get("rmse", 0.0) if ses_metrics.get("rmse") is not None else 0.0
        # ses_mape = ses_metrics.get("mape", 0.0) if ses_metrics.get("mape") is not None else 0.0
        #
        # # Prepare data for bar chart
        # metrics = ['MAE', 'RMSE', 'MAPE']
        # lstm_values = [lstm_mae, lstm_rmse, lstm_mape]
        # ses_values = [ses_mae, ses_rmse, ses_mape]
        #
        # # Filter out zero values for display (only show if at least one method has metrics)
        # if any(v > 0 for v in lstm_values) or any(v > 0 for v in ses_values):
        #     fig_metrics = go.Figure(data=[
        #         go.Bar(name='LSTM', x=metrics, y=lstm_values, marker_color='#1f77b4'),
        #         go.Bar(name='SES', x=metrics, y=ses_values, marker_color='#ff7f0e')
        #     ])
        #
        #     fig_metrics.update_layout(
        #         title='Perbandingan Metrik LSTM vs SES',
        #         xaxis_title='Metrik Evaluasi',
        #         yaxis_title='Nilai',
        #         barmode='group',
        #         height=400,
        #         showlegend=True,
        #         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        #         template="plotly_white"
        #     )
        #
        #     st.plotly_chart(fig_metrics, use_container_width=True)
        #
        #     # Display comparison table
        #     def format_metric_value(val, metric_name):
        #         if val <= 0:
        #             return "N/A"
        #         if metric_name == "MAPE":
        #             return f"{val:.2f}%"
        #         return f"{val:,.4f}"
        #
        #     comparison_df = pd.DataFrame({
        #         'Metrik': metrics,
        #         'LSTM': [format_metric_value(v, m) for v, m in zip(lstm_values, metrics)],
        #         'SES': [format_metric_value(v, m) for v, m in zip(ses_values, metrics)],
        #         'Selisih': [
        #             format_metric_value(abs(l - s), m) if l > 0 and s > 0 else "N/A"
        #             for l, s, m in zip(lstm_values, ses_values, metrics)
        #         ]
        #     })
        #
        #     st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        #
        #     # Show note if metrics are not fully available
        #     if lstm_mae == 0.0 and lstm_rmse == 0.0:
        #         st.info("ℹ️ Metrik LSTM tidak tersedia dari training diagnostics. Pastikan model sudah ditraining.")
        #     if ses_mae == 0.0 and ses_rmse == 0.0:
        #         st.info("ℹ️ Metrik SES belum tersedia. Metrik SES dihitung dari data forecast vs aktual. Pastikan file Excel data historis tersedia.")
        # else:
        #     st.warning("⚠️ Metrik evaluasi tidak tersedia untuk kedua metode. Pastikan model sudah ditraining dan data metrik tersedia.")
        #
        st.divider()
        # # Product selector with intersection
        # if lstm_per.exists() and ses_per.exists():
        #     l_df = pd.read_csv(lstm_per)
        #     s_df = pd.read_csv(ses_per)
        #     lp = next((c for c in l_df.columns if 'product' in c.lower()), l_df.columns[0])
        #     sp = next((c for c in s_df.columns if 'product' in c.lower()), s_df.columns[0])
        #     common = sorted(set(l_df[lp]).intersection(set(s_df[sp])))
        #     sel_products = st.multiselect("Pilih Produk untuk dibandingkan", options=common, default=common[:1])
        #     # Overlay for each product
        #     for prod in sel_products:
        #         lpf = l_df[l_df[lp] == prod]
        #         spf = s_df[s_df[sp] == prod]
        #         dcol_l = next((c for c in lpf.columns if 'date' in c.lower()), lpf.columns[1])
        #         dcol_s = next((c for c in spf.columns if 'date' in c.lower()), spf.columns[1])
        #         ycol_l = next((c for c in lpf.columns if c not in (dcol_l, lp)), lpf.columns[-1])
        #         ycol_s = next((c for c in spf.columns if c not in (dcol_s, sp)), spf.columns[-1])
        #         fig = go.Figure()
        #         fig.add_trace(go.Scatter(x=pd.to_datetime(lpf[dcol_l]), y=lpf[ycol_l], mode='lines+markers', name='LSTM', line=dict(color='blue')))
        #         fig.add_trace(go.Scatter(x=pd.to_datetime(spf[dcol_s]), y=spf[ycol_s], mode='lines+markers', name='SES', line=dict(color='orange')))
        #         fig.update_layout(title=f"Perbandingan Forecast: {prod}")
        #         st.plotly_chart(fig, use_container_width=True)
        # Aggregate comparison
        # lt = pd.read_csv(lstm_ft)
        # st_df = pd.read_csv(ses_ft)
        # dcol_l = lt.columns[0]
        # dcol_s = st_df.columns[0]
        # ycol_l = next((c for c in lt.columns if c != dcol_l), lt.columns[-1])
        # ycol_s = next((c for c in st_df.columns if c != dcol_s), st_df.columns[-1])
        # fig2 = go.Figure()
        # fig2.add_trace(go.Scatter(x=pd.to_datetime(lt[dcol_l]), y=lt[ycol_l], mode='lines', name='Total LSTM', line=dict(color='blue')))
        # fig2.add_trace(go.Scatter(x=pd.to_datetime(st_df[dcol_s]), y=st_df[ycol_s], mode='lines', name='Total SES', line=dict(color='orange')))
        # st.plotly_chart(fig2, use_container_width=True)
        # # Simple insight
        # mean_l = lt[ycol_l].mean()
        # mean_s = st_df[ycol_s].mean()
        # diff_pct = (mean_s - mean_l) / mean_l * 100 if mean_l else 0.0
        # if diff_pct < 0:
        #     st.info(f"Metode SES cenderung lebih konservatif dengan rata-rata {abs(diff_pct):.1f}% lebih rendah dari LSTM.")
        # else:
        #     st.info(f"Metode SES cenderung lebih tinggi dengan rata-rata {abs(diff_pct):.1f}% di atas LSTM.")
 