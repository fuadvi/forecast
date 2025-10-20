from pathlib import Path
import io
import zipfile

import pandas as pd
import streamlit as st

from config.settings import (
    FORECAST_TOTAL,
    TOPN_PER_MONTH,
    FORECAST_PER_PRODUCT,
    FORECAST_DIAGNOSTICS,
    SKIPPED_PRODUCTS_LOG,
)
from utils.chart_generator import (
    generate_total_forecast_chart,
    generate_top_products_chart,
    generate_product_detail_chart,
    generate_diagnostics_charts,
)

st.set_page_config(page_title="Results & Analytics", page_icon="📊", layout="wide")
st.title("📊 Results & Analytics")
st.caption("Visualisasi interaktif dan analitik mendalam untuk hasil forecast.")

# Load available data
ft_path = Path(FORECAST_TOTAL)
topn_path = Path(TOPN_PER_MONTH)
per_path = Path(FORECAST_PER_PRODUCT)
diag_path = Path(FORECAST_DIAGNOSTICS)

if not ft_path.exists():
    st.error("❌ File forecast_total_24m.csv tidak ditemukan. Jalankan forecast terlebih dahulu.")
    if st.button("🔮 Ke Halaman Forecast"):
        st.switch_page("pages/3_Generate_Forecast.py")
    st.stop()

# Tabs
T1, T2, T3, T4 = st.tabs(["Total Forecast", "Top Products", "Product Details", "Diagnostics"])

with T1:
    st.subheader("Total Sales Forecast (24 Months)")
    st.caption("Aggregate forecast across all products")
    total_df = pd.read_csv(ft_path)
    show_ci = st.checkbox("Tampilkan Confidence Interval", value=False)
    fig = generate_total_forecast_chart(total_df, show_ci=show_ci)
    st.plotly_chart(fig, use_container_width=True)
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    num_cols = total_df.select_dtypes(include="number").columns
    ssum = float(total_df[num_cols].sum(numeric_only=True).sum()) if len(num_cols) else 0.0
    avg = ssum / max(1, len(total_df))
    with col1:
        st.metric("Total Forecast Sum", f"{ssum:,.0f}")
    with col2:
        st.metric("Average Monthly Forecast", f"{avg:,.0f}")
    # Peak and lowest
    ycol = num_cols[0] if len(num_cols) else None
    if ycol:
        idx_max = int(total_df[ycol].idxmax())
        idx_min = int(total_df[ycol].idxmin())
        with col3:
            st.metric("Peak Month", str(total_df.iloc[idx_max, 0]))
        with col4:
            st.metric("Lowest Month", str(total_df.iloc[idx_min, 0]))
    # Downloads
    st.download_button("Download CSV", data=total_df.to_csv(index=False).encode("utf-8"), file_name=ft_path.name)

with T2:
    st.subheader("Top Performing Products by Month")
    if not topn_path.exists():
        st.info("File topN tidak ditemukan. Jalankan forecast untuk menghasilkan.")
    else:
        topn_df = pd.read_csv(topn_path)
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
        # Export to Excel
        toexcel = io.BytesIO()
        with pd.ExcelWriter(toexcel, engine='openpyxl') as writer:
            topn_df.to_excel(writer, index=False)
        st.download_button("Export to Excel", data=toexcel.getvalue(), file_name="topN_per_month_24m.xlsx")

with T3:
    st.subheader("Product Details")
    if not per_path.exists():
        st.info("File forecast per product tidak ditemukan.")
    else:
        per_df = pd.read_csv(per_path)
        prod_col = next((c for c in per_df.columns if "product" in c.lower()), per_df.columns[0])
        products = sorted(per_df[prod_col].unique())
        prod = st.selectbox("Pilih Produk", options=products)
        p_df = per_df[per_df[prod_col] == prod]
        fig = generate_product_detail_chart(None, p_df, prod)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Product Data (CSV)", data=p_df.to_csv(index=False).encode("utf-8"), file_name=f"{prod}_forecast.csv")

with T4:
    st.subheader("Training & Forecast Diagnostics")
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        st.dataframe(diag, use_container_width=True)
        figs = generate_diagnostics_charts(diag)
        for key in ["cv_hist", "per_cat", "pie"]:
            if key in figs:
                st.plotly_chart(figs[key], use_container_width=True)
        st.download_button("Download Full Diagnostics (CSV)", data=diag.to_csv(index=False).encode("utf-8"), file_name=diag_path.name)
    else:
        st.info("Diagnostics belum tersedia.")
    if Path(SKIPPED_PRODUCTS_LOG).exists():
        with st.expander("View Skipped Products"):
            st.code(Path(SKIPPED_PRODUCTS_LOG).read_text(encoding="utf-8"))

# Download all results as ZIP
with st.expander("Download Semua Hasil (ZIP)"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for p in [ft_path, topn_path, per_path, diag_path]:
            if p.exists():
                z.write(p, arcname=p.name)
    st.download_button("Download All Results", data=buf.getvalue(), file_name="forecast_results.zip")
