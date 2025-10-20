import io
from pathlib import Path
import pandas as pd
import streamlit as st

from config.settings import UPLOAD_FOLDER
from utils.data_handler import load_excel_data, save_uploaded_file, detect_column_names, validate_columns, get_data_stats, quality_checks
from utils.validators import validate_file_size, validate_extension

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")
st.title("📤 Upload Data Penjualan")
st.caption("Unggah file Excel (.xlsx) dan lakukan validasi sebelum training.")

# Help text and template download
with st.expander("Lihat Instruksi & Template", expanded=False):
    st.info("Kolom yang dibutuhkan: Date, Quantity, Product, Category (boleh dengan variasi nama). Format tanggal: YYYY-MM-DD.")
    template = pd.DataFrame({
        "Transaction Date": ["2023-01-01", "2023-02-01"],
        "Product Name": ["Produk A", "Produk B"],
        "Quantity": [10, 15],
        "Product Category": ["Kategori 1", "Kategori 2"],
    })
    csv = template.to_csv(index=False).encode("utf-8")
    st.download_button("Unduh Template CSV", data=csv, file_name="template_sales.csv", mime="text/csv")

uploaded = st.file_uploader("Tarik & letakkan file Excel di sini", type=["xlsx"], accept_multiple_files=False,
                            help="Hanya mendukung .xlsx")

if uploaded is not None:
    # Validate size and extension
    if not validate_extension(uploaded.name):
        st.error("❌ Tipe file tidak didukung. Gunakan .xlsx")
        st.stop()
    if not validate_file_size(uploaded.getbuffer()):
        st.error("❌ Ukuran file terlalu besar.")
        st.stop()

    with st.spinner("Membaca file Excel..."):
        df = pd.read_excel(uploaded, engine="openpyxl")

    st.success("✅ File berhasil dibaca. Preview di bawah.")

    # Column validation
    mapping = detect_column_names(df)
    status = validate_columns(df)

    st.subheader("🔎 Validasi Kolom")
    cols = st.columns(4)
    labels = {
        "date": "Date",
        "quantity": "Quantity",
        "product": "Product",
        "category": "Category",
    }
    for i, key in enumerate(["date", "quantity", "product", "category"]):
        with cols[i]:
            if status[key]["found"]:
                st.success(f"✅ {labels[key]}: {status[key]['name']} ({status[key]['dtype']})")
            else:
                st.error(f"❌ {labels[key]}: Tidak ditemukan")

    # Data Preview with pagination
    st.subheader("👀 Preview Data")
    show = st.checkbox("Tampilkan preview", value=True)
    if show:
        total = len(df)
        page_size = st.slider("Rows per page", 10, 100, 20)
        page = st.number_input("Halaman", min_value=1, value=1)
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        st.caption(f"Menampilkan baris {start+1}-{end} dari {total}")
        st.dataframe(df.iloc[start:end])

    # Basic stats
    st.subheader("📊 Statistik Dasar")
    stats = get_data_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows", stats.get("n_rows", 0))
    with c2:
        st.metric("Unique Products", stats.get("n_products", 0))
    with c3:
        dr = stats.get("date_range", (None, None))
        st.metric("Date Range", f"{dr[0] or '-'} s/d {dr[1] or '-'}")
    with c4:
        st.metric("Total Columns", stats.get("n_cols", 0))

    # Quality checks
    st.subheader("🧪 Data Quality Checks")
    if all(status[k]["found"] for k in ["date", "quantity", "product", "category"]):
        mapping2 = {
            "date": mapping["date"],
            "qty": mapping["quantity"],
            "product": mapping["product"],
            "category": mapping["category"],
        }
        qc = quality_checks(df, mapping2)
        q1, q2, q3 = st.columns(3)
        with q1:
            st.write("Missing values:", qc["missing_values"])  # add icons if needed
        with q2:
            st.write("Duplicate rows:", qc["duplicate_rows"])    
        with q3:
            st.write("Invalid dates:", qc["invalid_dates"])    
    else:
        st.warning("⚠️ Tidak semua kolom yang dibutuhkan ditemukan.")

    st.divider()
    disabled = not all(status[k]["found"] for k in ["date", "quantity", "product", "category"])
    if st.button("💾 Simpan Data ke Sistem", type="primary", disabled=disabled):
        dest = save_uploaded_file(uploaded)
        st.session_state["data_uploaded"] = True
        st.session_state["data_file"] = str(dest)
        st.success(f"✅ Data disimpan ke: {dest}")
        st.balloons()
        st.info("Langkah berikutnya: Pergi ke halaman Train Models")
        st.button("🚀 Ke Halaman Training", on_click=lambda: st.switch_page("pages/2_Train_Models.py"))
else:
    st.info("Silakan unggah file Excel (.xlsx). Hindari baris kosong di akhir file dan pastikan kolom numerik hanya berisi angka.")
