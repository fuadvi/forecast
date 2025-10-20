from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config.settings import (
    UPLOAD_FOLDER,
    DEFAULT_EXCEL,
)

# Caching loaders for performance
@st.cache_data(show_spinner=False)
def load_excel_data(path: str | os.PathLike) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return df
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file Excel: {e}")


def save_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, target_name: Optional[str] = None) -> Path:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix
    filename = target_name if target_name else Path(uploaded_file.name).stem + suffix
    dest = UPLOAD_FOLDER / filename
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


CANONICAL_COLUMNS = {
    "date": [
        "Tanggal Transaksi", "Transaction Date", "date", "Date", "Tanggal",
        "transaction_date", "tanggal_transaksi"
    ],
    "quantity": [
        "Jumlah", "Quantity", "sales", "Qty", "qty", "quantity"
    ],
    "product": [
        "Nama Produk", "Product Name", "product_name", "Product", "Produk"
    ],
    "category": [
        "Kategori Barang", "Product Category", "category", "Kategori", "product_category"
    ],
}


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")


def detect_column_names(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    normalized_cols = { _norm(c): c for c in df.columns }
    result: Dict[str, Optional[str]] = {"date": None, "quantity": None, "product": None, "category": None}
    for key, candidates in CANONICAL_COLUMNS.items():
        for cand in candidates:
            nc = _norm(cand)
            # exact or partial match
            if nc in normalized_cols:
                result[key] = normalized_cols[nc]
                break
            for ncol, orig in normalized_cols.items():
                if nc in ncol or ncol in nc:
                    result[key] = orig
                    break
            if result[key] is not None:
                break
    return result


def validate_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
    mapping = detect_column_names(df)
    status: Dict[str, Dict[str, Optional[str]]] = {}
    for k in ["date", "quantity", "product", "category"]:
        col = mapping.get(k)
        dtype = str(df[col].dtype) if col else None
        status[k] = {"found": bool(col), "name": col, "dtype": dtype}
    return status


def get_data_stats(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    if not mapping:
        m = detect_column_names(df)
        if not all(m.values()):
            return {"n_rows": len(df), "n_products": 0, "date_range": (None, None), "n_cols": df.shape[1]}
        mapping = {
            "date": m["date"],
            "qty": m["quantity"],
            "product": m["product"],
            "category": m["category"],
        }
    dcol, qcol, pcol = mapping["date"], mapping["qty"], mapping["product"]
    n_rows = len(df)
    n_cols = df.shape[1]
    n_products = df[pcol].nunique() if pcol in df.columns else 0
    dates = pd.to_datetime(df[dcol], errors="coerce") if dcol in df.columns else pd.Series(dtype="datetime64[ns]")
    date_min = pd.to_datetime(dates.min()).date().isoformat() if not dates.empty and pd.notna(dates.min()) else None
    date_max = pd.to_datetime(dates.max()).date().isoformat() if not dates.empty and pd.notna(dates.max()) else None
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_products": int(n_products),
        "date_range": (date_min, date_max),
    }


def quality_checks(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, object]:
    dcol, qcol, pcol = mapping["date"], mapping["qty"], mapping["product"]
    res = {}
    # Missing values
    res["missing_values"] = int(df[[dcol, qcol, pcol]].isna().sum().sum())
    # Duplicate rows
    res["duplicate_rows"] = int(df.duplicated().sum())
    # Date validity
    dates = pd.to_datetime(df[dcol], errors="coerce")
    res["invalid_dates"] = int(dates.isna().sum())
    return res
