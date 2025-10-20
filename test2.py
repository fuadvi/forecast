# Python
import warnings
from typing import List, Tuple
import os

import numpy as np
import pandas as pd
from keras import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

warnings.filterwarnings(action="ignore", category=UserWarning, module="sklearn")

# --- 1) KONFIGURASI YANG DIPERBAIKI ---
FILE_PATH = "Data_Penjualan_Dengan_ID_Pelanggan.xlsx"
# Ubah horizon menjadi bulanan (24 bulan)
FORECAST_WEEKS = 24  # memakai nama variabel lama untuk perubahan minimal
TIME_STEPS = 6  # Dikurangi dari 12 untuk menangani data terbatas
# Batas minimum jumlah bulan dengan transaksi > 0 untuk memakai model
MIN_NONZERO_TRANSACTIONS = 3

# Fast mode to speed up training significantly
FAST_MODE = False  # set to False for highest accuracy
EPOCHS = 30 if FAST_MODE else 100
BATCH_SIZE = 32 if FAST_MODE else 8
# Minimum titik data bulanan yang lebih realistis
MIN_DATA_POINTS = 8  # Dikurangi dari 18 untuk menangani data terbatas
VALIDATION_SPLIT = 0.2 if FAST_MODE else 0.2

# Plot control for fast mode
SKIP_PLOTS_IN_FAST_MODE = False

column_mapping = {
    "Tanggal Transaksi": "date",
    # "Jumlah Unit Terjual": "sales",
    "Jumlah": "sales",
    "Total Harga": "total_price",
    "Kategori Barang": "category",
    "Nama Produk": "product_name",
    "Kota": "city",
    "Jenis Pembelian": "purchase_type",
    "Instansi": "institution",
    "Bahan": "material",
    "Warna": "color",
}

# Helper normalization and fuzzy utilities
import re

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def normalize_product_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_attr_text(val: str) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    # Common normalizations
    replacements = {
        "e katalog": "E-Katalog",
        "e-katalog": "E-Katalog",
        "e. katalog": "E-Katalog",
        "ritel": "Retail",
        "retail": "Retail",
        "powder coating": "Powder Coating",
        "powder-coating": "Powder Coating",
        "aluminium": "Aluminium",
        "alumunium": "Aluminium",
        "aluminum": "Aluminium",
        "hitam": "Hitam",
        "merah": "Merah",
        "biru": "Biru",
    }
    # direct map if available
    if s in replacements:
        return replacements[s]
    # title case as fallback
    return s.title()


def build_fuzzy_canonical_map(names: List[str], threshold: int = 90) -> dict:
    """Build mapping from similar names to a canonical name using simple frequency-based selection.
    Uses rapidfuzz if available; otherwise returns identity mapping.
    """
    if fuzz is None:
        return {n: n for n in names}
    # simple greedy clustering
    names_sorted = sorted(set(names))
    canonical = {}
    for n in names_sorted:
        if n in canonical:
            continue
        canonical[n] = n
        for m in names_sorted:
            if m in canonical and canonical[m] != m:
                continue
            if m == n:
                continue
            try:
                score = fuzz.ratio(n, m)
            except Exception:
                score = 0
            if score >= threshold:
                canonical[m] = n
    # ensure identity for any missing
    for n in names_sorted:
        canonical.setdefault(n, n)
    return canonical


# --- 2) UTILITAS ---
def create_sequences(data: np.ndarray, time_steps: int, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(data) <= time_steps:
        return np.array([]), np.array([])

    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        Xs.append(data[i: (i + time_steps)])
        ys.append(data[i + time_steps, target_idx])
    return np.array(Xs), np.array(ys)


def calculate_trend_and_seasonality(sales: pd.Series, dates: pd.Series) -> dict:
    """Hitung trend linear sederhana dan pola musiman bulanan (1..12)."""
    # Trend linear sederhana pada urutan waktu
    if len(sales) > 1:
        x = np.arange(len(sales))
        try:
            trend_slope = np.polyfit(x, sales.values, 1)[0]
        except:
            trend_slope = 0.0
    else:
        trend_slope = 0.0

    # Seasonality berbasis bulan (1..12)
    seasonal_pattern = None
    try:
        if len(sales) >= 6 and hasattr(dates, 'dt'):  # Dikurangi dari 12
            month_numbers = dates.dt.month.astype(int)
            df_tmp = pd.DataFrame({"month": month_numbers, "sales": sales.values})
            seasonal_pattern = df_tmp.groupby("month")["sales"].mean()
            seasonal_pattern.index = seasonal_pattern.index.astype(int)
    except:
        seasonal_pattern = None

    return {
        "trend_slope": float(trend_slope),
        "seasonal_pattern": seasonal_pattern,
    }


# Stabilization utilities for post-forecast smoothing and MoM limiting

# --- Hybrid helpers: SMA baseline, RFE, metrics ---
def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _select_sma_window(series: pd.Series, windows=(3, 4, 5, 6)) -> int:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if len(s) < 6:
        return 3
    best_w, best_err = 3, float('inf')
    for w in windows:
        if w >= len(s):
            continue
        sma = s.rolling(window=w, min_periods=1).mean()
        # validate on last 6 points or half
        val_n = max(3, min(6, len(s) // 2))
        err = _smape(s[-val_n:], sma[-val_n:])
        if err < best_err:
            best_err, best_w = err, w
    return best_w


def build_baseline_series(sales: pd.Series, dates: pd.Series) -> Tuple[pd.Series, dict]:
    s = pd.to_numeric(sales, errors="coerce").fillna(0.0).astype(float)
    w = _select_sma_window(s)
    level = s.rolling(window=w, min_periods=1).mean()
    seasonal = pd.Series(0.0, index=s.index)
    meta = {"sma_window": w, "seasonal_active": False}
    if len(s) >= 12:
        month_idx = dates.dt.month.astype(int)
        month_means = s.groupby(month_idx).transform('mean')
        # center seasonal by removing overall mean effect
        overall = float(s.mean()) if float(s.mean()) != 0 else 1.0
        seasonal = (month_means - overall)
        meta["seasonal_active"] = True
    baseline = level + seasonal
    return baseline.clip(lower=0.0), meta


def forecast_baseline_forward(last_hist_date: pd.Timestamp, hist_sales: pd.Series, hist_dates: pd.Series, horizon: int,
                              meta: dict = None) -> pd.Series:
    baseline_hist, meta2 = build_baseline_series(hist_sales, hist_dates)
    if meta is None:
        meta = meta2
    w = meta.get("sma_window", 3)
    level_hist = pd.to_numeric(hist_sales, errors="coerce").fillna(0.0).astype(float).rolling(window=w,
                                                                                              min_periods=1).mean()
    seasonal_active = meta.get("seasonal_active", len(hist_sales) >= 12)
    future_dates = pd.date_range(start=last_hist_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    baseline_future = []
    history_vals = hist_sales.astype(float).tolist()
    for i, d in enumerate(future_dates):
        # level via last SMA value
        lv = float(np.mean(history_vals[-w:])) if history_vals else 0.0
        seas = 0.0
        if seasonal_active:
            month = int(d.month)
            # compute seasonal offset as mean for that month minus overall mean from history
            month_mask = hist_dates.dt.month.astype(int) == month
            if month_mask.any():
                month_mean = float(hist_sales[month_mask].mean())
                overall = float(hist_sales.mean()) if float(hist_sales.mean()) != 0 else 1.0
                seas = month_mean - overall
        baseline_future.append(max(0.0, lv + seas))
        # append naive baseline as proxy to propagate level
        history_vals.append(lv)
    return pd.Series(baseline_future, index=future_dates)


def rfe_select_features(X: pd.DataFrame, y: pd.Series, min_features: int = 2, max_features: int = 6,
                        random_state: int = 42) -> Tuple[List[str], List[str], List[dict]]:
    # Simple RFE using RF importance; X columns assumed numeric-encoded already
    rng = np.random.RandomState(random_state)
    candidates = list(X.columns)
    elimination_order = []
    history = []
    best_subset = candidates.copy()
    best_err = float('inf')

    while len(candidates) > min_features:
        # Train RF
        try:
            rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
            rf.fit(X[candidates], y)
            importances = pd.Series(rf.feature_importances_, index=candidates)
        except Exception:
            # fallback: keep time features
            time_feats = [c for c in candidates if c in ("month_sin", "month_cos", "trend")]
            return time_feats[:max(min_features, len(time_feats))], elimination_order, history
        # eliminate weakest
        weakest = importances.idxmin()
        elimination_order.append(weakest)
        # evaluate error after removing weakest via simple CV (last 20%)
        cand_try = [c for c in candidates if c != weakest]
        split = max(1, int(len(X) * 0.8))
        X_tr, X_va = X[cand_try].iloc[:split], X[cand_try].iloc[split:]
        y_tr, y_va = y.iloc[:split], y.iloc[split:]
        if len(X_va) >= 1:
            rf2 = RandomForestRegressor(n_estimators=200, random_state=random_state)
            rf2.fit(X_tr, y_tr)
            pred = rf2.predict(X_va)
            err = _smape(y_va.values, pred)
        else:
            err = float('inf')
        history.append({"removed": weakest, "err": err, "features_left": len(cand_try)})
        if err <= best_err and len(cand_try) <= max_features:
            best_err, best_subset = err, cand_try.copy()
        candidates = cand_try
        if len(candidates) <= max_features:
            break
    if len(best_subset) < min_features:
        best_subset = candidates[:min_features]
    return best_subset, elimination_order, history


def _compute_hist_cv(series: pd.Series) -> float:
    try:
        s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        s = s[s > 0]
        if len(s) < 2:
            return 0.0
        m = float(s.mean())
        if m <= 0:
            return 0.0
        sd = float(s.std(ddof=1))
        return float(sd / m)
    except Exception:
        return 0.0


def stabilize_forecast(values: List[float], floor_val: float, cap_val: float, hist_cv: float) -> tuple:
    """Stabilize sequence by limiting month-over-month change based on historical CV
    and blending with EMA smoothing. Returns (stabilized_values, applied_mom_limit, smoothing_applied).
    """
    if not values:
        return values, False, False

    # Determine allowable relative change bound from volatility
    # Base bound 30% plus up to 30% from CV, capped at 60%
    rel_bound = min(0.60, 0.30 + max(0.0, float(hist_cv)))

    applied_limit = False

    stabilized = [float(max(floor_val, min(cap_val, values[0])))]
    for i in range(1, len(values)):
        prev = stabilized[-1]
        lo = prev * (1.0 - rel_bound)
        hi = prev * (1.0 + rel_bound)
        v = float(values[i])
        v_clamped = max(lo, min(hi, v))
        if abs(v_clamped - v) > 1e-12:
            applied_limit = True
        # Ensure within global floor/cap
        v_clamped = float(max(floor_val, min(cap_val, v_clamped)))
        stabilized.append(v_clamped)

    # EMA smoothing blend to reduce zig-zag but preserve level
    smoothing_applied = False
    if len(stabilized) >= 3:
        alpha = 0.5  # moderate smoothing
        ema = []
        for i, v in enumerate(stabilized):
            if i == 0:
                ema.append(v)
            else:
                ema.append(alpha * v + (1 - alpha) * ema[-1])
        # Blend original stabilized with EMA to keep some dynamics
        blended = [0.7 * s + 0.3 * e for s, e in zip(stabilized, ema)]
        # Re-apply floor/cap to be safe
        blended = [float(max(floor_val, min(cap_val, b))) for b in blended]
        smoothing_applied = True
        stabilized = blended

    return stabilized, applied_limit, smoothing_applied


# --- 3) BACA & PREPROCESS DATA ---
try:
    df = pd.read_excel(FILE_PATH)
    print(f"Data berhasil dibaca: {len(df)} baris")
except Exception as e:
    raise ValueError(f"Gagal membaca file {FILE_PATH}: {e}")

existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
df = df.rename(columns=existing_columns)
print(f"Kolom yang ditemukan: {list(existing_columns.values())}")

if "product_name" not in df.columns:
    raise ValueError("Kolom 'Nama Produk' tidak ditemukan di file sumber.")

if "date" not in df.columns:
    raise ValueError("Kolom tanggal tidak ditemukan setelah mapping.")

# Simpan tanggal mentah sebelum konversi untuk logging invalid dates
df["date_raw"] = df["date"].astype(str)

# Konversi tipe data dengan error handling yang lebih baik
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["sales"] = pd.to_numeric(df.get("sales", pd.Series(dtype=float)), errors="coerce")

# Log baris dengan tanggal invalid
invalid_mask = df["date"].isna()
if invalid_mask.any():
    invalid_rows = df.loc[invalid_mask].copy()
    cols_to_save = [c for c in ["date_raw", "product_name", "sales", "category", "purchase_type", "material", "color",
                                "institution", "city"] if c in invalid_rows.columns]
    try:
        invalid_rows[cols_to_save].to_csv("invalid_dates.csv", index=False)
        print(f"Tersimpan {len(invalid_rows)} baris dengan tanggal invalid ke invalid_dates.csv")
    except Exception as e:
        print(f"Gagal menulis invalid_dates.csv: {e}")

# Bersihkan data
initial_rows = len(df)
df = df.dropna(subset=["date", "product_name"])
df["sales"] = df["sales"].fillna(0)
df = df.sort_values("date")

if len(df) == 0:
    raise ValueError("Tidak ada data yang valid setelah pembersihan")

# Standarisasi nama produk yang lebih kuat
df["product_name"] = df["product_name"].apply(normalize_product_name)

# Opsional: fuzzy merge nama produk serupa (>0.9)
try:
    name_map = build_fuzzy_canonical_map(df["product_name"].unique().tolist(), threshold=90)
    df["product_name"] = df["product_name"].map(name_map)
except Exception as e:
    print(f"Lewati fuzzy merge nama produk: {e}")

unique_products = df["product_name"].unique()
print(f"Data setelah pembersihan: {len(df)} baris (dari {initial_rows})")
print(f"Jumlah produk unik: {len(unique_products)})")

# Global last date to align all product forecasts
GLOBAL_LAST_DATE = df["date"].max()
print(f"Tanggal terakhir data: {GLOBAL_LAST_DATE}")

# --- 4) AGREGASI BULANAN PER PRODUK ---
attr_cols = [col for col in ["purchase_type", "material", "color", "category", "city", "institution"] if
             col in df.columns]

# Normalisasi teks atribut untuk konsistensi
for col in attr_cols:
    df[col] = df[col].apply(normalize_attr_text)

# Hitung statistik kategori untuk clamp baseline/floor bila tersedia
category_stats = {}
if "category" in df.columns:
    try:
        temp = df.copy()
        temp_nonzero = temp[temp["sales"] > 0]
        if not temp_nonzero.empty:
            for cat, sub in temp_nonzero.groupby("category"):
                sales_nonzero = sub["sales"].astype(float)
                category_stats[cat] = {
                    "q10": float(sales_nonzero.quantile(0.10)),
                    "q90": float(sales_nonzero.quantile(0.90)),
                    "median": float(sales_nonzero.median()),
                    "mean": float(sales_nonzero.mean()),
                }
    except Exception as e:
        print(f"Gagal menghitung statistik kategori: {e}")

# Global stats
_nonzero_global = df.loc[df["sales"] > 0, "sales"].astype(float)
GLOBAL_STATS = {
    "q10": float(_nonzero_global.quantile(0.10)) if not _nonzero_global.empty else 0.0,
    "q90": float(_nonzero_global.quantile(0.90)) if not _nonzero_global.empty else 0.0,
    "median": float(_nonzero_global.median()) if not _nonzero_global.empty else 0.0,
    "mean": float(_nonzero_global.mean()) if not _nonzero_global.empty else 0.0,
}


def _most_common_nonnull(series: pd.Series) -> str:
    try:
        s = series.dropna().astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            return ""
        modes = s.mode()
        return str(modes.iloc[0]) if not modes.empty else ""
    except:
        return ""


attr_map = {}
for prod in unique_products:
    sub = df[df["product_name"] == prod]
    attr_map[prod] = {col: _most_common_nonnull(sub[col]) for col in attr_cols}

results_list: List[pd.DataFrame] = []
skipped_products = []
diagnostics_records: List[dict] = []

print(f"Memulai peramalan {len(unique_products)} produk untuk {FORECAST_WEEKS} bulan ke depan...")

for product in tqdm(unique_products, desc="Forecast per product"):
    try:
        product_df = df[df["product_name"] == product].copy()

        # Agregasi bulanan
        weekly = product_df.groupby(pd.Grouper(key="date", freq="MS"))["sales"].sum().reset_index()
        weekly.columns = ["date", "sales_raw"]

        print(f"Produk '{product}': {len(weekly)} titik data bulanan")

        # Skip produk dengan data terlalu sedikit
        if len(weekly) < MIN_DATA_POINTS:
            skipped_products.append(f"{product} (hanya {len(weekly)} bulan data)")
            # Meskipun titik bulan sedikit, tetap cek transaksi non-zero untuk fallback konservatif
            # Lanjut ke pengecekan low-data di bawah.

        # Jika jumlah bulan dengan transaksi > 0 sangat sedikit, jangan latih model LSTM.
        nonzero_count = int((weekly["sales_raw"] > 0).sum())
        if nonzero_count < MIN_NONZERO_TRANSACTIONS:
            # Fallback konservatif namun tidak datar sepenuhnya
            nz = weekly.loc[weekly["sales_raw"] > 0, "sales_raw"].astype(float)
            prod_q10 = float(nz.quantile(0.10)) if len(nz) > 0 else 0.0
            prod_q90 = float(nz.quantile(0.90)) if len(nz) > 0 else 0.0
            prod_median = float(nz.median()) if len(nz) > 0 else 0.0
            # Kategori clamp jika ada
            cat = None
            if "category" in df.columns:
                try:
                    cat = df.loc[df["product_name"] == product, "category"].dropna().astype(str)
                    cat = cat.mode().iloc[0] if not cat.empty else None
                except Exception:
                    cat = None
            cat_stats = category_stats.get(cat, {}) if cat is not None else {}
            base_q10 = prod_q10 if prod_q10 > 0 else float(cat_stats.get("q10", GLOBAL_STATS["q10"]))
            base_q90 = prod_q90 if prod_q90 > 0 else float(cat_stats.get("q90", GLOBAL_STATS["q90"]))
            base_med = prod_median if prod_median > 0 else float(cat_stats.get("median", GLOBAL_STATS["median"]))
            # baseline awal pakai median
            baseline = base_med if base_med > 0 else (base_q10 + base_q90) / 2.0
            # clamp ke [q10, q90]
            q10c, q90c = base_q10, base_q90
            if q90c < q10c:
                q90c = q10c
            baseline = float(np.clip(baseline, q10c if q10c > 0 else 1e-6,
                                     q90c if q90c > 0 else baseline * 1.2 if baseline > 0 else 1.0))
            # Buat tapering 2-5% untuk hindai baseline konstan
            rng = np.random.RandomState(abs(hash(product)) % (2 ** 32))
            amps = rng.uniform(0.02, 0.05)
            future_dates = pd.date_range(
                start=GLOBAL_LAST_DATE + pd.offsets.MonthBegin(1),
                periods=FORECAST_WEEKS,
                freq="MS",
            )
            taper = 1.0 + amps * np.sin(2 * np.pi * (np.arange(FORECAST_WEEKS) / 12.0))
            mean_preds = (baseline * taper)
            # Simple uncertainty: +/- 15% band around mean using product/category variability
            var_scale = max(0.10,
                            min(0.25, float(_compute_hist_cv(weekly['sales_raw'])) if 'weekly' in locals() else 0.15))
            p10 = mean_preds * (1.0 - var_scale)
            p50 = mean_preds
            p90 = mean_preds * (1.0 + var_scale)
            # QC non-negatif
            mean_preds = np.clip(mean_preds, 1e-6, None)
            p10 = np.clip(p10, 1e-6, None)
            p50 = np.clip(p50, 1e-6, None)
            p90 = np.clip(p90, 1e-6, None)
            attrs = attr_map.get(product, {})
            temp_df = pd.DataFrame({
                "date": future_dates,
                "product_name": product,
                "forecast_mean": mean_preds,
                "forecast_p10": p10,
                "forecast_p50": p50,
                "forecast_p90": p90,
                "purchase_type": [attrs.get("purchase_type", "")] * FORECAST_WEEKS,
                "material": [attrs.get("material", "")] * FORECAST_WEEKS,
                "color": [attrs.get("color", "")] * FORECAST_WEEKS,
                "category": [attrs.get("category", "")] * FORECAST_WEEKS,
                "city": [attrs.get("city", "")] * FORECAST_WEEKS,
                "institution": [attrs.get("institution", "")] * FORECAST_WEEKS,
                "note": [f"fallback_low-data: nz={nonzero_count}; clamp+taper"] * FORECAST_WEEKS,
            })
            results_list.append(temp_df)
            skipped_products.append(
                f"{product} (low-data: hanya {nonzero_count} transaksi non-zero, pakai baseline clamp+taper)")
            continue

        # --- 5) PREPROCESSING YANG DIPERBAIKI ---
        # Handle missing values dan outliers dengan lebih hati-hati
        sales_series = weekly["sales_raw"].copy()

        # Fill zeros dengan interpolasi jika memungkinkan
        if len(sales_series[sales_series > 0]) >= 2:
            sales_series = sales_series.replace(0, np.nan)
            sales_series = sales_series.interpolate(method="linear", limit_direction="both")

        # Fill remaining NaN dengan median
        nonzero_vals = sales_series.dropna()
        if not nonzero_vals.empty:
            fill_value = nonzero_vals.median()
        else:
            fill_value = 1.0  # Fallback value
        sales_series = sales_series.fillna(fill_value)

        # Outlier handling yang lebih konservatif
        if len(sales_series) >= 4:
            Q1 = sales_series.quantile(0.25)
            Q3 = sales_series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                upper_bound = Q3 + 2.0 * IQR  # Lebih konservatif
                lower_bound = max(0, Q1 - 2.0 * IQR)
                sales_series = sales_series.clip(lower=lower_bound, upper=upper_bound)

        # Smoothing
        alpha = 0.4
        weekly["sales"] = sales_series.ewm(alpha=alpha, adjust=False).mean()
        weekly["sales"] = weekly["sales"].clip(lower=0)

        # --- 6) HYBRID FEATURE ENGINEERING (SMA baseline + residual LSTM) ---
        # Time features
        weekly["month"] = weekly["date"].dt.month
        weekly["month_sin"] = np.sin(2 * np.pi * weekly["month"] / 12)
        weekly["month_cos"] = np.cos(2 * np.pi * weekly["month"] / 12)
        weekly["trend"] = range(len(weekly))
        if len(weekly) < 12:
            weekly["month_sin"] = 0.0
            weekly["month_cos"] = 0.0

        # Build baseline and residual
        baseline_hist, baseline_meta = build_baseline_series(weekly["sales"], weekly["date"])
        weekly["residual"] = weekly["sales"].astype(float) - baseline_hist.astype(float)

        # Candidate exogenous (mostly constant per SKU, RFE expected to remove if unhelpful)
        cand_feats = ["month_sin", "month_cos", "trend"]
        for col in ["purchase_type", "material", "color", "category", "city", "institution"]:
            if col in attr_cols:
                val = attr_map.get(product, {}).get(col, "")
                # encode as simple binary (presence of dominant attr); time-constant
                weekly[f"{col}_bin"] = 1.0 if val != "" else 0.0
                cand_feats.append(f"{col}_bin")

        # RFE selection on residuals using RF
        X_rfe = weekly[cand_feats].copy()
        y_rfe = weekly["residual"].astype(float)
        selected_feats, elim_order, rfe_hist = rfe_select_features(X_rfe, y_rfe, min_features=2, max_features=6)
        rfe_note = f"RFE_keep={selected_feats}; removed={elim_order}"

        # Prepare sequences for LSTM on residuals
        features_for_lstm = ["residual"] + selected_feats
        data_matrix = weekly[features_for_lstm].astype(float).copy()

        # Scaling
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        data_scaled = scaler.fit_transform(data_matrix.values)

        # sequences with target_idx=0 (residual)
        target_col_index = 0
        X_seq, y_seq = create_sequences(data_scaled, TIME_STEPS, target_col_index)
        if len(X_seq) < TIME_STEPS or len(X_seq) < 4:
            skipped_products.append(f"{product} (sequences residual terlalu sedikit)")
            continue

        split_idx = max(1, int(len(X_seq) * (1 - VALIDATION_SPLIT)))
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_val = X_seq[split_idx:] if split_idx < len(X_seq) else X_seq[-1:]
        y_val = y_seq[split_idx:] if split_idx < len(y_seq) else y_seq[-1:]

        # LSTM ringan untuk residual
        model = Sequential([
            Input(shape=(TIME_STEPS, len(features_for_lstm))),
            LSTM(16, return_sequences=False, activation="tanh"),
            Dropout(0.2),
            Dense(8, activation="relu"),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer="adam", loss="mse")
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
        if len(X_val) > 0:
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=min(8, len(X_train)),
                      callbacks=[early_stopping], verbose=0)
        else:
            model.fit(X_train, y_train, epochs=30, batch_size=min(8, len(X_train)), callbacks=[early_stopping],
                      verbose=0)

        # In-sample residual pred to derive noise for quantiles
        ins_pred = model.predict(X_train, verbose=0).flatten()
        ins_true = y_train.flatten()
        resid_err = (ins_true - ins_pred)
        if len(resid_err) == 0:
            resid_err = np.array([0.0])

        # Forecast horizon
        future_dates = pd.date_range(start=GLOBAL_LAST_DATE + pd.offsets.MonthBegin(1), periods=FORECAST_WEEKS,
                                     freq="MS")

        # Baseline forward
        baseline_future = forecast_baseline_forward(GLOBAL_LAST_DATE, weekly["sales"], weekly["date"], FORECAST_WEEKS,
                                                    baseline_meta)

        # Iterative residual forecast
        last_window_scaled = list(X_seq[-1])
        residual_forecast = []
        for i, d in enumerate(future_dates):
            try:
                window_array = np.array(last_window_scaled).reshape(1, TIME_STEPS, len(features_for_lstm))
                pred_scaled = model.predict(window_array, verbose=0)[0][0]
                # inverse scale for residual (index 0)
                if hasattr(scaler, "scale_") and scaler.scale_[0] != 0:
                    pred_resid = (pred_scaled - scaler.min_[0]) / scaler.scale_[0]
                else:
                    pred_resid = float(pred_scaled)
                residual_forecast.append(float(pred_resid))

                # build next feature row (time features + constant bins)
                new_vals = {"residual": pred_resid}
                new_vals["month_sin"] = np.sin(2 * np.pi * d.month / 12) if len(weekly) >= 12 else 0.0
                new_vals["month_cos"] = np.cos(2 * np.pi * d.month / 12) if len(weekly) >= 12 else 0.0
                new_vals["trend"] = len(weekly) + i + 1
                for col in ["purchase_type", "material", "color", "category", "city", "institution"]:
                    if f"{col}_bin" in selected_feats:
                        new_vals[f"{col}_bin"] = 1.0
                # align order of selected features
                new_row_unscaled = [new_vals.get(col, 0.0) for col in features_for_lstm]
                new_scaled = scaler.transform([new_row_unscaled])[0]
                last_window_scaled.pop(0)
                last_window_scaled.append(new_scaled)
            except Exception as e:
                print(f"Error forecasting residual for {product} at step {i}: {e}")
                residual_forecast.append(0.0)

        # Combine baseline + residual for mean forecast
        point_forecast = (baseline_future.values + np.array(residual_forecast))
        point_forecast = np.maximum(point_forecast, 1e-6)

        # Uncertainty via bootstrap of residual errors
        rng = np.random.RandomState(abs(hash(product)) % (2 ** 32))
        samples = 300 if not FAST_MODE else 100
        draws = rng.choice(resid_err, size=(samples, len(future_dates)), replace=True)
        sim = point_forecast + draws
        sim = np.clip(sim, 1e-6, None)
        p10 = np.percentile(sim, 10, axis=0)
        p50 = np.percentile(sim, 50, axis=0)
        p90 = np.percentile(sim, 90, axis=0)

        # QC clamp and stabilization on mean forecast
        hist_nonzero = weekly.loc[weekly["sales"] > 0, "sales"].astype(float)
        hist_mean = float(hist_nonzero.mean()) if not hist_nonzero.empty else float(weekly["sales"].mean())
        hist_q10 = float(hist_nonzero.quantile(0.10)) if not hist_nonzero.empty else 0.0
        hist_q90 = float(hist_nonzero.quantile(0.90)) if not hist_nonzero.empty else hist_mean * 1.2
        hist_median = float(hist_nonzero.median()) if not hist_nonzero.empty else hist_mean

        floor_val = max(hist_q10 * 0.8, 1e-6,
                        0.005 * (hist_median if hist_median > 0 else GLOBAL_STATS.get("median", 0.0)))
        cap_val = min(hist_mean * 3.0 if hist_mean > 0 else (GLOBAL_STATS.get("mean", 1.0) * 3.0),
                      hist_q90 * 1.2 if hist_q90 > 0 else (GLOBAL_STATS.get("q90", 1.0) * 1.2))
        if cap_val < floor_val:
            cap_val = floor_val * 1.2

        clamped_mean = [min(max(float(p), floor_val), cap_val) for p in point_forecast]
        hist_cv = _compute_hist_cv(weekly["sales"].astype(float))
        stabilized_mean, applied_mom_limit, smoothing_applied = stabilize_forecast(clamped_mean, floor_val, cap_val,
                                                                                   hist_cv)
        stabilized_mean = np.array(stabilized_mean)

        # Apply same ratio to quantiles for consistency
        ratio = stabilized_mean / np.maximum(point_forecast, 1e-6)
        p10_adj = (p10 * ratio)
        p50_adj = (p50 * ratio)
        p90_adj = (p90 * ratio)
        p10_adj = np.clip(p10_adj, 1e-6, None)
        p50_adj = np.clip(p50_adj, 1e-6, None)
        p90_adj = np.clip(p90_adj, 1e-6, None)

        # Material-risk flag
        attrs = attr_map.get(product, {})
        mat = (attrs.get("material", "") or "").lower()
        risk_material = any(k in mat for k in ["roda", "roda besi", "selongsong", "besi", "aluminium", "alum"])
        sharp_increase = False
        if len(stabilized_mean) >= 2:
            inc = (stabilized_mean[1:] - stabilized_mean[:-1]) / np.maximum(stabilized_mean[:-1], 1e-6)
            sharp_increase = np.any(inc > min(0.6, 0.3 + hist_cv))
        note_flags = []
        if risk_material and sharp_increase:
            note_flags.append("material-risk")

        note = ("hybrid_SMA+LSTM; " + rfe_note + ("; " + ",".join(note_flags) if note_flags else ""))

        # --- 12) HASIL PER PRODUK (mean + quantiles) ---
        temp_df = pd.DataFrame({
            "date": future_dates,
            "product_name": product,
            "forecast_mean": stabilized_mean,
            "forecast_p10": p10_adj,
            "forecast_p50": p50_adj,
            "forecast_p90": p90_adj,
            "purchase_type": [attrs.get("purchase_type", "")] * len(future_dates),
            "material": [attrs.get("material", "")] * len(future_dates),
            "color": [attrs.get("color", "")] * len(future_dates),
            "category": [attrs.get("category", "")] * len(future_dates),
            "city": [attrs.get("city", "")] * len(future_dates),
            "institution": [attrs.get("institution", "")] * len(future_dates),
            "note": [note] * len(future_dates),
        })
        results_list.append(temp_df)

        diagnostics_records.append({
            "product_name": product,
            "hist_cv": float(hist_cv),
            "floor": float(floor_val),
            "cap": float(cap_val),
            "applied_mom_limit": bool(applied_mom_limit),
            "smoothing_applied": bool(smoothing_applied),
            "nonzero_hist_points": int(len(hist_nonzero)),
            "hist_mean": float(hist_mean),
            "hist_q10": float(hist_q10),
            "hist_q90": float(hist_q90),
            "hist_median": float(hist_median),
            "rfe_keep": ",".join(selected_feats),
            "rfe_removed_order": ",".join(elim_order),
        })

    except Exception as e:
        print(f"Error processing product {product}: {e}")
        skipped_products.append(f"{product} (error: {str(e)[:50]})")
        continue

# --- 13) HASIL AKHIR ---
print(f"\nProduk yang diproses: {len(results_list)}")
print(f"Produk yang dilewati: {len(skipped_products)}")

if skipped_products:
    print("\nProduk yang dilewati (maks 20):")
    for skip_reason in skipped_products[:20]:
        print(f"  - {skip_reason}")
    if len(skipped_products) > 20:
        print(f"  ... dan {len(skipped_products) - 20} lainnya")
    # Simpan ke file
    try:
        pd.DataFrame({"skipped_product": skipped_products}).to_csv("skipped_products.csv", index=False)
        print("Daftar produk yang dilewati disimpan ke skipped_products.csv")
    except Exception as e:
        print(f"Gagal menyimpan skipped_products.csv: {e}")

if not results_list:
    print("\n" + "=" * 50)
    print("SOLUSI UNTUK MASALAH INI:")
    print("=" * 50)
    print("1. Periksa format tanggal dalam file Excel")
    print("2. Pastikan ada kolom 'Jumlah' atau 'Jumlah Unit Terjual'")
    print("3. Pastikan data memiliki minimal 8 bulan riwayat per produk")
    print("4. Periksa apakah ada data penjualan (bukan semua 0)")
    print("=" * 50)

    # Buat data dummy untuk demonstrasi
    print("Membuat contoh forecast dengan data dummy...")
    future_dates = pd.date_range(
        start=pd.Timestamp.now().replace(day=1),
        periods=FORECAST_WEEKS,
        freq="MS"
    )

    dummy_df = pd.DataFrame({
        "date": future_dates,
        "product_name": "sample_product",
        "forecasted_sales": np.random.uniform(10, 100, FORECAST_WEEKS),
        "purchase_type": "",
        "material": "",
        "color": "",
    })
    results_list.append(dummy_df)

final_df = pd.concat(results_list, ignore_index=True)
final_df = final_df.sort_values(["product_name", "date"]).reset_index(drop=True)

# Save per-product forecasts (24 months) with quantiles
per_product_csv = "forecast_per_product_24m.csv"
final_df.to_csv(per_product_csv, index=False)

# Total forecast per month (sum of all SKUs)
agg = (final_df.groupby("date", as_index=False)
       .agg({
    "forecast_mean": "sum",
    "forecast_p10": "sum",
    "forecast_p50": "sum",
    "forecast_p90": "sum",
})
       .sort_values("date"))
agg_csv = "forecast_total_24m.csv"
agg.to_csv(agg_csv, index=False)

# Top-5 SKU per month by forecast_mean
ranked = final_df.sort_values(["date", "forecast_mean"], ascending=[True, False])
ranked["rank"] = ranked.groupby("date").cumcount() + 1
topN = ranked[ranked["rank"] <= 5].copy()
topN_csv = "topN_per_month_24m.csv"
topN.to_csv(topN_csv, index=False)

# Save diagnostics if available
if diagnostics_records:
    try:
        pd.DataFrame(diagnostics_records).to_csv("forecast_diagnostics.csv", index=False)
        print("Diagnostics per-produk disimpan ke forecast_diagnostics.csv")
    except Exception as e:
        print(f"Gagal menyimpan forecast_diagnostics.csv: {e}")

print(f"\nSelesai! File disimpan:")
print(f" - {per_product_csv}")
print(f" - {agg_csv}")
print(f" - {topN_csv}")
print(f"Total produk yang berhasil diforecast: {len(final_df['product_name'].unique())}")
print(f"Range forecast_mean: {final_df['forecast_mean'].min():.2f} - {final_df['forecast_mean'].max():.2f}")

# === Monthly Top-5 Charts ===

def generate_monthly_top5_charts(
    topN_df: pd.DataFrame = None,
    csv_path: str = "topN_per_month_24m.csv",
    value_col: str = "forecast_mean",
    output_dir: str = None,
    palette_name: str = "tab10",
):
    """
    Generate one bar chart per month showing up to top-5 products by forecast value.
    - topN_df: optional preloaded DataFrame. If None, will read from csv_path.
    - value_col: column name for the value to plot (e.g., "forecast_mean" or "forecast_p50").
    - output_dir: directory to save charts (default: forecast_plots/bulan).
    - palette_name: matplotlib colormap name for consistent colors across months.
    """
    try:
        if topN_df is None:
            if csv_path is None or not os.path.exists(csv_path):
                print(f"Top-N CSV tidak ditemukan: {csv_path}")
                return
            topN_df = pd.read_csv(csv_path)
        # Ensure date type
        if not np.issubdtype(topN_df["date"].dtype, np.datetime64):
            topN_df["date"] = pd.to_datetime(topN_df["date"], errors="coerce")
        # Validate columns
        if "product_name" not in topN_df.columns:
            print("Kolom 'product_name' tidak ditemukan pada data Top-N.")
            return
        if value_col not in topN_df.columns:
            print(f"Kolom nilai '{value_col}' tidak ditemukan. Gunakan value_col yang benar.")
            return
        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "forecast_plots", "bulan")
        os.makedirs(output_dir, exist_ok=True)
        # Consistent palette
        try:
            cmap = plt.get_cmap(palette_name)
            base_colors = cmap.colors if hasattr(cmap, "colors") else [cmap(i) for i in np.linspace(0, 1, 5)]
        except Exception:
            base_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
        # Iterate per month
        month_groups = topN_df.dropna(subset=["date"]).groupby(pd.Grouper(key="date", freq="MS"))
        for month, sub in month_groups:
            if sub.empty:
                continue
            # pick top 5 by value_col (already Top-N is fine, but re-ensure ordering)
            sub2 = sub.sort_values(value_col, ascending=False).head(5).copy()
            # X labels are product names
            x_labels = sub2["product_name"].astype(str).tolist()
            y_vals = pd.to_numeric(sub2[value_col], errors="coerce").fillna(0.0).values
            n = len(x_labels)
            colors = list(base_colors)[:n]
            # Create bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(n), y_vals, color=colors)
            plt.xticks(range(n), x_labels, rotation=30, ha="right")
            plt.ylabel("Nilai Forecast")
            # Title with month-year
            if pd.isna(month):
                title_txt = "Top-5 Produk per Bulan"
                fname = "top5_unknown.png"
            else:
                title_txt = month.strftime("Top-5 Produk Tertinggi - %B %Y")
                fname = f"top5_{month.strftime('%Y_%m')}.png"
            plt.title(title_txt)
            plt.grid(axis="y", alpha=0.2)
            # Add value labels on top of bars
            for i, b in enumerate(bars):
                val = float(y_vals[i])
                plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{val:,.0f}",
                         ha="center", va="bottom", fontsize=9, rotation=0)
            plt.tight_layout()
            out_path = os.path.join(output_dir, fname)
            try:
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                print(f"Gagal menyimpan chart bulan {month}: {e}")
            plt.close()
        print(f"Chart Top-5 per bulan disimpan di: {output_dir}")
    except Exception as e:
        print(f"Gagal membuat chart Top-5 per bulan: {e}")


def generate_grouped_top5_24m_chart(
    topN_df: pd.DataFrame = None,
    csv_path: str = "topN_per_month_24m.csv",
    value_col: str = "forecast_mean",
    output_path: str = None,
    palette_name: str = "tab20",
    width: float = 0.15,
    annotate_only_max: bool = True,
):
    """
    Create a single grouped bar chart covering 24 months.
    - Up to Top-5 products per month (sorted high->low within month).
    - Consistent color per product across all months.
    - X-axis labels as MMM-YYYY with rotation.
    - Light Y-grid, tight layout, save one PNG.
    """
    try:
        # Load data if needed
        if topN_df is None:
            if csv_path is None or not os.path.exists(csv_path):
                print(f"Top-N CSV tidak ditemukan: {csv_path}")
                return
            topN_df = pd.read_csv(csv_path)
        if topN_df.empty:
            print("Data Top-N kosong.")
            return
        # Ensure correct dtypes
        if not np.issubdtype(topN_df["date"].dtype, np.datetime64):
            topN_df["date"] = pd.to_datetime(topN_df["date"], errors="coerce")
        topN_df = topN_df.dropna(subset=["date", "product_name"]).copy()
        # Normalize to month start (convert to monthly period, then to timestamp at period start)
        topN_df["date"] = topN_df["date"].dt.to_period("M").dt.to_timestamp()
        if value_col not in topN_df.columns:
            print(f"Kolom nilai '{value_col}' tidak ditemukan. Gunakan value_col yang benar.")
            return
        topN_df[value_col] = pd.to_numeric(topN_df[value_col], errors="coerce").fillna(0.0)
        # Determine 24 chronological months
        months = sorted(topN_df["date"].unique())
        # If more than 24 (safety), keep first 24 chronologically
        if len(months) > 24:
            months = months[:24]
        # If less than 24, keep whatever exists but still plot
        # Determine products that appear at least once in Top-5 in these months
        filtered = topN_df[topN_df["date"].isin(months)].copy()
        # Re-rank within each month to ensure top-5 by chosen value
        filtered = (filtered.sort_values(["date", value_col], ascending=[True, False])
                             .groupby("date", as_index=False).head(5))
        products_in_legend = filtered["product_name"].unique().tolist()
        # Build color map (stable per product)
        try:
            cmap = plt.get_cmap(palette_name)
            # sample enough colors for all products; fall back to cycling
            base_colors = list(getattr(cmap, "colors", []))
            if not base_colors:
                base_colors = [cmap(i) for i in np.linspace(0, 1, max(5, len(products_in_legend)))]
        except Exception:
            base_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                           "#9c755f", "#bab0ab"]
        # Deterministic assignment using sorted product names
        prods_sorted = sorted(set(products_in_legend))
        color_map = {}
        for i, p in enumerate(prods_sorted):
            color_map[p] = base_colors[i % len(base_colors)]
        # Prepare plotting positions
        n_months = len(months)
        if n_months == 0:
            print("Tidak ada bulan untuk dipetakan.")
            return
        x = np.arange(n_months)
        max_bars_per_month = 5
        # Center offsets for up to 5 bars
        offsets = np.linspace(-(max_bars_per_month - 1) / 2.0, (max_bars_per_month - 1) / 2.0, max_bars_per_month) * width
        # Figure
        plt.figure(figsize=(20, 8))
        handles = {}
        # Plot per month
        for mi, m in enumerate(months):
            sub = filtered[filtered["date"] == m].copy()
            # sort within month high->low
            sub = sub.sort_values(value_col, ascending=False).head(max_bars_per_month)
            # place bars across available offsets
            vals = sub[value_col].values
            names = sub["product_name"].astype(str).values
            # annotate only for the highest bar per month (optional)
            max_idx = int(np.argmax(vals)) if len(vals) > 0 else -1
            for k in range(len(vals)):
                p = names[k]
                v = float(vals[k])
                pos = x[mi] + offsets[k]
                bar = plt.bar(pos, v, width=width, color=color_map.get(p, "#999999"), edgecolor="white")
                # collect legend handle once per product
                if p not in handles:
                    handles[p] = bar[0]
                if not annotate_only_max or k == max_idx:
                    # concise number formatting: no decimals or 1 decimal depending on scale
                    if v >= 100:
                        label = f"{v:,.0f}"
                    else:
                        label = f"{v:,.1f}"
                    plt.text(pos, v, label, ha="center", va="bottom", fontsize=8, rotation=0)
        # Axes formatting
        month_labels = [pd.Timestamp(m).strftime("%b-%Y") for m in months]
        plt.xticks(x, month_labels, rotation=50, ha="right")
        plt.ylabel("Nilai Forecast")
        plt.title("Top-5 Produk per Bulan (Grouped) - 24 Bulan")
        plt.grid(axis="y", alpha=0.25)
        # Legend only for products that appear at least once
        if handles:
            # preserve sorted legend by product name
            ordered_names = sorted(handles.keys())
            ordered_handles = [handles[n] for n in ordered_names]
            plt.legend(ordered_handles, ordered_names, title="Produk", ncol=4, fontsize=8, title_fontsize=9, frameon=False)
        plt.tight_layout()
        # Output path
        if output_path is None:
            output_dir = os.path.join(os.getcwd(), "forecast_plots", "bulan")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "top5_grouped_24m.png")
        try:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Chart grouped Top-5 24 bulan disimpan ke: {output_path}")
        except Exception as e:
            print(f"Gagal menyimpan chart grouped 24 bulan: {e}")
        finally:
            plt.close()
    except Exception as e:
        print(f"Gagal membuat chart grouped Top-5 24 bulan: {e}")

# --- 14) PLOT HASIL ---
# Generate single grouped Top-5 chart across 24 months
try:
    generate_grouped_top5_24m_chart(csv_path=topN_csv, value_col="forecast_mean")
except Exception as e:
    print(f"Gagal membuat chart Top-5 grouped 24 bulan: {e}")
if FAST_MODE and SKIP_PLOTS_IN_FAST_MODE:
    print("FAST_MODE aktif: melewati pembuatan plot untuk mempercepat eksekusi.")
else:
    plots_dir = os.path.join(os.getcwd(), "forecast_plots", "bulan")
    os.makedirs(plots_dir, exist_ok=True)

    unique_forecast_products = final_df["product_name"].unique()
    print("Plot per-produk dinonaktifkan: menggunakan chart Top-5 per bulan saja sesuai spesifikasi.")

    # Plot agregat total semua produk (24 bulan)
    try:
        agg_df = agg.copy()

        plt.figure(figsize=(12, 6))
        plt.plot(agg_df['date'], agg_df['forecast_mean'], marker='o', linewidth=1.8, markersize=3,
                 label='Total Semua Produk')
        plt.fill_between(agg_df['date'], agg_df['forecast_p10'], agg_df['forecast_p90'], color='gray', alpha=0.2,
                         label='p10-p90')
        plt.title('Forecast Total Semua Produk - 24 Bulan')
        plt.xlabel('Tanggal (Bulanan)')
        plt.ylabel('Total Perkiraan Penjualan')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        mean_total = agg_df['forecast_mean'].mean()
        plt.axhline(y=mean_total, color='red', linestyle='--', alpha=0.7, label=f'Mean Total: {mean_total:.2f}')
        plt.legend()
        plt.tight_layout()

        out_all_path = os.path.join(plots_dir, 'forecast_all_products_total_2y.png')
        plt.savefig(out_all_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Gagal membuat plot agregat total semua produk: {e}")

    print(f"Plot disimpan di folder: {plots_dir}")

# === 15) Pembersihan Agregat 24 Bulan + Outlier Handling & Plot Bersih ===
try:
    # Helper: enforce monotonic quantiles and clamp non-negative
    def _enforce_quantile_order(df_in: pd.DataFrame) -> pd.DataFrame:
        df2 = df_in.copy()
        for col in ["forecast_p10", "forecast_p50", "forecast_p90", "forecast_mean"]:
            if col in df2.columns:
                df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0.0).clip(lower=0.0)
        # enforce per-row
        arr = df2[["forecast_p10", "forecast_p50", "forecast_p90", "forecast_mean"]].values
        p10 = arr[:, 0];
        p50 = arr[:, 1];
        p90 = arr[:, 2];
        meanv = arr[:, 3]
        p10 = np.maximum(p10, 0.0)
        # ensure order
        p50 = np.maximum(p50, p10)
        p90 = np.maximum(p90, p50)
        # ensure mean within [p10, p90]
        meanv = np.minimum(np.maximum(meanv, p10), p90)
        df2["forecast_p10"] = p10
        df2["forecast_p50"] = p50
        df2["forecast_p90"] = p90
        df2["forecast_mean"] = meanv
        return df2


    # Build historical stats per SKU
    hist_monthly = (df.groupby(["product_name", pd.Grouper(key="date", freq="MS")])["sales"].sum()
                    .reset_index())
    hist_stats = {}
    for sku, sub in hist_monthly.groupby("product_name"):
        s = pd.to_numeric(sub["sales"], errors='coerce').fillna(0.0)
        s_pos = s[s > 0]
        meanv = float(s_pos.mean()) if len(s_pos) > 0 else float(s.mean()) if len(s) > 0 else 0.0
        q10 = float(s_pos.quantile(0.10)) if len(s_pos) > 0 else 0.0
        q90 = float(s_pos.quantile(0.90)) if len(s_pos) > 0 else max(meanv * 1.2, q10)
        cv = _compute_hist_cv(s)
        hist_stats[sku] = {"hist_mean": meanv, "hist_q10": q10, "hist_q90": q90, "hist_cv": cv}

    kept_rows = []
    excluded_records = []

    # Global typical scale for sanity check
    global_typical = max(1.0, float(final_df["forecast_mean"].median()))

    for sku, sub in final_df.groupby("product_name"):
        sub = sub.sort_values("date").copy()
        # Validate monthly 24 points
        valid_dates = sub["date"].dropna()
        valid_len = len(valid_dates)
        reasons = []
        if valid_len != FORECAST_WEEKS:
            reasons.append(f"invalid_points:{valid_len}")
        # enforce monotonic quantiles per-row
        sub = _enforce_quantile_order(sub)
        # MoM cap on mean using hist CV (<= 60%)
        hs = hist_stats.get(sku, {"hist_mean": 0.0, "hist_q10": 0.0, "hist_q90": 0.0, "hist_cv": 0.0})
        floor_v = max(hs["hist_q10"] * 0.8, 0.0)
        cap_v = max(floor_v, hs["hist_q90"] * 1.2 if hs["hist_q90"] > 0 else (
            hs["hist_mean"] * 3.0 if hs["hist_mean"] > 0 else global_typical * 3.0))
        mean_vals = sub["forecast_mean"].astype(float).tolist()
        stabilized_vals, applied_mom, _ = stabilize_forecast(mean_vals, floor_v, cap_v, hs["hist_cv"])
        stabilized_vals = np.array(stabilized_vals)
        # adjust quantiles proportionally
        denom = np.maximum(np.array(mean_vals), 1e-9)
        ratio = stabilized_vals / denom
        sub["forecast_mean"] = stabilized_vals
        sub["forecast_p10"] = np.maximum(0.0, sub["forecast_p10"].values * ratio)
        sub["forecast_p50"] = np.maximum(0.0, sub["forecast_p50"].values * ratio)
        sub["forecast_p90"] = np.maximum(0.0, sub["forecast_p90"].values * ratio)
        sub = _enforce_quantile_order(sub)

        # Outlier detection metrics
        avg_p90_over_mean = float(np.nanmean(
            np.where(sub["forecast_mean"] > 0, sub["forecast_p90"] / np.maximum(sub["forecast_mean"], 1e-9), np.nan)))
        hist_mean = hs["hist_mean"]
        any_10x_hist = False
        if hist_mean and hist_mean > 0:
            any_10x_hist = bool((sub["forecast_mean"] > 10.0 * hist_mean).any())
        max_p50 = float(sub["forecast_p50"].max())
        max_p90 = float(sub["forecast_p90"].max())
        extreme_scale = (max_p50 >= 1e5) or (max_p90 >= 1e5) or (max_p90 > 200.0 * max(1.0, global_typical))
        band_median_ratio = float(np.median(np.where(sub["forecast_mean"] > 0,
                                                     (sub["forecast_p90"] - sub["forecast_p10"]) / np.maximum(
                                                         sub["forecast_mean"], 1e-9), 0.0)))
        too_wide_band = band_median_ratio > 3.0

        is_anomaly = (avg_p90_over_mean > 5.0) or any_10x_hist or extreme_scale or too_wide_band

        if is_anomaly:
            reasons.append(
                ";".join([
                    f"avg(p90/mean)={avg_p90_over_mean:.2f}",
                    f">10x_hist={any_10x_hist}",
                    f"max_p50={max_p50:.1f}",
                    f"max_p90={max_p90:.1f}",
                    f"band_med_ratio={band_median_ratio:.2f}"
                ])
            )
            excluded_records.append({
                "product_name": sku,
                "reason": ";".join(reasons) if reasons else "anomaly",
                "avg_p90_over_mean": round(avg_p90_over_mean, 4),
                "any_mean_gt_10x_hist": bool(any_10x_hist),
                "max_p50": float(max_p50),
                "max_p90": float(max_p90),
                "band_median_ratio": float(band_median_ratio),
                "hist_mean": float(hist_mean),
            })
        else:
            kept_rows.append(sub)

    if kept_rows:
        clean_df = pd.concat(kept_rows, ignore_index=True)
    else:
        clean_df = final_df.copy()

    # Save excluded SKUs report
    excl_df = pd.DataFrame(excluded_records)
    excl_path = "excluded_skus.csv"
    excl_df.to_csv(excl_path, index=False)
    print(f"Excluded SKUs disimpan: {len(excl_df)} ke {excl_path}")

    # Recalculate aggregate on cleaned data
    agg_clean = (clean_df.groupby("date", as_index=False)
                 .agg({
        "forecast_mean": "sum",
        "forecast_p10": "sum",
        "forecast_p50": "sum",
        "forecast_p90": "sum",
    })
                 .sort_values("date"))

    # Enforce aggregate quantile order
    agg_clean = _enforce_quantile_order(agg_clean)

    # Save cleaned aggregate CSV
    agg_clean_csv = "forecast_total_24m_clean.csv"
    agg_clean.to_csv(agg_clean_csv, index=False)

    # Guardrail: check band width extreme per month
    width = agg_clean["forecast_p90"] - agg_clean["forecast_p10"]
    extreme_month_mask = width > (3.0 * np.maximum(agg_clean["forecast_mean"], 1e-9))
    extreme_shade = bool(extreme_month_mask.any())

    # Compute excluded contribution annotation
    if excl_df.empty:
        avg_excluded_contrib = 0.0
        n_excl = 0
    else:
        # compute excluded total mean per month
        excluded_merge = final_df.merge(clean_df[["product_name", "date"]].drop_duplicates(),
                                        on=["product_name", "date"], how="outer", indicator=True)
        only_excl = excluded_merge[excluded_merge["_merge"] == "left_only"][final_df.columns]
        excl_month_sum = only_excl.groupby("date")["forecast_mean"].sum()
        avg_excluded_contrib = float(excl_month_sum.mean()) if not excl_month_sum.empty else 0.0
        n_excl = int(excl_df["product_name"].nunique())

    # Identify top-5 contributors to band width if extreme
    top5_msg = ""
    if extreme_shade:
        contrib = (final_df.groupby(["product_name", "date"])
                   .agg(band=lambda x: 0).reset_index())
        kept = clean_df.copy()
        kept["band_contrib"] = kept["forecast_p90"] - kept["forecast_p10"]
        contrib = kept.groupby("product_name")["band_contrib"].mean().sort_values(ascending=False)
        top5 = contrib.head(5)
        top5_msg = ", ".join([f"{k}({v:.1f})" for k, v in top5.items()])

    # Plot cleaned aggregate (linear)
    plots_dir = os.path.join(os.getcwd(), "forecast_plots", "bulan")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(agg_clean['date'], agg_clean['forecast_mean'], marker='o', linewidth=1.8, markersize=3,
             label='Total Mean (Bersih)')

    if not extreme_shade:
        plt.fill_between(agg_clean['date'], agg_clean['forecast_p10'], agg_clean['forecast_p90'], color='steelblue',
                         alpha=0.18, label='p10–p90 (bersih)')
    else:
        # error bars tipis bila band ekstrem
        y = agg_clean['forecast_mean'].values
        yerr = np.vstack([
            np.maximum(y - agg_clean['forecast_p10'].values, 0.0),
            np.maximum(agg_clean['forecast_p90'].values - y, 0.0)
        ])
        plt.errorbar(agg_clean['date'], y, yerr=yerr, fmt='o-', elinewidth=0.8, capsize=2, alpha=0.9,
                     label='Mean dengan p10/p90')

    plt.title('Forecast Total Semua Produk - 24 Bulan (Bersih)')
    plt.xlabel('Tanggal (Bulanan)')
    plt.ylabel('Total Perkiraan Penjualan')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    mean_total_clean = float(agg_clean['forecast_mean'].mean())
    plt.axhline(y=mean_total_clean, color='red', linestyle='--', alpha=0.7, label=f'Mean Total: {mean_total_clean:.2f}')

    # Annotation about exclusions
    annot = f"Excluded SKU: {n_excl}; Avg excluded mean/bulan: {avg_excluded_contrib:.1f}"
    if extreme_shade and top5_msg:
        annot += f"\nBand ekstrem: top-5 kontributor: {top5_msg}"
    plt.annotate(annot, xy=(0.01, 0.97), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

    plt.tight_layout()
    out_clean_linear = os.path.join(plots_dir, 'forecast_total_24m_clean.png')
    plt.savefig(out_clean_linear, dpi=150, bbox_inches='tight')
    plt.close()

    # Optional log10 plot if band still very wide
    if extreme_shade:
        plt.figure(figsize=(12, 6))
        plt.plot(agg_clean['date'], agg_clean['forecast_mean'], marker='o', linewidth=1.8, markersize=3,
                 label='Total Mean (Bersih)')
        plt.fill_between(agg_clean['date'], agg_clean['forecast_p10'], agg_clean['forecast_p90'], color='steelblue',
                         alpha=0.18, label='p10–p90 (bersih)')
        plt.yscale('log')
        plt.title('Forecast Total Semua Produk - 24 Bulan (Bersih, Log10)')
        plt.xlabel('Tanggal (Bulanan)')
        plt.ylabel('Total Perkiraan Penjualan (log10)')
        plt.grid(True, alpha=0.3, which='both')
        plt.xticks(rotation=45)
        plt.axhline(y=max(mean_total_clean, 1e-6), color='red', linestyle='--', alpha=0.7,
                    label=f'Mean Total: {mean_total_clean:.2f}')
        plt.tight_layout()
        out_clean_log = os.path.join(plots_dir, 'forecast_total_24m_clean_log.png')
        plt.savefig(out_clean_log, dpi=150, bbox_inches='tight')
        plt.close()

    # Quality summary markdown
    band_width = (agg_clean['forecast_p90'] - agg_clean['forecast_p10']).values
    mean_vals = agg_clean['forecast_mean'].values
    bw_stats = {
        'min': float(np.nanmin(band_width)),
        'mean': float(np.nanmean(band_width)),
        'max': float(np.nanmax(band_width)),
    }
    agg_mean_stats = {
        'min': float(np.nanmin(mean_vals)),
        'mean': float(np.nanmean(mean_vals)),
        'max': float(np.nanmax(mean_vals)),
    }
    # Quantile inversion check
    no_inversion = bool(((agg_clean['forecast_p10'] <= agg_clean['forecast_p50']) & (
                agg_clean['forecast_p50'] <= agg_clean['forecast_p90'])).all())

    summary_lines = []
    summary_lines.append('# Ringkasan Kualitas Forecast (Bersih)')
    summary_lines.append('')
    summary_lines.append('## SKU yang di-exclude')
    if excl_df.empty:
        summary_lines.append('Tidak ada SKU yang di-exclude.')
    else:
        summary_lines.append(
            'sku,reason,avg_p90_over_mean,any_mean_gt_10x_hist,max_p50,max_p90,band_median_ratio,hist_mean')
        for _, r in excl_df.iterrows():
            summary_lines.append(
                f"{r['product_name']},{r['reason']},{r.get('avg_p90_over_mean', np.nan)},{r.get('any_mean_gt_10x_hist', False)},{r.get('max_p50', np.nan)},{r.get('max_p90', np.nan)},{r.get('band_median_ratio', np.nan)},{r.get('hist_mean', np.nan)}")
    summary_lines.append('')
    summary_lines.append('## Statistik agregat final (bersih)')
    summary_lines.append(
        f"Mean total (min/mean/max): {agg_mean_stats['min']:.2f}/{agg_mean_stats['mean']:.2f}/{agg_mean_stats['max']:.2f}")
    summary_lines.append(
        f"Lebar band p10–p90 (min/mean/max): {bw_stats['min']:.2f}/{bw_stats['mean']:.2f}/{bw_stats['max']:.2f}")
    summary_lines.append(f"Quantile order valid (p10 ≤ p50 ≤ p90): {no_inversion}")
    summary_lines.append(f"Jumlah SKU dalam agregat final: {clean_df['product_name'].nunique()}")
    if extreme_shade and top5_msg:
        summary_lines.append(f"Peringatan: Ada bulan dengan band p10–p90 > 3× mean. Kontributor top-5: {top5_msg}")

    with open('quality_summary.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print("\nOutput bersih:")
    print(f" - forecast_total_24m_clean.csv")
    print(f" - forecast_total_24m_clean.png" + (" dan forecast_total_24m_clean_log.png" if extreme_shade else ""))
    print(f" - excluded_skus.csv")
    print(f" - quality_summary.md")

except Exception as e:
    print(f"Gagal melakukan pembersihan agregat bersih: {e}")
