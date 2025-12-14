import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

# Optional reuse from in-repo utilities
# try:
#     import test2
# except Exception:
#     test2 = None
test2 = None

# ------------------------
# Configuration
# ------------------------
FORECAST_HORIZON_MONTHS = 24
MODELS_DIR = os.path.join(os.getcwd(), "trained_models")
DEFAULT_EXCEL = os.path.join(os.getcwd(), "Data_Penjualan_Dengan_ID_Pelanggan.xlsx")

# Outputs
OUT_FORECAST_PER_PRODUCT = os.path.join(os.getcwd(), "forecast_per_product_24m.csv")
OUT_FORECAST_TOTAL = os.path.join(os.getcwd(), "forecast_total_24m.csv")
OUT_TOPN = os.path.join(os.getcwd(), "topN_per_month_24m.csv")
OUT_DIAG = os.path.join(os.getcwd(), "forecast_diagnostics.csv")
PLOTS_DIR = os.path.join(os.getcwd(), "forecast_plots", "bulan")

# Quarterly/Yearly outputs
PLOTS_DIR_QUARTERLY = os.path.join(os.getcwd(), "forecast_plots")
OUT_QUARTERLY_TOP5_TEMPLATE = os.path.join(os.getcwd(), "quarterly_top5_{year}.csv")
OUT_YEARLY_TOP5_TEMPLATE = os.path.join(os.getcwd(), "yearly_top5_borda_{year}.csv")

# Experimental/diagnostic flags (Priority-1 fixes)
# Set to True to disable the respective post-processing; helps avoid over-constraining forecasts
DISABLE_STABILIZATION = False  # ENABLED: apply stabilization for consistency with SES
DISABLE_CLAMPING = False       # ENABLED: apply clamping for consistency with SES
NOISE_INJECTION = True         # add small noise to residual forward loop to break perfect cycles

# ------------------------
# Helpers
# ------------------------

def normalize_product_name(name: str) -> str:
    if test2 and hasattr(test2, "normalize_product_name"):
        return test2.normalize_product_name(name)
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s


def read_excel_latest(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    
    # Print actual columns for debugging
    print(f"Columns found in Excel file: {df.columns.tolist()}")
    
    # Normalize column names for comparison (lowercase, strip whitespace)
    def normalize_col_name(name):
        return str(name).lower().strip().replace('_', ' ')
    
    # Expanded candidates with more variations
    candidates = [
        {"date": "Transaction Date", "qty": "Quantity", "product": "Product Name", "category": "Product Category"},
        {"date": "Tanggal Transaksi", "qty": "Jumlah", "product": "Nama Produk", "category": "Kategori Barang"},
        {"date": "Date", "qty": "Qty", "product": "Product", "category": "Category"},
        {"date": "Tanggal", "qty": "Jumlah", "product": "Produk", "category": "Kategori"},
        {"date": "transaction_date", "qty": "quantity", "product": "product_name", "category": "product_category"},
        {"date": "tanggal_transaksi", "qty": "jumlah", "product": "nama_produk", "category": "kategori_barang"},
    ]
    
    # Try fuzzy matching
    normalized_cols = {normalize_col_name(col): col for col in df.columns}
    mapping = None
    actual_mapping = {}
    
    for cand in candidates:
        actual_mapping = {}
        match_count = 0
        for key, expected_name in cand.items():
            normalized_expected = normalize_col_name(expected_name)
            # Direct match
            if normalized_expected in normalized_cols:
                actual_mapping[key] = normalized_cols[normalized_expected]
                match_count += 1
            else:
                # Partial match - check if expected name is contained in any column
                for norm_col, orig_col in normalized_cols.items():
                    if normalized_expected in norm_col or norm_col in normalized_expected:
                        actual_mapping[key] = orig_col
                        match_count += 1
                        break
        
        # If all 4 columns were matched, we found a valid mapping
        if match_count == 4:
            mapping = actual_mapping
            break
    
    # Fallback to test2 column mapping if available
    if mapping is None and test2 and hasattr(test2, "column_mapping"):
        mapping = {"date": test2.column_mapping.get("Tanggal Transaksi", "date"),
                   "qty": test2.column_mapping.get("Jumlah", "sales"),
                   "product": test2.column_mapping.get("Nama Produk", "product_name"),
                   "category": test2.column_mapping.get("Kategori Barang", "category")}
        # Verify these columns actually exist
        if not all(col in df.columns for col in mapping.values()):
            mapping = None
    
    if mapping is None:
        # Provide helpful error message
        error_msg = f"Could not infer column mapping for Excel file.\n"
        error_msg += f"Found columns: {df.columns.tolist()}\n"
        error_msg += f"Expected columns similar to:\n"
        error_msg += f"  - Date column (e.g., 'Transaction Date', 'Tanggal Transaksi')\n"
        error_msg += f"  - Quantity column (e.g., 'Quantity', 'Jumlah')\n"
        error_msg += f"  - Product column (e.g., 'Product Name', 'Nama Produk')\n"
        error_msg += f"  - Category column (e.g., 'Product Category', 'Kategori Barang')"
        raise ValueError(error_msg)

    print(f"Using column mapping: {mapping}")

    df = df[[mapping["date"], mapping["qty"], mapping["product"], mapping["category"]]].copy()
    df.columns = ["date", "sales", "product_name", "category"]
    df.dropna(subset=["date"], inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors='coerce')  # preserve time zone naive, coerce invalid dates
    
    # Remove rows with invalid dates
    df = df.dropna(subset=["date"])
    
    df["product_norm"] = df["product_name"].apply(normalize_product_name)

    # IQR-based clipping per product (consistent with SES method)
    def _clip_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < 4:
            return g
        # Use IQR method (same as SES)
        q1 = g["sales"].quantile(0.25)
        q3 = g["sales"].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = max(0.0, q1 - 1.5 * iqr)
            upper = q3 + 1.5 * iqr
            g["sales"] = g["sales"].clip(lower=lower, upper=upper)
        else:
            g["sales"] = g["sales"].clip(lower=0)
        return g

    df.sort_values(["product_norm", "date"], inplace=True)
    df = df.groupby("product_norm", group_keys=False).apply(_clip_group, include_groups=False)
    return df


def monthly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure product_norm column exists
    if 'product_norm' not in df.columns:
        if 'product_name' in df.columns:
            df["product_norm"] = df["product_name"].apply(normalize_product_name)
        else:
            raise ValueError("Neither 'product_norm' nor 'product_name' column found in DataFrame")
    
    df = df.copy()
    df["month"] = df["date"].values.astype("datetime64[M]")
    m = (df.groupby(["product_norm", "category", "month"])
           .agg(qty=("sales", "sum"))
           .reset_index())
    return m


def build_time_features(dates: pd.Series) -> pd.DataFrame:
    month = pd.to_datetime(dates).dt.month
    feat = pd.DataFrame({
        "month": month.values,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "trend": np.arange(1, len(month) + 1, dtype=float)
    }, index=pd.to_datetime(dates))
    if len(feat) < 12:
        feat["month_sin"] = 0.0
        feat["month_cos"] = 0.0
    return feat


def load_models_metadata(models_dir: str) -> dict:
    meta_path = os.path.join(models_dir, "models_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing models_metadata.json in {models_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_global_category_stats(models_dir: str) -> Tuple[dict, dict]:
    with open(os.path.join(models_dir, "global_stats.json"), "r", encoding="utf-8") as f:
        g = json.load(f)
    with open(os.path.join(models_dir, "category_stats.json"), "r", encoding="utf-8") as f:
        c = json.load(f)
    return g, c


def load_product_artifacts(models_dir: str, product_norm: str):
    safe = product_norm.replace("/", "-").replace("\\", "-").replace(" ", "_")
    model_path = os.path.join(models_dir, f"product_{safe}_model.pkl")
    scaler_path = os.path.join(models_dir, f"product_{safe}_scaler.pkl")
    features_path = os.path.join(models_dir, f"product_{safe}_features.json")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        return None, None, None
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    model = model_from_json(blob["model_json"])  # type: ignore
    model.set_weights(blob["weights"])  # type: ignore
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        feats = json.load(f)
    return model, scaler, feats


def forecast_baseline_forward(last_hist_date: pd.Timestamp,
                              hist_sales: pd.Series,
                              hist_dates: pd.Series,
                              horizon: int,
                              meta: dict) -> pd.Series:
    # Try reuse
    if test2 and hasattr(test2, "forecast_baseline_forward"):
        return test2.forecast_baseline_forward(last_hist_date, hist_sales, hist_dates, horizon, meta)  # type: ignore
    
    # Minimal baseline: just use last value with small random variation
    s = pd.Series(hist_sales.values, index=pd.to_datetime(hist_dates))
    last_value = float(s.iloc[-1]) if len(s) > 0 else 1.0
    
    # Calculate trend from recent data
    if len(s) >= 6:
        recent_trend = (s.tail(3).mean() - s.tail(6).mean()) / 3
    else:
        recent_trend = 0
    
    idx = pd.date_range(start=pd.to_datetime(last_hist_date) + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    vals = []
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    for i, d in enumerate(idx):
        # Conservative baseline with moderate trend (consistent with SES)
        trend_component = recent_trend * (i + 1) * 0.3  # Moderate trend (reduced from 0.5)
        random_component = rng.normal(0, last_value * 0.05)  # 5% random variation (reduced from 15%)
        
        # Add monthly variation component
        month_variation = 1.0 + 0.05 * np.sin(2 * np.pi * d.month / 12)  # Reduced from 0.1
        
        # Add additional random walk component (minimal)
        walk_component = rng.normal(0, last_value * 0.02)  # Reduced from 0.05
        
        val = (last_value + trend_component + random_component + walk_component) * month_variation
        vals.append(max(0.1, float(val)))  # Ensure positive
    
    return pd.Series(vals, index=idx)


def build_sequences_for_inference(hist_df: pd.DataFrame, features_for_lstm: List[str], scaler, time_steps: int) -> np.ndarray:
    """Build the last scaled window for inference from historical data.
    hist_df has columns: date(index), residual, month_sin, month_cos, trend
    """
    data = hist_df[features_for_lstm].astype(float).values
    data_scaled = scaler.transform(data)
    if len(data_scaled) < time_steps:
        # pad with first row if too short
        pad = np.tile(data_scaled[0], (time_steps - len(data_scaled), 1))
        seq = np.vstack([pad, data_scaled])
    else:
        seq = data_scaled[-time_steps:]
    return seq


def bootstrap_quantiles(mean_series: np.ndarray, resid_errors: np.ndarray, n_samples: int = 400, qs=(0.1, 0.5, 0.9)):
    if resid_errors is None or len(resid_errors) == 0:
        resid_errors = np.array([0.0])
    rng = np.random.RandomState(123)
    draws = rng.choice(resid_errors, size=(n_samples, len(mean_series)), replace=True)
    sims = mean_series[None, :] + draws
    q_vals = np.quantile(sims, qs, axis=0)
    return q_vals  # shape (len(qs), horizon)


def enforce_quantile_order(p10, p50, p90):
    p10c = np.minimum(p10, p50)
    p50c = np.minimum(np.maximum(p50, p10c), p90)
    p90c = np.maximum(p90, p50c)
    return p10c, p50c, p90c


def stabilize_series(mean_forecast: np.ndarray, hist_values: np.ndarray) -> np.ndarray:
    """Apply moderate MoM change limits to preserve forecast dynamics while ensuring stability.
    - Balanced bounds to avoid extreme spikes but allow reasonable variation
    - More conservative than previous version for consistency with SES
    """
    if len(hist_values) == 0:
        return np.maximum(mean_forecast, 0.0)
    
    last = float(hist_values[-1])
    cv = float(np.std(hist_values) / (np.mean(hist_values) + 1e-6)) if len(hist_values) > 1 else 0.5
    
    # More conservative bounds: allow 50-150% monthly change based on CV
    k = 2.0  # Reduced from 5.0 for more stability
    max_monthly_change = max(0.5, min(1.5, k * cv))  # Between 50% and 150%
    
    out = []
    prev = last
    for m in mean_forecast:
        upper = prev * (1 + max_monthly_change)
        lower = prev * max(0.1, (1 - max_monthly_change))  # Allow down to 10% of previous (more conservative)
        val = float(np.clip(m, lower, upper))
        out.append(val)
        prev = val
    
    return np.array(out)


def clamp_with_historical_quantiles(arr: np.ndarray, hist_values: np.ndarray) -> np.ndarray:
    """Apply moderate bounds based on historical data for consistency with SES.
    Reasonable bounds using IQR-based approach similar to SES outlier capping.
    """
    if len(hist_values) == 0:
        return np.maximum(arr, 0.0)
    
    # Use IQR-based bounds (similar to SES method)
    q1 = np.quantile(hist_values, 0.25)
    q3 = np.quantile(hist_values, 0.75)
    iqr = q3 - q1
    mean_hist = np.mean(hist_values)
    
    # Allow forecasts within reasonable bounds: Q1-1.5*IQR to Q3+3*IQR (generous upper bound for growth)
    if iqr > 0:
        lower_bound = max(0.0, q1 - 1.5 * iqr)
        upper_bound = max(q3 + 3.0 * iqr, mean_hist * 3.0)  # Allow up to 3x historical mean
    else:
        # Fallback if no variance
        lower_bound = 0.0
        upper_bound = mean_hist * 3.0
    
    return np.clip(arr, lower_bound, upper_bound)


def forecast_for_product(prod: str,
                         hist_g: pd.DataFrame,
                         model,
                         scaler,
                         feats_meta: dict) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    # hist_g columns: product_norm, category, month, qty
    hist_g = hist_g.sort_values("month").copy()
    dates = pd.to_datetime(hist_g["month"])  # Month start
    sales = hist_g["qty"].astype(float).values

    # Determine mode: direct forecasting if model trained without residuals
    features_for_lstm = feats_meta.get("features_for_lstm", ["residual", "month_sin", "month_cos", "trend"]) if feats_meta else ["residual", "month_sin", "month_cos", "trend"]
    direct_mode = bool(feats_meta.get("direct_mode", False)) if feats_meta else ("residual" not in features_for_lstm)

    tf = build_time_features(dates)

    future_index = pd.date_range(start=dates.iloc[-1] + pd.offsets.MonthBegin(1), periods=FORECAST_HORIZON_MONTHS, freq="MS")

    if direct_mode:
        # Build historical feature frame for direct sales forecasting
        s = pd.Series(sales, index=dates)
        hist_feat = pd.DataFrame(index=dates)
        first_col = features_for_lstm[0]
        if first_col == "sales":
            hist_feat[first_col] = s.values
        else:
            # fallback naming
            hist_feat[first_col] = s.values
        for c in features_for_lstm[1:]:
            if c == "lag_1":
                hist_feat[c] = s.shift(1).bfill().values
            elif c == "lag_2":
                hist_feat[c] = s.shift(2).bfill().values
            elif c == "lag_3":
                hist_feat[c] = s.shift(3).bfill().values
            elif c == "rolling_mean_3":
                hist_feat[c] = s.rolling(3, min_periods=1).mean().shift(1).bfill().values
            elif c == "rolling_std_3":
                hist_feat[c] = s.rolling(3, min_periods=1).std(ddof=0).shift(1).fillna(0.0).values
            elif c == "trend":
                hist_feat[c] = np.arange(1, len(s) + 1, dtype=float)
            elif c in ("month_sin", "month_cos"):
                t2 = tf.reindex(dates)
                if len(s) >= 12 and c in t2.columns:
                    hist_feat[c] = t2[c].values
                else:
                    hist_feat[c] = 0.0
            else:
                hist_feat[c] = 0.0

        # Last scaled window
        last_window = build_sequences_for_inference(hist_feat, features_for_lstm, scaler, int(feats_meta.get("time_steps", 3)))

        # Iterative direct forecast with enhanced variation
        preds = []
        lw = last_window.copy()
        augmented = list(s.values)
        rng = np.random.RandomState(42)
        
        for i, d in enumerate(future_index):
            pred_scaled = float(model.predict(lw.reshape(1, lw.shape[0], lw.shape[1]), verbose=0)[0][0])
            try:
                pred_val = (pred_scaled - scaler.min_[0]) / scaler.scale_[0]
            except Exception:
                pred_val = pred_scaled
            
            # Add small noise to break perfect cycles (reduced for consistency with SES)
            if NOISE_INJECTION:
                noise_factor = rng.normal(0, 0.05)  # 5% noise (reduced from 15%)
                pred_val = pred_val * (1 + noise_factor)
            
            preds.append(max(0.1, pred_val))
            # Build next feature row using prior history before adding current pred
            lag1 = augmented[-1] if len(augmented) >= 1 else pred_val
            lag2 = augmented[-2] if len(augmented) >= 2 else lag1
            lag3 = augmented[-3] if len(augmented) >= 3 else lag2
            roll_vals = augmented[-3:] if len(augmented) >= 3 else augmented
            rmean = float(np.mean(roll_vals)) if len(roll_vals) > 0 else 0.0
            rstd = float(np.std(roll_vals)) if len(roll_vals) > 1 else 0.0
            new_row = {
                features_for_lstm[0]: pred_val,
                "lag_1": lag1,
                "lag_2": lag2,
                "lag_3": lag3,
                "rolling_mean_3": rmean,
                "rolling_std_3": rstd,
                "trend": float(len(hist_feat) + i + 1),
                "month_sin": np.sin(2 * np.pi * d.month / 12) if len(hist_feat) >= 12 else 0.0,
                "month_cos": np.cos(2 * np.pi * d.month / 12) if len(hist_feat) >= 12 else 0.0,
            }
            unscaled_vec = [new_row.get(col, 0.0) for col in features_for_lstm]
            next_scaled = scaler.transform([unscaled_vec])[0]
            lw = np.vstack([lw[1:], next_scaled])
            augmented.append(pred_val)

        mean_forecast = np.maximum(np.array(preds, dtype=float), 0.0)

        # Errors for quantiles approximation using sales differences
        resid_arr = s.values
        if len(resid_arr) >= 4:
            errs = resid_arr[1:] - resid_arr[:-1]
        else:
            errs = resid_arr - np.mean(resid_arr)

    else:
        # Residual mode (legacy)
        baseline_meta = feats_meta.get("baseline_meta", {}) if feats_meta else {}
        base_future = forecast_baseline_forward(dates.iloc[-1], pd.Series(sales), dates, FORECAST_HORIZON_MONTHS, baseline_meta)

        kept = [c for c in features_for_lstm if c != "residual"]

        # Baseline history (SMA fallback)
        if test2 and hasattr(test2, "build_baseline_series"):
            baseline_hist, _ = test2.build_baseline_series(pd.Series(sales), dates)  # type: ignore
        else:
            baseline_hist = pd.Series(sales).rolling(3, min_periods=1).mean().shift(1).bfill()
        residual_series = pd.Series(sales, index=dates) - baseline_hist.values

        hist_feat = pd.DataFrame({"residual": residual_series.values}, index=dates)
        if "month_sin" in kept or "month_cos" in kept or "trend" in kept:
            tf2 = tf.reindex(dates)
            for c in kept:
                if c in tf2.columns:
                    hist_feat[c] = tf2[c].values
                else:
                    hist_feat[c] = 0.0

        last_window = build_sequences_for_inference(hist_feat, features_for_lstm, scaler, int(feats_meta.get("time_steps", 6)))

        residual_forecast = []
        lw = last_window.copy()
        resid_std = float(np.std(hist_feat["residual"])) if "residual" in hist_feat.columns and len(hist_feat) > 1 else 0.0
        rng = np.random.RandomState(2025)
        for i, d in enumerate(future_index):
            pred_scaled = float(model.predict(lw.reshape(1, lw.shape[0], lw.shape[1]), verbose=0)[0][0])
            try:
                pred_resid = (pred_scaled - scaler.min_[0]) / scaler.scale_[0]
            except Exception:
                pred_resid = pred_scaled
            if NOISE_INJECTION and resid_std > 0:
                # Small noise to break perfect cycles (reduced for consistency)
                pred_resid = float(pred_resid + rng.normal(0.0, resid_std * 0.15))
            residual_forecast.append(pred_resid)
            new_row = {"residual": pred_resid, "month_sin": np.sin(2 * np.pi * d.month / 12) if len(hist_feat) >= 12 else 0.0,
                       "month_cos": np.cos(2 * np.pi * d.month / 12) if len(hist_feat) >= 12 else 0.0,
                       "trend": float(len(hist_feat) + i + 1)}
            unscaled_vec = [new_row.get(col, 0.0) for col in features_for_lstm]
            next_scaled = scaler.transform([unscaled_vec])[0]
            lw = np.vstack([lw[1:], next_scaled])

        mean_forecast = np.maximum(base_future.values + np.array(residual_forecast), 0.0)

        resid_arr = residual_series.values
        if len(resid_arr) >= 4:
            errs = resid_arr[1:] - resid_arr[:-1]
        else:
            errs = resid_arr - np.mean(resid_arr)

    q10, q50, q90 = bootstrap_quantiles(mean_forecast, errs)
    q10, q50, q90 = enforce_quantile_order(q10, q50, q90)

    # Stabilization & QC (toggleable)
    mean_stab = mean_forecast.copy()
    if not DISABLE_STABILIZATION:
        mean_stab = stabilize_series(mean_stab, sales)
    if not DISABLE_CLAMPING:
        mean_stab = clamp_with_historical_quantiles(mean_stab, sales)
        q10 = clamp_with_historical_quantiles(q10, sales)
        q50 = clamp_with_historical_quantiles(q50, sales)
        q90 = clamp_with_historical_quantiles(q90, sales)
        q10, q50, q90 = enforce_quantile_order(q10, q50, q90)
    else:
        # Ensure non-negative if clamping disabled
        mean_stab = np.maximum(mean_stab, 0.0)
        q10 = np.maximum(q10, 0.0)
        q50 = np.maximum(q50, 0.0)
        q90 = np.maximum(q90, 0.0)
        q10, q50, q90 = enforce_quantile_order(q10, q50, q90)

    # Diagnostics
    if 'residual_series' in locals():
        resid_std_val = float(np.std(residual_series.values)) if len(residual_series.values) > 1 else 0.0
    else:
        resid_std_val = float(np.std(sales)) if len(sales) > 1 else 0.0
    diag = {
        "hist_n": int(len(sales)),
        "hist_cv": float(np.std(sales) / (np.mean(sales) + 1e-6)) if len(sales) > 1 else 0.0,
        "used_model": True,
        "residual_std": resid_std_val,
        "disable_stabilization": bool(DISABLE_STABILIZATION),
        "disable_clamping": bool(DISABLE_CLAMPING),
        "noise_injection": bool(NOISE_INJECTION),
        "direct_mode": bool(direct_mode),
        "features_count": int(len(features_for_lstm) if isinstance(features_for_lstm, list) else 0),
        "time_steps": int(feats_meta.get("time_steps", 3) if feats_meta else 3),
    }

    return future_index, mean_stab, q10, q50, q90, diag


def fallback_forecast(prod: str, hist_g: pd.DataFrame, category: str, global_stats: dict, cat_stats: dict, forecast_start_date: pd.Timestamp = None):
    """
    Simple LSTM-style fallback forecast WITHOUT SES smoothing.
    Uses last historical value or historical mean with minimal variation.
    """
    hist_g = hist_g.sort_values("month").copy()
    
    # Use provided forecast_start_date or calculate from data
    if forecast_start_date is None:
        dates = pd.to_datetime(hist_g["month"]) if len(hist_g) else pd.to_datetime([global_stats.get("last_date", datetime.utcnow().strftime("%Y-%m-01"))])
        last_date = dates.iloc[-1]
        forecast_start_date = last_date + pd.offsets.MonthBegin(1)
    
    future_index = pd.date_range(start=forecast_start_date, periods=FORECAST_HORIZON_MONTHS, freq="MS")
    
    # STRATEGY: Use historical data directly, no SES smoothing
    if len(hist_g) > 0:
        # Use historical mean as base (more stable than last value)
        base = float(hist_g["qty"].mean())
        # Get last value for reference
        last_value = float(hist_g["qty"].iloc[-1])
        
        # If last value is significantly different, blend it
        if abs(last_value - base) / (base + 1e-6) > 0.5:
            base = 0.7 * base + 0.3 * last_value  # Blend: 70% mean, 30% last
    else:
        # Fallback to global median (conservative, not category median which is too high)
        base = global_stats.get("global_median", 8.0)
    
    # Simple forward projection with minimal noise (LSTM-style, not SES)
    mean = []
    rng = np.random.RandomState(42)
    
    # NO TREND for fallback (conservative approach)
    # NO SEASONALITY for fallback (too complex without model)
    # Just slight random walk
    
    current_value = base
    for i, d in enumerate(future_index):
        # Add small random walk (simulating LSTM uncertainty)
        noise = rng.normal(0, 0.02 * current_value)  # 2% noise per step
        
        # Slight mean reversion (drift back to base)
        mean_reversion = 0.05 * (base - current_value)
        
        current_value = current_value + noise + mean_reversion
        current_value = max(0.1, current_value)  # Ensure positive
        
        mean.append(float(current_value))
    
    mean = np.array(mean)
    
    # Uncertainty bands based on historical volatility
    if len(hist_g) > 1:
        hist_std = float(np.std(hist_g["qty"]))
        qspread = max(0.2 * base, hist_std * 0.5)
    else:
        qspread = 0.2 * base
    
    p10 = np.maximum(mean - 0.8 * qspread, 0.1)  # Minimum 0.1 to avoid zeros
    p50 = mean.copy()
    p90 = mean + 0.8 * qspread
    
    diag = {
        "hist_n": int(len(hist_g)),
        "hist_cv": float(np.std(hist_g["qty"]) / (np.mean(hist_g["qty"]) + 1e-6)) if len(hist_g) > 1 else 0.0,
        "used_model": False,
        "fallback_mode": "simple_random_walk",
    }
    return future_index, mean, p10, p50, p90, diag


def ensure_dirs():
    """Create necessary output directories."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR_QUARTERLY, exist_ok=True)


# ------------------------
# Quarterly Aggregation & Borda Count Functions
# ------------------------

def get_quarter(month: int) -> str:
    """
    Map month number (1-12) to quarter string.
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Quarter string (Q1, Q2, Q3, or Q4)
    """
    if month in [1, 2, 3]:
        return "Q1"
    elif month in [4, 5, 6]:
        return "Q2"
    elif month in [7, 8, 9]:
        return "Q3"
    else:
        return "Q4"


def aggregate_to_quarterly(
    monthly_df: pd.DataFrame,
    top_n: int = 5
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Aggregate monthly forecast data into quarterly rankings.
    
    Groups monthly forecasts by quarter and calculates top N products
    for each quarter based on the sum of monthly forecast values.
    
    Args:
        monthly_df: DataFrame with columns ['product', 'category', 'date', 'mean']
                   containing monthly forecast data.
        top_n: Number of top products to return per quarter (default: 5)
        
    Returns:
        Nested dictionary: {year: {quarter: DataFrame with top N products}}
        Each DataFrame has columns: ['product', 'category', 'quarterly_sum', 'rank']
        
    Example:
        >>> result = aggregate_to_quarterly(forecast_df, top_n=5)  # doctest: +SKIP
        >>> result[2025]['Q1']  # Top 5 products for Q1 2025  # doctest: +SKIP
    """
    print("\n=== Aggregating Monthly Data to Quarterly ===")
    
    # Ensure date is datetime
    df = monthly_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Extract year and quarter
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['month'].apply(get_quarter)
    
    # Get unique years
    years = sorted(df['year'].unique())
    print(f"Years found: {years}")
    
    result: Dict[int, Dict[str, pd.DataFrame]] = {}
    
    for year in years:
        year_data = df[df['year'] == year]
        result[year] = {}
        
        # Get quarters available for this year
        quarters_available = sorted(year_data['quarter'].unique(), 
                                    key=lambda q: int(q[1]))
        print(f"\nYear {year} - Quarters available: {quarters_available}")
        
        for quarter in quarters_available:
            quarter_data = year_data[year_data['quarter'] == quarter]
            
            # Aggregate by product: sum of mean forecasts across months in quarter
            quarterly_agg = (
                quarter_data
                .groupby(['product', 'category'])
                .agg(quarterly_sum=('mean', 'sum'))
                .reset_index()
            )
            
            # Sort and get top N
            quarterly_agg = quarterly_agg.sort_values('quarterly_sum', ascending=False)
            top_products = quarterly_agg.head(top_n).copy()
            top_products['rank'] = range(1, len(top_products) + 1)
            
            result[year][quarter] = top_products
            
            print(f"  {quarter}: Top {top_n} products identified")
            for _, row in top_products.iterrows():
                print(f"    Rank {row['rank']}: {row['product'][:40]}... = {row['quarterly_sum']:.2f}")
    
    return result


def borda_count_ranking(
    quarterly_rankings: Dict[str, pd.DataFrame],
    year: int,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Calculate Borda Count ranking from quarterly rankings.
    
    Implements Borda Count Voting where:
    - Rank 1 = top_n points (e.g., 5 points for top_n=5)
    - Rank 2 = top_n - 1 points
    - ...
    - Rank top_n = 1 point
    - Products not in top N for a quarter = 0 points
    
    Args:
        quarterly_rankings: Dictionary {quarter: DataFrame with ranked products}
        year: Year for logging purposes
        top_n: Number of top products considered in ranking (default: 5)
        
    Returns:
        DataFrame with columns:
        ['product', 'category', 'total_score', 'total_forecast', 'rank',
         'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score',
         'Q1_rank', 'Q2_rank', 'Q3_rank', 'Q4_rank']
         
    Notes:
        - Ties in total_score are broken by total_forecast (descending)
        - Products not appearing in any quarter's top N are excluded
    """
    print(f"\n=== Calculating Borda Count for Year {year} ===")
    
    # Collect all products that appeared in any quarter
    all_products: Dict[str, Dict] = {}
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    for quarter in quarters:
        if quarter not in quarterly_rankings:
            print(f"  {quarter}: No data available")
            continue
            
        quarter_df = quarterly_rankings[quarter]
        
        for _, row in quarter_df.iterrows():
            product = row['product']
            category = row.get('category', '')
            rank = row['rank']
            quarterly_sum = row['quarterly_sum']
            
            # Calculate Borda score: top_n points for rank 1, 1 point for rank top_n
            borda_score = max(0, top_n - rank + 1)
            
            if product not in all_products:
                all_products[product] = {
                    'category': category,
                    'total_score': 0,
                    'total_forecast': 0,
                    'Q1_score': 0, 'Q2_score': 0, 'Q3_score': 0, 'Q4_score': 0,
                    'Q1_rank': 0, 'Q2_rank': 0, 'Q3_rank': 0, 'Q4_rank': 0,
                }
            
            all_products[product]['total_score'] += borda_score
            all_products[product]['total_forecast'] += quarterly_sum
            all_products[product][f'{quarter}_score'] = borda_score
            all_products[product][f'{quarter}_rank'] = rank
    
    # Convert to DataFrame
    rows = []
    for product, data in all_products.items():
        rows.append({
            'product': product,
            **data
        })
    
    if not rows:
        print("  No products found in quarterly rankings!")
        return pd.DataFrame(columns=[
            'product', 'category', 'total_score', 'total_forecast', 'rank',
            'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score',
            'Q1_rank', 'Q2_rank', 'Q3_rank', 'Q4_rank'
        ])
    
    result_df = pd.DataFrame(rows)
    
    # Sort by total_score (descending), then by total_forecast (descending) for tiebreaker
    result_df = result_df.sort_values(
        ['total_score', 'total_forecast'], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    # Assign final rank
    result_df['rank'] = range(1, len(result_df) + 1)
    
    # Reorder columns
    cols = ['product', 'category', 'total_score', 'total_forecast', 'rank',
            'Q1_score', 'Q2_score', 'Q3_score', 'Q4_score',
            'Q1_rank', 'Q2_rank', 'Q3_rank', 'Q4_rank']
    result_df = result_df[cols]
    
    # Print top N results
    print(f"\nTop {top_n} Products for {year} (Borda Count):")
    for _, row in result_df.head(top_n).iterrows():
        score_breakdown = f"Q1:{row['Q1_score']} Q2:{row['Q2_score']} Q3:{row['Q3_score']} Q4:{row['Q4_score']}"
        print(f"  Rank {row['rank']}: {row['product'][:40]}...")
        print(f"         Total Score: {row['total_score']} ({score_breakdown})")
    
    return result_df


def plot_yearly_top5(
    borda_results: pd.DataFrame,
    year: int,
    top_n: int = 5,
    output_dir: str = PLOTS_DIR_QUARTERLY
) -> str:
    """
    Create horizontal bar chart visualizing top N products for the year.
    
    Args:
        borda_results: DataFrame from borda_count_ranking with Borda scores
        year: Year for the plot title and filename
        top_n: Number of top products to display (default: 5)
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    print(f"\n=== Creating Yearly Top-{top_n} Plot for {year} ===")
    
    # Get top N products
    top_products = borda_results.head(top_n).copy()
    
    if len(top_products) == 0:
        print(f"  No data available for {year}")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data - reverse for horizontal bar (top to bottom)
    products = top_products['product'].tolist()[::-1]
    scores = top_products['total_score'].tolist()[::-1]
    ranks = top_products['rank'].tolist()[::-1]
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(products)))
    
    # Create horizontal bars
    y_pos = np.arange(len(products))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Customize y-axis
    ax.set_yticks(y_pos)
    # Truncate long product names
    product_labels = [p[:50] + '...' if len(p) > 50 else p for p in products]
    ax.set_yticklabels(product_labels, fontsize=10)
    
    # Add rank labels on the left
    for i, (rank, product) in enumerate(zip(ranks, products)):
        ax.text(-max(scores)*0.02, i, f"#{rank}", 
                ha='right', va='center', fontweight='bold', fontsize=11)
    
    # Add score labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(score)} pts', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Total Borda Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top-{top_n} Produk Tahun {year}\n(Berdasarkan Borda Count Voting)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid and styling
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"top5_yearly_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


def plot_quarterly_top5(
    quarterly_rankings: Dict[str, pd.DataFrame],
    year: int,
    top_n: int = 5,
    output_dir: str = PLOTS_DIR_QUARTERLY
) -> str:
    """
    Create 2x2 subplot grid showing top N products for each quarter.
    
    Args:
        quarterly_rankings: Dictionary {quarter: DataFrame} from aggregate_to_quarterly
        year: Year for the plot title and filename
        top_n: Number of top products to display per quarter (default: 5)
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    print(f"\n=== Creating Quarterly Top-{top_n} Plot for {year} ===")
    
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get all unique products for consistent coloring
    all_products = set()
    for quarter in quarters:
        if quarter in quarterly_rankings:
            all_products.update(quarterly_rankings[quarter]['product'].tolist())
    
    # Create color map
    product_list = sorted(all_products)
    cmap = plt.cm.tab20
    color_map = {prod: cmap(i % 20) for i, prod in enumerate(product_list)}
    
    for idx, quarter in enumerate(quarters):
        ax = axes[idx]
        
        if quarter not in quarterly_rankings or len(quarterly_rankings[quarter]) == 0:
            ax.text(0.5, 0.5, f'{quarter}\nNo Data', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            continue
        
        # Get top N for this quarter
        quarter_data = quarterly_rankings[quarter].head(top_n).copy()
        
        # Prepare data - reverse for better visualization (top to bottom)
        products = quarter_data['product'].tolist()[::-1]
        values = quarter_data['quarterly_sum'].tolist()[::-1]
        ranks = quarter_data['rank'].tolist()[::-1]
        
        # Get colors
        colors = [color_map.get(p, '#999999') for p in products]
        
        # Create horizontal bars
        y_pos = np.arange(len(products))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Truncate long names
        product_labels = [p[:40] + '...' if len(p) > 40 else p for p in products]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(product_labels, fontsize=9)
        
        # Add rank labels
        for i, rank in enumerate(ranks):
            ax.text(-max(values)*0.02, i, f"#{rank}", 
                   ha='right', va='center', fontweight='bold', fontsize=9)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            # Format value
            if val >= 1000:
                label = f'{val/1000:.1f}K'
            else:
                label = f'{val:.0f}'
            ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=8)
        
        # Styling
        ax.set_xlabel('Total Forecast (3 bulan)', fontsize=10)
        ax.set_title(f'{quarter} {year}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle(f'Top-{top_n} Produk Per Kuartal - Tahun {year}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    output_path = os.path.join(output_dir, f"top5_quarterly_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


def plot_borda_process(
    borda_results: pd.DataFrame,
    year: int,
    top_n: int = 5,
    output_dir: str = PLOTS_DIR_QUARTERLY
) -> str:
    """
    Create stacked horizontal bar chart showing Borda score contributions by quarter.
    
    Args:
        borda_results: DataFrame from borda_count_ranking with quarterly score breakdown
        year: Year for the plot title and filename
        top_n: Number of top products to display (default: 5)
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    print(f"\n=== Creating Borda Process Visualization for {year} ===")
    
    # Get top N products
    top_products = borda_results.head(top_n).copy()
    
    if len(top_products) == 0:
        print(f"  No data available for {year}")
        return ""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data - reverse for better visualization
    products = top_products['product'].tolist()[::-1]
    q1_scores = top_products['Q1_score'].tolist()[::-1]
    q2_scores = top_products['Q2_score'].tolist()[::-1]
    q3_scores = top_products['Q3_score'].tolist()[::-1]
    q4_scores = top_products['Q4_score'].tolist()[::-1]
    total_scores = top_products['total_score'].tolist()[::-1]
    ranks = top_products['rank'].tolist()[::-1]
    
    # Truncate product names
    product_labels = [p[:45] + '...' if len(p) > 45 else p for p in products]
    
    # Y positions
    y_pos = np.arange(len(products))
    
    # Colors for each quarter
    colors_q = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']  # Blue, Green, Orange, Red
    
    # Create stacked bars
    bars1 = ax.barh(y_pos, q1_scores, color=colors_q[0], alpha=0.9, label='Q1', edgecolor='white')
    bars2 = ax.barh(y_pos, q2_scores, left=q1_scores, color=colors_q[1], alpha=0.9, label='Q2', edgecolor='white')
    
    # Calculate cumulative for Q3
    left_q3 = [q1 + q2 for q1, q2 in zip(q1_scores, q2_scores)]
    bars3 = ax.barh(y_pos, q3_scores, left=left_q3, color=colors_q[2], alpha=0.9, label='Q3', edgecolor='white')
    
    # Calculate cumulative for Q4
    left_q4 = [q1 + q2 + q3 for q1, q2, q3 in zip(q1_scores, q2_scores, q3_scores)]
    bars4 = ax.barh(y_pos, q4_scores, left=left_q4, color=colors_q[3], alpha=0.9, label='Q4', edgecolor='white')
    
    # Set y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(product_labels, fontsize=10)
    
    # Add rank labels on the left
    for i, rank in enumerate(ranks):
        ax.text(-max(total_scores)*0.03, i, f"#{rank}", 
               ha='right', va='center', fontweight='bold', fontsize=11)
    
    # Add total score labels at the end of bars
    for i, total in enumerate(total_scores):
        ax.text(total + max(total_scores)*0.01, i, f'{int(total)} pts',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add quarter score labels inside bars (if space allows)
    for i in range(len(products)):
        quarters_data = [
            (q1_scores[i], 0, 'Q1'),
            (q2_scores[i], q1_scores[i], 'Q2'),
            (q3_scores[i], left_q3[i], 'Q3'),
            (q4_scores[i], left_q4[i], 'Q4')
        ]
        
        for score, left_pos, label in quarters_data:
            if score > 0:  # Only show if there's a score
                center = left_pos + score / 2
                ax.text(center, i, f'{int(score)}', 
                       ha='center', va='center', color='white', 
                       fontsize=9, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Total Borda Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Analisis Borda Count - Top {top_n} Produk Tahun {year}\n(Kontribusi Skor Per Kuartal)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    
    # Grid and styling
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"borda_count_process_{year}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


# ==============================================================================
# COMBINED VISUALIZATION FUNCTIONS (2025 & 2026 in single plots)
# ==============================================================================

def plot_yearly_top5_combined(
    yearly_results: Dict[int, pd.DataFrame],
    top_n: int = 5,
    output_dir: str = PLOTS_DIR_QUARTERLY
) -> str:
    """
    Plot top 5 produk tahunan dalam 1 figure (side-by-side subplots).
    
    Args:
        yearly_results: Dictionary {year: DataFrame} dari borda_count_ranking
        top_n: Number of top products to display (default: 5)
        output_dir: Directory untuk menyimpan plot
        
    Returns:
        Path to saved plot file
    """
    # Ambil semua tahun yang tersedia (maksimal 2 tahun pertama)
    years = sorted(yearly_results.keys())[:2]
    
    if len(years) == 0:
        print("  No data available for yearly visualization")
        return ""
    
    print(f"\n=== Creating Combined Yearly Top-{top_n} Plot ({'-'.join(map(str, years))}) ===")
    
    # Color palettes untuk setiap tahun (dinamis)
    color_palettes = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples]
    year_colors = {
        year: color_palettes[i % len(color_palettes)](np.linspace(0.4, 0.85, top_n))[::-1]
        for i, year in enumerate(years)
    }
    
    # Create figure dengan side-by-side subplots
    fig, axes = plt.subplots(1, len(years), figsize=(16, 7), sharey=False)
    if len(years) == 1:
        axes = [axes]
    
    max_score = 0
    for year in years:
        if year in yearly_results:
            max_score = max(max_score, yearly_results[year]['total_score'].max())
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        
        if year not in yearly_results or len(yearly_results[year]) == 0:
            ax.text(0.5, 0.5, f'{year}\nNo Data', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Get top N products
        top_products = yearly_results[year].head(top_n).copy()
        
        # Prepare data - reverse for horizontal bar (top to bottom)
        products = top_products['product'].tolist()[::-1]
        scores = top_products['total_score'].tolist()[::-1]
        ranks = top_products['rank'].tolist()[::-1]
        
        # Get colors for this year
        colors = year_colors.get(year, plt.cm.Greys(np.linspace(0.4, 0.85, len(products)))[::-1])
        
        # Create horizontal bars
        y_pos = np.arange(len(products))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.9, 
                       edgecolor='white', linewidth=2, height=0.7)
        
        # Customize y-axis
        ax.set_yticks(y_pos)
        product_labels = [p[:35] + '...' if len(p) > 35 else p for p in products]
        ax.set_yticklabels(product_labels, fontsize=10, fontweight='medium')
        
        # Add rank badges on the left (warna dinamis)
        badge_colors_list = ['#2c3e50', '#c0392b', '#196f3d', '#6c3483']
        badge_color = badge_colors_list[idx % len(badge_colors_list)]
        for i, (rank, product) in enumerate(zip(ranks, products)):
            ax.text(-max_score*0.08, i, f"#{rank}", 
                   ha='center', va='center', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor=badge_color, 
                            edgecolor='white', linewidth=1.5),
                   color='white')
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + max_score*0.02, bar.get_y() + bar.get_height()/2,
                   f'{int(score)} pts', ha='left', va='center', 
                   fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Labels and styling (warna dinamis)
        ax.set_xlabel('Total Borda Score', fontsize=12, fontweight='bold')
        title_colors_list = ['#1a5f7a', '#d35400', '#196f3d', '#6c3483']
        title_color = title_colors_list[idx % len(title_colors_list)]
        ax.set_title(f'Tahun {year}', fontsize=14, fontweight='bold', 
                    color=title_color, pad=15)
        
        # Grid and styling
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='#bdc3c7')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_xlim(0, max_score * 1.25)
    
    # Main title (dinamis berdasarkan tahun)
    years_str = '-'.join(map(str, years))
    fig.suptitle(f'Top {top_n} Produk Berdasarkan Borda Count Voting ({years_str})', 
                fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
    
    # Footer note
    fig.text(0.5, 0.01, 'Metode: Borda Count Voting dari Q1-Q4', 
            ha='center', fontsize=10, style='italic', color='#7f8c8d')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    output_path = os.path.join(output_dir, "top5_yearly.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


def plot_quarterly_top5_combined(
    quarterly_data: Dict[int, Dict[str, pd.DataFrame]],
    top_n: int = 5,
    output_dir: str = PLOTS_DIR_QUARTERLY
) -> str:
    """
    Plot top 5 produk per kuartal dalam 1 figure (NxM grid).
    
    Args:
        quarterly_data: Dictionary {year: {quarter: DataFrame}} dari aggregate_to_quarterly
        top_n: Number of top products to display per quarter (default: 5)
        output_dir: Directory untuk menyimpan plot
        
    Returns:
        Path to saved plot file
    """
    # Ambil tahun yang tersedia (maksimal 2)
    years = sorted(quarterly_data.keys())[:2]
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    if len(years) == 0:
        print("  No data available for quarterly visualization")
        return ""
    
    years_str = '-'.join(map(str, years))
    print(f"\n=== Creating Combined Quarterly Top-{top_n} Plot ({years_str}) ===")
    
    # Create subplot grid (rows = years, cols = quarters)
    n_rows = len(years)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Color palettes untuk setiap tahun (dinamis)
    palette_list = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples]
    year_palettes = {
        year: palette_list[i % len(palette_list)]
        for i, year in enumerate(years)
    }
    
    # Get all unique products for consistent coloring
    all_products = set()
    for year in years:
        if year in quarterly_data:
            for quarter in quarters:
                if quarter in quarterly_data[year]:
                    all_products.update(quarterly_data[year][quarter]['product'].tolist())
    
    # Find global max for consistent x-axis scale
    global_max = 0
    for year in years:
        if year in quarterly_data:
            for quarter in quarters:
                if quarter in quarterly_data[year]:
                    max_val = quarterly_data[year][quarter]['quarterly_sum'].max()
                    global_max = max(global_max, max_val)
    
    for row, year in enumerate(years):
        palette = year_palettes.get(year, plt.cm.Greys)
        
        for col, quarter in enumerate(quarters):
            ax = axes[row, col]
            
            # Check data availability
            if year not in quarterly_data or quarter not in quarterly_data[year]:
                ax.text(0.5, 0.5, f'{quarter} {year}\nNo Data', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes,
                       color='#7f8c8d')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            
            quarter_df = quarterly_data[year][quarter]
            if len(quarter_df) == 0:
                ax.text(0.5, 0.5, f'{quarter} {year}\nNo Data', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes,
                       color='#7f8c8d')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            
            # Get top N for this quarter
            top_data = quarter_df.head(top_n).copy()
            
            # Prepare data - reverse for better visualization
            products = top_data['product'].tolist()[::-1]
            values = top_data['quarterly_sum'].tolist()[::-1]
            ranks = top_data['rank'].tolist()[::-1]
            
            # Create gradient colors
            colors = palette(np.linspace(0.3, 0.85, len(products)))[::-1]
            
            # Create horizontal bars
            y_pos = np.arange(len(products))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.9, 
                          edgecolor='white', linewidth=1.5, height=0.7)
            
            # Truncate long names
            product_labels = [p[:25] + '...' if len(p) > 25 else p for p in products]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(product_labels, fontsize=8, fontweight='medium')
            
            # Add rank badges (warna dinamis per tahun)
            badge_colors = ['#2980b9', '#d35400', '#27ae60', '#8e44ad']
            badge_color = badge_colors[row % len(badge_colors)]
            for i, rank in enumerate(ranks):
                ax.text(-global_max*0.05, i, f"#{rank}", 
                       ha='center', va='center', fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='circle,pad=0.2', facecolor=badge_color, 
                                edgecolor='white', linewidth=1),
                       color='white')
            
            # Add value labels at end of bars
            for bar, val in zip(bars, values):
                width = bar.get_width()
                if val >= 1000:
                    label = f'{val/1000:.1f}K'
                else:
                    label = f'{val:.0f}'
                ax.text(width + global_max*0.02, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center', fontsize=8, fontweight='bold')
            
            # Styling (warna dinamis)
            title_colors = ['#1a5f7a', '#c0392b', '#196f3d', '#6c3483']
            title_color = title_colors[row % len(title_colors)]
            ax.set_title(f'{quarter} {year}', fontsize=12, fontweight='bold', 
                        color=title_color, pad=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--', color='#bdc3c7')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0, global_max * 1.2)
            
            # Only show x-label on bottom row
            if row == n_rows - 1:
                ax.set_xlabel('Total Forecast', fontsize=9)
    
    # Main title (dinamis)
    fig.suptitle(f'Top {top_n} Produk Per Kuartal ({years_str})', 
                fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')
    
    # Year labels on the left (dinamis)
    year_label_colors = ['#1a5f7a', '#c0392b', '#196f3d', '#6c3483']
    for i, year in enumerate(years):
        y_pos = 1 - (i + 0.5) / n_rows
        fig.text(0.01, y_pos, str(year), fontsize=16, fontweight='bold', 
                color=year_label_colors[i % len(year_label_colors)], rotation=90, va='center')
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.95])
    
    # Save plot
    output_path = os.path.join(output_dir, "top5_quarterly.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


def plot_borda_count_process_combined(
    yearly_results: Dict[int, pd.DataFrame],
    top_n: int = 5,
    output_dir: str = PLOTS_DIR_QUARTERLY
) -> str:
    """
    Create combined Borda Count process visualization.
    Shows stacked horizontal bars with quarterly score contributions for all years.
    
    Args:
        yearly_results: Dictionary {year: DataFrame} dari borda_count_ranking
        top_n: Number of top products to display (default: 5)
        output_dir: Directory untuk menyimpan plot
        
    Returns:
        Path to saved plot file
    """
    # Ambil semua tahun yang tersedia (maksimal 2)
    years = sorted(yearly_results.keys())[:2]
    
    if len(years) == 0:
        print("  No data available for Borda Count visualization")
        return ""
    
    years_str = '-'.join(map(str, years))
    print(f"\n=== Creating Combined Borda Count Process Visualization ({years_str}) ===")
    
    # Colors for each quarter - theme per year (dinamis)
    quarter_color_themes = [
        ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c'],  # Blues/Greens theme
        ['#e74c3c', '#f39c12', '#e91e63', '#ff5722'],  # Oranges/Reds theme
        ['#16a085', '#27ae60', '#2980b9', '#8e44ad'],  # Teal/Purple theme
        ['#d35400', '#c0392b', '#7b241c', '#943126'],  # Warm theme
    ]
    quarter_colors = {
        year: quarter_color_themes[i % len(quarter_color_themes)]
        for i, year in enumerate(years)
    }
    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Create figure with side-by-side subplots
    fig, axes = plt.subplots(1, len(years), figsize=(9 * len(years), 8))
    if len(years) == 1:
        axes = [axes]
    
    # Find global max score for consistent scale
    global_max = 0
    for year in years:
        if year in yearly_results:
            global_max = max(global_max, yearly_results[year]['total_score'].max())
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        
        if year not in yearly_results or len(yearly_results[year]) == 0:
            ax.text(0.5, 0.5, f'{year}\nNo Data', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes)
            continue
        
        # Get top N products
        top_products = yearly_results[year].head(top_n).copy()
        
        # Prepare data - reverse for better visualization
        products = top_products['product'].tolist()[::-1]
        q1_scores = top_products['Q1_score'].tolist()[::-1]
        q2_scores = top_products['Q2_score'].tolist()[::-1]
        q3_scores = top_products['Q3_score'].tolist()[::-1]
        q4_scores = top_products['Q4_score'].tolist()[::-1]
        total_scores = top_products['total_score'].tolist()[::-1]
        ranks = top_products['rank'].tolist()[::-1]
        
        # Truncate product names
        product_labels = [p[:35] + '...' if len(p) > 35 else p for p in products]
        
        # Y positions
        y_pos = np.arange(len(products))
        colors = quarter_colors.get(year, ['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        
        # Create stacked bars
        bars1 = ax.barh(y_pos, q1_scores, color=colors[0], alpha=0.9, 
                       label='Q1', edgecolor='white', linewidth=1, height=0.7)
        bars2 = ax.barh(y_pos, q2_scores, left=q1_scores, color=colors[1], 
                       alpha=0.9, label='Q2', edgecolor='white', linewidth=1, height=0.7)
        
        # Calculate cumulative for Q3
        left_q3 = [q1 + q2 for q1, q2 in zip(q1_scores, q2_scores)]
        bars3 = ax.barh(y_pos, q3_scores, left=left_q3, color=colors[2], 
                       alpha=0.9, label='Q3', edgecolor='white', linewidth=1, height=0.7)
        
        # Calculate cumulative for Q4
        left_q4 = [q1 + q2 + q3 for q1, q2, q3 in zip(q1_scores, q2_scores, q3_scores)]
        bars4 = ax.barh(y_pos, q4_scores, left=left_q4, color=colors[3], 
                       alpha=0.9, label='Q4', edgecolor='white', linewidth=1, height=0.7)
        
        # Set y-axis
        ax.set_yticks(y_pos)
        ax.set_yticklabels(product_labels, fontsize=10, fontweight='medium')
        
        # Add rank badges on the left (warna dinamis)
        badge_colors_list = ['#2c3e50', '#c0392b', '#196f3d', '#6c3483']
        badge_color = badge_colors_list[idx % len(badge_colors_list)]
        for i, rank in enumerate(ranks):
            ax.text(-global_max*0.08, i, f"#{rank}", 
                   ha='center', va='center', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor=badge_color, 
                            edgecolor='white', linewidth=1.5),
                   color='white')
        
        # Add total score labels at the end of bars
        for i, total in enumerate(total_scores):
            ax.text(total + global_max*0.02, i, f'{int(total)} pts',
                   ha='left', va='center', fontsize=10, fontweight='bold', color='#2c3e50')
        
        # Add quarter score labels inside bars (if space allows)
        all_quarter_data = [
            (q1_scores, [0]*len(products)),
            (q2_scores, q1_scores),
            (q3_scores, left_q3),
            (q4_scores, left_q4)
        ]
        
        for q_idx, (scores, lefts) in enumerate(all_quarter_data):
            for i in range(len(products)):
                score = scores[i]
                left = lefts[i]
                if score >= 1:  # Only show if there's a meaningful score
                    center = left + score / 2
                    ax.text(center, i, f'{int(score)}', 
                           ha='center', va='center', color='white', 
                           fontsize=9, fontweight='bold')
        
        # Labels and title (warna dinamis)
        ax.set_xlabel('Total Borda Score', fontsize=12, fontweight='bold')
        title_colors_list = ['#1a5f7a', '#c0392b', '#196f3d', '#6c3483']
        title_color = title_colors_list[idx % len(title_colors_list)]
        ax.set_title(f'Tahun {year}', fontsize=14, fontweight='bold', 
                    color=title_color, pad=15)
        
        # Legend
        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10,
                 fancybox=True, framealpha=0.9)
        
        # Grid and styling
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='#bdc3c7')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, global_max * 1.25)
    
    # Main title (dinamis)
    fig.suptitle(f'Analisis Borda Count - Top {top_n} Produk ({years_str})\n(Kontribusi Skor Per Kuartal)', 
                fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save plot
    output_path = os.path.join(output_dir, "borda_count_process.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


def run_forecast(excel_path: str = DEFAULT_EXCEL, models_dir: str = MODELS_DIR, top_n: int = 10):
    ensure_dirs()

    models_meta = load_models_metadata(models_dir)
    global_stats, cat_stats = load_global_category_stats(models_dir)

    df = read_excel_latest(excel_path)
    monthly = monthly_aggregate(df)

    # Find the global most recent date across all products
    global_last_date = pd.to_datetime(monthly["month"]).max()
    print(f"Most recent data date across all products: {global_last_date.strftime('%Y-%m-%d')}")
    
    # Define a single forecast period starting from the next month after global_last_date
    forecast_start = global_last_date + pd.offsets.MonthBegin(1)
    forecast_index = pd.date_range(start=forecast_start, periods=FORECAST_HORIZON_MONTHS, freq="MS")
    print(f"Forecasting period: {forecast_index[0].strftime('%Y-%m-%d')} to {forecast_index[-1].strftime('%Y-%m-%d')}")

    products = sorted(monthly["product_norm"].unique().tolist())

    rows = []
    diag_rows = []

    for prod in products:
        g = monthly[monthly["product_norm"] == prod]
        category = g["category"].iloc[0] if len(g) else ""
        model, scaler, feats = load_product_artifacts(models_dir, prod)
        
        # Generate forecast regardless of model availability
        if model is not None and scaler is not None and feats is not None:
            try:
                idx, mean, p10, p50, p90, diag = forecast_for_product(prod, g, model, scaler, feats)
            except Exception:
                idx, mean, p10, p50, p90, diag = fallback_forecast(prod, g, category, global_stats, cat_stats, forecast_start)
        else:
            idx, mean, p10, p50, p90, diag = fallback_forecast(prod, g, category, global_stats, cat_stats, forecast_start)

        # Use the global forecast_index instead of product-specific index
        for i, d in enumerate(forecast_index):
            rows.append({
                "product": prod,
                "category": category,
                "date": d.strftime("%Y-%m-01"),
                "mean": float(mean[i]) if i < len(mean) else 0.0,
                "p10": float(p10[i]) if i < len(p10) else 0.0,
                "p50": float(p50[i]) if i < len(p50) else 0.0,
                "p90": float(p90[i]) if i < len(p90) else 0.0,
            })
        diag_rows.append({
            "product": prod,
            "category": category,
            **diag,
        })

    # Save per-product forecast
    fpp = pd.DataFrame(rows)
    fpp.sort_values(["product", "date"], inplace=True)
    fpp.to_csv(OUT_FORECAST_PER_PRODUCT, index=False)

    # Total aggregate across all products
    total = (fpp.groupby("date")["mean"].sum().reset_index().rename(columns={"mean": "total_mean"}))
    total.to_csv(OUT_FORECAST_TOTAL, index=False)

    # Top-N per month by mean
    topn_rows = []
    for d, g in fpp.groupby("date"):
        topg = g.sort_values("mean", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(topg.iterrows(), start=1):
            topn_rows.append({
                "date": d,
                "rank": rank,
                "product": row["product"],
                "category": row["category"],
                "mean": row["mean"],
            })
    pd.DataFrame(topn_rows).to_csv(OUT_TOPN, index=False)

    # Diagnostics
    pd.DataFrame(diag_rows).to_csv(OUT_DIAG, index=False)

    # Plots
    try:
        # Generate grouped bar chart: 24 months on X-axis, each month shows top-5 products
        topn_df = pd.read_csv(OUT_TOPN)
        if not topn_df.empty:
            # Ensure date is datetime
            topn_df["date"] = pd.to_datetime(topn_df["date"], errors="coerce")
            topn_df = topn_df.dropna(subset=["date"]).copy()
            
            # Get unique months sorted
            unique_months = sorted(topn_df["date"].unique())
            n_months = len(unique_months)
            
            if n_months > 0:
                # Prepare data structure: {month: [(product, value), ...]}
                month_data = {}
                for month in unique_months:
                    month_sub = topn_df[topn_df["date"] == month].sort_values("mean", ascending=False).head(5)
                    month_data[month] = list(zip(month_sub["product"].tolist(), 
                                                  month_sub["mean"].astype(float).tolist()))
                
                # Get all unique products across all months for consistent coloring
                all_products = set()
                for month_products in month_data.values():
                    for prod, _ in month_products:
                        all_products.add(prod)
                
                # Assign colors to products
                product_list = sorted(all_products)
                cmap = plt.get_cmap("tab20")  # Use tab20 for more colors
                color_map = {prod: cmap(i % 20) for i, prod in enumerate(product_list)}
                
                # Create improved grouped bar chart
                # Dynamic figure size: width scales with months
                fig_width = max(16, n_months * 0.8)
                fig, ax = plt.subplots(figsize=(fig_width, 8))
                
                # Bar layout configuration
                bar_width = 0.18  # wider bars for readability
                month_positions = np.arange(n_months)
                # Center 5 bars within each month range with small gaps
                offsets = np.linspace(-0.36, 0.36, 5)
                
                # Helper: smart number formatting (K, M)
                def _fmt_val(v: float) -> str:
                    try:
                        v = float(v)
                    except Exception:
                        return str(v)
                    if v >= 1_000_000:
                        return f"{v/1_000_000:.1f}M"
                    if v >= 1_000:
                        return f"{v/1_000:.1f}K"
                    return f"{v:.0f}"
                
                # Pre-compute totals per product for legend ordering
                totals: Dict[str, float] = {p: 0.0 for p in product_list}
                for month in unique_months:
                    for prod, val in month_data.get(month, []):
                        totals[prod] = totals.get(prod, 0.0) + float(val)
                
                # Draw bars per rank position
                for rank_idx in range(5):  # Top 5 per month
                    positions = []
                    heights = []
                    colors = []
                    for month_idx, month in enumerate(unique_months):
                        products_values = month_data.get(month, [])
                        if rank_idx < len(products_values):
                            prod, val = products_values[rank_idx]
                            positions.append(month_idx + offsets[rank_idx])
                            heights.append(val)
                            colors.append(color_map.get(prod, "#999999"))
                        else:
                            positions.append(month_idx + offsets[rank_idx])
                            heights.append(0.0)
                            colors.append("#e5e7eb")  # light gray for missing
                    ax.bar(positions, heights, bar_width, color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
                
                # Annotate only the top (rank 1) per month to reduce clutter
                for month_idx, month in enumerate(unique_months):
                    products_values = month_data.get(month, [])
                    if not products_values:
                        continue
                    top_prod, top_val = products_values[0]
                    x_pos = month_idx + offsets[0]
                    ax.text(x_pos, float(top_val) * 1.01, _fmt_val(top_val), ha="center", va="bottom", fontsize=9)
                
                # X-axis labels: compact month format and rotation
                month_labels = [pd.to_datetime(m).strftime("%b'%y") for m in unique_months]
                ax.set_xticks(month_positions)
                ax.set_xticklabels(month_labels, fontsize=10, rotation=50, ha='right')
                
                # Axis labels and title
                ax.set_ylabel("Nilai Forecast", fontsize=11)
                ax.set_xlabel("Bulan", fontsize=11)
                ax.set_title("Top-5 Produk per Bulan (24 Bulan Forecast)", fontsize=13, fontweight='bold')
                
                # Grid and spines
                ax.grid(axis='y', color="#e5e7eb", alpha=0.8)
                ax.set_axisbelow(True)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                
                # Legend: only products that appear, ordered by total forecast, max 10 entries
                from matplotlib.patches import Patch
                appeared = [p for p in product_list if totals.get(p, 0.0) > 0]
                appeared_sorted = sorted(appeared, key=lambda p: totals.get(p, 0.0), reverse=True)[:10]
                legend_elements = [Patch(facecolor=color_map[p], label=p) for p in appeared_sorted]
                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                              fontsize=9, title="Produk", ncol=1, frameon=False)
                
                plt.tight_layout()
                out_path = os.path.join(PLOTS_DIR, "top5_grouped_24m.png")
                plt.savefig(out_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                
                print(f"Chart Top-5 grouped disimpan di: {out_path}")

        # Total aggregate plot
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(pd.to_datetime(total["date"]), total["total_mean"], marker='o')
        ax2.set_title('Total Forecast (24m)')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Total Mean Quantity')
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(PLOTS_DIR, "forecast_total_24m_clean.png"))
        plt.close(fig2)
    except Exception as e:
        # Plotting should not fail the pipeline
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        pass

    # ------------------------
    # Quarterly Aggregation and Borda Count Ranking
    # ------------------------
    print("\n" + "="*60)
    print("QUARTERLY AGGREGATION & BORDA COUNT RANKING")
    print("="*60)
    
    try:
        # Read forecast data
        forecast_data = pd.read_csv(OUT_FORECAST_PER_PRODUCT)
        
        # Aggregate to quarterly with top 5 per quarter
        quarterly_data = aggregate_to_quarterly(forecast_data, top_n=5)
        
        # Process each year
        yearly_results = {}
        for year in sorted(quarterly_data.keys()):
            print(f"\n{'='*60}")
            print(f"Processing Year: {year}")
            print(f"{'='*60}")
            
            # Save quarterly rankings to CSV
            quarterly_csv_path = OUT_QUARTERLY_TOP5_TEMPLATE.format(year=year)
            quarterly_combined = []
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                if quarter in quarterly_data[year]:
                    quarter_df = quarterly_data[year][quarter].copy()
                    quarter_df['year'] = year
                    quarter_df['quarter'] = quarter
                    quarterly_combined.append(quarter_df)
            
            if quarterly_combined:
                quarterly_df_full = pd.concat(quarterly_combined, ignore_index=True)
                quarterly_df_full = quarterly_df_full[['year', 'quarter', 'rank', 'product', 'category', 'quarterly_sum']]
                quarterly_df_full.to_csv(quarterly_csv_path, index=False)
                print(f"\nQuarterly rankings saved: {quarterly_csv_path}")
            
            # Calculate Borda Count ranking
            borda_results = borda_count_ranking(quarterly_data[year], year=year, top_n=5)
            yearly_results[year] = borda_results
            
            # Save Borda results to CSV
            yearly_csv_path = OUT_YEARLY_TOP5_TEMPLATE.format(year=year)
            borda_results.to_csv(yearly_csv_path, index=False)
            print(f"Yearly Borda rankings saved: {yearly_csv_path}")
        
        # ------------------------
        # Generate COMBINED Visualizations (2025 & 2026 in single plots)
        # ------------------------
        print("\n" + "="*60)
        print("GENERATING COMBINED VISUALIZATIONS (2025-2026)")
        print("="*60)
        
        # 1. Combined Yearly Top 5 Plot (side-by-side 2025 & 2026)
        plot_yearly_top5_combined(yearly_results, top_n=5, output_dir=PLOTS_DIR_QUARTERLY)
        
        # 2. Combined Quarterly Top 5 Plot (2x4 grid for all quarters)
        plot_quarterly_top5_combined(quarterly_data, top_n=5, output_dir=PLOTS_DIR_QUARTERLY)
        
        # 3. Combined Borda Count Process Visualization
        plot_borda_count_process_combined(yearly_results, top_n=5, output_dir=PLOTS_DIR_QUARTERLY)
        
        print("\n" + "="*60)
        print("QUARTERLY AGGREGATION & VISUALIZATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nOutput files generated:")
        print(f"  - {os.path.join(PLOTS_DIR_QUARTERLY, 'top5_yearly.png')}")
        print(f"  - {os.path.join(PLOTS_DIR_QUARTERLY, 'top5_quarterly.png')}")
        print(f"  - {os.path.join(PLOTS_DIR_QUARTERLY, 'borda_count_process.png')}")
        
    except Exception as e:
        print(f"\nError during quarterly aggregation: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with regular forecast output...")

    return {
        "per_product_csv": OUT_FORECAST_PER_PRODUCT,
        "total_csv": OUT_FORECAST_TOTAL,
        "topn_csv": OUT_TOPN,
        "diagnostics_csv": OUT_DIAG,
        "plots_dir": PLOTS_DIR,
        "quarterly_plots_dir": PLOTS_DIR_QUARTERLY,
    }


if __name__ == "__main__":
    import sys
    # Jika dijalankan dengan argumen "test", jalankan doctest
    # Jika tidak, jalankan forecast
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import doctest
        doctest.testmod(verbose=True)
    else:
        print("="*60)
        print("RUNNING LSTM FORECAST")
        print("="*60)
        info = run_forecast(DEFAULT_EXCEL, MODELS_DIR)
        print("\nForecasting completed:", info)
