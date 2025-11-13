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

# Experimental/diagnostic flags (Priority-1 fixes)
# Set to True to disable the respective post-processing; helps avoid over-constraining forecasts
DISABLE_STABILIZATION = True   # bypass stabilize_series during testing
DISABLE_CLAMPING = True        # bypass clamp_with_historical_quantiles during testing
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

    # simple clipping per product to avoid pathological spikes
    def _clip_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < 3:
            return g
        upper = g["sales"].quantile(0.99)
        g["sales"] = g["sales"].clip(lower=0)
        g["sales"] = np.minimum(g["sales"], upper)
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
        # Enhanced baseline with stronger trend and more random variation
        trend_component = recent_trend * (i + 1) * 0.5  # Much stronger trend
        random_component = rng.normal(0, last_value * 0.15)  # 15% random variation
        
        # Add monthly variation component
        month_variation = 1.0 + 0.1 * np.sin(2 * np.pi * d.month / 12)
        
        # Add additional random walk component
        walk_component = rng.normal(0, last_value * 0.05)
        
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
    """Apply very generous MoM change limits to preserve forecast dynamics.
    - Much more generous bounds to avoid killing natural variation
    - Only apply minimal constraints for extreme outliers
    """
    if len(hist_values) == 0:
        return np.maximum(mean_forecast, 0.0)
    
    last = float(hist_values[-1])
    cv = float(np.std(hist_values) / (np.mean(hist_values) + 1e-6)) if len(hist_values) > 1 else 0.5
    
    # Much more generous bounds: allow 200-500% monthly change based on CV
    k = 5.0  # Increased significantly from 2.5
    max_monthly_change = max(2.0, min(5.0, k * cv))  # Between 200% and 500%
    
    out = []
    prev = last
    for m in mean_forecast:
        upper = prev * (1 + max_monthly_change)
        lower = prev * max(0.01, (1 - max_monthly_change))  # Allow down to 1% of previous
        val = float(np.clip(m, lower, upper))
        out.append(val)
        prev = val
    
    return np.array(out)


def clamp_with_historical_quantiles(arr: np.ndarray, hist_values: np.ndarray) -> np.ndarray:
    """Apply extremely generous bounds based on historical data.
    Very wide bounds to avoid killing natural variation: q001–q999 and 0.01x–100x.
    """
    if len(hist_values) == 0:
        return np.maximum(arr, 0.0)
    
    # Use extremely wide bounds
    lo = np.quantile(hist_values, 0.001)
    hi = np.quantile(hist_values, 0.999)
    mean_hist = np.mean(hist_values)
    
    # Allow forecasts to be 0.01x to 100x of historical central tendency/range
    lower_bound = max(0.0, lo * 0.01, mean_hist * 0.01)
    upper_bound = max(hi * 100.0, mean_hist * 100.0)
    
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
            
            # Add significant noise to break perfect cycles
            if NOISE_INJECTION:
                noise_factor = rng.normal(0, 0.15)  # 15% noise
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
                # Much stronger noise to break perfect cycles
                pred_resid = float(pred_resid + rng.normal(0.0, resid_std * 0.5))
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
    hist_g = hist_g.sort_values("month").copy()
    
    # Use provided forecast_start_date or calculate from data
    if forecast_start_date is None:
        dates = pd.to_datetime(hist_g["month"]) if len(hist_g) else pd.to_datetime([global_stats.get("last_date", datetime.utcnow().strftime("%Y-%m-01"))])
        last_date = dates.iloc[-1]
        forecast_start_date = last_date + pd.offsets.MonthBegin(1)
    
    future_index = pd.date_range(start=forecast_start_date, periods=FORECAST_HORIZON_MONTHS, freq="MS")
    
    # Use historical mean if available, otherwise category median
    if len(hist_g) > 0 and hist_g["qty"].mean() > 0:
        base = float(hist_g["qty"].mean())
    else:
        base = cat_stats.get(category, {}).get("median", global_stats.get("global_median", 1.0))
    
    # Add realistic variation based on month and trend
    mean = []
    rng = np.random.RandomState(42)
    base_trend = 0.02 if len(hist_g) > 6 else 0  # 2% monthly growth trend
    
    for i, d in enumerate(future_index):
        # Add trend component
        trend_factor = 1 + base_trend * i
        
        # Add small random variation (no strong seasonality)
        random_factor = rng.normal(1.0, 0.1)  # 10% random variation
        
        # Add very mild month-based variation (not perfect seasonal)
        month_variation = 1.0 + 0.05 * np.sin(2 * np.pi * (d.month - 1) / 12)
        
        val = base * trend_factor * random_factor * month_variation
        mean.append(max(0.1, float(val)))
    
    mean = np.array(mean)
    
    # Uncertainty based on category or global stats
    qspread = max(0.2 * base, cat_stats.get(category, {}).get("std", global_stats.get("global_std", 0.2 * base)))
    
    p10 = np.maximum(mean - 0.8 * qspread, 0.1)  # Minimum 0.1 to avoid zeros
    p50 = mean.copy()
    p90 = mean + 0.8 * qspread
    diag = {
        "hist_n": int(len(hist_g)),
        "hist_cv": float(np.std(hist_g["qty"]) / (np.mean(hist_g["qty"]) + 1e-6)) if len(hist_g) > 1 else 0.0,
        "used_model": False,
    }
    return future_index, mean, p10, p50, p90, diag


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


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

    return {
        "per_product_csv": OUT_FORECAST_PER_PRODUCT,
        "total_csv": OUT_FORECAST_TOTAL,
        "topn_csv": OUT_TOPN,
        "diagnostics_csv": OUT_DIAG,
        "plots_dir": PLOTS_DIR,
    }


if __name__ == "__main__":
    info = run_forecast(DEFAULT_EXCEL, MODELS_DIR)
    print("Forecasting completed:", info)
