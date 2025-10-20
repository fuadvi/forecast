import os
import json
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import warnings
import numpy as np
import pandas as pd

# Suppress sklearn and general warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
import pandas as pd
from keras import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Optional reuse from in-repo utilities
# try:
#     import test2
# except Exception:
test2 = None

# ------------------------
# Configuration
# ------------------------
FORECAST_HORIZON_MONTHS = 24
TIME_STEPS = 6
MIN_DATA_POINTS_MONTHS = 8
MIN_NONZERO_TRANSACTIONS = 3
OUTPUT_DIR = os.path.join(os.getcwd(), "trained_models")
DIAG_CSV = os.path.join(OUTPUT_DIR, "training_diagnostics.csv")
SKIPPED_LOG = os.path.join(OUTPUT_DIR, "skipped_products.log")
MODELS_META_PATH = os.path.join(OUTPUT_DIR, "models_metadata.json")
GLOBAL_STATS_PATH = os.path.join(OUTPUT_DIR, "global_stats.json")
CATEGORY_STATS_PATH = os.path.join(OUTPUT_DIR, "category_stats.json")

# Input Excel file (project root)
DEFAULT_EXCEL = os.path.join(os.getcwd(), "Data_Penjualan_Dengan_ID_Pelanggan.xlsx")

# Outputs
OUT_FORECAST_PER_PRODUCT = os.path.join(os.getcwd(), "forecast_per_product_24m.csv")
OUT_FORECAST_TOTAL = os.path.join(os.getcwd(), "forecast_total_24m.csv")
OUT_TOPN = os.path.join(os.getcwd(), "topN_per_month_24m.csv")
OUT_DIAG = os.path.join(os.getcwd(), "forecast_diagnostics.csv")
PLOTS_DIR = os.path.join(os.getcwd(), "forecast_plots", "bulan")

# ------------------------
# Utility functions (fallbacks if test2 is absent)
# ------------------------

def _normalize_product_name(name: str) -> str:
    if test2 and hasattr(test2, "normalize_product_name"):
        return test2.normalize_product_name(name)
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s

def _create_sequences(data: np.ndarray, time_steps: int, target_idx: int):
    if test2 and hasattr(test2, "create_sequences"):
        return test2.create_sequences(data, time_steps, target_idx)
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, target_idx])
    return np.array(X), np.array(y)


def _select_sma_window(series: pd.Series, windows=(3, 4, 5, 6)) -> int:
    # try to reuse from test2
    if test2 and hasattr(test2, "_select_sma_window"):
        return test2._select_sma_window(series, windows)
    best_w, best_mae = windows[0], float("inf")
    s = series.astype(float).values
    for w in windows:
        if len(s) < w + 2:
            continue
        sma = pd.Series(s).rolling(w, min_periods=1).mean().shift(1).bfill().values
        mae = np.mean(np.abs(s[w:] - sma[w:]))
        if mae < best_mae:
            best_mae, best_w = mae, w
    return best_w


def _build_baseline_series(sales: pd.Series, dates: pd.Series) -> Tuple[pd.Series, dict]:
    if test2 and hasattr(test2, "build_baseline_series"):
        return test2.build_baseline_series(sales, dates)
    # Simple SMA baseline with selected window + simple seasonal adjustment
    s = sales.astype(float).values
    w = _select_sma_window(sales)
    sma = pd.Series(s).rolling(w, min_periods=1).mean().shift(1)
    # seasonality (month-of-year) from last available years
    month = pd.to_datetime(dates).dt.month
    df = pd.DataFrame({"sales": sales.values, "month": month.values})
    seas = df.groupby("month")["sales"].mean()
    seas = seas / seas.mean() if seas.mean() != 0 else seas / (seas.mean() + 1e-6)
    # build baseline aligned to weekly length
    baseline = []
    for i, m in enumerate(month):
        base = (sma.iloc[i] if not np.isnan(sma.iloc[i]) else np.nanmean(s[:i + 1]))
        mult = seas.loc[m] if m in seas.index else 1.0
        baseline.append(float(base) * float(mult))
    baseline_series = pd.Series(baseline, index=pd.to_datetime(dates))
    meta = {"sma_window": int(w), "seasonality": seas.to_dict()}
    return baseline_series, meta


def _forecast_baseline_forward(last_hist_date: pd.Timestamp,
                              hist_sales: pd.Series,
                              hist_dates: pd.Series,
                              horizon: int,
                              meta: dict = None) -> pd.Series:
    if test2 and hasattr(test2, "forecast_baseline_forward"):
        return test2.forecast_baseline_forward(last_hist_date, hist_sales, hist_dates, horizon, meta)
    # Extend baseline with SMA + seasonal profile
    if meta is None:
        base_hist, meta = _build_baseline_series(hist_sales, hist_dates)
    seas = meta.get("seasonality", {})
    w = int(meta.get("sma_window", _select_sma_window(hist_sales)))
    sales = pd.Series(hist_sales.values, index=pd.to_datetime(hist_dates))
    sma = sales.rolling(w, min_periods=1).mean().iloc[-1]
    future_index = pd.date_range(start=pd.to_datetime(last_hist_date) + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future_vals = []
    for d in future_index:
        m = d.month
        mult = seas.get(m, 1.0)
        future_vals.append(float(sma) * float(mult))
    return pd.Series(future_vals, index=future_index)


def rfe_select_features(X: pd.DataFrame, y: pd.Series, min_features: int = 2, max_features: int = 6, random_state: int = 42) -> Tuple[List[str], List[str], List[dict]]:
    """Use RandomForest + RFE to select features.
    Returns kept_features, eliminated_order, history.
    """
    # Reuse if available
    if test2 and hasattr(test2, "rfe_select_features"):
        return test2.rfe_select_features(X, y, min_features=min_features, max_features=max_features, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rfe = RFE(estimator=rf, n_features_to_select=max(min_features, 1))
    rfe.fit(X, y)
    support = rfe.support_
    kept = [c for c, s in zip(X.columns.tolist(), support) if s]
    elim = [c for c in X.columns.tolist() if c not in kept]
    return kept, elim, []


def serialize_keras_model(model: Sequential) -> dict:
    """Serialize a Keras model into a pickle-friendly dict."""
    return {
        "model_json": model.to_json(),
        "weights": model.get_weights(),
    }


def build_time_features(dates: pd.Series) -> pd.DataFrame:
    month = pd.to_datetime(dates).dt.month
    df = pd.DataFrame({
        "month": month.values,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "trend": np.arange(1, len(month) + 1, dtype=float),
    }, index=pd.to_datetime(dates))
    # If less than 12 points, drop seasonality signals to avoid leakage
    if len(df) < 12:
        df["month_sin"] = 0.0
        df["month_cos"] = 0.0
    return df


def read_and_preprocess(excel_path: str) -> pd.DataFrame:
    # Try to handle both English and Indonesian column names
    df = pd.read_excel(excel_path)
    
    # Debug: Print actual column names
    print(f"Actual columns in Excel file: {list(df.columns)}")
    
    cols = {c.lower().strip(): c for c in df.columns}
    
    # More flexible column mapping - case insensitive and partial matching
    def find_column(possible_names, df_columns):
        for possible in possible_names:
            # Exact match (case insensitive)
            for col in df_columns:
                if col.lower().strip() == possible.lower().strip():
                    return col
            # Partial match (case insensitive)
            for col in df_columns:
                if possible.lower().strip() in col.lower().strip():
                    return col
        return None
    
    # Extended candidates with more variations
    date_candidates = ["Transaction Date", "Tanggal Transaksi", "date", "tanggal", "Date", "Tanggal"]
    qty_candidates = ["Quantity", "Jumlah", "sales", "qty", "Sales", "Qty", "Jumlah Unit Terjual"]
    product_candidates = ["Product Name", "Nama Produk", "product", "Product", "product_name"]
    category_candidates = ["Product Category", "Kategori Barang", "category", "Category", "Kategori"]
    
    # Find actual column names
    date_col = find_column(date_candidates, df.columns)
    qty_col = find_column(qty_candidates, df.columns)
    product_col = find_column(product_candidates, df.columns)
    category_col = find_column(category_candidates, df.columns)
    
    # Check if all required columns were found
    missing_cols = []
    if not date_col:
        missing_cols.append("date column")
    if not qty_col:
        missing_cols.append("quantity/sales column")
    if not product_col:
        missing_cols.append("product name column")
    if not category_col:
        missing_cols.append("category column")
    
    if missing_cols:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Could not find columns for: {', '.join(missing_cols)}. "
                        f"Please check that your Excel file contains the required columns.")
    
    print(f"Mapped columns: date='{date_col}', qty='{qty_col}', product='{product_col}', category='{category_col}'")
    
    # Select and rename columns
    df = df[[date_col, qty_col, product_col, category_col]].copy()
    df.columns = ["date", "sales", "product_name", "category"]

    # Normalize product names
    df["product_norm"] = df["product_name"].apply(_normalize_product_name)

    # Outlier handling: clip per product at 99th percentile
    def _clip_group(g):
        if len(g) < 3:
            return g
        up = g["sales"].quantile(0.99)
        g["sales"] = g["sales"].clip(lower=0)
        g["sales"] = np.minimum(g["sales"], up)
        return g

    df = df.dropna(subset=["date"]).copy()
    
    # Use safe date parsing instead of direct pd.to_datetime
    df["date"] = df["date"].apply(safe_parse_date)
    
    # Remove rows where date parsing failed
    before_count = len(df)
    df = df.dropna(subset=["date"])
    after_count = len(df)
    
    if before_count > after_count:
        print(f"Removed {before_count - after_count} rows with invalid dates")
        # Save invalid dates for review
        invalid_rows = df[df["date"].isna()]
        if len(invalid_rows) > 0:
            invalid_rows.to_csv("invalid_dates.csv", index=False)
            print("Invalid date entries saved to invalid_dates.csv")
    
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
    
    # Aggregate per product per month (sum of quantities)
    df["month"] = df["date"].values.astype("datetime64[M]")
    agg = (df.groupby(["product_norm", "category", "month"])['sales']
             .sum()
             .reset_index())
    agg.rename(columns={"sales": "qty"}, inplace=True)
    return agg


def filter_eligible_products(monthly_df: pd.DataFrame) -> Tuple[List[str], Dict[str, dict]]:
    eligible = []
    stats = {}
    for prod, g in monthly_df.groupby("product_norm"):
        months = len(g)
        nonzero = int((g["qty"] > 0).sum())
        if months >= MIN_DATA_POINTS_MONTHS and nonzero >= MIN_NONZERO_TRANSACTIONS:
            eligible.append(prod)
        else:
            stats[prod] = {"months": months, "nonzero": nonzero}
    return eligible, stats


def train_per_product(product: str, g: pd.DataFrame) -> dict:
    # g has columns: product_norm, category, month, qty
    g = g.sort_values("month").copy()
    dates = pd.to_datetime(g["month"])  # Month start
    sales = g["qty"].astype(float)

    # Time features
    time_feats = build_time_features(dates)

    # Baseline and residual
    baseline_hist, baseline_meta = _build_baseline_series(sales, dates)
    residual = sales.values - baseline_hist.values

    # Candidate features
    Xc = time_feats[["month_sin", "month_cos", "trend"]].copy()
    Xc = Xc.reset_index(drop=True)
    y = pd.Series(residual, index=Xc.index)

    # RFE selection (use RF based on residuals)
    kept, elim, rfe_hist = rfe_select_features(Xc, y, min_features=2, max_features=min(6, Xc.shape[1]))

    # Prepare matrix for LSTM: target residual + kept features
    features_for_lstm = ["residual"] + kept
    data_mat = pd.DataFrame({"residual": y.values}, index=Xc.index)
    for c in kept:
        data_mat[c] = Xc[c].values

    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    data_scaled = scaler.fit_transform(data_mat.values.astype(float))

    target_idx = 0
    X_seq, y_seq = _create_sequences(data_scaled, TIME_STEPS, target_idx)
    if len(X_seq) < max(4, TIME_STEPS):
        raise ValueError("insufficient sequences for LSTM training")

    split_idx = max(1, int(len(X_seq) * 0.8))
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_val, y_val = (X_seq[split_idx:], y_seq[split_idx:]) if split_idx < len(X_seq) else (X_seq[-1:], y_seq[-1:])

    # Lightweight LSTM
    model = Sequential([
        LSTM(16, input_shape=(TIME_STEPS, data_mat.shape[1]), return_sequences=False, activation="tanh"),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=min(8, len(X_train)), callbacks=[es], verbose=0)

    # Validate model: simple forward pass
    _ = model.predict(X_train[:1], verbose=0)

    # Build forecast horizon via iterative residual prediction
    last_hist_date = dates.iloc[-1]
    future_dates = pd.date_range(start=last_hist_date + pd.offsets.MonthBegin(1), periods=FORECAST_HORIZON_MONTHS, freq="MS")

    baseline_future = _forecast_baseline_forward(last_hist_date, sales, dates, FORECAST_HORIZON_MONTHS, baseline_meta)

    # Start from last window
    last_window = X_seq[-1]
    residual_forecast = []
    for i, d in enumerate(future_dates):
        pred_resid_scaled = float(model.predict(last_window.reshape(1, TIME_STEPS, data_mat.shape[1]), verbose=0)[0][0])
        # inverse scaling for residual (feature index 0)
        if hasattr(scaler, "scale_") and scaler.scale_[0] != 0:
            pred_resid = (pred_resid_scaled - scaler.min_[0]) / scaler.scale_[0]
        else:
            pred_resid = pred_resid_scaled
        residual_forecast.append(pred_resid)
        # Compose next feature row
        new_feats = {
            "residual": pred_resid,
            "month_sin": np.sin(2 * np.pi * d.month / 12) if len(time_feats) >= 12 else 0.0,
            "month_cos": np.cos(2 * np.pi * d.month / 12) if len(time_feats) >= 12 else 0.0,
            "trend": float(len(time_feats) + i + 1),
        }
        next_row_unscaled = [new_feats.get(col, 0.0) for col in features_for_lstm]
        next_row_scaled = scaler.transform([next_row_unscaled])[0]
        # roll window
        last_window = np.vstack([last_window[1:], next_row_scaled])

    point_forecast = np.maximum(baseline_future.values + np.array(residual_forecast), 0.0)

    # Diagnostics on residual fit
    ins_pred = model.predict(X_train, verbose=0).flatten()
    ins_true = y_train.flatten()
    mae = float(mean_absolute_error(ins_true, ins_pred)) if len(ins_true) > 0 else None
    rmse = float(np.sqrt(mean_squared_error(ins_true, ins_pred))) if len(ins_true) > 0 else None

    result = {
        "model": model,
        "scaler": scaler,
        "features": {
            "selected_features": kept,
            "features_for_lstm": features_for_lstm,
            "baseline_meta": baseline_meta,
            "time_steps": TIME_STEPS,
            "horizon": FORECAST_HORIZON_MONTHS,
        },
        "diagnostics": {
            "train_mae_residual": mae,
            "train_rmse_residual": rmse,
            "n_points": int(len(g)),
        },
        "forecast": pd.Series(point_forecast, index=future_dates),
    }
    return result


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_per_product(product: str, category: str, res: dict):
    safe_name = product.replace("/", "-").replace("\\", "-").replace(" ", "_")
    model_path = os.path.join(OUTPUT_DIR, f"product_{safe_name}_model.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, f"product_{safe_name}_scaler.pkl")
    features_path = os.path.join(OUTPUT_DIR, f"product_{safe_name}_features.json")

    # Serialize model into pickle-friendly dict
    model_blob = serialize_keras_model(res["model"])  # type: ignore
    with open(model_path, "wb") as f:
        pickle.dump(model_blob, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(res["scaler"], f)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({
            **res["features"],
            "category": category,
        }, f, indent=2)

    return model_path, scaler_path, features_path


def compute_global_and_category_stats(monthly_df: pd.DataFrame) -> Tuple[dict, dict]:
    # monthly_df columns: product_norm, category, month, qty
    overall = {
        "global_mean": float(monthly_df["qty"].mean()),
        "global_median": float(monthly_df["qty"].median()),
        "global_std": float(monthly_df["qty"].std() if monthly_df["qty"].std() == monthly_df["qty"].std() else 0.0),
        "n_rows": int(len(monthly_df)),
        "n_products": int(monthly_df["product_norm"].nunique()),
        "last_date": pd.to_datetime(monthly_df["month"]).max().strftime("%Y-%m-01"),
    }
    cat_stats = {}
    for cat, g in monthly_df.groupby("category"):
        cat_stats[cat] = {
            "mean": float(g["qty"].mean()),
            "median": float(g["qty"].median()),
            "std": float(g["qty"].std() if g["qty"].std() == g["qty"].std() else 0.0),
            "n_rows": int(len(g)),
            "n_products": int(g["product_norm"].nunique()),
        }
    return overall, cat_stats


def train_all(excel_path: str = DEFAULT_EXCEL):
    ensure_output()
    skipped: List[str] = []
    diagnostics_rows: List[dict] = []
    models_meta: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_months": FORECAST_HORIZON_MONTHS,
        "time_steps": TIME_STEPS,
        "min_data_points_months": MIN_DATA_POINTS_MONTHS,
        "min_nonzero_transactions": MIN_NONZERO_TRANSACTIONS,
        "products": {}
    }

    # Read and preprocess
    df = read_and_preprocess(excel_path)
    monthly_df = monthly_aggregate(df)

    # compute stats for fallback
    global_stats, cat_stats = compute_global_and_category_stats(monthly_df)

    eligible, ineligible_stats = filter_eligible_products(monthly_df)

    # Persist stats
    with open(GLOBAL_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)
    with open(CATEGORY_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(cat_stats, f, indent=2)

    # Iterate products
    for prod, g in monthly_df.groupby("product_norm"):
        category = g["category"].iloc[0] if len(g) > 0 else ""
        if prod not in eligible:
            reason_stats = ineligible_stats.get(prod, {})
            skipped.append(f"{prod}: insufficient data (months={reason_stats.get('months')}, nonzero={reason_stats.get('nonzero')})")
            continue
        try:
            res = train_per_product(prod, g)
            model_path, scaler_path, features_path = save_per_product(prod, category, res)

            # Update metadata
            models_meta["products"][prod] = {
                "category": category,
                "model_path": os.path.basename(model_path),
                "scaler_path": os.path.basename(scaler_path),
                "features_path": os.path.basename(features_path),
                "diagnostics": res["diagnostics"],
                "n_months": int(len(g)),
            }

            # Diagnostics row
            diagnostics_rows.append({
                "product": prod,
                "category": category,
                "n_months": int(len(g)),
                "train_mae_residual": res["diagnostics"]["train_mae_residual"],
                "train_rmse_residual": res["diagnostics"]["train_rmse_residual"],
            })
        except Exception as e:
            skipped.append(f"{prod}: error during training - {e}")
            continue

    # Save metadata and diagnostics
    with open(MODELS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(models_meta, f, indent=2)
    if diagnostics_rows:
        pd.DataFrame(diagnostics_rows).to_csv(DIAG_CSV, index=False)

    if skipped:
        with open(SKIPPED_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped))

    return {
        "eligible_count": len(models_meta["products"]),
        "skipped_count": len(skipped),
        "output_dir": OUTPUT_DIR,
    }

def safe_parse_date(date_value):
    """Safely parse date values with overflow protection"""
    if pd.isna(date_value):
        return pd.NaT
    
    date_str = str(date_value).strip()
    if not date_str or date_str.lower() == 'nan':
        return pd.NaT
    
    try:
        # First attempt: normal parsing with dayfirst=True for dd/mm/yyyy format
        parsed = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
        if pd.isna(parsed):
            return pd.NaT
            
        # Check if year is reasonable (between 1900 and 2100)
        if parsed.year < 1900 or parsed.year > 2100:
            # Try to fix common 2-digit year issues
            if len(date_str.split('/')) == 3 or len(date_str.split('-')) == 3:
                parts = date_str.replace('-', '/').split('/')
                if len(parts) == 3:
                    try:
                        year = int(parts[2])
                        # Fix 2-digit years
                        if year < 50:
                            year += 2000
                        elif year < 100:
                            year += 1900
                        
                        # Reconstruct date string
                        fixed_date_str = f"{parts[0]}/{parts[1]}/{year}"
                        fixed_parsed = pd.to_datetime(fixed_date_str, errors='coerce')
                        
                        if not pd.isna(fixed_parsed) and 1900 <= fixed_parsed.year <= 2100:
                            return fixed_parsed
                    except (ValueError, IndexError):
                        pass
            
            return pd.NaT
        
        return parsed
    except (OverflowError, ValueError, TypeError):
        return pd.NaT

import re
import pandas as pd

def normalize_product_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def read_excel_latest(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    # detect columns
    candidates = [
        {"date": "Transaction Date", "qty": "Quantity", "product": "Product Name", "category": "Product Category"},
        {"date": "Tanggal Transaksi", "qty": "Jumlah", "product": "Nama Produk", "category": "Kategori Barang"},
    ]
    mapping = None
    for cand in candidates:
        ok = all(any(c2.lower() == v.lower() for c2 in df.columns) for v in cand.values())
        if ok:
            mapping = cand
            break
    if mapping is None and test2 and hasattr(test2, "column_mapping"):
        mapping = {"date": test2.column_mapping.get("Tanggal Transaksi", "date"),
                   "qty": test2.column_mapping.get("Jumlah", "sales"),
                   "product": test2.column_mapping.get("Nama Produk", "product_name"),
                   "category": test2.column_mapping.get("Kategori Barang", "category")}
    if mapping is None:
        raise ValueError("Could not infer column mapping for Excel file.")

    df = df[[mapping["date"], mapping["qty"], mapping["product"], mapping["category"]]].copy()
    df.columns = ["date", "sales", "product_name", "category"]
    df.dropna(subset=["date"], inplace=True)
    
    # Use safe date parsing instead of direct pd.to_datetime
    df["date"] = df["date"].apply(safe_parse_date)
    
    # Remove rows where date parsing failed
    before_count = len(df)
    df = df.dropna(subset=["date"])
    after_count = len(df)
    
    if before_count > after_count:
        print(f"Removed {before_count - after_count} rows with invalid dates")
    
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
    df = df.groupby("product_norm", group_keys=False).apply(_clip_group)
    return df


if __name__ == "__main__":
    info = train_all(DEFAULT_EXCEL)
    print("Training completed:", info)
