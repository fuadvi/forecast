"""
SES Monthly Product Forecast 24 Months

Deskripsi:
- Pipeline peramalan alternatif menggunakan Simple Exponential Smoothing (SES)
- Menghasilkan forecast per produk per bulan 24 bulan ke depan
- Menyediakan 4 file CSV output utama + 1 log skip + 1 evaluasi metrik + 1 gambar chart grouped Top-5

Keluaran:
1) forecast_per_product_ses_24m.csv
   Kolom: date, product_name, forecast, method="SES"
2) forecast_total_ses_24m.csv
   Kolom: date, total_forecast
3) topN_per_month_ses_24m.csv
   Kolom: date, product_name, forecast, rank
4) ses_skipped_products.csv
   Kolom: product_name, reason
5) ses_evaluation_metrics.csv
   Kolom: product_name, mae, rmse, mape, n_validation_points, n_train_points, method_used
6) forecast_plots/bulan/top5_grouped_24m_ses.png

Catatan & Ketangguhan:
- Minimal titik data per produk: MIN_DATA_POINTS (default 6). Jika kurang, skip & log.
- Reindex ke frekuensi bulanan konsisten (freq="MS"). Bulan hilang diisi 0.
- Outlier capping ringan per produk (opsional, default aktif): clip ke [Q1-1.5*IQR, Q3+1.5*IQR].
- SES: gunakan statsmodels.tsa.holtwinters.SimpleExpSmoothing, optimized=True untuk alpha.
- Jika optimasi alpha gagal, fallback alpha=0.3.
- Jika terjadi error library atau kasus ekstrim, fallback forecast nol datar.
- Jika seri semua nol, boleh fit; jika error, fallback nol datar.
- Tanggal keluaran 24 titik bulanan selaras (awal bulan, freq="MS").
- Satu plot grouped per 24 bulan, hingga Top-5 produk per bulan.

Opsional:
- Simpan alpha per produk ke ses_model_params.csv (jika tersedia dari model).
- Argparse untuk override parameter (FILE_PATH, OUT_DIR, TOP_K, FORECAST_MONTHS, MIN_DATA_POINTS, APPLY_OUTLIER_CAPPING)

"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
except Exception as e:  # pragma: no cover
    SimpleExpSmoothing = None  # type: ignore
    ExponentialSmoothing = None  # type: ignore

# ==========================
# Konfigurasi (dapat dioverride via CLI)
# ==========================
FILE_PATH = "Data_Penjualan_Dengan_ID_Pelanggan.xlsx"
OUT_DIR = "."  # direktori untuk CSV output (default: root project)
FORECAST_MONTHS = 24
TOP_K = 5
MIN_DATA_POINTS = 6
APPLY_OUTLIER_CAPPING = True
FREQ = "MS"  # awal bulan
SEED = 42

# Lokasi plot sesuai spesifikasi
PLOTS_DIR = os.path.join("forecast_plots", "bulan")
PLOT_PATH = os.path.join(PLOTS_DIR, "top5_grouped_24m_ses.png")

# Nama file output
CSV_PER_PRODUCT = "forecast_per_product_ses_24m.csv"
CSV_TOTAL = "forecast_total_ses_24m.csv"
CSV_TOPN = "topN_per_month_ses_24m.csv"
CSV_SKIPPED = "ses_skipped_products.csv"
CSV_MODEL_PARAMS = "ses_model_params.csv"  # opsional
CSV_EVALUATION_METRICS = "ses_evaluation_metrics.csv"  # evaluasi metrik

COLUMN_MAPPING = {
    "Tanggal Transaksi": "date",
    "Nama Produk": "product_name",
    # Izinkan dua kemungkinan nama kolom kuantitas
    "Jumlah": "sales",
    "Jumlah Unit Terjual": "sales",
}


# ==========================
# Utilitas
# ==========================

def normalize_product_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ==========================
# 1) Load & Prepare Data
# ==========================

def load_and_prepare_data(file_path: str = FILE_PATH) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File Excel tidak ditemukan: {file_path}")

    df = pd.read_excel(file_path)
    # rename mapping jika ada
    existing = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=existing)

    # Validasi kolom wajib
    if "date" not in df.columns:
        raise ValueError("Kolom tanggal tidak ditemukan setelah mapping (butuh 'Tanggal Transaksi').")
    if "product_name" not in df.columns:
        raise ValueError("Kolom 'Nama Produk' tidak ditemukan setelah mapping.")

    # Tipe data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    else:
        # jika kolom sales tidak ditemukan, buat sales=1 (fallback di beberapa dataset)
        df["sales"] = 1.0

    # Drop baris invalid
    df = df.dropna(subset=["date", "product_name"]).copy()
    # Normalisasi nama produk
    df["product_name"] = df["product_name"].apply(normalize_product_name)
    df = df[df["product_name"] != ""].copy()

    # Fill NaN sales ke 0
    df["sales"] = df["sales"].fillna(0.0).astype(float)

    # Sortir
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ==========================
# 2) Monthly Aggregation
# ==========================

def aggregate_monthly_per_product(df: pd.DataFrame, freq: str = FREQ,
                                  outlier_capping: bool = APPLY_OUTLIER_CAPPING) -> pd.DataFrame:
    # Agregasi ke level bulanan (sum per product_name, month)
    monthly = (df.groupby(["product_name", pd.Grouper(key="date", freq=freq)])
                 ["sales"].sum().reset_index())

    # Lengkapi seri per produk dengan bulan hilang -> 0
    completed: List[pd.DataFrame] = []
    for prod, sub in monthly.groupby("product_name"):
        sub = sub.sort_values("date").reset_index(drop=True)
        if sub.empty:
            continue
        start = sub["date"].min().to_period("M").to_timestamp()
        end = sub["date"].max().to_period("M").to_timestamp()
        idx = pd.date_range(start=start, end=end, freq=freq)
        sub2 = sub.set_index("date").reindex(idx).rename_axis("date").reset_index()
        sub2["product_name"] = prod
        sub2["sales"] = pd.to_numeric(sub2["sales"], errors="coerce").fillna(0.0)

        # Outlier capping ringan per produk
        if outlier_capping and len(sub2) >= 4:
            s = sub2["sales"].astype(float)
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                low = max(0.0, q1 - 1.5 * iqr)
                high = q3 + 1.5 * iqr
                sub2["sales"] = s.clip(lower=low, upper=high)
        completed.append(sub2[["date", "product_name", "sales"]])

    if completed:
        res = pd.concat(completed, ignore_index=True)
    else:
        res = monthly.copy()
    # Pastikan tipe
    res["date"] = pd.to_datetime(res["date"])
    res["sales"] = pd.to_numeric(res["sales"], errors="coerce").fillna(0.0)
    return res.sort_values(["product_name", "date"]).reset_index(drop=True)


# ==========================
# 3) Evaluation Metrics
# ==========================

def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    return float(np.mean(np.abs(actual - predicted)))


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


def calculate_evaluation_metrics(monthly_df: pd.DataFrame, validation_split: float = 0.2,
                                 min_points: int = MIN_DATA_POINTS) -> pd.DataFrame:
    """
    Calculate evaluation metrics (MAE, RMSE, MAPE) using time-series validation split.
    
    Args:
        monthly_df: DataFrame dengan kolom date, product_name, sales
        validation_split: Proporsi data untuk validation (default 0.2 = 20%)
        min_points: Minimal titik data per produk untuk evaluasi
    
    Returns:
        DataFrame dengan kolom: product_name, mae, rmse, mape, n_validation_points
    """
    metrics_rows: List[Dict] = []
    
    for prod, sub in monthly_df.groupby("product_name"):
        sub = sub.sort_values("date").reset_index(drop=True)
        s = pd.to_numeric(sub["sales"], errors="coerce").fillna(0.0)
        
        # Validasi minimal data
        non_na_points = int(s.notna().sum())
        if non_na_points < min_points:
            continue
        
        # Time-series split: gunakan data terakhir sebagai validation
        n_total = len(sub)
        n_train = max(min_points, int(n_total * (1 - validation_split)))
        
        if n_train >= n_total:
            # Tidak cukup data untuk split, skip evaluasi
            continue
        
        # Split data
        train_data = sub.iloc[:n_train]
        val_data = sub.iloc[n_train:]
        
        if len(val_data) == 0:
            continue
        
        # Fit model pada training data
        train_series = pd.to_numeric(train_data["sales"], errors="coerce").fillna(0.0)
        val_actual = pd.to_numeric(val_data["sales"], errors="coerce").fillna(0.0).values
        
        # Forecast untuk validation period
        val_horizon = len(val_data)
        try:
            fc, params, err = fit_hw_or_ses_forecast(train_series, steps=val_horizon)
            if fc is None or len(fc) != val_horizon:
                continue
            
            # Clip non-negative
            val_predicted = np.clip(np.asarray(fc, dtype=float), 0.0, None)
            
            # Hitung metrik
            mae = calculate_mae(val_actual, val_predicted)
            rmse = calculate_rmse(val_actual, val_predicted)
            mape = calculate_mape(val_actual, val_predicted)
            
            metrics_rows.append({
                "product_name": prod,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "n_validation_points": len(val_data),
                "n_train_points": len(train_data),
                "method_used": params.get("method_used", "SES")
            })
        except Exception:
            # Skip jika error
            continue
    
    if metrics_rows:
        return pd.DataFrame(metrics_rows)
    else:
        return pd.DataFrame(columns=["product_name", "mae", "rmse", "mape", "n_validation_points", "n_train_points", "method_used"])


# ==========================
# 4) HW/SES Fit & Forecast
# ==========================

def fit_ses_and_forecast(series: pd.Series, steps: int, alpha: Optional[float] = None,
                         random_state: int = SEED) -> Tuple[np.ndarray, Dict[str, Optional[float]], Optional[str]]:
    """
    Backward-compatible SES forecaster (kept for reuse). Returns (fc, params, reason).
    params: {'alpha': float|None}
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    if len(s) < 1 or SimpleExpSmoothing is None:
        last = float(s.iloc[-1]) if len(s) > 0 else 0.0
        return np.full(steps, last if np.isfinite(last) else 0.0, dtype=float), {"alpha": None}, (
            "empty_series" if len(s) < 1 else "statsmodels_not_available")
    try:
        if alpha is None:
            model = SimpleExpSmoothing(s, initialization_method="heuristic").fit(optimized=True)
        else:
            model = SimpleExpSmoothing(s, initialization_method="heuristic").fit(
                smoothing_level=float(alpha), optimized=False
            )
        fc = np.asarray(model.forecast(steps), dtype=float)
        smoothing_level = None
        try:
            smoothing_level = float(getattr(model, "params", {}).get("smoothing_level", None))
        except Exception:
            smoothing_level = float(getattr(model, "smoothing_level", None)) if hasattr(model, "smoothing_level") else None
        return fc, {"alpha": smoothing_level}, None
    except Exception:
        # fallback alpha=0.3
        try:
            model = SimpleExpSmoothing(s, initialization_method="heuristic").fit(smoothing_level=0.3, optimized=False)
            fc = np.asarray(model.forecast(steps), dtype=float)
            return fc, {"alpha": 0.3}, None
        except Exception as e2:
            last = float(s.iloc[-1]) if len(s) > 0 else 0.0
            return np.full(steps, last if np.isfinite(last) else 0.0, dtype=float), {"alpha": None}, f"fallback: {e2}"

def fit_hw_or_ses_forecast(series: pd.Series, steps: int) -> Tuple[np.ndarray, Dict[str, Optional[float]], Optional[str]]:
    """
    Try Holt-Winters (additive seasonality, damped trend) when len(series) >= 24.
    Fallback chain: HW -> SES optimized -> SES alpha=0.3 -> last level.
    Returns (forecast, params, fallback_reason)
    params contains keys: alpha, beta, gamma, method_used, fallback_reason (optional)
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    n = int(len(s))

    # Default result container
    params: Dict[str, Optional[float]] = {"alpha": None, "beta": None, "gamma": None, "method_used": None}  # type: ignore

    # Prefer HW if enough data and library available
    if n >= 24 and ExponentialSmoothing is not None:
        try:
            hw_model = ExponentialSmoothing(
                s,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = np.asarray(hw_model.forecast(steps), dtype=float)
            # Extract smoothing params if available
            alpha = None
            beta = None
            gamma = None
            try:
                p = getattr(hw_model, "params", {})
                alpha = float(p.get("smoothing_level", None)) if p is not None else None
                beta = float(p.get("smoothing_trend", None)) if p is not None else None
                gamma = float(p.get("smoothing_seasonal", None)) if p is not None else None
            except Exception:
                pass
            params.update({"alpha": alpha, "beta": beta, "gamma": gamma, "method_used": "HW"})
            return fc, params, None
        except Exception as e_hw:
            # Fall through to SES, but record reason
            params["fallback_reason"] = f"HW_failed: {e_hw}"  # type: ignore

    # SES path (for n < 24 or HW failed/unavailable)
    fc, ses_params, ses_reason = fit_ses_and_forecast(s, steps=steps, alpha=None)
    params.update({"alpha": ses_params.get("alpha"), "method_used": (params.get("method_used") or "SES")})
    if ses_reason:
        params["fallback_reason"] = (params.get("fallback_reason") or ses_reason)  # type: ignore
    return fc, params, None


# ==========================
# 5) Build outputs
# ==========================

def build_forecast_frames(monthly_df: pd.DataFrame, horizon: int = FORECAST_MONTHS,
                          min_points: int = MIN_DATA_POINTS) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Return: per_product_df, total_df, topN_df, skipped_df, model_params_records
    """
    if monthly_df.empty:
        raise ValueError("Data bulanan kosong.")

    # Tanggal referensi global: gunakan tanggal maksimum dari seluruh dataset
    global_last_date = monthly_df["date"].max()
    future_dates = pd.date_range(start=global_last_date + pd.offsets.MonthBegin(1), periods=horizon, freq=FREQ)

    per_product_rows: List[pd.DataFrame] = []
    skipped_rows: List[Dict[str, str]] = []
    model_params: List[Dict] = []

    for prod, sub in monthly_df.groupby("product_name"):
        sub = sub.sort_values("date")
        s = pd.to_numeric(sub["sales"], errors="coerce").fillna(0.0)
        # valid minimal data
        non_na_points = int(s.notna().sum())
        if non_na_points < min_points:
            skipped_rows.append({"product_name": prod, "reason": f"insufficient data (<{min_points})"})
            continue

        # Fit per produk: coba Holt-Winters bila cukup data; fallback ke SES
        fc, params, err = fit_hw_or_ses_forecast(s, steps=horizon)

        # Jika error fatal, catat dan lanjutkan
        if fc is None or len(fc) != horizon:
            skipped_rows.append({"product_name": prod, "reason": err or params.get("fallback_reason", "forecast_failed")})
            continue

        # Non-negatif
        fc = np.clip(np.asarray(fc, dtype=float), 0.0, None)

        # Simpan baris per-produk untuk 24 bulan
        df_prod = pd.DataFrame({
            "date": future_dates,
            "product_name": prod,
            "forecast": fc,
            "method": ["SES"] * horizon,
        })
        per_product_rows.append(df_prod)

        # Simpan parameter smoothing jika ada (alpha/beta/gamma) + method_used
        rec = {
            "product_name": prod,
            "alpha": params.get("alpha", None),
            "beta": params.get("beta", None),
            "gamma": params.get("gamma", None),
            "method_used": params.get("method_used", None),
        }
        # tambahkan fallback_reason jika ada (opsional kolom)
        if params.get("fallback_reason"):
            rec["fallback_reason"] = params.get("fallback_reason")
        model_params.append(rec)

    if per_product_rows:
        per_product_df = pd.concat(per_product_rows, ignore_index=True)
    else:
        per_product_df = pd.DataFrame(columns=["date", "product_name", "forecast", "method"])  # empty

    # Total agregat per bulan
    if not per_product_df.empty:
        total_df = (per_product_df.groupby("date", as_index=False)["forecast"].sum()
                    .rename(columns={"forecast": "total_forecast"}))
    else:
        total_df = pd.DataFrame({"date": future_dates, "total_forecast": [0.0] * len(future_dates)})

    # Top-N per bulan
    if not per_product_df.empty:
        ranked = per_product_df.sort_values(["date", "forecast"], ascending=[True, False])
        ranked["rank"] = ranked.groupby("date").cumcount() + 1
        topN_df = ranked[ranked["rank"] <= TOP_K].copy()
    else:
        topN_df = pd.DataFrame(columns=["date", "product_name", "forecast", "rank"])  # empty

    skipped_df = pd.DataFrame(skipped_rows, columns=["product_name", "reason"]) if skipped_rows else pd.DataFrame(
        columns=["product_name", "reason"])
    return per_product_df, total_df, topN_df, skipped_df, model_params


# ==========================
# 6) Plot Grouped Top-5
# ==========================

def plot_grouped_top5(topN_df: pd.DataFrame, out_path: str = PLOT_PATH,
                      width: float = 0.18, annotate_only_max: bool = True) -> None:
    if topN_df is None or topN_df.empty:
        ensure_dir(os.path.dirname(out_path))
        plt.figure(figsize=(20, 8))
        plt.title("Top-5 Produk per Bulan (SES) - 24 Bulan")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        return

    df = topN_df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "product_name"])  # type: ignore

    # Ambil hingga 24 bulan berurutan
    months = sorted(df["date"].dt.to_period("M").dt.to_timestamp().unique())
    if len(months) > 24:
        months = months[:24]

    # Filter top-5 per bulan dan daftar produk yang muncul
    filtered = df[df["date"].isin(months)].copy()
    filtered = (filtered.sort_values(["date", "forecast"], ascending=[True, False])
                        .groupby("date", as_index=False).head(5))
    prods = sorted(filtered["product_name"].unique().tolist())

    # Warna stabil dan kontras per produk
    try:
        cmap = plt.get_cmap("tab20")
        base_colors = list(getattr(cmap, "colors", []))
        if not base_colors:
            base_colors = [cmap(i) for i in np.linspace(0, 1, max(5, len(prods)))]
    except Exception:
        base_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                       "#9c755f", "#bab0ab"]
    color_map = {p: base_colors[i % len(base_colors)] for i, p in enumerate(prods)}

    ensure_dir(os.path.dirname(out_path))

    x = np.arange(len(months))
    offsets = np.linspace(-0.36, 0.36, 5)  # center 5 bars

    # Dynamic figure width based on months
    fig_width = max(16, len(months) * 0.8)
    plt.figure(figsize=(fig_width, 8))

    # Helper for compact number formatting
    def _fmt(v: float) -> str:
        v = float(v)
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:.0f}"

    # Totals per product for legend ordering
    totals: Dict[str, float] = {p: 0.0 for p in prods}
    for _, row in filtered.iterrows():
        totals[row["product_name"]] += float(row["forecast"])  # type: ignore

    # Draw bars and annotate only top per month
    for mi, m in enumerate(months):
        sub = filtered[filtered["date"] == m].sort_values("forecast", ascending=False).head(5)
        vals = sub["forecast"].astype(float).values
        names = sub["product_name"].astype(str).values
        for k in range(len(vals)):
            p = names[k]
            v = float(vals[k])
            pos = x[mi] + offsets[k]
            plt.bar(pos, v, width=width, color=color_map.get(p, "#999999"),
                    edgecolor="white", linewidth=0.5, alpha=0.9)
        # annotate top-1 only
        if len(vals):
            plt.text(x[mi] + offsets[0], float(vals[0]) * 1.01, _fmt(vals[0]), ha="center", va="bottom", fontsize=9)

    month_labels = [pd.Timestamp(m).strftime("%b'%y") for m in months]
    plt.xticks(x, month_labels, rotation=50, ha="right", fontsize=10)
    plt.ylabel("Forecast")
    plt.title("Top-5 Produk per Bulan (Grouped) - 24 Bulan - SES")
    plt.grid(axis="y", color="#e5e7eb", alpha=0.8)

    # Legend: only appeared products, sort by totals and limit to 10 entries
    ordered = sorted([p for p in prods if totals.get(p, 0.0) > 0], key=lambda p: totals[p], reverse=True)[:10]
    if ordered:
        from matplotlib.patches import Patch
        plt.legend([Patch(facecolor=color_map[p], label=p) for p in ordered], ordered,
                   title="Produk", ncol=1, fontsize=9, title_fontsize=10, frameon=False,
                   loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ==========================
# 7) Orkestrasi
# ==========================

def main(file_path: str = FILE_PATH,
         out_dir: str = OUT_DIR,
         top_k: int = TOP_K,
         forecast_months: int = FORECAST_MONTHS,
         min_points: int = MIN_DATA_POINTS,
         outlier_capping: bool = APPLY_OUTLIER_CAPPING) -> None:
    global TOP_K, FORECAST_MONTHS
    TOP_K = top_k
    FORECAST_MONTHS = forecast_months

    # Load
    df = load_and_prepare_data(file_path)
    if df.empty:
        raise ValueError("Dataset kosong setelah pembersihan.")

    # Aggregasi bulanan
    monthly_df = aggregate_monthly_per_product(df, freq=FREQ, outlier_capping=outlier_capping)

    # Build forecasts
    per_product_df, total_df, topN_df, skipped_df, model_params = build_forecast_frames(
        monthly_df, horizon=forecast_months, min_points=min_points
    )

    # Calculate evaluation metrics
    print("Menghitung evaluasi metrik...")
    evaluation_metrics_df = calculate_evaluation_metrics(
        monthly_df, 
        validation_split=0.2, 
        min_points=min_points
    )

    # Simpan CSV ke out_dir
    ensure_dir(out_dir)
    per_product_path = os.path.join(out_dir, CSV_PER_PRODUCT)
    total_path = os.path.join(out_dir, CSV_TOTAL)
    topN_path = os.path.join(out_dir, CSV_TOPN)
    skipped_path = os.path.join(out_dir, CSV_SKIPPED)
    params_path = os.path.join(out_dir, CSV_MODEL_PARAMS)
    metrics_path = os.path.join(out_dir, CSV_EVALUATION_METRICS)

    per_product_df.to_csv(per_product_path, index=False)
    total_df.to_csv(total_path, index=False)
    topN_df.to_csv(topN_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    
    # Simpan evaluasi metrik
    if not evaluation_metrics_df.empty:
        evaluation_metrics_df.to_csv(metrics_path, index=False)
        print(f"Evaluasi metrik disimpan: {metrics_path}")
        # Print summary
        if len(evaluation_metrics_df) > 0:
            avg_mae = evaluation_metrics_df["mae"].mean()
            avg_rmse = evaluation_metrics_df["rmse"].mean()
            avg_mape = evaluation_metrics_df["mape"].mean()
            print(f"Rata-rata metrik evaluasi:")
            print(f"  MAE: {avg_mae:.4f}")
            print(f"  RMSE: {avg_rmse:.4f}")
            print(f"  MAPE: {avg_mape:.2f}%")
    else:
        print("Tidak ada metrik evaluasi yang dapat dihitung (data tidak cukup untuk validasi).")

    # Simpan alpha jika tersedia
    try:
        if model_params:
            pd.DataFrame(model_params).to_csv(params_path, index=False)
    except Exception:
        pass

    # Plot grouped top-5
    plot_grouped_top5(topN_df, out_path=PLOT_PATH)

    # Validasi dasar
    # Pastikan horizon 24 titik untuk setiap produk yang ter-forecast
    if not per_product_df.empty:
        counts = per_product_df.groupby("product_name")["date"].count().unique().tolist()
        if counts != [forecast_months]:
            warnings.warn(f"Tidak semua produk memiliki {forecast_months} titik forecast. Counts unik: {counts}")

    print("Selesai! File disimpan:")
    print(f" - {per_product_path}")
    print(f" - {total_path}")
    print(f" - {topN_path}")
    print(f" - {skipped_path}")
    if not evaluation_metrics_df.empty:
        print(f" - {metrics_path}")
    print(f" - {PLOT_PATH}")


if __name__ == "__main__":
    # Argparse sederhana
    import argparse

    parser = argparse.ArgumentParser(description="SES Monthly Product Forecast 24 Months")
    parser.add_argument("--file", dest="file_path", default=FILE_PATH, help="Path Excel input")
    parser.add_argument("--out", dest="out_dir", default=OUT_DIR, help="Direktori output CSV")
    parser.add_argument("--topk", dest="top_k", type=int, default=TOP_K, help="Top-K per bulan untuk chart/CSV TopN")
    parser.add_argument("--months", dest="forecast_months", type=int, default=FORECAST_MONTHS, help="Horizon bulan")
    parser.add_argument("--minpts", dest="min_points", type=int, default=MIN_DATA_POINTS, help="Minimal titik data per produk")
    parser.add_argument("--nocap", dest="no_capping", action="store_true", help="Matikan outlier capping per produk")

    args = parser.parse_args()
    main(
        file_path=args.file_path,
        out_dir=args.out_dir,
        top_k=args.top_k,
        forecast_months=args.forecast_months,
        min_points=args.min_points,
        outlier_capping=(not args.no_capping),
    )
