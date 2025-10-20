from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import (
    FORECAST_PER_PRODUCT,
    FORECAST_TOTAL,
    TOPN_PER_MONTH,
    FORECAST_DIAGNOSTICS,
)

_proc: Optional[subprocess.Popen] = None


def run_forecast() -> Tuple[bool, str]:
    global _proc
    if _proc and _proc.poll() is None:
        return False, "Forecast sedang berjalan"
    py = sys.executable
    cmd = [py, "-u", "forecast.py"]
    _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return True, "Forecast dimulai"


def check_forecast_status() -> str:
    if _proc is None:
        return "idle"
    code = _proc.poll()
    if code is None:
        return "running"
    return "finished" if code == 0 else "failed"


def load_forecast_results() -> Dict[str, Optional[pd.DataFrame]]:
    res: Dict[str, Optional[pd.DataFrame]] = {"per_product": None, "total": None, "topn": None, "diagnostics": None}
    try:
        if FORECAST_PER_PRODUCT.exists():
            res["per_product"] = pd.read_csv(FORECAST_PER_PRODUCT)
    except Exception:
        pass
    try:
        if FORECAST_TOTAL.exists():
            res["total"] = pd.read_csv(FORECAST_TOTAL)
    except Exception:
        pass
    try:
        if TOPN_PER_MONTH.exists():
            res["topn"] = pd.read_csv(TOPN_PER_MONTH)
    except Exception:
        pass
    try:
        if FORECAST_DIAGNOSTICS.exists():
            res["diagnostics"] = pd.read_csv(FORECAST_DIAGNOSTICS)
    except Exception:
        pass
    return res


def get_forecast_summary() -> Dict[str, object]:
    res = load_forecast_results()
    summary: Dict[str, object] = {}
    total_df = res.get("total")
    if total_df is not None and not total_df.empty:
        summary["n_months"] = int(total_df.shape[0])
        if "date" in total_df.columns:
            try:
                dt = pd.to_datetime(total_df["date"])  # assume column name
                summary["period"] = (dt.min().date().isoformat(), dt.max().date().isoformat())
            except Exception:
                summary["period"] = None
        summary["sum"] = float(total_df.select_dtypes(include="number").sum(numeric_only=True).sum())
    per_df = res.get("per_product")
    if per_df is not None:
        summary["n_products"] = int(per_df["product_name"].nunique()) if "product_name" in per_df.columns else None
    return summary
