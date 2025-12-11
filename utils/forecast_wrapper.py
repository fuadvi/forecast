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
    ROOT_DIR,
)

_proc: Optional[subprocess.Popen] = None
_log_file: Optional[Path] = None
_log_handle = None


def run_forecast() -> Tuple[bool, str]:
    """Run forecast.py as subprocess with proper working directory."""
    global _proc, _log_file, _log_handle
    
    # Close previous log handle if exists
    if _log_handle:
        try:
            _log_handle.close()
        except Exception:
            pass
    
    if _proc and _proc.poll() is None:
        return False, "Forecast sedang berjalan"
    
    py = sys.executable
    forecast_script = ROOT_DIR / "forecast.py"
    
    # Log file untuk debug
    _log_file = ROOT_DIR / "forecast_run.log"
    
    # Pastikan script ada
    if not forecast_script.exists():
        return False, f"Script tidak ditemukan: {forecast_script}"
    
    try:
        # Buka log file untuk output
        _log_handle = open(_log_file, 'w', encoding='utf-8', buffering=1)  # Line buffered
        
        cmd = [py, "-u", str(forecast_script)]
        _proc = subprocess.Popen(
            cmd, 
            stdout=_log_handle,
            stderr=subprocess.STDOUT, 
            text=True,
            cwd=str(ROOT_DIR),  # Set working directory ke project root
            env={**os.environ, "PYTHONUNBUFFERED": "1"}  # Unbuffered output
        )
        return True, f"Forecast dimulai (PID: {_proc.pid})"
    except Exception as e:
        return False, f"Error starting forecast: {e}"


def check_forecast_status() -> str:
    """Check if forecast process is still running."""
    global _log_handle
    
    if _proc is None:
        return "idle"
    code = _proc.poll()
    if code is None:
        return "running"
    
    # Process finished, close log handle
    if _log_handle:
        try:
            _log_handle.close()
            _log_handle = None
        except Exception:
            pass
    
    return "finished" if code == 0 else "failed"


def get_forecast_log() -> str:
    """Get the last N lines from the forecast log file."""
    if _log_file and _log_file.exists():
        try:
            content = _log_file.read_text(encoding='utf-8')
            # Return last 50 lines
            lines = content.strip().split('\n')
            return '\n'.join(lines[-50:])
        except Exception:
            return ""
    return ""


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
