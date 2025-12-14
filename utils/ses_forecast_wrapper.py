from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
from datetime import datetime

import pandas as pd

from config.settings import (
    DEFAULT_EXCEL,
    SES_FORECAST_PER_PRODUCT,
    SES_FORECAST_TOTAL,
    SES_TOPN_PER_MONTH,
    SES_SKIPPED_PRODUCTS,
    SES_MODEL_PARAMS,
    SES_GROUPED_TOP5_PNG,
    ROOT_DIR,
)

# Global process handle
_proc: Optional[subprocess.Popen] = None
_last_cmd: Optional[str] = None
_last_cwd: Optional[str] = None

# Log file path
SES_LOG_FILE = Path(ROOT_DIR) / "ses_forecast_run.log"


def run_ses_forecast(
    file_path: Optional[str | os.PathLike] = None,
    top_k: int = 5,
    forecast_months: int = 24,
    min_points: int = 6,
    outlier_capping: bool = True,
) -> Tuple[bool, str]:
    """
    Jalankan SES forecasting script via subprocess.
    Returns (success, message).
    Logs saved to: ses_forecast_run.log
    """
    global _proc
    if _proc and _proc.poll() is None:
        return False, "Proses SES sudah berjalan"

    # Initialize log file
    try:
        with open(SES_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"SES FORECAST LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
    except Exception as e:
        return False, f"Failed to initialize log file: {e}"

    py = sys.executable
    # Resolve absolute paths
    script = (Path(__file__).resolve().parents[1] / "ses_monthly_product_forecast_24m.py").resolve()
    
    # Log script path
    _log_to_file(f"Python executable: {py}")
    _log_to_file(f"Script path: {script}")
    
    if not script.exists():
        msg = f"Script SES tidak ditemukan: {script}"
        _log_to_file(f"ERROR: {msg}")
        return False, msg

    data_path = Path(file_path) if file_path else Path(DEFAULT_EXCEL)
    data_path = data_path.resolve()
    
    _log_to_file(f"Data path: {data_path}")
    
    if not data_path.exists():
        msg = f"File data Excel tidak ditemukan: {data_path}"
        _log_to_file(f"ERROR: {msg}")
        return False, msg

    out_dir = Path(ROOT_DIR).resolve()
    _log_to_file(f"Output directory: {out_dir}")

    # Build command to match argparse in SES script
    cmd = [
        py,
        "-u",
        str(script),
        "--file",
        str(data_path),
        "--out",
        str(out_dir),
        "--topk",
        str(int(top_k)),
        "--months",
        str(int(forecast_months)),
        "--minpts",
        str(int(min_points)),
    ]
    # SES script uses --nocap (flag) to disable capping
    if not outlier_capping:
        cmd.append("--nocap")

    # Log command and cwd for debugging via stream
    global _last_cmd, _last_cwd
    _last_cmd = " ".join(cmd)
    _last_cwd = str(out_dir)
    
    _log_to_file(f"\nCommand: {_last_cmd}")
    _log_to_file(f"Working directory: {_last_cwd}\n")
    _log_to_file(f"{'='*80}\n")
    _log_to_file("PROCESS OUTPUT:\n")
    _log_to_file(f"{'='*80}\n")

    try:
        _proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(out_dir),
        )
        _log_to_file(f"Process started with PID: {_proc.pid}\n")
        return True, "Proses SES dimulai"
    except Exception as e:
        _proc = None
        msg = f"Gagal menjalankan SES: {e}"
        _log_to_file(f"ERROR: {msg}")
        return False, msg


def _log_to_file(message: str):
    """Helper function to append message to log file"""
    try:
        with open(SES_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    except Exception:
        pass  # Silent fail untuk logging


def check_ses_status() -> str:
    global _proc
    if _proc is None:
        return "idle"
    code = _proc.poll()
    if code is None:
        return "running"
    return "finished" if code == 0 else "failed"


def stream_ses_logs() -> Generator[str, None, None]:
    global _proc, _last_cmd, _last_cwd
    # Emit wrapper debug info first
    if _last_cwd:
        yield f"[wrapper] Working directory: {_last_cwd}"
    if _last_cmd:
        yield f"[wrapper] Command: {_last_cmd}"
    if _proc is None or _proc.stdout is None:
        return
    try:
        for line in _proc.stdout:
            if line is None:
                break
            line = line.rstrip("\n")
            # Log to file AND yield for streaming
            _log_to_file(line)
            yield line
        # After stream ends, emit simple completion summary
        code = _proc.poll()
        if code is not None:
            exit_msg = f"[wrapper] Process exited with code {code}"
            _log_to_file(f"\n{'='*80}")
            _log_to_file(exit_msg)
            yield exit_msg
            
            # quick existence checks
            ok_files = []
            for p in [SES_FORECAST_PER_PRODUCT, SES_FORECAST_TOTAL, SES_TOPN_PER_MONTH]:
                exists = Path(p).exists()
                ok_files.append(exists)
                _log_to_file(f"File check: {Path(p).name} - {'EXISTS' if exists else 'MISSING'}")
            
            if not all(ok_files):
                warn_msg = "[wrapper] Warning: Not all expected output files were found after completion."
                _log_to_file(warn_msg)
                yield warn_msg
            else:
                success_msg = "[wrapper] All output files generated successfully."
                _log_to_file(success_msg)
                yield success_msg
            
            _log_to_file(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            _log_to_file(f"{'='*80}\n")
            _log_to_file(f"Log saved to: {SES_LOG_FILE}\n")
    except Exception as e:
        error_msg = f"[wrapper] Log streaming error: {e}"
        _log_to_file(f"ERROR: {error_msg}")
        yield error_msg
        return


def load_ses_results() -> Dict[str, Optional[pd.DataFrame]]:
    res: Dict[str, Optional[pd.DataFrame]] = {
        "per_product": None,
        "total": None,
        "topn": None,
        "skipped": None,
        "params": None,
    }
    try:
        if Path(SES_FORECAST_PER_PRODUCT).exists():
            res["per_product"] = pd.read_csv(SES_FORECAST_PER_PRODUCT)
    except Exception:
        pass
    try:
        if Path(SES_FORECAST_TOTAL).exists():
            res["total"] = pd.read_csv(SES_FORECAST_TOTAL)
    except Exception:
        pass
    try:
        if Path(SES_TOPN_PER_MONTH).exists():
            res["topn"] = pd.read_csv(SES_TOPN_PER_MONTH)
    except Exception:
        pass
    try:
        if Path(SES_SKIPPED_PRODUCTS).exists():
            res["skipped"] = pd.read_csv(SES_SKIPPED_PRODUCTS)
    except Exception:
        pass
    try:
        if Path(SES_MODEL_PARAMS).exists():
            res["params"] = pd.read_csv(SES_MODEL_PARAMS)
    except Exception:
        pass
    return res


def get_ses_summary() -> Dict[str, object]:
    res = load_ses_results()
    summary: Dict[str, object] = {
        "n_products": None,
        "n_skipped": None,
        "n_hw": None,
        "n_ses": None,
        "period": None,
    }
    per_df = res.get("per_product")
    if per_df is not None and not per_df.empty:
        pcol = next((c for c in per_df.columns if "product" in c.lower()), None)
        if pcol:
            summary["n_products"] = int(per_df[pcol].nunique())
    skip_df = res.get("skipped")
    if skip_df is not None and not skip_df.empty:
        summary["n_skipped"] = int(skip_df.shape[0])
    params_df = res.get("params")
    if params_df is not None and not params_df.empty:
        # Assume a column 'method' indicates 'HW' or 'SES'
        mcol = next((c for c in params_df.columns if c.lower() in ("method", "model", "algo")), None)
        if mcol:
            vc = params_df[mcol].astype(str).str.upper().value_counts()
            summary["n_hw"] = int(vc.get("HW", 0) + vc.get("HOLT-WINTERS", 0))
            summary["n_ses"] = int(vc.get("SES", 0))
    total_df = res.get("total")
    if total_df is not None and not total_df.empty:
        dcol = next((c for c in total_df.columns if "date" in c.lower() or "bulan" in c.lower()), None)
        try:
            dt = pd.to_datetime(total_df[dcol]) if dcol else None
            if dt is not None:
                summary["period"] = (dt.min().date().isoformat(), dt.max().date().isoformat())
        except Exception:
            pass
    return summary
