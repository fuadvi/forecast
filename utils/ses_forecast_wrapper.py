from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

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
    """
    global _proc
    if _proc and _proc.poll() is None:
        return False, "Proses SES sudah berjalan"

    py = sys.executable
    # Resolve absolute paths
    script = (Path(__file__).resolve().parents[1] / "ses_monthly_product_forecast_24m.py").resolve()
    if not script.exists():
        return False, f"Script SES tidak ditemukan: {script}"

    data_path = Path(file_path) if file_path else Path(DEFAULT_EXCEL)
    data_path = data_path.resolve()
    if not data_path.exists():
        return False, f"File data Excel tidak ditemukan: {data_path}"

    out_dir = Path(ROOT_DIR).resolve()

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

    try:
        _proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(out_dir),
        )
        return True, "Proses SES dimulai"
    except Exception as e:
        _proc = None
        return False, f"Gagal menjalankan SES: {e}"


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
            yield line.rstrip("\n")
        # After stream ends, emit simple completion summary
        code = _proc.poll()
        if code is not None:
            yield f"[wrapper] Process exited with code {code}"
            # quick existence checks
            ok_files = []
            for p in [SES_FORECAST_PER_PRODUCT, SES_FORECAST_TOTAL, SES_TOPN_PER_MONTH]:
                ok_files.append(Path(p).exists())
            if not all(ok_files):
                yield "[wrapper] Warning: Not all expected output files were found after completion."
    except Exception as e:
        yield f"[wrapper] Log streaming error: {e}"
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
