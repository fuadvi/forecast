from __future__ import annotations
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import json

from config.settings import (
    MODELS_FOLDER,
    TRAINING_DIAGNOSTICS,
)

_process: Optional[subprocess.Popen] = None
_log_path = MODELS_FOLDER / "training_stream.log"


def run_training(
    excel_path: Optional[Path] = None,
    forecast_horizon: int = 24,
    time_steps: int = 6,
    min_points: int = 8,
    min_nonzero: int = 3,
    epochs: int = 100,
    batch_size: int = 8,
) -> Tuple[bool, str]:
    """Run train_models.py in a subprocess. Return (started, message)."""
    global _process
    if _process and _process.poll() is None:
        return False, "Training sudah berjalan"

    # Pass params via env vars to avoid modifying the script
    env = os.environ.copy()
    env["FORECAST_HORIZON_MONTHS"] = str(forecast_horizon)
    env["TIME_STEPS"] = str(time_steps)
    env["MIN_DATA_POINTS_MONTHS"] = str(min_points)
    env["MIN_NONZERO_TRANSACTIONS"] = str(min_nonzero)
    env["EPOCHS"] = str(epochs)
    env["BATCH_SIZE"] = str(batch_size)
    if excel_path:
        env["EXCEL_PATH"] = str(excel_path)

    py = sys.executable
    cmd = [py, "-u", "train_models.py"]

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    # Open file to capture logs
    f = open(_log_path, "w", buffering=1, encoding="utf-8")
    _process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(Path.cwd()), env=env)
    return True, "Training dimulai"


def stream_training_logs(tail: int = 50) -> Generator[str, None, None]:
    """Yield log lines as they are written by the training process."""
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = _log_path
    path.touch(exist_ok=True)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # go to near end
        try:
            lines = f.readlines()
        except Exception:
            lines = []
        start = max(0, len(lines) - tail)
        for line in lines[start:]:
            yield line.rstrip("\n")
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.3)
                f.seek(where)
            else:
                yield line.rstrip("\n")
            # stop if process finished and no new lines
            if _process is None or _process.poll() is not None:
                # Drain remaining
                rest = f.readlines()
                for l in rest:
                    yield l.rstrip("\n")
                break


def check_training_status() -> str:
    if _process is None:
        return "idle"
    code = _process.poll()
    if code is None:
        return "running"
    return "finished" if code == 0 else "failed"


def get_training_results() -> dict:
    res = {
        "diagnostics": None,
        "n_models": 0,
        "skipped": [],
    }
    if TRAINING_DIAGNOSTICS.exists():
        try:
            import pandas as pd
            diag = pd.read_csv(TRAINING_DIAGNOSTICS)
            res["diagnostics"] = diag
            res["n_models"] = int((diag.get("Model Used") == True).sum()) if "Model Used" in diag.columns else len(diag)
        except Exception:
            pass
    skip_log = MODELS_FOLDER / "skipped_products.log"
    if skip_log.exists():
        try:
            res["skipped"] = skip_log.read_text(encoding="utf-8").splitlines()
        except Exception:
            pass
    meta = MODELS_FOLDER / "models_metadata.json"
    if meta.exists():
        try:
            res["metadata"] = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            pass
    return res


def stop_training() -> bool:
    global _process
    if _process and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
        finally:
            _process = None
        return True
    return False
