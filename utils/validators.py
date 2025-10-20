from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import (
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    MODELS_METADATA,
    TRAINING_DIAGNOSTICS,
    SKIPPED_PRODUCTS_LOG,
    FORECAST_TOTAL,
)


def validate_file_size(file_bytes: bytes) -> bool:
    return len(file_bytes) <= MAX_FILE_SIZE


def validate_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_date_column(series: pd.Series) -> Dict[str, int | bool]:
    dt = pd.to_datetime(series, errors="coerce")
    invalid = int(dt.isna().sum())
    return {"all_valid": invalid == 0, "invalid_count": invalid}


def validate_numeric_column(series: pd.Series) -> Dict[str, int | bool]:
    # try convert to numeric
    cn = pd.to_numeric(series, errors="coerce")
    invalid = int(cn.isna().sum())
    return {"all_valid": invalid == 0, "invalid_count": invalid}


def check_data_quality(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, object]:
    dcol, qcol, pcol = mapping["date"], mapping["qty"], mapping["product"]
    res = {}
    res["missing_values"] = int(df[[dcol, qcol, pcol]].isna().sum().sum())
    res["duplicates"] = int(df.duplicated().sum())
    dates = pd.to_datetime(df[dcol], errors="coerce")
    res["invalid_dates"] = int(dates.isna().sum())
    qn = pd.to_numeric(df[qcol], errors="coerce")
    res["invalid_numeric"] = int(qn.isna().sum())
    return res


def check_required_files() -> Dict[str, bool]:
    return {
        "models_metadata": MODELS_METADATA.exists(),
        "training_diagnostics": TRAINING_DIAGNOSTICS.exists(),
        "skipped_log": SKIPPED_PRODUCTS_LOG.exists(),
        "forecast_total": FORECAST_TOTAL.exists(),
    }
