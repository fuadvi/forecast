from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_FOLDER = ROOT_DIR / "uploads"
MODELS_FOLDER = ROOT_DIR / "trained_models"
OUTPUT_FOLDER = ROOT_DIR / "outputs"
ASSETS_FOLDER = ROOT_DIR / "assets"

# Ensure folders exist
for p in [UPLOAD_FOLDER, MODELS_FOLDER, OUTPUT_FOLDER, ASSETS_FOLDER]:
    p.mkdir(parents=True, exist_ok=True)

# Application settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".xlsx"}

# Common filenames produced by the existing LSTM scripts
FORECAST_PER_PRODUCT = ROOT_DIR / "forecast_per_product_24m.csv"
FORECAST_TOTAL = ROOT_DIR / "forecast_total_24m.csv"
TOPN_PER_MONTH = ROOT_DIR / "topN_per_month_24m.csv"
FORECAST_DIAGNOSTICS = ROOT_DIR / "forecast_diagnostics.csv"
TRAINING_DIAGNOSTICS = MODELS_FOLDER / "training_diagnostics.csv"
SKIPPED_PRODUCTS_LOG = MODELS_FOLDER / "skipped_products.log"
MODELS_METADATA = MODELS_FOLDER / "models_metadata.json"

# LSTM Combined Visualization Plots (Quarterly Analysis)
LSTM_PLOTS_DIR = ROOT_DIR / "forecast_plots"
LSTM_TOP5_YEARLY_PNG = LSTM_PLOTS_DIR / "top5_yearly.png"
LSTM_TOP5_QUARTERLY_PNG = LSTM_PLOTS_DIR / "top5_quarterly.png"
LSTM_BORDA_COUNT_PNG = LSTM_PLOTS_DIR / "borda_count_process.png"
LSTM_GROUPED_TOP5_PNG = LSTM_PLOTS_DIR / "bulan" / "top5_grouped_24m.png"

# LSTM Quarterly/Yearly CSV outputs
LSTM_QUARTERLY_TOP5_2025 = ROOT_DIR / "quarterly_top5_2025.csv"
LSTM_QUARTERLY_TOP5_2026 = ROOT_DIR / "quarterly_top5_2026.csv"
LSTM_YEARLY_TOP5_2025 = ROOT_DIR / "yearly_top5_borda_2025.csv"
LSTM_YEARLY_TOP5_2026 = ROOT_DIR / "yearly_top5_borda_2026.csv"

# SES outputs and plots
SES_FORECAST_PER_PRODUCT = ROOT_DIR / "forecast_per_product_ses_24m.csv"
SES_FORECAST_TOTAL = ROOT_DIR / "forecast_total_ses_24m.csv"
SES_TOPN_PER_MONTH = ROOT_DIR / "topN_per_month_ses_24m.csv"
SES_SKIPPED_PRODUCTS = ROOT_DIR / "ses_skipped_products.csv"
SES_MODEL_PARAMS = ROOT_DIR / "ses_model_params.csv"
SES_PLOTS_DIR = ROOT_DIR / "forecast_plots" / "bulan"
SES_GROUPED_TOP5_PNG = SES_PLOTS_DIR / "top5_grouped_24m_ses.png"

# SES default parameters
SES_DEFAULTS = {
    "top_k": 5,
    "forecast_months": 24,
    "min_points": 6,
    "outlier_capping": True,
}

# Default data file at project root (legacy)
DEFAULT_EXCEL = ROOT_DIR / "Data_Penjualan_Dengan_ID_Pelanggan.xlsx"
