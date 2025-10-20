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

# Common filenames produced by the existing scripts
FORECAST_PER_PRODUCT = ROOT_DIR / "forecast_per_product_24m.csv"
FORECAST_TOTAL = ROOT_DIR / "forecast_total_24m.csv"
TOPN_PER_MONTH = ROOT_DIR / "topN_per_month_24m.csv"
FORECAST_DIAGNOSTICS = ROOT_DIR / "forecast_diagnostics.csv"
TRAINING_DIAGNOSTICS = MODELS_FOLDER / "training_diagnostics.csv"
SKIPPED_PRODUCTS_LOG = MODELS_FOLDER / "skipped_products.log"
MODELS_METADATA = MODELS_FOLDER / "models_metadata.json"

# Default data file at project root (legacy)
DEFAULT_EXCEL = ROOT_DIR / "Data_Penjualan_Dengan_ID_Pelanggan.xlsx"
