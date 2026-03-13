"""
config.py
---------
Central configuration for the Carbon Intensity forecasting package.
Edit paths and constants here rather than inside individual modules.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR        = _ROOT / "data"
PROCESSED_DIR   = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
ASSETS_DIR      = _ROOT / "assets"

CARBON_PARQUET = PROCESSED_DIR / "df_carbon.parquet"
TEMP_PARQUET   = PROCESSED_DIR / "uk_temp_hourly.parquet"
PREDS_PARQUET  = PREDICTIONS_DIR / "ci_predictions.parquet"

# ---------------------------------------------------------------------------
# Dataset date range
# ---------------------------------------------------------------------------
TRAIN_START = "2020-01-01"
DATA_CUTOFF = "2025-09-30 23:00:00"   # last available temperature record

TRAIN_YEARS = (2020, 2023)
VAL_YEAR    = 2024
TEST_YEAR   = 2025

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TARGET_COL = "CARBON_INTENSITY"

CI_LAGS = (1, 24, 168)   # 1 h, 24 h, 1 week

MIX_COLS = [
    "FOSSIL", "COAL", "GAS", "NUCLEAR", "STORAGE",
    "GENERATION", "WIND", "HYDRO", "SOLAR", "BIOMASS",
    "RENEWABLE", "OTHER",
]

# Columns excluded from the feature matrix (raw target + raw mix)
EXCLUDE_FROM_FEATURES = [TARGET_COL] + MIX_COLS + ["LOW_CARBON", "ZERO_CARBON"]

# Temperature-derived columns dropped after experimentation
TEMP_COLS_TO_DROP = [
    "temp_2m_lag_1",
    "temp_2m_lag_24",
    "temp_rollmean_24",
    "heating_degree_18",
    "cooling_degree_22",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
HGB_PARAMS = dict(
    max_depth=8,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
)

RIDGE_ALPHA = 1.0

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COLORS = {
    "blue":   "#1764AB",
    "green":  "#4CA466",
    "orange": "#D55E00",
    "grey":   "#6E7D8C",
}
