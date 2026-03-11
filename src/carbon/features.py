"""
features.py
-----------
Feature engineering pipeline for the Carbon Intensity forecasting project.

Public API
~~~~~~~~~~
    build_features(df_carbon, df_temp) -> pd.DataFrame
        Full pipeline: merge → time features → lag features → mix lags → cleanup.

    add_time_features(df)   -> pd.DataFrame
    add_lag_features(df)    -> pd.DataFrame
    add_mix_lag1(df)        -> pd.DataFrame
    split_X_y(df_model)     -> (X, y, feature_cols)
    time_split(X, y, df)    -> (X_train, y_train, X_val, y_val, X_test, y_test)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .config import (
    TARGET_COL,
    CI_LAGS,
    MIX_COLS,
    EXCLUDE_FROM_FEATURES,
    TEMP_COLS_TO_DROP,
    DATA_CUTOFF,
    TRAIN_START,
    TRAIN_YEARS,
    VAL_YEAR,
    TEST_YEAR,
)


# ---------------------------------------------------------------------------
# Individual feature-engineering steps
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day, day-of-week, and month cyclical features."""
    df = df.copy()
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = CI_LAGS,
) -> pd.DataFrame:
    """
    Add lagged carbon-intensity columns and a 24-hour rolling mean.

    Parameters
    ----------
    df   : DataFrame with a ``CARBON_INTENSITY`` column.
    lags : Lag periods in hours (default: 1, 24, 168).
    """
    df = df.copy()
    for lag in lags:
        df[f"ci_lag_{lag}"] = df[TARGET_COL].shift(lag)
    df["ci_rollmean_24"] = df[TARGET_COL].rolling(24).mean()
    return df


def add_mix_lag1(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-hour lags for every generation-mix column (avoids data leakage)."""
    df = df.copy()
    for col in MIX_COLS:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    return df


def _add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature-derived columns.
    Columns that proved negligible (see config.TEMP_COLS_TO_DROP) are
    created then immediately removed so the logic stays in one place.
    """
    df = df.copy()
    df["temp_2m_lag_1"]     = df["temp_2m"].shift(1)
    df["temp_2m_lag_24"]    = df["temp_2m"].shift(24)
    df["heating_degree_18"] = (18 - df["temp_2m"]).clip(lower=0)
    df["cooling_degree_22"] = (df["temp_2m"] - 22).clip(lower=0)
    df["temp_rollmean_24"]  = df["temp_2m"].rolling(24).mean()

    df = df.drop(columns=TEMP_COLS_TO_DROP, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(
    df_carbon: pd.DataFrame,
    df_temp:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge carbon and temperature data, then apply the full feature
    engineering pipeline.

    Parameters
    ----------
    df_carbon : Hourly carbon-intensity DataFrame (datetime index, UTC).
    df_temp   : Hourly temperature DataFrame  (datetime index, UTC).

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame ready for ``split_X_y``.
    """
    # --- Date filtering ---
    df_carbon = df_carbon[df_carbon.index >= TRAIN_START].sort_index()

    df_temp = df_temp.copy()
    df_temp.index = pd.to_datetime(df_temp.index, utc=True)
    df_temp = df_temp.sort_index()

    # --- Merge ---
    df = df_carbon.join(df_temp, how="left")

    # --- Cap to last available temperature record ---
    cutoff = pd.Timestamp(DATA_CUTOFF, tz="UTC")
    df = df.loc[:cutoff].copy()

    # --- Feature engineering ---
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_mix_lag1(df)
    df = _add_temperature_features(df)

    return df


# ---------------------------------------------------------------------------
# Feature / target split & train/val/test split
# ---------------------------------------------------------------------------

def split_X_y(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Drop rows with NaNs (created by shifts/rolling), then separate X and y.

    Returns
    -------
    X            : Feature matrix.
    y            : Target series (CARBON_INTENSITY).
    feature_cols : List of column names used as features.
    """
    df_model = df.dropna().copy()
    y = df_model[TARGET_COL]

    feature_cols = [
        col for col in df_model.columns
        if col not in EXCLUDE_FROM_FEATURES
    ]

    X = df_model[feature_cols]
    return X, y, feature_cols


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_model: pd.DataFrame | None = None,
) -> tuple:
    """
    Temporal train / validation / test split based on calendar year.

    Years used (see config):
        Train : TRAIN_YEARS[0] – TRAIN_YEARS[1]  (default 2020–2023)
        Val   : VAL_YEAR                          (default 2024)
        Test  : TEST_YEAR                         (default 2025)

    Parameters
    ----------
    X        : Feature matrix (datetime index).
    y        : Target series  (same datetime index).
    df_model : Optional — only used if X.index is not already a DatetimeIndex.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    idx = X.index

    train_mask = (idx.year >= TRAIN_YEARS[0]) & (idx.year <= TRAIN_YEARS[1])
    val_mask   = idx.year == VAL_YEAR
    test_mask  = idx.year == TEST_YEAR

    return (
        X[train_mask], y[train_mask],
        X[val_mask],   y[val_mask],
        X[test_mask],  y[test_mask],
    )
