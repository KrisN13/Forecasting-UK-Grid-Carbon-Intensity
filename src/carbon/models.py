"""
models.py
---------
Model training, evaluation, and persistence for the Carbon Intensity
forecasting project.

Public API
~~~~~~~~~~
    train_hgb(X_train, y_train, **kwargs)                -> HistGradientBoostingRegressor
    train_hgb_quantiles(X_train, y_train, quantiles)     -> dict[str, model]
    train_ridge(X_train, y_train, **kwargs)               -> Ridge
    evaluate_model(name, model, ...)                      -> dict
    evaluate_interval_coverage(df_preds, q_low, q_high)  -> dict
    build_predictions_df(model, X, y, splits)             -> pd.DataFrame
    build_quantile_predictions_df(models, X, y, splits)  -> pd.DataFrame
    feature_importance(model, X_test, y_test)             -> pd.Series
    save_predictions(df_preds, path)
    load_predictions(path)                                -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import HGB_PARAMS, RIDGE_ALPHA, PREDS_PARQUET

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> HistGradientBoostingRegressor:
    """
    Fit a HistGradientBoostingRegressor (squared error loss).

    Default hyperparameters come from config.HGB_PARAMS; any keyword
    argument passed here will override them.
    """
    params = {**HGB_PARAMS, **kwargs}
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_hgb_quantiles(
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    **kwargs:  Any,
) -> dict[str, HistGradientBoostingRegressor]:
    """
    Train one HGB model per quantile using native quantile loss.

    All models share the same hyperparameters from config.HGB_PARAMS,
    with loss='quantile' and the appropriate quantile value added.

    Parameters
    ----------
    X_train   : Training feature matrix.
    y_train   : Training target series.
    quantiles : Quantiles to train (default: 0.1, 0.5, 0.9).

    Returns
    -------
    dict mapping quantile label (e.g. 'q10') to fitted model.

    Example
    -------
    >>> models = train_hgb_quantiles(X_train, y_train)
    >>> models['q10'], models['q50'], models['q90']
    """
    fitted = {}
    for q in quantiles:
        label = f"q{int(q * 100)}"
        print(f"  Training HGB quantile={q} ({label})...")
        params = {**HGB_PARAMS, "loss": "quantile", "quantile": q, **kwargs}
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        fitted[label] = model
    return fitted


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = RIDGE_ALPHA,
    **kwargs: Any,
) -> Ridge:
    """Fit a Ridge regression model."""
    model = Ridge(alpha=alpha, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def evaluate_model(
    name:    str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a fitted model on train / val / test splits.

    Returns a dictionary of MAE and RMSE for each split, plus the model
    name, suitable for collecting results across multiple models.
    """
    mae_tr,  rmse_tr  = _metrics(y_train, model.predict(X_train))
    mae_val, rmse_val = _metrics(y_val,   model.predict(X_val))
    mae_te,  rmse_te  = _metrics(y_test,  model.predict(X_test))

    results = {
        "name":    name,
        "mae_tr":  mae_tr,  "rmse_tr":  rmse_tr,
        "mae_val": mae_val, "rmse_val": rmse_val,
        "mae_te":  mae_te,  "rmse_te":  rmse_te,
    }

    if verbose:
        print(f"\n=== {name} ===")
        print(f"  Train : MAE={mae_tr:.2f}  RMSE={rmse_tr:.2f}")
        print(f"  Val   : MAE={mae_val:.2f}  RMSE={rmse_val:.2f}")
        print(f"  Test  : MAE={mae_te:.2f}  RMSE={rmse_te:.2f}")

    return results


def evaluate_interval_coverage(
    df_preds: pd.DataFrame,
    q_low:    str = "CI_pred_q10",
    q_high:   str = "CI_pred_q90",
    actual:   str = "CI_actual",
) -> dict[str, float]:
    """
    Evaluate the empirical coverage of a prediction interval.

    For an 80% interval (q10-q90), well-calibrated coverage should be
    close to 80%. Coverage below 80% means the interval is too narrow;
    above means it is too wide.

    Returns
    -------
    dict with keys: coverage, mean_width, target_coverage
    """
    df = df_preds[[actual, q_low, q_high]].dropna()

    within     = (df[actual] >= df[q_low]) & (df[actual] <= df[q_high])
    coverage   = within.mean()
    mean_width = (df[q_high] - df[q_low]).mean()

    try:
        lo = int(q_low.split("q")[-1])
        hi = int(q_high.split("q")[-1])
        target = (hi - lo) / 100
    except Exception:
        target = None

    result = {
        "coverage":        round(float(coverage), 4),
        "mean_width":      round(float(mean_width), 2),
        "target_coverage": target,
    }

    print(f"Interval coverage : {coverage:.1%}  (target: {target:.0%})")
    print(f"Mean interval width: {mean_width:.1f} gCO2/kWh")

    return result


def summarise_results(results_list: list[dict]) -> pd.DataFrame:
    """Convert a list of evaluate_model dicts into a tidy comparison table."""
    return pd.DataFrame(results_list).set_index("name")


# ---------------------------------------------------------------------------
# Predictions DataFrames
# ---------------------------------------------------------------------------

def build_predictions_df(
    model,
    X:       pd.DataFrame,
    y:       pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> pd.DataFrame:
    """
    Build a DataFrame with actual and point-predicted carbon intensity.
    For quantile predictions use build_quantile_predictions_df.
    """
    df_preds = pd.DataFrame({"CI_actual": y})
    df_preds["CI_pred"] = np.nan

    df_preds.loc[y_train.index, "CI_pred"] = model.predict(X_train)
    df_preds.loc[y_val.index,   "CI_pred"] = model.predict(X_val)
    df_preds.loc[y_test.index,  "CI_pred"] = model.predict(X_test)

    return df_preds


def build_quantile_predictions_df(
    models:  dict[str, HistGradientBoostingRegressor],
    X:       pd.DataFrame,
    y:       pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> pd.DataFrame:
    """
    Build a predictions DataFrame with one column per quantile model.

    Columns produced
    ----------------
    CI_actual, CI_pred_q10, CI_pred_q50, CI_pred_q90

    Use CI_pred_q50 as the point forecast in downstream scenario analysis.

    Parameters
    ----------
    models : Dict returned by train_hgb_quantiles.
    """
    df_preds = pd.DataFrame({"CI_actual": y})

    for label, model in models.items():
        col = f"CI_pred_{label}"
        df_preds[col] = np.nan
        df_preds.loc[y_train.index, col] = model.predict(X_train)
        df_preds.loc[y_val.index,   col] = model.predict(X_val)
        df_preds.loc[y_test.index,  col] = model.predict(X_test)

    return df_preds


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance(
    model,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    n_repeats:    int = 10,
    random_state: int = 42,
    top_n:        int = 20,
) -> pd.Series:
    """
    Compute permutation feature importance on the test set.

    Returns
    -------
    pd.Series — top top_n features sorted by mean importance (descending).
    """
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    fi = pd.Series(perm.importances_mean, index=X_test.columns)
    return fi.sort_values(ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_predictions(
    df_preds: pd.DataFrame,
    path: str | Path = PREDS_PARQUET,
) -> None:
    """Save the predictions DataFrame to a parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(path)
    print(f"Predictions saved -> {path}")


def load_predictions(path: str | Path = PREDS_PARQUET) -> pd.DataFrame:
    """Load a previously saved predictions parquet file."""
    return pd.read_parquet(path)
