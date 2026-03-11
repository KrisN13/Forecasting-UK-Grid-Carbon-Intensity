"""
models.py
---------
Model training, evaluation, and persistence for the Carbon Intensity
forecasting project.

Public API
~~~~~~~~~~
    train_hgb(X_train, y_train, **kwargs)      -> HistGradientBoostingRegressor
    train_ridge(X_train, y_train, **kwargs)     -> Ridge
    evaluate_model(name, model, ...)            -> dict
    build_predictions_df(model, X, y, splits)  -> pd.DataFrame
    feature_importance(model, X_test, y_test)  -> pd.Series
    save_predictions(df_preds, path)
    load_predictions(path)                      -> pd.DataFrame
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> HistGradientBoostingRegressor:
    """
    Fit a HistGradientBoostingRegressor.

    Default hyperparameters come from ``config.HGB_PARAMS``; any keyword
    argument passed here will override them.
    """
    params = {**HGB_PARAMS, **kwargs}
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


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


def summarise_results(results_list: list[dict]) -> pd.DataFrame:
    """Convert a list of evaluate_model dicts into a tidy comparison table."""
    return pd.DataFrame(results_list).set_index("name")


# ---------------------------------------------------------------------------
# Predictions DataFrame
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
    Build a single DataFrame with actual and predicted carbon intensity
    aligned on the original datetime index.

    Predictions are written only for the rows in each split; all other
    rows get NaN in the ``CI_pred`` column.
    """
    df_preds = pd.DataFrame({"CI_actual": y})
    df_preds["CI_pred"] = np.nan

    df_preds.loc[y_train.index, "CI_pred"] = model.predict(X_train)
    df_preds.loc[y_val.index,   "CI_pred"] = model.predict(X_val)
    df_preds.loc[y_test.index,  "CI_pred"] = model.predict(X_test)

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
    pd.Series
        Top ``top_n`` features sorted by mean importance (descending).
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
    print(f"Predictions saved → {path}")


def load_predictions(path: str | Path = PREDS_PARQUET) -> pd.DataFrame:
    """Load a previously saved predictions parquet file."""
    return pd.read_parquet(path)
