"""
carbon
======
Forecasting package for UK Grid Carbon Intensity.

Modules
-------
config    – paths, constants, hyperparameters
features  – feature engineering pipeline
models    – training, evaluation, persistence
scenarios – household demand-shifting scenario engine
"""

from .config import COLORS, TARGET_COL, MIX_COLS  # noqa: F401

from .features import (  # noqa: F401
    build_features,
    add_time_features,
    add_lag_features,
    add_mix_lag1,
    split_X_y,
    time_split,
)

from .models import (  # noqa: F401
    train_hgb,
    train_ridge,
    evaluate_model,
    summarise_results,
    build_predictions_df,
    feature_importance,
    save_predictions,
    load_predictions,
)

from .scenarios import (  # noqa: F401
    generate_household_load,
    generate_ev_load,
    generate_total_load,
    compute_renewable_share,
    get_day_slice,
    select_shift_hours,
    run_shift_scenario,
    compute_daily_reductions_over_range,
    summarize_reductions,
    style_reduction_summary,
)

__all__ = [
    # config
    "COLORS", "TARGET_COL", "MIX_COLS",
    # features
    "build_features", "add_time_features", "add_lag_features",
    "add_mix_lag1", "split_X_y", "time_split",
    # models
    "train_hgb", "train_ridge", "evaluate_model", "summarise_results",
    "build_predictions_df", "feature_importance",
    "save_predictions", "load_predictions",
    # scenarios
    "generate_household_load", "generate_ev_load", "generate_total_load",
    "compute_renewable_share", "get_day_slice", "select_shift_hours",
    "run_shift_scenario", "compute_daily_reductions_over_range",
    "summarize_reductions", "style_reduction_summary",
]
