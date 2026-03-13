"""
scenarios.py
------------
Household demand-shifting scenario engine for the Carbon Intensity
forecasting project.

Public API
~~~~~~~~~~
    generate_household_load(date, daily_kwh)       -> pd.Series
    generate_ev_load(date, daily_ev_kwh)            -> pd.Series
    generate_total_load(date, base_kwh, ev_kwh)    -> (base, ev, total)
    compute_renewable_share(df_day)                 -> pd.Series
    select_shift_hours(ci_day_df, strategy, n_hours) -> DatetimeIndex
    run_shift_scenario(...)                          -> dict
    get_day_slice(df, date)                          -> pd.DataFrame
    compute_daily_reductions_over_range(...)         -> pd.DataFrame
    summarize_reductions(df_red)                     -> pd.DataFrame
"""

from __future__ import annotations

import datetime
import logging

import pandas as pd

logger = logging.getLogger(__name__)

DateLike = str | datetime.date | pd.Timestamp

# ---------------------------------------------------------------------------
# Load profile generators
# ---------------------------------------------------------------------------

def generate_household_load(
    date: DateLike,
    daily_kwh: float = 8.5,
) -> pd.Series:
    """
    Baseline hourly household load profile for a single UTC day.

    Distributes ``daily_kwh`` across four usage bands:
      - Night   00–05 : low background load
      - Morning 06–09 : heating / shower peak
      - Day     10–16 : low daytime load
      - Evening 17–22 : high demand peak

    Parameters
    ----------
    date      : Anything accepted by ``pd.Timestamp`` (str, date, Timestamp).
    daily_kwh : Total energy to distribute across the 24 hours.

    Returns
    -------
    pd.Series
        Hourly load (kWh), indexed 00:00–23:00 UTC.
    """
    date = pd.Timestamp(date)
    if date.tzinfo is None:
        date = date.tz_localize("UTC")

    hours = pd.date_range(date, date + pd.Timedelta("23h"), freq="h", tz="UTC")
    load = pd.Series(0.0, index=hours)

    load.loc[load.between_time("00:00", "05:00").index] = 0.5
    load.loc[load.between_time("06:00", "09:00").index] = 1.2
    load.loc[load.between_time("10:00", "16:00").index] = 0.8
    load.loc[load.between_time("17:00", "22:00").index] = 1.5

    total = load.sum()
    if total > 0:
        load *= daily_kwh / total

    return load


def generate_ev_load(
    date: DateLike,
    daily_ev_kwh: float = 0.0,
) -> pd.Series:
    """
    EV charging profile for a single day.

    Charging is distributed evenly across 18:00–23:00 UTC.
    Returns all zeros when ``daily_ev_kwh <= 0``.
    """
    date = pd.Timestamp(date)
    if date.tzinfo is None:
        date = date.tz_localize("UTC")

    hours = pd.date_range(date, date + pd.Timedelta("23h"), freq="h", tz="UTC")
    ev_profile = pd.Series(0.0, index=hours)

    if daily_ev_kwh <= 0:
        return ev_profile

    charging_idx = ev_profile.between_time("18:00", "23:00").index
    if len(charging_idx) == 0:
        return ev_profile

    ev_profile.loc[charging_idx] = daily_ev_kwh / len(charging_idx)
    return ev_profile


def generate_total_load(
    date: DateLike,
    base_kwh: float = 8.5,
    ev_kwh:   float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Combine household and EV profiles.

    Returns
    -------
    base_load, ev_load, total_load  (all pd.Series on the same hourly index)
    """
    base_load = generate_household_load(date, daily_kwh=base_kwh)
    ev_load   = generate_ev_load(date, daily_ev_kwh=ev_kwh)
    ev_load   = ev_load.reindex(base_load.index).fillna(0.0)
    return base_load, ev_load, base_load + ev_load


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def compute_renewable_share(df_day: pd.DataFrame) -> pd.Series:
    """
    Compute renewable generation share (0–1) from RENEWABLE and GENERATION columns.

    Parameters
    ----------
    df_day : Slice of df_carbon for a single day.
    """
    share = df_day["RENEWABLE"] / df_day["GENERATION"]
    return share.clip(lower=0.0, upper=1.0)


def get_day_slice(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Return the 24-row slice for a given calendar date.

    Raises
    ------
    ValueError if the slice does not contain exactly 24 rows.
    """
    day_df = df.loc[date:date]
    if len(day_df) != 24:
        raise ValueError(f"Expected 24 rows for {date}, got {len(day_df)}")
    return day_df


# ---------------------------------------------------------------------------
# Shift-hour selection
# ---------------------------------------------------------------------------

def select_shift_hours(
    ci_day_df: pd.DataFrame,
    strategy:  str = "low_intensity",
    n_hours:   int = 4,
) -> pd.DatetimeIndex:
    """
    Choose the ``n_hours`` best hours to shift flexible load into.

    Strategies
    ----------
    ``low_intensity``  : hours with the lowest carbon intensity.
    ``max_renewable``  : hours with the highest renewable share
                         (requires a ``RENEWABLE`` column).

    Parameters
    ----------
    ci_day_df : DataFrame with at minimum ``CARBON_INTENSITY`` and,
                for max_renewable, ``RENEWABLE`` columns.
    strategy  : ``'low_intensity'`` or ``'max_renewable'``.
    n_hours   : Number of target hours.
    """
    if strategy == "low_intensity":
        order = ci_day_df["CARBON_INTENSITY"].sort_values()
    elif strategy == "max_renewable":
        if "RENEWABLE" not in ci_day_df.columns:
            raise ValueError("RENEWABLE column required for 'max_renewable' strategy.")
        order = ci_day_df["RENEWABLE"].sort_values(ascending=False)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. "
                         "Choose 'low_intensity' or 'max_renewable'.")

    return order.index[:n_hours]


# ---------------------------------------------------------------------------
# Single-day scenario
# ---------------------------------------------------------------------------

def run_shift_scenario(
    date=None,
    ci_day_df=None,
    # Legacy keyword arguments kept for backwards compatibility
    ci_series=None,
    renewable_share=None,
    strategy: str = "low_intensity",
    base_kwh: float = 8.5,
    daily_kwh: float | None = None,   # alias → base_kwh
    flexible_share: float = 0.30,
    ev_kwh: float = 0.0,
    n_target_hours: int = 4,
) -> dict:
    """
    Compute baseline and demand-shifted emissions for a single day.

    Calling conventions
    -------------------
    **New style** (preferred)::

        run_shift_scenario(
            date=pd.Timestamp("2024-03-05", tz="UTC"),
            ci_day_df=df_carbon.loc["2024-03-05"],
            strategy="low_intensity",
            base_kwh=8.5,
            flexible_share=0.3,
            ev_kwh=7.0,
        )

    **Legacy style** (backwards compatible)::

        run_shift_scenario(
            ci_series=ci_actual_day,
            renewable_share=renewable_share_day,
            daily_kwh=8.5,
            flexible_share=0.3,
            strategy="low_intensity",
        )

    Parameters
    ----------
    date           : Day to model (tz-aware Timestamp or str ``'YYYY-MM-DD'``).
    ci_day_df      : DataFrame containing ``CARBON_INTENSITY`` and ``RENEWABLE``
                     columns for the day (new style).
    ci_series      : Carbon intensity Series  (legacy).
    renewable_share: Renewable share Series   (legacy).
    strategy       : ``'low_intensity'`` or ``'max_renewable'``.
    base_kwh       : Total household energy consumption per day (kWh).
    daily_kwh      : Alias for ``base_kwh`` (legacy).
    flexible_share : Fraction of base load that is shiftable (0–1).
    ev_kwh         : Additional daily EV energy (kWh); fully flexible.
    n_target_hours : Number of hours into which flexible load is redistributed.

    Returns
    -------
    dict with keys:
        ``index``, ``ci``, ``baseline_load``, ``shifted_load``,
        ``baseline_emissions``, ``shifted_emissions``,
        ``total_baseline_emissions``, ``total_shifted_emissions``,
        ``relative_reduction``, ``base_load``, ``ev_load``.
    """
    # ---- Build ci_day_df from legacy keyword arguments ----
    if ci_day_df is None:
        if ci_series is None or renewable_share is None:
            raise ValueError(
                "Provide either ci_day_df, or both ci_series and renewable_share."
            )
        ci_day_df = pd.DataFrame({
            "CARBON_INTENSITY": ci_series,
            "RENEWABLE":        renewable_share,
        }).sort_index()

    # ---- Infer date from index if not supplied ----
    if date is None:
        date = ci_day_df.index[0].normalize()

    date = pd.Timestamp(date)
    if date.tzinfo is None:
        date = date.tz_localize("UTC")

    # ---- Legacy alias ----
    if daily_kwh is not None:
        base_kwh = daily_kwh

    ci_day_df = ci_day_df.sort_index()
    ci = ci_day_df["CARBON_INTENSITY"]

    # 1. Load profiles
    base_load, ev_load, _ = generate_total_load(date, base_kwh=base_kwh, ev_kwh=ev_kwh)
    base_load  = base_load.reindex(ci.index).fillna(0.0)
    ev_load    = ev_load.reindex(ci.index).fillna(0.0)

    # 2. Split into flexible / inflexible
    flexible_base   = base_load * flexible_share
    inflexible_base = base_load - flexible_base
    flexible_ev     = ev_load.copy()       # EV is 100 % flexible
    inflexible_ev   = ev_load * 0.0

    baseline_load = inflexible_base + flexible_base + inflexible_ev + flexible_ev

    # 3. Guard: nothing to shift
    total_flexible_energy = (flexible_base + flexible_ev).sum()
    if total_flexible_energy <= 1e-6:
        baseline_emissions = baseline_load * ci
        total_be = baseline_emissions.sum()
        return {
            "index": ci.index,
            "ci": ci,
            "baseline_load": baseline_load,
            "shifted_load": baseline_load.copy(),
            "baseline_emissions": baseline_emissions,
            "shifted_emissions": baseline_emissions.copy(),
            "total_baseline_emissions": total_be,
            "total_shifted_emissions": total_be,
            "relative_reduction": 0.0,
            "base_load": base_load,
            "ev_load": ev_load,
        }

    # 4. Select target hours and redistribute flexible energy
    target_hours = select_shift_hours(ci_day_df, strategy=strategy, n_hours=n_target_hours)
    shifted_flexible = pd.Series(0.0, index=ci.index)
    shifted_flexible.loc[target_hours] = total_flexible_energy / len(target_hours)

    shifted_load = inflexible_base + inflexible_ev + shifted_flexible

    # 5. Emissions
    baseline_emissions = baseline_load * ci
    shifted_emissions  = shifted_load  * ci

    total_be = baseline_emissions.sum()
    total_se = shifted_emissions.sum()

    relative_reduction = (
        (total_be - total_se) / total_be if total_be > 0 else 0.0
    )

    return {
        "index": ci.index,
        "ci": ci,
        "baseline_load": baseline_load,
        "shifted_load": shifted_load,
        "baseline_emissions": baseline_emissions,
        "shifted_emissions": shifted_emissions,
        "total_baseline_emissions": total_be,
        "total_shifted_emissions": total_se,
        "relative_reduction": relative_reduction,
        "base_load": base_load,
        "ev_load": ev_load,
    }


# ---------------------------------------------------------------------------
# Multi-day analysis
# ---------------------------------------------------------------------------

def compute_daily_reductions_over_range(
    start_date: str,
    end_date:   str,
    df_carbon:  pd.DataFrame,
    df_preds:   pd.DataFrame,
    flexible_share:  float = 0.3,
    daily_kwh:       float = 8.5,
    ev_kwh:          float = 0.0,
    strategy_list:   tuple = ("low_intensity", "max_renewable"),
) -> pd.DataFrame:
    """
    Compute daily CO₂ reduction fractions for every day in a date range.

    Parameters
    ----------
    start_date, end_date : Inclusive date range (``'YYYY-MM-DD'``).
    df_carbon            : Raw grid DataFrame (provides RENEWABLE / GENERATION).
    df_preds             : Predictions DataFrame with ``CI_actual`` column.
    flexible_share       : Fraction of base household load that is shiftable.
    daily_kwh            : Household base consumption per day.
    ev_kwh               : Additional EV daily energy (0 = no EV).
    strategy_list        : Strategies to evaluate for each day.

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns ``strategy`` and ``reduction`` (fraction).
        Incomplete days are silently skipped.
    """
    records = []

    for d in pd.date_range(start_date, end_date, freq="D"):
        date_str = d.strftime("%Y-%m-%d")
        try:
            df_day     = get_day_slice(df_carbon, date_str)
            ci_day     = df_preds.loc[date_str, "CI_actual"]
            renew_share = compute_renewable_share(df_day)

            for strategy in strategy_list:
                res = run_shift_scenario(
                    ci_series=ci_day,
                    renewable_share=renew_share,
                    daily_kwh=daily_kwh,
                    flexible_share=flexible_share,
                    strategy=strategy,
                    ev_kwh=ev_kwh,
                )
                records.append({
                    "date":     d,
                    "strategy": strategy,
                    "reduction": res["relative_reduction"],
                })
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping {date_str}: {e}")
            continue  # skip incomplete days

    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_reductions(df_red: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot reduction results and produce a summary statistics table (%).

    Returns
    -------
    pd.DataFrame with index ``strategy`` and columns
    ``mean_%``, ``std_%``, ``min_%``, ``max_%``, ``n_days``.
    """
    df_pivot = df_red.pivot(columns="strategy", values="reduction")
    rows = []
    for strategy in df_pivot.columns:
        vals = df_pivot[strategy].dropna() * 100
        rows.append({
            "strategy": strategy,
            "mean_%":   vals.mean(),
            "std_%":    vals.std(),
            "min_%":    vals.min(),
            "max_%":    vals.max(),
            "n_days":   len(vals),
        })
    return pd.DataFrame(rows).set_index("strategy")


def style_reduction_summary(summary_df: pd.DataFrame):
    """
    Return a Jupyter-friendly styled version of a ``summarize_reductions`` table.
    """
    return (
        summary_df.style
        .format({"mean_%": "{:.2f}", "std_%": "{:.2f}",
                 "min_%":  "{:.2f}", "max_%": "{:.2f}", "n_days": "{:.0f}"})
        .background_gradient(subset=["mean_%", "max_%"], cmap="Greens")
        .set_caption("Summary of daily CO₂ reduction by strategy")
    )
