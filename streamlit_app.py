import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Data loading and caching
# =========================

@st.cache_data
def load_data():
    df_carbon = pd.read_parquet("data/processed/df_carbon.parquet")
    df_carbon = df_carbon.sort_index()
    df_carbon = df_carbon.asfreq("H")
    df_carbon = df_carbon[df_carbon.index >= "2020-01-01"]

    df_preds = pd.read_parquet("data/predictions/ci_predictions.parquet")
    df_preds = df_preds.sort_index()

    # Align on common hourly index to avoid surprises
    common_idx = df_preds.index.intersection(df_carbon.index)
    df_carbon = df_carbon.loc[common_idx]
    df_preds = df_preds.loc[common_idx]

    # Available dates for the selector
    available_dates = pd.to_datetime(df_preds.index.date).unique()

    return df_carbon, df_preds, available_dates


# =========================
# Helper functions
# =========================

HOUSEHOLD_PROFILE_RAW = np.array([
    0.25, 0.23, 0.22, 0.22, 0.25,        # 00–04
    0.35, 0.55, 0.65, 0.60,              # 05–08
    0.55, 0.50, 0.48, 0.47, 0.50, 0.55,  # 09–14
    0.60, 0.75, 1.10, 1.20, 1.05,        # 15–19
    0.70, 0.55, 0.40, 0.30               # 20–23
])


def make_household_profile(daily_kwh: float = 14.0) -> pd.Series:
    """Return 24-element Series of hourly kWh, scaled to daily_kwh."""
    raw = HOUSEHOLD_PROFILE_RAW.astype(float)
    scale = daily_kwh / raw.sum()
    return pd.Series(raw * scale, index=range(24))


def compute_renewable_share(df_day: pd.DataFrame) -> pd.Series:
    """RENEWABLE / GENERATION, clipped between 0 and 1."""
    share = df_day["RENEWABLE"] / df_day["GENERATION"]
    return share.clip(lower=0.0, upper=1.0)


def get_day_slice(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Return 24h slice for a given YYYY-MM-DD."""
    day_df = df.loc[date_str:date_str]
    if len(day_df) != 24:
        raise ValueError(f"Expected 24 rows for {date_str}, got {len(day_df)}")
    return day_df


def run_shift_scenario(
    ci_series: pd.Series,
    daily_kwh: float = 14.0,
    flexible_share: float = 0.3,
    strategy: str = "low_intensity",
    renewable_share: pd.Series | None = None,
    n_target_hours: int = 4,
) -> dict:
    """
    Run a load-shifting scenario for a single day.

    strategy: "low_intensity" or "max_renewable"
    """
    ci_series = ci_series.sort_index()
    if len(ci_series) != 24:
        raise ValueError(f"ci_series must contain exactly 24 hourly values, got {len(ci_series)}")

    index = ci_series.index

    if strategy == "max_renewable":
        if renewable_share is None:
            raise ValueError("renewable_share is required for 'max_renewable' strategy")
        renewable_share = renewable_share.loc[index].sort_index()

    # Baseline load profile aligned to timestamps
    base_profile = make_household_profile(daily_kwh)
    baseline_load = pd.Series(base_profile.values, index=index)

    # Split into non-flexible and flexible components
    nonflex_load = baseline_load * (1.0 - flexible_share)
    flex_load = baseline_load * flexible_share
    total_flex_energy = flex_load.sum()

    # Choose target hours
    if strategy == "low_intensity":
        sort_key = ci_series
        target_hours = sort_key.sort_values(ascending=True).index[:n_target_hours]
    elif strategy == "max_renewable":
        sort_key = renewable_share
        target_hours = sort_key.sort_values(ascending=False).index[:n_target_hours]
    else:
        raise ValueError("strategy must be 'low_intensity' or 'max_renewable'")

    # Allocate flex energy evenly into target hours
    shifted_flex = pd.Series(0.0, index=index)
    per_hour = total_flex_energy / len(target_hours)
    for ts in target_hours:
        shifted_flex[ts] += per_hour

    shifted_load = nonflex_load + shifted_flex

    # Emissions (gCO2) = kWh * gCO2/kWh
    ci_array = ci_series.values
    baseline_emissions = baseline_load.values * ci_array
    shifted_emissions = shifted_load.values * ci_array

    total_baseline_emissions = baseline_emissions.sum()
    total_shifted_emissions = shifted_emissions.sum()
    relative_reduction = (
        (total_baseline_emissions - total_shifted_emissions) / total_baseline_emissions
    )

    return {
        "index": index,
        "ci": ci_array,
        "baseline_load": baseline_load.values,
        "shifted_load": shifted_load.values,
        "baseline_emissions": baseline_emissions,
        "shifted_emissions": shifted_emissions,
        "total_baseline_emissions": total_baseline_emissions,
        "total_shifted_emissions": total_shifted_emissions,
        "relative_reduction": relative_reduction,
    }


# =========================
# Streamlit app
# =========================

def main():
    st.set_page_config(
        page_title="UK Grid Carbon Intensity – Household Shifting",
        layout="wide",
    )

    st.markdown("""
    ### UK Carbon Intensity Forecasting & Household Impact

    This tool simulates how shifting flexible household electricity usage into 
    lower-carbon hours changes daily CO₂ emissions.

    **Data:** UK Grid Generation Mix & Carbon Intensity (2020–2025)  
    **Models:** Gradient Boosting (HGBRegressor) + baseline benchmarks  
    **Scenarios:** Low-carbon hours vs high-renewable hours  
    """)

    st.title("UK Grid Carbon Intensity – Household Load Shifting Simulator")

    df_carbon, df_preds, available_dates = load_data()

    # Sidebar controls
    st.sidebar.header("Scenario settings")

    min_date = pd.to_datetime(available_dates.min())
    max_date = pd.to_datetime(available_dates.max())
    default_date = pd.to_datetime("2024-02-05")
    if not (min_date <= default_date <= max_date):
        default_date = min_date

    selected_date = st.sidebar.date_input(
        "Select date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
    )
    date_str = selected_date.strftime("%Y-%m-%d")

    ci_source = st.sidebar.radio(
        "Carbon intensity source",
        ["Historical (actual)", "Model prediction"],
        index=0,
    )

    strategy_label = st.sidebar.radio(
        "Shifting strategy",
        ["Lowest-intensity hours", "Highest-renewables hours"],
        index=0,
    )
    strategy = "low_intensity" if strategy_label.startswith("Lowest") else "max_renewable"

    daily_kwh = st.sidebar.slider(
        "Daily household consumption (kWh)",
        min_value=5.0,
        max_value=30.0,
        value=14.0,
        step=0.5,
    )

    flexible_share = st.sidebar.slider(
        "Flexible share of daily load",
        min_value=0.0,
        max_value=0.8,
        value=0.3,
        step=0.05,
        help="Fraction of daily consumption that can be shifted (e.g. laundry, dishwasher, EV charging).",
    )

    n_target_hours = st.sidebar.slider(
        "Number of target hours",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="How many of the best hours to concentrate shifted load into.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: UK grid carbon intensity & generation mix, 2020–2025")

    # Fetch day data
    try:
        df_day_carbon = get_day_slice(df_carbon, date_str)
        if ci_source.startswith("Historical"):
            ci_day = df_preds.loc[date_str, "CI_actual"]
            source_label = "Historical carbon intensity"
        else:
            ci_day = df_preds.loc[date_str, "CI_pred"]
            source_label = "Model-predicted carbon intensity"

        renewable_share_day = compute_renewable_share(df_day_carbon)

    except Exception as e:
        st.error(f"Could not load a complete day for {date_str}: {e}")
        return

    # Run scenario
    scenario = run_shift_scenario(
        ci_series=ci_day,
        daily_kwh=daily_kwh,
        flexible_share=flexible_share,
        strategy=strategy,
        renewable_share=renewable_share_day,
        n_target_hours=n_target_hours,
    )

    # =========================
    # Metrics
    # =========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total baseline emissions",
            f"{scenario['total_baseline_emissions']:.0f} gCO₂",
        )
    with col2:
        st.metric(
            "Total shifted emissions",
            f"{scenario['total_shifted_emissions']:.0f} gCO₂",
        )
    with col3:
        st.metric(
            "Relative reduction",
            f"{scenario['relative_reduction'] * 100:.2f} %",
        )

    st.caption(
        f"{source_label} · Strategy: {strategy_label.lower()} · "
        f"Daily load: {daily_kwh:.1f} kWh · Flexible: {flexible_share*100:.0f}%"
    )

    # =========================
    # Plots
    # =========================
    idx = scenario["index"]
    hours = [ts.strftime("%H:%M") for ts in idx]

    # Plot 1: Load before/after
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hours, scenario["baseline_load"], label="Baseline load", marker="o")
    ax1.plot(hours, scenario["shifted_load"], label="Shifted load", marker="o")
    ax1.set_ylabel("Load (kWh)")
    ax1.set_xlabel("Hour of day")
    ax1.set_title("Household load before and after shifting")
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot 2: Carbon intensity & emissions
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(hours, scenario["ci"], label="Carbon intensity (gCO₂/kWh)", marker="o")
    ax2.set_ylabel("gCO₂/kWh")
    ax2.set_xlabel("Hour of day")
    ax2.set_title("Carbon intensity profile")

    ax3 = ax2.twinx()
    ax3.bar(
        hours,
        scenario["baseline_emissions"],
        alpha=0.3,
        label="Baseline emissions",
    )
    ax3.bar(
        hours,
        scenario["shifted_emissions"],
        alpha=0.3,
        label="Shifted emissions",
    )
    ax3.set_ylabel("Emissions (gCO₂)")

    # Build combined legend
    lines, labels = ax2.get_legend_handles_labels()
    bars, bar_labels = ax3.get_legend_handles_labels()
    ax2.legend(lines + bars, labels + bar_labels, loc="upper right")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig1)
    st.pyplot(fig2)


if __name__ == "__main__":
    main()