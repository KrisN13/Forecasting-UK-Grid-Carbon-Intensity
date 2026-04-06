import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Site colour palette (krisnoondata.com green theme) ───────────────────────
G_DEEP  = "#1c3a2a"
G_MID   = "#2a5040"
G_PANEL = "#243d2f"
CREAM   = "#f5eedc"
CREAM_D = "#c4b99a"
GOLD    = "#c8a84b"
TEAL    = "#7eb8a4"
GRID    = "#3d5c47"
BORDER  = "#3d5c47"

mpl.rcParams.update({
    "figure.facecolor":  G_PANEL,
    "axes.facecolor":    G_PANEL,
    "savefig.facecolor": G_PANEL,
    "text.color":        CREAM,
    "axes.labelcolor":   GOLD,
    "axes.titlecolor":   CREAM,
    "xtick.color":       CREAM_D,
    "ytick.color":       CREAM_D,
    "axes.edgecolor":    GRID,
    "axes.linewidth":    0.8,
    "grid.color":        GRID,
    "grid.alpha":        0.5,
    "legend.facecolor":  G_MID,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": CREAM_D,
    "figure.dpi":        130,
})

# Add src to path so we can import the carbon package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from carbon.scenarios import (
    run_shift_scenario,
    get_day_slice,
    compute_renewable_share
)

# =========================
# Data loading and caching
# =========================

@st.cache_data
def load_data():
    df_carbon = pd.read_parquet("data/processed/df_carbon.parquet")
    df_carbon = df_carbon.sort_index()
    df_carbon = df_carbon.asfreq("H")
    df_carbon = df_carbon[df_carbon.index >= "2022-01-01"]

    df_preds = pd.read_parquet("data/predictions/ci_predictions.parquet")
    df_preds = df_preds.sort_index()

    # Align on common hourly index
    common_idx = df_preds.index.intersection(df_carbon.index)
    df_carbon = df_carbon.loc[common_idx]
    df_preds = df_preds.loc[common_idx]

    available_dates = pd.to_datetime(df_preds.index.date).unique()

    return df_carbon, df_preds, available_dates


# =========================
# Streamlit app
# =========================

def main():
    st.set_page_config(
        page_title="UK Grid Carbon Intensity – Household Shifting",
        layout="wide",
    )

    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{
        background-color: {G_DEEP};
        color: {CREAM};
        font-family: 'DM Mono', monospace;
      }}
      section[data-testid="stSidebar"] {{
        background-color: {G_MID};
      }}
      section[data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
      .stMetric label {{ color: {CREAM_D} !important; font-size: 11px; }}
      .stMetric [data-testid="metric-container"] {{ background: {G_PANEL}; border-top: 2px solid {GOLD}; padding: 12px 16px; border-radius: 2px; }}
      h1, h2, h3 {{ color: {CREAM} !important; }}
      .stMarkdown a {{ color: {GOLD}; }}
      #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### UK Carbon Intensity Forecasting & Household Impact

    This tool simulates how shifting flexible household electricity usage into 
    lower-carbon hours changes daily CO₂ emissions.

    **Data:** UK Grid Generation Mix & Carbon Intensity (2020 to 2025)  
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
        help="How many hours to concentrate the shifted load into.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: UK grid carbon intensity & generation mix, 2020 to 2025")

    # Fetch day data
    try:
        df_day_carbon = get_day_slice(df_carbon, date_str)
        if ci_source.startswith("Historical"):
            ci_day = df_preds.loc[date_str, "CI_actual"]
            source_label = "Historical carbon intensity"
        else:
            ci_day = df_preds.loc[date_str, "CI_pred_q50"]
            source_label = "Model predicted carbon intensity"

        renewable_share_day = compute_renewable_share(df_day_carbon)

    except Exception as e:
        st.error(f"Could not load a complete day of data for {date_str}. ({e})")
        return

    # Run scenario
    try:
        # Note: The package run_shift_scenario accepts base_kwh (legacy: daily_kwh)
        results = run_shift_scenario(
            ci_series=ci_day,
            renewable_share=renewable_share_day,
            daily_kwh=daily_kwh,
            flexible_share=flexible_share,
            strategy=strategy,
            n_target_hours=n_target_hours,
        )
    except Exception as e:
        st.error(f"Scenario failed: {e}")
        return

    # Metrics
    col1, col2, col3 = st.columns(3)
    
    baseline_co2 = results["total_baseline_emissions"]
    shifted_co2 = results["total_shifted_emissions"]
    reduction_pct = results["relative_reduction"] * 100.0
    
    with col1:
        st.metric("Baseline Emissions", f"{baseline_co2:.0f} gCO₂")
    with col2:
        st.metric("Shifted Emissions", f"{shifted_co2:.0f} gCO₂")
    with col3:
        st.metric("Reduction", f"{reduction_pct:.2f}%", delta_color="normal")

    # Plots
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = results["index"].hour

    # Carbon intensity as a filled background band — cream at low opacity
    ax2 = ax.twinx()
    ax2.fill_between(hours, results["ci"], color=CREAM, alpha=0.06, zorder=0)
    ax2.plot(hours, results["ci"], color=CREAM_D, alpha=0.5, linewidth=1,
             linestyle=":", label="Carbon Intensity (gCO₂/kWh)", zorder=1)
    ax2.set_ylabel("Carbon Intensity (gCO₂/kWh)", color=CREAM_D)
    ax2.tick_params(axis="y", colors=CREAM_D)
    ax2.spines["right"].set_color(GRID)
    ax2.spines["left"].set_color(GRID)
    ax2.spines["top"].set_color(GRID)
    ax2.spines["bottom"].set_color(GRID)

    # Load lines — teal for baseline, gold for shifted
    ax.plot(hours, results["baseline_load"], label="Baseline Load (kWh)",
            color=TEAL, linestyle="--", linewidth=1.8, alpha=0.85, zorder=3)
    ax.plot(hours, results["shifted_load"],  label="Shifted Load (kWh)",
            color=GOLD, linewidth=2.2, zorder=4)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Load (kWh)")
    ax.set_title(f"Load Shifting Scenario: {date_str} ({source_label})")
    ax.set_xlim(0, 23)
    ax.grid(True, axis="y")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    st.pyplot(fig)

    # Show data
    with st.expander("See detailed data"):
        df_display = pd.DataFrame({
            "Hour": results["index"],
            "Carbon Intensity": results["ci"],
            "Baseline Load": results["baseline_load"],
            "Shifted Load": results["shifted_load"],
            "Baseline Emissions": results["baseline_emissions"],
            "Shifted Emissions": results["shifted_emissions"],
        })
        st.dataframe(df_display)

if __name__ == "__main__":
    main()