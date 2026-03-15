import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    df_carbon = df_carbon[df_carbon.index >= "2020-01-01"]

    df_preds = pd.read_parquet("data/predictions/ci_predictions.parquet")
    df_preds = df_preds.sort_index()

    # Load feature importance
    fi_path = "data/predictions/feature_importance.parquet"
    if os.path.exists(fi_path):
        df_fi = pd.read_parquet(fi_path).sort_values("importance", ascending=True)
    else:
        df_fi = None

    # Extend predictions to the full carbon data range (new dates will be NaN)
    df_preds = df_preds.reindex(df_carbon.index)

    # Available dates come from carbon data (updated in real-time)
    available_dates = pd.to_datetime(df_carbon.index.date).unique()

    return df_carbon, df_preds, available_dates, df_fi


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

    **Data:** UK Grid Generation Mix & Carbon Intensity (2020 to present)
    **Models:** Gradient Boosting (HGBRegressor) with Probabilistic Forecasting (q10/q50/q90)  
    **Scenarios:** Low-carbon hours vs high-renewable hours  
    """)

    st.title("UK Grid Carbon Intensity – Household Load Shifting Simulator")

    df_carbon, df_preds, available_dates, df_fi = load_data()

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
        ["Historical (actual)", "Model prediction (with uncertainty)"],
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
        min_value=2.0,
        max_value=20.0,
        value=8.5,
        step=0.1,
        help="Default 8.5 kWh/day based on BEIS/Elexon UK average (approx. 7.5-10 kWh).",
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
    st.sidebar.caption("Data: UK grid carbon intensity & generation mix, 2020 to present")

    # Fetch day data
    try:
        df_day_carbon = get_day_slice(df_carbon, date_str)
        q10 = None
        q90 = None
        
        preds_available = not df_preds.loc[date_str, "CI_pred_q50"].isna().all()

        if ci_source.startswith("Historical") or not preds_available:
            if not preds_available and not ci_source.startswith("Historical"):
                st.warning(
                    f"Model predictions are not available for {date_str} "
                    "(beyond the trained model's range). Showing historical actuals instead."
                )
            ci_day = df_preds.loc[date_str, "CI_actual"]
            source_label = "Historical carbon intensity"
        else:
            # Use Median (q50) for prediction and capture q10/q90 for uncertainty
            ci_day = df_preds.loc[date_str, "CI_pred_q50"]
            q10 = df_preds.loc[date_str, "CI_pred_q10"]
            q90 = df_preds.loc[date_str, "CI_pred_q90"]
            source_label = "Model predicted carbon intensity (q50)"

        renewable_share_day = compute_renewable_share(df_day_carbon)

    except Exception as e:
        st.error(f"Could not load a complete day of data for {date_str}. ({e})")
        return

    # Run scenario
    try:
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

    # Interactive Plotly chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    hours = results["index"].hour
    
    # Add load profiles
    fig.add_trace(
        go.Scatter(x=hours, y=results["baseline_load"], name="Baseline Load (kWh)", 
                   line=dict(dash='dash', color='blue', width=2), opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=hours, y=results["shifted_load"], name="Shifted Load (kWh)", 
                   line=dict(color='blue', width=4)),
        secondary_y=False,
    )
    
    # Add Carbon Intensity
    if q10 is not None and q90 is not None:
        # Prediction interval
        fig.add_trace(
            go.Scatter(
                x=list(hours) + list(hours)[::-1],
                y=list(q90) + list(q10)[::-1],
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="Prediction Interval (q10-q90)"
            ),
            secondary_y=True,
        )
        
    fig.add_trace(
        go.Scatter(x=hours, y=results["ci"], name="Carbon Intensity (gCO₂/kWh)", 
                   line=dict(color='grey', width=2), fill='tozeroy', fillcolor='rgba(128, 128, 128, 0.1)'),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text=f"Load Shifting Scenario: {date_str} ({source_label})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Hour of Day", tickmode='linear', tick0=0, dtick=1)
    fig.update_yaxes(title_text="Load (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Carbon Intensity (gCO₂/kWh)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

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
        if q10 is not None:
            df_display["CI_q10"] = q10.values
            df_display["CI_q90"] = q90.values
            
        st.dataframe(df_display)

    # Feature Importance Section
    if df_fi is not None:
        st.markdown("---")
        st.subheader("Model Insights: Feature Importance")
        st.markdown("""
        How much does each feature contribute to the carbon intensity forecast? 
        Calculated using **permutation importance** on the test set (neg. MAE).
        """)
        
        fig_fi = go.Figure(go.Bar(
            x=df_fi["importance"],
            y=df_fi["feature"],
            orientation='h',
            marker_color='blue'
        ))
        fig_fi.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Importance (Change in MAE)",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_fi, use_container_width=True)

if __name__ == "__main__":
    main()
