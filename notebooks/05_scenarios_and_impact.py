#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Consistent plotting style
plt.style.use("default")
sns.set_theme(style="whitegrid")

# Colour palette (clean energy theme)
COLORS = {
    "blue": "#1764AB",
    "green": "#4CA466",
    "orange": "#D55E00",
    "grey": "#6E7D8C",
}

# Make all figures transparent by default
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",

    # Grid defaults
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
})


# In[3]:


# File path for carbon intensity dataset
PATH = "../data/processed/df_carbon.parquet"

# Keeps the DATETIME column as an index
df_carbon = pd.read_parquet(PATH)

# Sort Ascending order by index (DATETIME)
df_carbon = df_carbon.sort_index()


# In[4]:


# File path for predictions dataset
PATH = "../data/predictions/ci_predictions.parquet"

df_preds = pd.read_parquet(PATH)

# Sort Ascending order by index (DATETIME)
df_preds = df_preds.sort_index()


# In[5]:


print(df_preds.index.min(), df_preds.index.max())
print(df_preds[["CI_actual", "CI_pred"]].head())
print(df_preds[["CI_actual", "CI_pred"]].tail())


# In[6]:


ci_actual_day = df_preds.loc["2024-02-05", "CI_actual"]
print(len(ci_actual_day), ci_actual_day.index[0], ci_actual_day.index[-1])


# # Base high-usage UK household profile (14kwh/day)

# In[7]:


def generate_household_load(date, daily_kwh=14.0):
    """
    Baseline household load profile for a single UTC day.
    Returns Series indexed hourly from 00:00 to 23:00.
    """
    hours = pd.date_range(date, date + pd.Timedelta("23h"), freq="h", tz="UTC")
    load = pd.Series(0.0, index=hours)

    night_hours   = load.between_time("00:00", "05:00").index
    morning_hours = load.between_time("06:00", "09:00").index
    day_hours     = load.between_time("10:00", "16:00").index
    eve_hours     = load.between_time("17:00", "22:00").index

    load.loc[night_hours]   = 0.5
    load.loc[morning_hours] = 1.2
    load.loc[day_hours]     = 0.8
    load.loc[eve_hours]     = 1.5

    total = load.sum()
    if total > 0:
        load *= daily_kwh / total

    return load


# In[8]:


def generate_ev_load(date, daily_ev_kwh=0.0):
    """
    EV charging profile for a single day.
    If daily_ev_kwh == 0, returns all zeros.
    Charging allowed between 18:00 and 23:00, evenly distributed.
    """
    hours = pd.date_range(date, date + pd.Timedelta("23h"), freq="h", tz="UTC")
    ev_profile = pd.Series(0.0, index=hours)

    if daily_ev_kwh <= 0:
        return ev_profile

    charging_idx = ev_profile.between_time("18:00", "23:00").index
    if len(charging_idx) == 0:
        return ev_profile

    kwh_per_hour = daily_ev_kwh / len(charging_idx)
    ev_profile.loc[charging_idx] = kwh_per_hour

    return ev_profile


# In[9]:


def generate_total_load(date, base_kwh=14.0, ev_kwh=0.0):
    """
    Returns base_load, ev_load, total_load as Series.
    """
    base_load = generate_household_load(date, daily_kwh=base_kwh)
    ev_load = generate_ev_load(date, daily_ev_kwh=ev_kwh)

    ev_load = ev_load.reindex(base_load.index).fillna(0.0)
    total_load = base_load + ev_load
    return base_load, ev_load, total_load


# A generic function that can:
# 
# - Take a carbon intensity series (24 hourly values)
# - Optionally a renewable share series (for the “renewables” strategy)
# - A flexible share (0 - 0.5)
# - A strategy: "low_intensity" or "max_renewable"
# 
# Returns:
# - baseline load + emissions
# - shifted load + emissions
# - summary stats

# In[10]:


def select_shift_hours(ci_day_df, strategy="low_intensity", n_hours=4):
    """
    ci_day_df: DataFrame with columns:
      - 'CARBON_INTENSITY'
      - 'RENEWABLE' (for max_renewable)
    Returns a DatetimeIndex of hours to shift INTO.
    """
    if strategy == "low_intensity":
        order = ci_day_df["CARBON_INTENSITY"].sort_values()
    elif strategy == "max_renewable":
        if "RENEWABLE" not in ci_day_df.columns:
            raise ValueError("RENEWABLE column is required for 'max_renewable' strategy.")
        order = ci_day_df["RENEWABLE"].sort_values(ascending=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return order.index[:n_hours]


# In[11]:


def compute_renewable_share(df_day: pd.DataFrame) -> pd.Series:
    """
    Compute renewable share (0–1) from RENEWABLE and GENERATION for a given day slice.
    """
    share = df_day["RENEWABLE"] / df_day["GENERATION"]
    return share.clip(lower=0.0, upper=1.0)


# In[12]:


def run_shift_scenario(
    date=None,
    ci_day_df=None,
    # backwards compatible:
    ci_series=None,
    renewable_share=None,
    strategy="low_intensity",
    base_kwh=14.0,
    daily_kwh=None,          # alias for base_kwh
    flexible_share=0.30,
    ev_kwh=0.0,
    n_target_hours=4,
):
    """
    Single-day scenario.

    NEW style:
      run_shift_scenario(
          date=day,
          ci_day_df=ci_day_df,   # has CARBON_INTENSITY, RENEWABLE
          strategy="low_intensity",
          base_kwh=14.0,
          flexible_share=0.3,
          ev_kwh=7.0,
      )

    OLD style (your previous calls):
      run_shift_scenario(
          ci_series=ci_actual_day,
          renewable_share=renewable_share_day,
          daily_kwh=14.0,
          flexible_share=0.3,
          strategy="low_intensity",
      )
    """

    # Build ci_day_df if only series are provided
    if ci_day_df is None:
        if ci_series is None or renewable_share is None:
            raise ValueError(
                "Either ci_day_df must be provided, or ci_series and renewable_share must both be given."
            )
        ci_day_df = pd.DataFrame({
            "CARBON_INTENSITY": ci_series,
            "RENEWABLE": renewable_share,
        })
        ci_day_df = ci_day_df.sort_index()

    # Infer date from index if not given
    if date is None:
        first_ts = ci_day_df.index[0]
        date = first_ts.normalize()

    # daily_kwh override
    if daily_kwh is not None:
        base_kwh = daily_kwh

    ci_day_df = ci_day_df.sort_index()
    ci = ci_day_df["CARBON_INTENSITY"]

    # 1) Loads
    base_load, ev_load, total_load = generate_total_load(date, base_kwh=base_kwh, ev_kwh=ev_kwh)

    base_load = base_load.reindex(ci.index).fillna(0.0)
    ev_load   = ev_load.reindex(ci.index).fillna(0.0)
    total_load = total_load.reindex(ci.index).fillna(0.0)

    # 2) Split into inflexible and flexible
    flexible_base   = base_load * flexible_share
    inflexible_base = base_load - flexible_base

    # EV: fully flexible
    flexible_ev   = ev_load.copy()
    inflexible_ev = ev_load * 0.0

    baseline_load = inflexible_base + flexible_base + inflexible_ev + flexible_ev

    # 3) Select shift hours
    target_hours = select_shift_hours(ci_day_df, strategy=strategy, n_hours=n_target_hours)

    # 4) Total flexible energy
    total_flexible_energy = (flexible_base + flexible_ev).sum()

    if total_flexible_energy <= 1e-6:
        baseline_emissions = baseline_load * ci
        total_baseline_emissions = baseline_emissions.sum()
        return {
            "index": ci.index,
            "ci": ci,
            "baseline_load": baseline_load,
            "shifted_load": baseline_load.copy(),
            "baseline_emissions": baseline_emissions,
            "shifted_emissions": baseline_emissions.copy(),
            "total_baseline_emissions": total_baseline_emissions,
            "total_shifted_emissions": total_baseline_emissions,
            "relative_reduction": 0.0,
            "base_load": base_load,
            "ev_load": ev_load,
        }

    # 5) Redistribute flexible energy into target hours
    shifted_flexible = pd.Series(0.0, index=ci.index)
    per_hour = total_flexible_energy / len(target_hours)
    shifted_flexible.loc[target_hours] = per_hour

    shifted_load = inflexible_base + inflexible_ev + shifted_flexible

    # 6) Emissions
    baseline_emissions = baseline_load * ci
    shifted_emissions  = shifted_load * ci

    total_baseline_emissions = baseline_emissions.sum()
    total_shifted_emissions  = shifted_emissions.sum()

    relative_reduction = (
        (total_baseline_emissions - total_shifted_emissions) / total_baseline_emissions
        if total_baseline_emissions > 0
        else 0.0
    )

    return {
        "index": ci.index,
        "ci": ci,
        "baseline_load": baseline_load,
        "shifted_load": shifted_load,
        "baseline_emissions": baseline_emissions,
        "shifted_emissions": shifted_emissions,
        "total_baseline_emissions": total_baseline_emissions,
        "total_shifted_emissions": total_shifted_emissions,
        "relative_reduction": relative_reduction,
        "base_load": base_load,
        "ev_load": ev_load,
    }


# Note: the allocation loop is intentionally simple. It can be redefined later (e.g., allocate equal proportion into lowest-intensity hours only, or cap hourly shifted load).

# # Historical mode

# In[13]:


def get_day_slice(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Return a single calendar day's worth of data (24h) from a datetime-indexed df.
    date: 'YYYY-MM-DD'
    """
    day_df = df.loc[date:date]
    if len(day_df) != 24:
        raise ValueError(f"Expected 24 rows for {date}, got {len(day_df)}")
    return day_df


# # Forecast mode - On a Specific Date

# In[14]:


date = "2024-01-01"

# Full grid data for the day (for RENEWABLE and GENERATION)
df_day = get_day_slice(df_carbon, date)

# Carbon intensity from actuals
ci_actual_day = df_preds.loc[date, "CI_actual"]

# Renewable share from RENEWABLE / GENERATION
renewable_share_day = compute_renewable_share(df_day)

scenario_low_hist = run_shift_scenario(
    ci_series=ci_actual_day,
    daily_kwh=14.0,
    flexible_share=0.3,
    strategy="low_intensity",
    renewable_share=renewable_share_day,
)

scenario_renew_hist = run_shift_scenario(
    ci_series=ci_actual_day,
    daily_kwh=14.0,
    flexible_share=0.3,
    strategy="max_renewable",
    renewable_share=renewable_share_day,
)


# In[15]:


def plot_scenario(result: dict, title: str = ""):
    idx = result["index"]

    plt.figure(figsize=(14, 6))
    plt.plot(idx, result["baseline_load"], label="Baseline load")
    plt.plot(idx, result["shifted_load"], label="Shifted load")
    plt.ylabel("Load (kWh)")
    plt.title(title or "Household load before/after shifting")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(idx, result["baseline_emissions"], label="Baseline emissions")
    plt.plot(idx, result["shifted_emissions"], label="Shifted emissions")
    plt.ylabel("Emissions (gCO$₂$)")
    plt.title("Emissions before/after shifting")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"Total baseline emissions: {result['total_baseline_emissions']:.0f} gCO2")
    print(f"Total shifted emissions:  {result['total_shifted_emissions']:.0f} gCO2")
    print(f"Relative reduction:       {result['relative_reduction']*100:.2f}%")

# Example usage:
plot_scenario(scenario_low_hist,   title="Strategy: Low-intensity hours (historical)")
plot_scenario(scenario_renew_hist, title="Strategy: Max renewables hours (historical)")


# # Reduction Over Many Days

# In[16]:


def daily_reduction_for_range(
    start_date: str,
    end_date: str,
    strategy: str,
    flexible_share: float = 0.3,
    daily_kwh: float = 14.0,
):
    reductions = []
    dates = pd.date_range(start_date, end_date, freq="D")

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        try:
            df_day = get_day_slice(df_carbon, date_str)
            ci_day = df_preds.loc[date_str, "CI_actual"]
            renew_share = compute_renewable_share(df_day)

            scenario = run_shift_scenario(
                ci_series=ci_day,
                daily_kwh=daily_kwh,
                flexible_share=flexible_share,
                strategy=strategy,
                renewable_share=renew_share,
            )
            reductions.append(scenario["relative_reduction"])
        except Exception:
            # skip incomplete days
            continue

    return dates[:len(reductions)], np.array(reductions)


# In[17]:


dates_low, red_low = daily_reduction_for_range(
    "2024-01-01", "2024-12-31", strategy="low_intensity", flexible_share=0.3
)

dates_ren, red_ren = daily_reduction_for_range(
    "2024-01-01", "2024-12-31", strategy="max_renewable", flexible_share=0.3
)

print("Low-intensity: mean reduction %:", red_low.mean() * 100)
print("Max-renewable: mean reduction %:", red_ren.mean() * 100)
print("Best day (low-intensity):", red_low.max() * 100)
print("Best day (max-renewable):", red_ren.max() * 100)


# # Daily Reductions over a Date Range

# In[18]:


def compute_daily_reductions_over_range(
    start_date,
    end_date,
    flexible_share=0.3,
    daily_kwh=14.0,
    ev_kwh=0.0,
    strategy_list=("low_intensity", "max_renewable"),
):
    """
    Loop over a date range and compute daily relative reductions for each strategy.

    Uses the same logic as daily_reduction_for_range:
      - CI from df_preds["CI_actual"]
      - RENEWABLE / GENERATION from df_carbon
      - skips incomplete days

    Returns a DataFrame indexed by date with columns:
      - 'strategy'
      - 'reduction' (fraction)
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    records = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        try:
            # Same as your original function:
            df_day = get_day_slice(df_carbon, date_str)
            ci_day = df_preds.loc[date_str, "CI_actual"]
            renew_share = compute_renewable_share(df_day)

            for strategy in strategy_list:
                res = run_shift_scenario(
                    ci_series=ci_day,
                    renewable_share=renew_share,
                    daily_kwh=daily_kwh,
                    flexible_share=flexible_share,
                    strategy=strategy,
                    ev_kwh=ev_kwh,   # EV OFF if 0.0, ON if >0
                )

                records.append({
                    "date": d,
                    "strategy": strategy,
                    "reduction": res["relative_reduction"],
                })
        except Exception:
            # Skip dates where either CI or generation mix is incomplete
            continue

    return pd.DataFrame(records).set_index("date")


# In[19]:


# Example for 2024
df_red_2024 = compute_daily_reductions_over_range(
    "2024-01-01", "2024-12-31",
    flexible_share=0.3,
    daily_kwh=14.0,
)


# In[20]:


def plot_yearly_reduction(df_red: pd.DataFrame, title: str = ""):
    # Pivot to have one column per strategy
    df_pivot = df_red.pivot(columns="strategy", values="reduction")

    plt.figure(figsize=(14, 6))
    plt.plot(df_pivot.index, df_pivot["low_intensity"] * 100, label="low_intensity", linestyle="-")
    plt.plot(df_pivot.index, df_pivot["max_renewable"] * 100, label="max_renewable", linestyle="--")


    plt.ylabel("Daily CO$₂$ reduction (%)")
    plt.xlabel("Date")
    plt.title(title or "Daily emission reduction over the year")
    plt.legend(title="Strategy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../assets/daily_reduction_2024.png", dpi=150)
    plt.show()

# Example:
plot_yearly_reduction(df_red_2024, title="Daily CO$₂$ reduction – Household Profile (2024)")


# In[21]:


df_pivot = df_red_2024.pivot(columns="strategy", values="reduction")

df_pivot["diff"] = df_pivot["low_intensity"] - df_pivot["max_renewable"]
print("Max abs diff:", df_pivot["diff"].abs().max())
print(df_pivot.head(12))


# This shows how savings fluctuate through the year and whether there’s any seasonal shape.

# In[22]:


df_red_ev_2024 = df_red_2024.copy()  # if you overwrote df_red_2024 for the EV run

# Prepare EV daily reductions for plotting
df_ev_plot = df_red_ev_2024.reset_index().rename(columns={"index": "date"})
df_ev_plot["reduction_pct"] = df_ev_plot["reduction"] * 100

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_ev_plot,
    x="date",
    y="reduction_pct",
    hue="strategy",
    style="strategy",
    linewidth=2,
)
plt.ylabel("Daily CO$₂$ reduction (%)")
plt.xlabel("")
plt.title("Daily CO$₂$ Reduction – EV Household (2024)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("../assets/daily_reduction_EV_2024.png", dpi=150)
plt.show()


# # Histogram of Daily Reductions for each Strategy

# In[23]:


def plot_reduction_histogram(df_red: pd.DataFrame, bins: int = 20, title: str = ""):
    df_pivot = df_red.pivot(columns="strategy", values="reduction")

    low_vals = (df_pivot["low_intensity"].dropna() * 100).values
    ren_vals = (df_pivot["max_renewable"].dropna() * 100).values

    # Shared bin edges so shapes align perfectly
    min_val = min(low_vals.min(), ren_vals.min())
    max_val = max(low_vals.max(), ren_vals.max())
    bins_edges = np.linspace(min_val, max_val, bins + 1)

    plt.figure(figsize=(10, 5))

    # Step histogram (outline only)
    plt.hist(
        low_vals, bins=bins_edges, histtype="step",
        linewidth=2, linestyle="-", color="blue",
        label="low_intensity"
    )

    plt.hist(
        ren_vals, bins=bins_edges, histtype="step",
        linewidth=2, linestyle="--", color="orange",
        label="max_renewable"
    )
    plt.xlabel("Daily CO$₂$ reduction (%)")
    plt.ylabel("Number of days")
    plt.title(title or "Distribution of Daily CO$₂$ Reductions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../assets/distribution_of_daily_CO2_2024.png", dpi=150)
    plt.show()

# Example:
plot_reduction_histogram(df_red_2024, bins=30, title="Distribution of daily CO$₂$ reductions – 2024")


# # Summary Table

# In[24]:


# 2. Summarise reductions (mean, std, min, max) in %
def summarize_reductions(df_red: pd.DataFrame) -> pd.DataFrame:
    df_pivot = df_red.pivot(columns="strategy", values="reduction")

    summary_rows = []
    for strategy in df_pivot.columns:
        vals = df_pivot[strategy].dropna() * 100  # convert to %
        summary_rows.append(
            {
                "strategy": strategy,
                "mean_%": vals.mean(),
                "std_%": vals.std(),
                "min_%": vals.min(),
                "max_%": vals.max(),
                "n_days": len(vals),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    return summary


# 3. Style the summary table (for Jupyter display)
def style_reduction_summary(summary_df: pd.DataFrame):
    styled = (
        summary_df.style
        .format(
            {
                "mean_%": "{:.2f}",
                "std_%": "{:.2f}",
                "min_%": "{:.2f}",
                "max_%": "{:.2f}",
                "n_days": "{:.0f}",
            }
        )
        .background_gradient(
            subset=["mean_%", "max_%"],
            cmap="Greens",
        )
        .set_caption("Summary of daily CO₂ reduction by strategy (2024)")
    )
    return styled


# In[25]:


df_red_std_2024 = compute_daily_reductions_over_range(
    "2024-01-01",
    "2024-12-31",
    flexible_share=0.3,
    daily_kwh=14.0,
    ev_kwh=0.0,
)

summary_std_2024 = summarize_reductions(df_red_std_2024)
summary_std_2024


# In[26]:


# EV Scenario
df_red_ev_2024 = compute_daily_reductions_over_range(
    "2024-01-01",
    "2024-12-31",
    flexible_share=0.3,
    daily_kwh=14.0,
    ev_kwh=7.0,
)

summary_ev_2024 = summarize_reductions(df_red_ev_2024)
summary_ev_2024


# # Daily CO₂ Reductions (Standard Household, Low-Intensity Strategy)

# In[27]:


# Figure 2: Daily CO₂ Reductions – Standard Household (Low-Intensity Strategy)
std_low = (
    df_red_std_2024[df_red_std_2024["strategy"] == "low_intensity"]
    .copy()
)
std_low["reduction_pct"] = std_low["reduction"] * 100

mean_red = std_low["reduction_pct"].mean()

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    std_low.index,
    std_low["reduction_pct"],
    color=COLORS["blue"],
    linewidth=1.5,
)

ax.axhline(
    mean_red,
    color=COLORS["grey"],
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {mean_red:.1f}%",
)

ax.set_title("Daily CO$₂$ Reductions – Standard Household (2024, Low-Intensity Strategy)")
ax.set_ylabel("CO$₂$ reduction (%) vs baseline")
ax.set_xlabel("Date")

ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Caption:
# "Daily percentage CO₂ reduction for a 14 kWh/day high-usage household in 2024
#  when 30% of load is shifted into the lowest carbon-intensity hours.
#  Average daily savings are around 10–11%, with best days above 20%."

# Alt text:
# "Time-series line plot showing daily CO₂ reduction percentages for a standard household,
#  with most values around 10 percent and occasional peaks above 20 percent.


# # Daily CO₂ Reductions (EV Household, Low-Intensity Strategy)

# In[28]:


# Figure 3: Daily CO₂ Reductions – EV Household (Low-Intensity Strategy)

ev_low = (
    df_red_ev_2024[df_red_ev_2024["strategy"] == "low_intensity"]
    .copy()
)
ev_low["reduction_pct"] = ev_low["reduction"] * 100

mean_ev = ev_low["reduction_pct"].mean()

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    ev_low.index,
    ev_low["reduction_pct"],
    color=COLORS["green"],
    linewidth=1.5,
)

ax.axhline(
    mean_ev,
    color=COLORS["grey"],
    linestyle="--",
    linewidth=1.5,
    label=f"Mean: {mean_ev:.1f}%",
)

ax.set_title("Daily CO$₂$ Reductions – EV Household (2024, Low-Intensity Strategy)")
ax.set_ylabel("CO$₂$ reduction (%) vs baseline")
ax.set_xlabel("Date")

ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Caption:
# "Daily CO₂ reductions for a household with 14 kWh/day base load and 7 kWh/day EV charging,
#  when flexible demand is shifted into the lowest carbon-intensity hours.
#  Typical savings are around 20%, with some days reaching reductions above 40%."

# Alt text:
# "Time-series line plot showing daily CO₂ reductions for an EV-owning household,
#  with average values near 20 percent and multiple peaks exceeding 40 percent."


# # Standard vs EV Overlay (Low-Intensity)

# In[29]:


# Figure 4: Standard vs EV – Daily CO₂ Reductions (Low-Intensity Strategy)

std_low = (
    df_red_std_2024[df_red_std_2024["strategy"] == "low_intensity"]
    .copy()
)
std_low["reduction_pct"] = std_low["reduction"] * 100

ev_low = (
    df_red_ev_2024[df_red_ev_2024["strategy"] == "low_intensity"]
    .copy()
)
ev_low["reduction_pct"] = ev_low["reduction"] * 100

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    std_low.index,
    std_low["reduction_pct"],
    color=COLORS["blue"],
    linewidth=1.3,
    label="Standard household",
)

ax.plot(
    ev_low.index,
    ev_low["reduction_pct"],
    color=COLORS["green"],
    linewidth=1.3,
    label="EV household",
    alpha=0.9,
)

ax.set_title("Daily CO$₂$ Reductions – Standard vs EV Household (2024, Low-Intensity)")
ax.set_ylabel("CO$₂$ reduction (%) vs baseline")
ax.set_xlabel("Date")

ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Caption:
# "Comparison of daily CO₂ reduction for a standard high-usage household and an EV-owning
#  household under the same low-intensity shifting rule. EV charging significantly amplifies
#  the achievable reductions."

# Alt text:
# "Overlayed line chart comparing daily CO₂ reductions for a standard household and an EV household,
#  showing consistently higher and more variable reductions for the EV case."


# # Load Profile Before vs After Shifting (Example Day)

# In[30]:


# Figure 5: Load Profile – Baseline vs Shifted (Example Day, Standard Household, Low-Intensity)

example_date = "2024-03-05"  # adjust as needed
day_ts = pd.to_datetime(example_date).tz_localize("UTC")

# Get CI + mix for the day
try:
    ci_day_df = df_carbon.loc[example_date].copy()
except KeyError:
    raise ValueError("No data for this date in df_carbon. Change example_date.")

scenario = run_shift_scenario(
    date=day_ts,
    ci_day_df=ci_day_df,
    strategy="low_intensity",
    base_kwh=14.0,
    flexible_share=0.3,
    ev_kwh=0.0,  # standard household
)

baseline_load = scenario["baseline_load"]
shifted_load = scenario["shifted_load"]

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    baseline_load.index,
    baseline_load.values,
    label="Baseline load",
    color=COLORS["grey"],
    linewidth=2,
)

ax.plot(
    shifted_load.index,
    shifted_load.values,
    label="Shifted load (low-intensity)",
    color=COLORS["blue"],
    linewidth=2,
    linestyle="--",
)

ax.set_title(f"Household Load Profile – Baseline vs Shifted ({example_date})")
ax.set_ylabel("Load (kWh per hour)")
ax.set_xlabel("Time (UTC)")

ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Caption:
# "Example day load profile for a standard household, showing the original baseline demand
#  and the shifted profile after moving flexible load into the hours with the lowest carbon intensity."

# Alt text:
# "Line chart showing two hourly load curves for the same day, one representing the original consumption
#  and the other the adjusted profile after demand shifting."


# # Carbon Intensity vs Renewable Share (Why Max-Renewable Can Fail)

# In[31]:


# Figure 6: Carbon Intensity vs Renewable Share (Scatter, 2024)

# Restrict to one year (for example 2024)
mask_2024 = (df_carbon.index.year == 2024)
carbon_2024 = df_carbon.loc[mask_2024].copy()

# Compute renewable share if not already present as fraction
if "renewable_share" not in carbon_2024.columns:
    carbon_2024["renewable_share"] = (
        carbon_2024["RENEWABLE"] / carbon_2024["GENERATION"]
    )

fig, ax = plt.subplots(figsize=(12, 10))

ax.scatter(
    carbon_2024["renewable_share"] * 100,
    carbon_2024["CARBON_INTENSITY"],
    s=8,
    alpha=0.3,
    color=COLORS["green"],
    edgecolors="none",
)

ax.set_title("Carbon Intensity vs Renewable Share (2024)")
ax.set_xlabel("Renewable share (%)")
ax.set_ylabel("Carbon intensity (gCO$₂$/kWh)")

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Caption:
# "Relationship between hourly renewable generation share and carbon intensity in 2024.
#  The relationship is noisy and far from perfectly linear, which helps explain why
#  a max-renewable strategy does not always minimise emissions."

# Alt text:
# "Scatter plot of hourly renewable share versus carbon intensity, showing a broad cloud of points
#  with only a loose negative relationship between the two."

