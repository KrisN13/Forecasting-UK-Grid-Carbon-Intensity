# Forecasting UK Grid Carbon Intensity  
And Quantifying the CO₂ Impact of Household Load Shifting

This project builds an end-to-end system to:
1. Forecast near-term UK grid carbon intensity.
2. Simulate realistic household electricity demand.
3. Quantify the CO₂ impact of shifting flexible load into cleaner hours.
4. Compare strategies based on carbon intensity vs renewable share.
5. Extend the analysis to EV-owner households.
6. Serve results through an interactive Streamlit app.

The goal is to move beyond simple “off-peak is greener” rules and show, with real data, how much timed consumption actually matters.

## Motivation
The UK power system is decarbonising quickly, but carbon intensity still varies hour by hour depending on the generation mix (renewables, gas, imports, etc.).

Questions this project answers:
- How large are emission reductions from household load shifting in practice?
- How do different strategies perform (lowest carbon intensity vs highest renewable share)?
- How much larger is the effect for EV owners, where a large part of demand is genuinely flexible?
- Can a simple forecast plus a scenario engine provide meaningful guidance?

The project is written in Python, built around Jupyter notebooks, and published as a reproducible, portfolio-ready workflow.

## Data Sources
Main datasets:
- Historic UK carbon intensity and generation mix (hourly).
- Derived renewable share (RENEWABLE / GENERATION).
- Temperature observations (loaded for future extensions, not yet used in the main model).

Key preprocessing steps:
- Convert all timestamps to UTC.
- Resample to hourly resolution.
- Handle missing periods and incomplete days.
- Derive renewable share and other engineered features.

## Project Pipeline (High-Level)
Simple pipeline:
1. Load raw grid and generation-mix data.
2. Clean and standardise timestamps; resample to hourly.
3. Explore and visualise historical carbon intensity and generation trends.
4. Build and benchmark forecasting models (baselines vs ML).
5. Design household load profiles and flexible demand assumptions.
6. Implement scenario engine to shift flexible load under different strategies.
7. Compute daily CO₂ reductions over a full year.
8. Extend to EV-owner scenarios with EV charging load.
9. Wrap the core logic into a Streamlit app for interactive use.

In code, this is split across notebooks:
- `01_creating_a_clean_carbon_intensity_dataset`
- `02_data_exploration_carbon_intensity`
- `03_modeling_and_forecasting_ci`
- `04_scenarios_and_impact`
- `streamlit_app.py` (app wrapper)

## Forecasting Models
The forecasting stage uses:

- Baseline models:
  - Naive: previous hour (t-1).
  - Daily: same hour previous day (t-24).
  - Weekly: same hour previous week (t-168).
- Machine learning model:
  - HistGradientBoostingRegressor (HGB), using:
    - Lagged carbon intensity.
    - Rolling means.
    - Calendar features (hour, day of week, month).
    - Generation-mix features (e.g. SOLAR_lag1, WIND_lag1, RENEWABLE_lag1, etc.).

Example performance on held-out test data:

| Model                   | MAE (gCO₂/kWh) |
|-------------------------|----------------|
| Naive baseline (t-1)    |    8.7         |
| Daily baseline (t-24)   |     38         |
| Weekly baseline (t-168) |     56         |
| HistGradientBoostingRegressor |  5.9     |

The HGB model reduces error by roughly 30–35% relative to the naive baseline and captures the main structure of the carbon-intensity curve.

## Household Scenario Engine

The scenario engine is built on three main components:

1. A realistic household load profile:
   - High-usage household: 14 kWh/day.
   - 30% of demand treated as flexible (appliances such as laundry, dishwasher, some discretionary use).
   - Fixed shape over the day (night/morning/day/evening weights).

2. A strategy for choosing target hours:
   - `low_intensity`: choose hours with the lowest carbon intensity.
   - `max_renewable`: choose hours with highest renewable share.
   - Both strategies operate on day-level information.

3. A flexible-load redistribution step:
   - Compute total flexible energy for the day.
   - Redistribute that energy into the selected target hours.
   - Keep total daily energy constant.
   - Recompute CO₂ emissions before and after shifting.

The engine runs over a full year (2024), producing a daily series of relative CO₂ reductions for each strategy.

## Results: Standard Household (14 kWh/day, 30% Flexible Load)
For the standard household, no EV load is included (EV disabled).

Two strategies:

- `low_intensity`: shift flexible load into hours with lowest carbon intensity.
- `max_renewable`: shift flexible load into hours with highest renewable share.

Summary for 2024:

| Strategy        | Mean % | Std % | Min %   | Max %   | n_days |
|-----------------|--------|-------|---------|---------|--------|
| low_intensity   | 10.48  | 3.72  | 1.26    | 21.21   | 366    |
| max_renewable   | 8.94   | 4.49  | -2.75   | 21.11   | 366    |

Interpretation:

- Using forecasted carbon intensity directly (low_intensity) yields an average daily CO₂ reduction of around 10 - 11%.
- On the cleanest days, reductions exceed 20% while preserving total energy use.
- The max_renewable strategy is weaker on average and has a non-trivial number of days where emissions increase (down to about −2.8%).
- Renewable share is not a perfect proxy for carbon intensity: high renewable share can coincide with relatively high emissions when balancing or backup generation is active.

## Results: EV Household (14 kWh/day + 7 kWh EV Charging)

A second profile models a household with an EV:

- Same base 14 kWh/day household load.
- Additional 7 kWh/day EV charging, treated as fully flexible within an evening/night window.
- EV charging is allocated initially in the early evening and then shifted according to the strategy.

Summary for 2024:

| Strategy        | Mean % | Std % | Min %    | Max %   | n_days |
|-----------------|--------|-------|----------|---------|--------|
| low_intensity   | 20.05  | 9.00  | 1.51     | 46.08   | 366    |
| max_renewable   | 17.35  | 10.21 | -8.79    | 44.79   | 366    |

Interpretation:

- Adding EV charging roughly doubles the achievable emission reductions because a large block of demand is highly flexible.
- Under low_intensity, the EV household can reach daily reductions above 40%, with a mean of around 20%.
- max_renewable still performs reasonably on average but is riskier, with some days where emissions are significantly higher than baseline (around −9%).
- The results highlight the importance of optimising directly for carbon intensity, especially when large flexible loads like EV charging are involved.

## Why These Reductions Are Plausible
The UK grid between 2020 and 2025:

- Has relatively high renewable penetration.
- Exhibits reduced frequency of extreme fossil-heavy periods compared to earlier years.
- Shows carbon-intensity curves that are smoother and less extreme overall.

For a standard household, this compresses the potential gains from shifting: the difference between 'bad' and 'good' hours is real but not extreme. A 10 - 11% daily saving under an aggressive shifting strategy is consistent with this structure.

For an EV household, the situation is different: 7 kWh/day of EV charging is a large load that can be moved almost entirely into the cleanest hours, which is why daily reductions can exceed 40% on the best days.

## Streamlit App
The project includes a Streamlit app that exposes the core logic:
- Select date ranges.
- Visualise actual and predicted carbon intensity.
- Configure household daily consumption.
- Adjust flexible share.
- Toggle EV charging and set daily EV kWh.
- Choose strategy (`low_intensity` vs `max_renewable`).
- View baseline vs shifted load profiles.
- View CO₂ reductions over time.

The app is built from `streamlit_app.py` and uses the cleaned parquet datasets and precomputed predictions.

## Limitations
Current limitations include:

- Single national carbon-intensity signal (no locational marginal emissions).
- Only two household profiles:
  - High-usage household.
  - High-usage household with EV.
- Fixed flexible share (30% for the household, 100% for EV).
- No behavioural modelling (no rebound, comfort constraints, or non-compliance).
- No explicit price-based optimisation (carbon-only objective).
- Temperature features and weather-driven modelling are not yet integrated into the main forecasting model.

Despite these limitations, the results are internally consistent and align with the expected behaviour of a decarbonising grid with moderate intraday variation.

## Future Work
Potential next steps:
- Integrate temperature and weather features into the forecasting model.
- Add additional household profiles (for example, electric heating).
- Compare HistGradientBoostingRegressor with gradient boosting libraries such as LightGBM or XGBoost.
- Use explainability tools (e.g. SHAP) to understand feature contributions.
- Implement multi-day ahead forecasts with uncertainty intervals.
- Extend the scenario engine to:
  - Combine multiple households.
  - Simulate price-based behaviour.
  - Explore aggregated flexibility potential.

## How to Run
1. Clone the repository:
   git clone https://github.com/KrisN13/Forecasting-UK-Grid-Carbon-Intensity.git
   cd Forecasting-UK-Grid-Carbon-Intensity
2. Create and activate a Python environment, then install dependencies:
    pip install -r requirements.txt
3. Run notebooks in order for end-to-end analysis:
    01_creating_a_clean_carbon_intensity_dataset
    02_data_exploration_carbon_intensity
    03_modeling_and_forecasting_ci
    04_scenarios_and_impact
4. Launch the Streamlit app:
    streamlit run streamlit_app.py

## License
This project is released under the MIT License.