# Forecasting-UK-Grid-Carbon-Intensity
Can we predict near-term UK carbon intensity and show people that when shifting their electricity usage, it could yield a bigger CO₂ reduction?

### Project motivation ###
#### Why Carbon Intensity ####
I have long been interested in how carbon emmissions plays a role in global warming. This then led me to start questioning my own carbon footprint, the biggest of all being my house.
An average household uses between 10 - 12 kWh/per day, more in the case of a family of 4 that also includes EV charging.
Although one household shifting their carbon load to more 'off-peak- hours will have little to no difference on the energy grid, a combined effort would be substantial. The aims of this project is to prove how we can all save on energy, as well as shifting our carbon loads to more tailored hours of the day, when the grid is much less in demand. However, this is not as simple as 'just set your washing machine to 3am start'. The predictions are both more subtle, and customisable using the app, to create a realistic expectation.

#### Data sources: ####
Historic generation mix and carbon intensity
Temperature Observations

Pipeline:
1. Pull raw data (carbon intensity, generation mix, temperature)
2. Clean and standardise data, including timestamps
3. Merge into unified hourly dataset
4. Feature engineering (lags, rolling windows, weather features)
5. Train and validate forecasting models
6. Generate scenario-based CO₂ calculations

### Workflow ###

        ┌──────────────────┐
        │  Raw data (CI,   │
        │  generation mix) │
        └─────────┬────────┘
                  │
     ┌────────────▼─────────────┐
     │   Data cleaning & merge   │
     │ (timestamps, resampling)  │
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │  Exploratory Data Analysis│
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │  Baseline models          │
     │  (t-1, t-24, t-168)       │
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │  HGB forecasting model    │
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │  Household scenario engine│
     └────────────┬─────────────┘
                  │
     ┌────────────▼─────────────┐
     │  Streamlit app            │
     └───────────────────────────┘

## Discussion and Limitations
### Structural insight: decarbonisation compresses the benefit of load shifting

The analysis shows that domestic load shifting delivers modest but consistent
CO₂ savings: ≈**1.6%** on an average day, and up to **5%** on the best days.
The key reason is structural rather than methodological:

- In the modern UK grid (2020–2025), **renewables dominate low-carbon periods**  
- As a result, the hours with the highest renewable share are essentially the same
  hours with the lowest carbon intensity  
- This makes the **low_intensity** and **max_renewable** strategies almost indistinguishable

In earlier, coal- and gas-heavy years, the spread between high- and low-carbon hours
was larger, and the theoretical benefit of shifting would have been higher. In that
sense, the relatively small gains observed here are a symptom of progress:
a cleaner, more stable grid leaves less “low-hanging fruit” for demand shifting.

### When shifting can make things worse

The presence of negative daily reductions (down to –2.5%) is not a modelling artefact.
On some days:

- The carbon intensity curve is flat, or  
- The household load profile is out of phase with clean periods  

Under those conditions, naively shifting a fraction of demand into nominally
“better” hours can increase total emissions.

This suggests that load shifting **should not** be applied as a static rule
(e.g., “always run appliances at night”), but instead be driven by **dynamic,
day-ahead signals** based on actual forecasted intensity.

**Key Results**
### **Forecasting**

- Naive baseline (t-1): ~8.7 MAE  
- Daily baseline (t-24): ~38 MAE  
- Weekly baseline (t-168): ~56 MAE  
- HGBRegressor: 5.9 MAE  

![Actual vs Predicted Carbon Intensity](assets/ci_actual_vs_pred_feb2024.png)
Two-week slice showing the model’s ability to follow real carbon intensity patterns.

> The machine-learning model outperforms all baselines meaningfully, reducing error by **~32% vs naive**.

### **Scenario Analysis (Household CO₂ Impact)**

Using:

- **14 kWh/day** high-usage household  
- **30% flexible load** (≈4.2 kWh)  
- Shifting into **4 cleanest hours** of each day  
- Year: **2024**  

![Daily CO₂ Reduction Throughout 2024](assets/daily_reduction_2024.png)
Daily CO₂ reduction under household load-shifting (2024). Most days show 1–2% savings, with best days reaching ~5% and occasional negative days when shifting increases emissions.

Results:

| Metric                  | Value |
|-------------------------|-------|
| **Mean daily reduction** | **1.65%** |
| **Max daily reduction**  | **4.93%** |
| **Min daily reduction**  | **–2.51%** |
| **Strategies identical** | Yes |

### Why are both strategies identical?

- The hours with the **lowest carbon intensity** are the same hours with the **highest renewable penetration**.
- In a decarbonising grid, renewables shape the cleanest periods.
- Therefore:  
  **“Shift to renewable hours” = “Shift to low-carbon hours”**

### Why are reductions modest?

Because the modern UK grid (2020–2025) has:

- flatter carbon intensity curves  
- high renewable penetration  
- fewer deep fossil-fuel peaks  

Small gradients → small opportunity for shifting.

> Even so, a 1–2% reduction **per household** scales to hundreds of thousands of tonnes of CO₂ annually at the national level.

![Daily CO₂ Reduction Throughout 2024](assets/distribution_of_daily_CO2_2024.png)
Distribution of daily CO₂ reductions. The distribution centers near 1.6%, with a right tail up towards 5% and a left tail down to -2.5%, demonstrating days when shifting increases emissions.

**Discussion & Limitations**
### Structural insight:  
**Decarbonisation compresses the benefit of load shifting.**

- Renewables dominate the cleanest periods  
- The two strategies collapse into the same behaviour  
- Grid is smoother → fewer “dirty peaks” → less advantage from timing shifts  

### Negative savings

Some days see **higher emissions** after shifting:

- Flat carbon intensity profiles  
- Household load out of phase with clean hours  

Dynamic signals are required — not static “off-peak” rules.

### Scope limitations

- Single household profile (14 kWh/day)  
- 30% flexible load  
- No temperature or weather exogenous variables  
- No behavioural rebound  
- No locational marginal emissions  
- Evaluated nationally, not regionally  

Despite this, the model aligns with power-system behaviour and gives a realistic upper bound on household load-shifting impact.

**Running the Streamlit App**
### Install dependencies
### Run the app

![Try the app](assets/streamlit_app_overview.png)
Interactive Streamlit application for exploring UK grid carbon intensity forecasts and household CO₂ reduction scenarios. Users can adjust dates, flexible load share, and demand settings to generate personalised results. The app opens automatically in your browser.

It allows users to:

- select a date (actual or model-predicted carbon intensity)  
- adjust daily consumption  
- adjust flexible share  
- choose shift strategy  
- view load, emissions, and CO₂ reduction  

**Future Enhancements**

- Temperature + weather regressors (improve forecasts)  
- LightGBM model for comparison  
- SHAP explainability  
- 48-hour forecasting  
- EV charging + heat-pump household profiles  
- Dynamic demand-flexibility simulation  
- Deployment to Streamlit Cloud  

**License**

MIT License  
Free for use and adaptation.