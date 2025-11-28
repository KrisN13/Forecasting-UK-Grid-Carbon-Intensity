# Forecasting-UK-Grid-Carbon-Intensity
Can we predict near-term UK carbon intensity and show people that when shifting their electricity usage, it could yield a bigger CO₂ reduction?

Data sources:
Historic generation mix and carbon intensity
Temperature Observations

Pipeline:
1. Pull raw data (carbon intensity, generation mix, temperature)
2. Clean and standardise data, including timestamps
3. Merge into unified hourly dataset
4. Feature engineering (lags, rolling windows, weather features)
5. Train and validate forecasting models
6. Generate scenario-based CO₂ calculations

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

### Scope and limitations

This analysis is explicitly scoped to:

- A single high-usage household profile (14 kWh/day)  
- 30% flexible demand, representing appliances such as laundry, dishwashers,
  and some discretionary usage  
- The 2024 UK grid, using historic carbon intensity and renewable mix

Key limitations:

- No modelling of intraday behaviour changes (behavioural rebound is ignored)  
- No explicit treatment of network constraints or locational marginal emissions  
- Flexible share is assumed homogeneous and perfectly controllable  
- Shifting is evaluated day-by-day, not as part of a portfolio of households or
  aggregated flexibility markets

Despite these simplifications, the results are consistent with the physical
behaviour of a decarbonising grid and provide a realistic upper-bound on the
benefit of naïve household load shifting at current UK emission levels.

### Running the Streamlit App

Install dependencies: