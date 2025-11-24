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
