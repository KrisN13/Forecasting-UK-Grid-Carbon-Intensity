# UK Grid Carbon Intensity Forecasting & Demand Flexibility Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://forecasting-uk-grid-carbon-intensity.streamlit.app/)

## The Challenge: Decarbonising the UK Grid
As the UK progresses toward its **Net Zero 2050** goal, the carbon intensity of the electricity grid is becoming increasingly volatile. National Grid ESO estimates that demand-side flexibility could provide up to **13 GW of peak reduction by 2030**, significantly reducing the need for fossil-fuel "peaker" plants.

This project addresses a critical question for the energy transition: **How much CO₂ can a typical UK household actually save by shifting demand?** By combining machine learning forecasts with realistic load-shifting scenarios, we quantify the potential of consumer flexibility to accelerate grid decarbonisation.

## Key Findings
- **Standard Household:** Shifting 30% of daily load into the cleanest hours reduces daily carbon emissions by an average of **10.5%**.
- **EV Households:** Households with an electric vehicle (7 kWh/day charging) can achieve **20.1% average reductions**, with peaks exceeding **40%** on high-renewables days.
- **Strategy Matters:** Optimising directly for *Carbon Intensity* (gCO₂/kWh) is 15-20% more effective at reducing emissions than simply chasing *Renewable Share*, which can occasionally correlate with high-emissions balancing generation.
- **Uncertainty is Critical:** Using probabilistic forecasting (quantile regression), we provide a 80% confidence interval, showing that "green" windows are often stable, but volatility spikes during winter demand peaks.

## Market Context: From Theory to Reality
The "low-intensity" windows identified by our models align with real-world UK energy mechanisms:
- **Octopus Agile:** A dynamic-pricing tariff that incentivises shifting to low-carbon (and low-cost) periods.
- **Demand Flexibility Service (DFS):** The National Grid ESO mechanism that pays consumers to reduce usage during peak "stress events."
- **REMA (Review of Electricity Market Arrangements):** Current UK policy reform aimed at strengthening these signals through locational pricing and enhanced flexibility markets.

## Technical Architecture
This project is engineered as a modular Python package supported by a research pipeline.

- **Data Ingestion:** Automated pipelines fetching from National Grid ESO (Carbon Intensity API) and Open-Meteo (Weather Archive).
- **Forecasting Engine:** A **HistGradientBoostingRegressor** with **Quantile Loss** (q10, q50, q90) to quantify prediction uncertainty.
- **Scenario Engine:** Realistic load profiles derived from **BEIS Energy Trends** and **Elexon** settlement data.
- **Interactive Dashboard:** A Plotly-powered Streamlit app for real-time scenario simulation.

### Project Structure
```
├── data/               # Processed grid data & model predictions
├── notebooks/          # 01-06 Research & Development workflow
├── src/carbon/         # Modular core logic (models, features, scenarios)
├── streamlit_app.py    # Plotly-based interactive dashboard
└── requirements.txt    # Pinned dependencies for reproducibility
```

## Household Scenario Engine
Our simulations use defensible UK assumptions:
1. **Load Profile:** Average consumption of **8.5 kWh/day** (Source: [DESNZ/BEIS Energy Trends](https://www.gov.uk/government/collections/energy-trends)).
2. **Flexibility:** **30% shiftable load** for standard homes (laundry, dishwashers) and **100% flexibility** for EV charging (7 kWh/day).
3. **Probabilistic Guidance:** Forecasts provide q10/q90 intervals, helping users identify not just "green" hours, but *reliably* green hours.

## Limitations & Real-World Constraints
While the technical potential is high, deployment faces several hurdles:
- **Consumer Behaviour:** Our model assumes "perfect compliance." In reality, "fatigue" and "rebound effects" (shifting too much load into a single hour) can create new local grid constraints.
- **Grid Constraints:** National carbon signals don't account for local distribution network (DNO) bottlenecks.
- **Thermal Comfort:** Demand shifting for space heating is limited by a home's building fabric and thermal retention.

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/KrisN13/Forecasting-UK-Grid-Carbon-Intensity.git
cd Forecasting-UK-Grid-Carbon-Intensity
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the Dashboard**
```bash
streamlit run streamlit_app.py
```

## License
This project is licensed under the MIT License.