# Project Review: Forecasting UK Grid Carbon Intensity

**Repository**: https://github.com/KrisN13/Forecasting-UK-Grid-Carbon-Intensity
**Reviewed**: 2026-03-13

**Brief Summary**: This project forecasts near-term UK electricity grid carbon intensity using a HistGradientBoostingRegressor with lagged features, generation mix data, and temperature, then simulates the CO2 reduction potential of household demand shifting and EV charging strategies. It includes a packaged Python codebase, numbered Jupyter notebooks, and a deployed Streamlit application.

---

## Scores

| Dimension | Score |
|---|---|
| Technical Skills | 7/10 |
| Presentation | 6/10 |
| Employability Impact | 6/10 |

---

## Technical Skills — 7/10

**What's working:**

- **Temporal split is done correctly.** Training on 2020–2023, validating on 2024, and holding out 2025 as a test set is exactly right for time-series forecasting. There is no data leakage from future observations. Many candidates get this wrong — this is the single most critical methodological requirement for time-series work.
- **Quantile regression is a genuine differentiator.** Training q10/q50/q90 models and evaluating interval coverage (76.8% achieved vs. 80% target) demonstrates real probabilistic forecasting literacy.
- **The `src/carbon/` package structure is well-engineered.** Separating `models.py`, `features.py`, `scenarios.py`, `config.py`, and `pipeline.py` into a proper Python package with type hints and docstrings shows software engineering maturity above a typical Jupyter-only submission.

**Weaknesses:**

- **No documented hyperparameter tuning.** `config.py` hardcodes `max_depth=8` and `learning_rate=0.05` with no explanation of how these were chosen. Without this, a hiring manager cannot distinguish between values arrived at through rigorous optimisation and values that were guessed and left in.
- **Dependencies are unpinned.** `requirements.txt` has zero version numbers — the project is not reproducible. A breaking change in scikit-learn or Streamlit will silently break the pipeline.
- **The 76.8% interval coverage is undercovering** (target: 80%) and is barely mentioned. Is the interval too narrow because the model underestimates uncertainty during high-volatility periods (e.g., winter demand spikes)? That analysis is absent.

**To reach a 10:**
- Add a hyperparameter tuning notebook or section (even a lightweight Optuna study) with documented results.
- Pin all dependencies with exact versions.
- Investigate and explain the interval undercoverage.
- Add MAPE alongside MAE/RMSE — it is more interpretable to non-technical stakeholders.
- Include a brief residual analysis: do errors cluster at high-carbon periods, seasonal extremes, or specific fuel mix states?

---

## Presentation — 6/10

**What's working:**

- The numbered notebook structure (01–06) is the right approach.
- The `src/` package as "a thin orchestration layer" over notebooks is architecturally sound.
- The Streamlit deployment demonstrates the work functions end-to-end.

**Weaknesses:**

- **The README lacks a narrative arc.** It reads as a technical specification. It does not ask: why does this problem matter? What does 10.48% CO2 reduction mean in real terms — for a household, for the UK grid, for a net-zero pathway? Numbers without context do not land.
- **The Streamlit app uses matplotlib**, which looks dated inside Streamlit. More critically, it does not expose the q10/q90 prediction intervals that the models actually produce. Uncertainty quantification is built into the backend and invisible to the user — a significant missed opportunity.
- **No feature importance visualisation is prominently displayed.** The `feature_importance()` function exists and uses permutation importance (the right choice), but results aren't shown in the README or app.
- **Notebook markdown commentary is thin.** A hiring manager reading the notebooks should be able to follow the reasoning without reading the code.

**To reach a 10:**
- Rewrite the README opening as a problem statement with real-world stakes (reference National Grid ESO's flexibility requirements or BEIS household energy statistics).
- Add a "Key Findings" section that translates the numbers into plain English.
- Upgrade the Streamlit app to Plotly charts with prediction interval bands shown.
- Add a feature importance visualisation to both the notebooks and the app.
- Ensure every notebook has substantive markdown commentary explaining the reasoning.

---

## Employability Impact (Climate Sector) — 6/10

**What's working:**

- **The topic is genuinely well-chosen.** Carbon intensity forecasting sits at the intersection of energy systems analysis, demand flexibility, and decarbonisation policy — all active hiring areas.
- **The Carbon Intensity API from National Grid ESO is a real, sector-recognised data source.** This is not a generic Kaggle dataset.
- **The EV load-shifting extension aligns with current grid flexibility policy priorities** (REMA, GB flexibility markets reform).

**Weaknesses:**

- **The household load profile lacks defensible assumptions.** The project uses 14 kWh/day — UK household average is approximately 7.5–10 kWh/day (BEIS Energy Trends). Using 30% flexible load without citing any demand response literature or Ofgem/BEIS flexibility trials is a red flag for a domain expert. Any interviewer from an energy consultancy will ask "where does that number come from?"
- **No connection to real UK energy market mechanisms.** There is no mention of Octopus Agile tariffs, the Demand Flexibility Service (DFS), or Dynamic Containment — the actual mechanisms through which demand shifting is incentivised. The analysis sits in a slightly abstract space.
- **No limitations section.** Real-world demand shifting faces constraints not modelled here: grid constraint costs, consumer behaviour, the rebound effect, thermal comfort limits. Not acknowledging these signals the candidate has not fully considered the gap between model assumptions and deployment reality — a standard expectation in climate consultancy outputs.

**To reach a 10:**
- Correct the household consumption figure to a defensible UK average and cite BEIS or Elexon.
- Add a section to the README explicitly connecting findings to a real-world mechanism (DFS, Agile tariffs, or smart meter rollout).
- Add a limitations section showing awareness of demand elasticity, grid constraints, and modelling assumptions.
- Cross-reference the 10.48% CO2 reduction figure against UK climate commitments or National Grid ESO flexibility ambitions.
- Consider adding a cost-benefit element: at a carbon price of £X/tonne, what is the annual monetary value of this shift to a household?

---

## Priority Action List

| Priority | Action | Effort |
|---|---|---|
| 1 | Correct the 14 kWh/day household energy assumption; cite BEIS or Elexon | ~30 min |
| 2 | Show q10/q90 prediction intervals in the Streamlit app; switch to Plotly | ~2–3 hrs |
| 3 | Rewrite README opening as a problem statement; connect results to DFS/Agile/REMA | ~1 hr |
| 4 | Pin all dependencies in `requirements.txt`; document hyperparameter selection | ~30 min |
| 5 | Add a limitations section (3 bullet points minimum) | ~20 min |

---

## Overall Verdict

This is a solid, technically competent project that is clearly above the median for early-career climate data analyst portfolios. The correct temporal split, quantile regression with interval coverage evaluation, and the packaged `src/` architecture all show genuine engineering and methodological care.

However, the project currently reads more like a well-structured ML exercise than a sector-aware analysis. The household assumptions lack defensible citations, there is no connection to real UK energy market mechanisms, and the presentation layer does not fully exploit the technical depth that is already built.

An interviewer at a climate consultancy, DSO, or flexibility aggregator would be broadly impressed but would push hard on the domain assumptions — and leave unsure whether you understand the policy and market context your technical work sits in.

**Fix the household energy figure, surface the prediction intervals in the app, and add one paragraph of market context to the README. Those three changes move this from a strong student project to a credible professional portfolio piece.**
