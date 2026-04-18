# Blend Café — Dynamic Pricing Recommendation Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Kali Linux](https://img.shields.io/badge/Kali_Linux-557C94?style=for-the-badge&logo=kali-linux&logoColor=white)](https://kali.org)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rushi-sagar-873b4b251/)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Rushi_Sagar221)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/Rushisagar221)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://reddit.com/user/wolf_sage_221)

</div>

---

> **Data Science Internship** | Blend Café, Deccan, Pune | July 1 – August 15, 2025
>
> **Headline result:** `+4.64%` projected revenue uplift on the test period (Jul 16–31),
> strongest condition: **Dinner × Clear After Rain (+10.9% per item)**

---

## What This Project Does

Blend Café (Deccan, Pune — adjacent to Fergusson College and BMCC) uses entirely
static pricing. Every item has one fixed price regardless of time of day, weather,
or demand pattern. This project builds a three-phase dynamic pricing recommendation
engine that analyses 3 months of transaction data and recommends item-level prices
per time slot, day type, and weather state.

**Key engineering constraint:** The café owner needed an explainable system.
A neural network outputting a price is not explainable. A majority vote of three
interpretable signals is.

---

## Quick Numbers

| Metric | Value |
|---|---|
| Revenue uplift (test period) | **+4.64%** (₹8,793 on ₹1,89,322) |
| Test period | Jul 16–31, 2024 — 671 transactions |
| Best single condition | Dinner × Clear After Rain **+10.9%** |
| Best slot | Dinner **+7.35%** |
| Best weather state | Clear After Rain **+7.60%** |
| Price recommendations generated | **6,400** (200 items × 32 conditions) |
| Floor violations | **0** |
| Premium items unflagged | **0** |
| HOLD rate | **84.7%** — intentional, brand-protective |
| LR test RMSE | 0.3972 |
| RF test RMSE | 0.4024 |

---

## Project Structure

```
blend_cafe_dp/
│
├── README.md                          ← You are here
│
├── data/
│   ├── obs/                           ← Raw input — DO NOT modify
│   │   └── BlendCafe_DynamicPricing_Data.xlsx
│   ├── data_analysis/                 ← Phase 1+2 outputs (auto-generated)
│   │   ├── charts/                    ← 18 visualisation charts
│   │   ├── abc_classified.csv
│   │   ├── correlation_matrix.csv
│   │   ├── elasticity_coefficients.csv
│   │   └── ...
│   ├── models/                        ← Phase 3 outputs (auto-generated)
│   │   ├── price_recommendations.csv  ← 6,400 recommendations
│   │   ├── lr_demand_model.pkl
│   │   ├── rf_demand_model.pkl
│   │   └── ...
│   └── reports/                       ← Final deliverables
│       ├── revenue_uplift_analysis.csv
│       └── project_evaluation_summary.csv
│
├── scripts/
│   ├── data_analysis/                 ← Phase 1+2 — run in order
│   │   ├── 01_data_loader.py
│   │   ├── 02_abc_analysis.py
│   │   ├── 03_pivot_heatmap.py
│   │   ├── 04_correlation_analysis.py
│   │   ├── 05_price_elasticity.py
│   │   ├── 06_pareto_analysis.py
│   │   ├── 07_time_series.py
│   │   └── 08_demand_segmentation.py
│   │
│   └── models/                        ← Phase 3 — run after data_analysis/
│       ├── 01_feature_engineering.py
│       ├── 02_demand_forecast.py
│       ├── 03_price_recommender.py
│       ├── 04_revenue_comparison.py
│       └── 05_model_evaluation.py
│
└── spec/
    ├── interview_project.yaml         ← Master technical specification
    ├── companion.md                   ← Interview narrative + failure stories
    ├── BLEND CAFE MENU.pdf            ← Original café menu
    └── requirements.txt
```

---

## Environment Setup

```bash
# Activate Python 3.11 environment
source ~/myenv311/bin/activate

# Install dependencies
pip install -r spec/requirements.txt
```

---

## How to Run

Run all scripts from the **project root** in order:

```bash
# Phase 1 + 2 — Data Analysis (8 scripts)
python scripts/data_analysis/01_data_loader.py
python scripts/data_analysis/02_abc_analysis.py
python scripts/data_analysis/03_pivot_heatmap.py
python scripts/data_analysis/04_correlation_analysis.py
python scripts/data_analysis/05_price_elasticity.py
python scripts/data_analysis/06_pareto_analysis.py
python scripts/data_analysis/07_time_series.py
python scripts/data_analysis/08_demand_segmentation.py

# Phase 3 — Machine Learning (5 scripts)
python scripts/models/01_feature_engineering.py
python scripts/models/02_demand_forecast.py
python scripts/models/03_price_recommender.py
python scripts/models/04_revenue_comparison.py
python scripts/models/05_model_evaluation.py
```

Each script prints a completion summary. All scripts are fully re-runnable —
outputs are overwritten cleanly on each run.

---

## Three Phases

### Phase 1 — Excel Analysis (Weeks 1–2)

Manual Excel work on `Raw_Transactions`. Pivot tables, ABC classification,
revenue heatmaps, AOV analysis. Python scripts replicate and extend this
in `scripts/data_analysis/01–03`.

**Key finding:** 95 items (47.5% of menu) drive 69.6% of revenue.
The classic Pareto 80/20 does not hold — Blend Café's revenue is broadly
distributed. This shaped the decision to build models for the full 200-item
menu rather than a short priority list.

---

### Phase 2 — Statistical Analysis (Weeks 3–4)

Correlation analysis, price elasticity, Pareto validation, time series
decomposition, demand segmentation.

**Key finding:** Weather is the strongest demand signal.
Cold Brew vs Temperature: `r = +0.503`. Footfall vs Revenue: `r = +0.906`.
38 items show a **rain-boost effect** — demand increases on Heavy Rain days
(Masala Chai, soups, hot chocolate). Time series decomposition revealed the
June monsoon dip and July college-return recovery cleanly in the trend component.

---

### Phase 3 — Machine Learning (Weeks 5–6)

25 engineered features, Linear Regression baseline, Random Forest primary model,
three-signal price recommender, revenue uplift comparison.

**Key finding and honest failure:**
Transaction-level quantity (`Quantity_Units`) had insufficient variance for
meaningful ML forecasting — 80% ones, 20% twos, std ≈ 0.40.
Both LR and RF converged to predict the mean (RMSE ≈ 0.40, R² ≈ 0).

> Predicting the mean of 1.2 every time gives RMSE of 0.40.
> That is almost exactly what both models produced.
> There was no signal to learn.

**Redesign decision:** The price recommender was rebuilt using three direct
demand signals — footfall vs baseline, weather state, and slot-day revenue
percentile — with a majority-vote verdict per recommendation.
More explainable, fully traceable, and does not depend on a near-constant target.

---

## Key Design Decisions

**Classical ML only — no deep learning**
3,818 rows, binary-range target, non-technical audience. Random Forest with
feature importances produces a chart explainable to a café owner.
PyTorch would have overfitted and added zero interpretability.

**Chronological train/test split**
Train: May 1 – July 15. Test: July 16–31.
Random split explicitly rejected — a pricing model must generalise to future
dates, not random past dates.

**85% price floor — data-driven, not gut-feel**
Derived from Phase 2 customer segment analysis: Premium items are bought by
customers for whom the price is part of the signal. Discounting below 85%
destroys the brand signal that drives full-price sales on high-value evenings.
Zero floor violations in 6,400 recommendations.

**84.7% HOLD rate — intentional**
Dynamic pricing does not mean constant change. The system only changes prices
when two of three signals strongly agree. 84.7% HOLD means the café charges
its normal price most of the time. This was a design choice, not a limitation.

---

## Data

All transaction data is **simulated**. Blend Café does not have a digital POS
system. The simulation covers May–July 2024 (Pune summer through full monsoon)
using historically grounded assumptions for Pune weather, Deccan-area footfall,
and Blend Café's own menu timing windows.

| Stat | Value |
|---|---|
| Simulation period | May 1 – July 31, 2024 (92 days) |
| Total transactions | 3,818 rows |
| Menu items | 200 across 24 categories |
| Total simulated revenue | ₹10,40,338 |
| Simulation seed | 42 (fully reproducible) |

The raw Excel has four sheets: `Raw_Transactions`, `Daily_Summary`,
`Item_Master`, and `README`. Only `data/obs/` is read-only ground truth.

---

## Key Outputs

| File | Location | Description |
|---|---|---|
| [`price_recommendations.csv`](data/models/price_recommendations.csv) | `data/models/` | 6,400 recommended prices |
| [`revenue_uplift_analysis.csv`](data/reports/revenue_uplift_analysis.csv) | `data/reports/` | Static vs dynamic comparison |
| [`project_evaluation_summary.csv`](data/reports/project_evaluation_summary.csv) | `data/reports/` | All headline metrics |
| [`project_summary.png`](data/data_analysis/charts/project_summary.png) | `data/data_analysis/charts/` | Four-panel project overview |
| [`demand_segmentation_2x2.png`](data/data_analysis/charts/demand_segmentation_2x2.png) | `data/data_analysis/charts/` | Item pricing segments |
| [`revenue_uplift_waterfall.png`](data/data_analysis/charts/revenue_uplift_waterfall.png) | `data/data_analysis/charts/` | Revenue impact waterfall chart |

---

## Specs and Documentation

| Document | Description |
|---|---|
| [`spec/interview_project.yaml`](spec/interview_project.yaml) | Master technical specification — all decisions, thresholds, and rationale |
| [`spec/companion.md`](spec/companion.md) | Interview narrative, failure stories, Q&A reference |

---

## About Me

Final-year CS student in Pune building toward ML engineering roles at YC-backed
startups. My work sits at the intersection of **deep learning**, **reinforcement
learning**, and **RAG/LLM systems** — all applied to financial markets.

**Other projects:**
- 🤗 [`Rushisagar221/dalal-street-financial-llm`](https://huggingface.co/Rushisagar221/dalal-street-financial-llm) — Fine-tuned Llama-3.2-3B for Indian equity analysis. Citation rate 0% → 100%.
- **Crypto Phase 2** — Regime detection system (LSTM + GNN + PPO meta-model), live paper trading on Binance
- **Poker PPO Bot** — Deployed RL agent with FastAPI backend + React frontend

<div align="center">

[![LinkedIn](https://img.shields.io/badge/Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rushi-sagar-873b4b251/)
[![X](https://img.shields.io/badge/Follow-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Rushi_Sagar221)
[![HuggingFace](https://img.shields.io/badge/🤗_Models-FFD21E?style=for-the-badge)](https://huggingface.co/Rushisagar221)
[![Reddit](https://img.shields.io/badge/wolf__sage__221-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://reddit.com/user/wolf_sage_221)

</div>

---

*Blend Café Dynamic Pricing Engine — Data Science Internship, July–August 2025, Pune*
