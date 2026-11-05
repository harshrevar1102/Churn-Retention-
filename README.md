# Customer Churn & Retention Strategy Analysis

**Stack:** Python · SQL (SQLite) · Pandas · scikit-learn · Streamlit · Plotly

An end-to-end analytics project that answers three business questions:

1. **Why** are customers leaving? (churn drivers, cohort analysis)
2. **Who** is at risk? (RFM segmentation + churn-probability model)
3. **What should we do?** (revenue at risk, retention-lever scenarios, prioritised recommendations)

Results are delivered as an executive-style **Streamlit dashboard**.

---

## 1. Dataset

**Recommended (real) dataset — Kaggle "E-Commerce Data" (Online Retail)**
<https://www.kaggle.com/datasets/carrie1/ecommerce-data> → file `data.csv`
~540k invoice lines from a UK online gift retailer, Dec 2010 – Dec 2011.

Why this dataset: it has invoice-level transactions with dates and customer IDs, which lets us do *real*
cohort analysis, RFM and a behavioural churn definition. (Popular "Telco churn" datasets lack transaction
history, so RFM/cohorts wouldn't be meaningful.)

| Column | Meaning |
|---|---|
| InvoiceNo | Invoice id (prefix `C` = cancellation/return) |
| StockCode / Description | Product |
| Quantity, UnitPrice | Units and price per unit |
| InvoiceDate | Timestamp |
| CustomerID | Customer (≈25% of rows are missing → dropped) |
| Country | Customer country |

### Where to put it
```
data/raw/data.csv      <-- download from Kaggle and place it here
```
Also auto-supported: *Online Retail II* (UCI) as `.csv`/`.xlsx` (columns are renamed automatically).

### No dataset? Synthetic fallback
If `data/raw/` contains no `.csv`/`.xlsx`, the pipeline generates a **synthetic dataset with the identical
schema** (`src/generate_synthetic.py` → `data/synthetic/`) so the whole project still runs. As soon as you
drop the real file into `data/raw/`, it is used instead. The dashboard header always shows which source was used.

> ⚠️ Numbers in this README's example section come from the **synthetic** data. Re-run on the real data and
> use *those* numbers on your resume / in interviews. Synthetic data is also "easier" — expect a lower
> ROC-AUC (~0.75–0.85) on the real dataset.

---

## 2. Quick start

```bash
python -m venv .venv && source .venv\Scripts\Activate.ps1      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional) put data.csv into data/raw/
python run_pipeline.py                                 # runs the full analysis (~10-60 s)
streamlit run app/streamlit_app.py                     # opens the dashboard
```
(If you skip `run_pipeline.py`, the dashboard runs it automatically on first launch.)

---

## 3. Project structure

```
churn-retention-project/
├── data/
│   ├── raw/                 <- PUT KAGGLE FILE HERE (data.csv)
│   ├── synthetic/           <- auto-generated fallback data
│   └── processed/           <- cleaned sales table
├── sql/
│   ├── 01_cohort_retention.sql   cohort x month-index retention
│   ├── 02_rfm_base.sql           recency / frequency / monetary per customer
│   └── 03_monthly_trend.sql      monthly revenue & active customers
├── src/
│   ├── config.py            churn window, margin, retention levers (edit assumptions here)
│   ├── data_loader.py       real file if present, else synthetic
│   ├── generate_synthetic.py
│   ├── cleaning.py          missing IDs, duplicates, non-product codes, sales vs returns
│   ├── sql_analytics.py     SQLite layer that runs the .sql files
│   ├── rfm.py               quintile scoring + 8 segments
│   ├── features.py          customer features + churn label (no leakage)
│   ├── churn_model.py       gradient boosting + permutation importance + driver bins
│   ├── scenarios.py         revenue-at-risk & lever economics
│   ├── recommendations.py   ranked action list
│   └── pipeline.py          orchestrates everything
├── app/streamlit_app.py     executive dashboard
├── reports/outputs/         CSV/JSON results consumed by the dashboard
├── notebooks/               (empty — add your EDA notebooks here)
├── run_pipeline.py
└── requirements.txt
```

---

## 4. Methodology

### 4.1 Cleaning
Drop rows without `CustomerID`, exact duplicates and non-product stock codes (POST, DOT, BANK CHARGES …).
Split into **sales** (positive qty & price) and **returns** (`C…` invoices). `Revenue = Quantity × UnitPrice`.

### 4.2 Churn definition (important — be ready to explain this)
There is no "cancel" button in retail data, so churn is **behavioural**:
> A customer **churned** if they placed **no order in the last 90 days** of the data.

To avoid **data leakage**, features are computed *only from data before* `cutoff = snapshot − 90 days`, and the
label looks at the 90 days *after* the cutoff. Customers must have ordered before the cutoff to enter the training set.

### 4.3 Cohort analysis (SQL)
Customers are grouped by month of first purchase. `01_cohort_retention.sql` counts how many are active
0, 1, 2 … N months later; dividing by month 0 gives the retention matrix (heatmap in the dashboard).

### 4.4 RFM segmentation
`02_rfm_base.sql` computes Recency, Frequency, Monetary as of the snapshot. Each is scored 1–5 by quintile
and mapped to segments: *Champions, Loyal Customers, New / Promising, Needs Attention, Can't Lose Them,
At Risk, Hibernating, Lost*.

### 4.5 Churn drivers
- 14 engineered features: recency, tenure, frequency, spend, AOV, basket size, product variety, order-gap
  regularity, "how overdue vs usual rhythm", recent vs previous-quarter activity, return rate, UK vs international.
- `HistGradientBoostingClassifier`, stratified 75/25 split, metrics: ROC-AUC and average precision.
- **Permutation importance** on the hold-out set ranks the drivers; **quintile churn-rate tables** make
  them explainable to non-technical stakeholders.

### 4.6 Revenue at risk & scenario analysis
```
revenue_at_risk   = P(churn) × revenue in last 12 months          (per customer)
saved_revenue     = save_rate × revenue_at_risk of targeted segments
gross_profit      = saved_revenue × (gross_margin − discount)
program_cost      = targeted_customers × cost_per_customer
net_benefit       = gross_profit − program_cost          ROI = net_benefit / program_cost
```
Five levers (outreach, discount, loyalty program, win-back, onboarding nurture) × three scenarios
(Conservative / Base / Optimistic). Edit them in `src/config.py` or live in the dashboard's simulator.

### 4.7 Recommendations
Levers are ranked by base-case net benefit → **P1 / P2 / P3 / Deprioritise**, with the conservative–optimistic
range and the top churn drivers shown alongside.

---

## 5. Dashboard tabs
Executive summary (KPIs, trends) · Cohorts (retention heatmap) · RFM segments · Churn drivers ·
Scenario simulator (sliders) · Recommendations (prioritised + top at-risk customers).

---

## 6. Example output (synthetic data — replace with your real results)
| Metric | Value |
|---|---|
| Customers | 2,500 |
| Champions share of customers / revenue | 26% / 56% |
| 90-day churn rate (training window) | 58% |
| Model ROC-AUC (hold-out) | 0.94 |
| Top driver | Recency (by a wide margin) |

---

## 7. Assumptions & limitations (say these in interviews — it shows maturity)
- Save rates, campaign costs and 30% margin are **assumptions**, not measured; validate through A/B tests.
- Churn is inferred from inactivity; B2B wholesalers with long order gaps may be flagged as churned too early.
- Only ~12 months of history: no seasonality (e.g., Christmas peak) adjustment; later cohorts have short histories.
- Customers already inactive have P(churn) ≈ 1, so their "revenue at risk" is really win-back potential.
- Model is correlational — drivers show *who* leaves, not that changing the driver *causes* retention.

---

## 8. Resume bullets (fill in with your real numbers)
- Analyzed ~4k customers / 400k+ transactions to identify churn drivers; segmented the base with RFM and cohort retention analysis (SQL, Pandas).
- Built a gradient-boosting churn model (ROC-AUC **X**) and quantified **£Y** revenue at risk across **N** high-risk customers.
- Modeled 5 retention levers under 3 scenarios; top lever delivered **£Z** projected net benefit at **R×** ROI.
- Delivered a Streamlit executive dashboard with an interactive what-if simulator and prioritised recommendations.

## 9. Ideas to extend
Survival analysis (Kaplan-Meier), CLV (BG/NBD), SHAP explanations, product-category churn drivers, A/B test design for the top lever.
