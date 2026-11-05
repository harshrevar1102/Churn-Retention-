"""Executive dashboard:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.scenarios import evaluate_lever

OUT = config.OUTPUT_DIR
st.set_page_config(page_title="Churn & Retention Strategy", page_icon="📉", layout="wide")


@st.cache_data(show_spinner=False)
def load_outputs():
    return {
        "kpis": json.loads((OUT / "kpis.json").read_text()),
        "customers": pd.read_csv(OUT / "customers_scored.csv"),
        "retention": pd.read_csv(OUT / "cohort_retention.csv", index_col=0),
        "cohort_counts": pd.read_csv(OUT / "cohort_counts.csv", index_col=0),
        "rfm": pd.read_csv(OUT / "rfm_summary.csv"),
        "importance": pd.read_csv(OUT / "driver_importance.csv"),
        "bins": pd.read_csv(OUT / "driver_bins.csv"),
        "scenarios": pd.read_csv(OUT / "scenarios.csv"),
        "recs": pd.read_csv(OUT / "recommendations.csv"),
        "trend": pd.read_csv(OUT / "monthly_trend.csv"),
    }


if not (OUT / "kpis.json").exists():
    with st.spinner("First run: executing the analysis pipeline ..."):
        from src.pipeline import run
        run()
        st.cache_data.clear()

d = load_outputs()
k, cust = d["kpis"], d["customers"]
money = lambda x: f"£{x:,.0f}"

st.title("📉 Customer Churn & Retention Strategy")
st.caption(f"Data: {k['data_source']} · Snapshot: {k['snapshot_date']} · "
           f"Churn = no order in {k['churn_window_days']} days")

tabs = st.tabs(["Executive summary", "Cohorts", "RFM segments", "Churn drivers",
                "Scenario simulator", "Recommendations"])

# ---------------------------------------------------------------- Summary
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Customers", f"{k['customers_total']:,}")
    c[1].metric("Active (last 90d)", f"{k['customers_active']:,}")
    c[2].metric("Churn rate (90d window)", f"{k['historical_churn_rate']:.1%}")
    c[3].metric("Revenue (last 12m)", money(k["revenue_last_12m"]))
    c[4].metric("Revenue at risk (active base)", money(k["revenue_at_risk_active"]),
                help="Sum of churn probability x last-12m revenue for customers who ordered in the last 90 days.")
    left, right = st.columns(2)
    fig = px.bar(d["trend"], x="month", y="revenue", title="Monthly revenue")
    left.plotly_chart(fig, width="stretch")
    fig = px.line(d["trend"], x="month", y="active_customers", markers=True, title="Monthly active customers")
    right.plotly_chart(fig, width="stretch")
    top = d["recs"].iloc[0]
    st.info(f"**Top recommendation:** {top['recommendation']}")

# ---------------------------------------------------------------- Cohorts
with tabs[1]:
    st.subheader("Cohort retention (share of a cohort active N months after first purchase)")
    ret = d["retention"]
    ret.columns = [int(c) for c in ret.columns]
    fig = px.imshow(ret * 100, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                    labels=dict(x="Months since first purchase", y="Cohort", color="% active"))
    st.plotly_chart(fig, width="stretch")
    avg = (ret * 100).mean().round(1)
    st.write("**Average retention curve:** " +
             ", ".join(f"M{m}: {v}%" for m, v in avg.items() if m in (1, 3, 6)))
    st.dataframe(d["cohort_counts"], width="stretch")

# ---------------------------------------------------------------- RFM
with tabs[2]:
    rfm = d["rfm"]
    a, b = st.columns(2)
    a.plotly_chart(px.bar(rfm, x="rfm_segment", y="customers", title="Customers per segment"),
                   width="stretch")
    b.plotly_chart(px.bar(rfm, x="rfm_segment", y="pct_revenue", title="Share of revenue per segment"),
                   width="stretch")
    st.dataframe(rfm.style.format({"avg_recency_days": "{:.0f}", "avg_frequency": "{:.1f}",
                                   "avg_monetary": "{:,.0f}", "total_revenue": "{:,.0f}",
                                   "pct_customers": "{:.1%}", "pct_revenue": "{:.1%}"}),
                 width="stretch")
    seg_risk = cust.groupby("rfm_segment")["revenue_at_risk"].sum().reset_index()
    st.plotly_chart(px.bar(seg_risk.sort_values("revenue_at_risk", ascending=False),
                           x="rfm_segment", y="revenue_at_risk", title="Revenue at risk by segment"),
                    width="stretch")

# ---------------------------------------------------------------- Drivers
with tabs[3]:
    imp = d["importance"].head(10).iloc[::-1]
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           title="Top churn drivers (permutation importance, drop in ROC-AUC)"),
                    width="stretch")
    m = k["model"]
    st.caption(f"Model ROC-AUC {m['roc_auc']:.3f} · Avg precision {m['avg_precision']:.3f} "
               f"(hold-out set of {m['test_rows']:,} customers)")
    feat = st.selectbox("Churn rate by feature quintile", d["bins"]["feature"].unique())
    sub = d["bins"][d["bins"]["feature"] == feat]
    st.plotly_chart(px.bar(sub, x="bin", y="churn_rate", text=sub["churn_rate"].map("{:.0%}".format),
                           title=f"Churn rate by {feat}"), width="stretch")

# ---------------------------------------------------------------- Simulator
with tabs[4]:
    st.subheader("What-if: retention lever economics")
    margin = st.slider("Gross margin", 0.05, 0.60, float(k["gross_margin"]), 0.01)
    lever = st.selectbox("Lever", list(config.LEVERS))
    cfg = config.LEVERS[lever]
    c1, c2, c3 = st.columns(3)
    save = c1.slider("Save rate (share of revenue at risk recovered)", 0.0, 0.5, float(cfg["save_rate"][1]), 0.01)
    cost = c2.slider("Cost per targeted customer", 0.0, 20.0, float(cfg["cost_per_customer"]), 0.5)
    disc = c3.slider("Discount on saved revenue", 0.0, 0.4, float(cfg["discount_pct"]), 0.01)
    segs = st.multiselect("Target segments", sorted(cust["rfm_segment"].unique()), default=cfg["segments"])
    res = evaluate_lever(cust, segs, save, cost, disc, margin)
    r = st.columns(5)
    r[0].metric("Targeted customers", f"{res['targeted_customers']:,}")
    r[1].metric("Revenue at risk", money(res["revenue_at_risk"]))
    r[2].metric("Revenue saved", money(res["revenue_saved"]))
    r[3].metric("Program cost", money(res["program_cost"]))
    r[4].metric("Net benefit", money(res["net_benefit"]),
                delta=None if res["program_cost"] == 0 else f"ROI {res['roi']:.1f}x")
# ---------------------------------------------------------------- Recs
with tabs[5]:
    st.subheader("Prioritised recommendations (ranked by base-case net benefit)")
    for _, r in d["recs"].iterrows():
        icon = {"P1": "🟢", "P2": "🟡", "P3": "🟠"}.get(r["priority"][:2], "🔴")
        st.markdown(f"{icon} **{r['priority']}** — {r['recommendation']}")
    st.caption(d["recs"]["top_drivers_context"].iloc[0])
    with st.expander("Top at-risk customers (active, highest revenue at risk)"):
        act = cust[cust["recency_days"] <= k["churn_window_days"]]
        st.dataframe(act.sort_values("revenue_at_risk", ascending=False)
                     [["CustomerID", "rfm_segment", "recency_days", "frequency", "annual_revenue",
                       "churn_prob", "revenue_at_risk"]].head(25), width="stretch")
    with st.expander("Assumptions & limitations"):
        st.markdown(f"""
- Revenue at risk = P(churn) × revenue in last 12 months; gross margin {k['gross_margin']:.0%}.
- Save rates / costs are **assumptions** (see `src/config.py`) — validate with A/B tests.
- Currency shown as £ (UK retailer); synthetic data uses the same unit.
- Customers already inactive have P(churn)≈1, so their "risk" is really win-back potential.
""")
