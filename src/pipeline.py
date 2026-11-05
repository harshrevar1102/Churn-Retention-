"""End-to-end pipeline: load -> clean -> SQL -> features -> model -> scenarios -> outputs."""
import json
import pandas as pd
from src import config
from src.churn_model import driver_bins, score_customers, train_model
from src.cleaning import clean
from src.data_loader import load_raw
from src.features import build_features, label_churn
from src.recommendations import build_recommendations
from src.rfm import score_rfm, segment_summary
from src.scenarios import add_risk_columns, scenario_table
from src.sql_analytics import cohort_retention, make_connection, run_sql


def run():
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading data ...")
    raw, source = load_raw()
    print(f"      source: {source}  rows: {len(raw):,}")

    print("[2/7] Cleaning ...")
    sales, returns, stats = clean(raw)
    sales.to_csv(config.PROCESSED_DIR / "sales_clean.csv", index=False)

    max_date = sales["InvoiceDate"].max()
    snapshot = max_date.normalize() + pd.Timedelta(days=1)
    cutoff = snapshot - pd.Timedelta(days=config.CHURN_WINDOW_DAYS)

    print("[3/7] SQL analytics (cohorts, RFM base, monthly trend) ...")
    conn = make_connection(sales)
    retention, cohort_counts = cohort_retention(conn)
    retention.to_csv(out / "cohort_retention.csv")
    cohort_counts.to_csv(out / "cohort_counts.csv")
    trend = run_sql(conn, "03_monthly_trend.sql")
    trend.to_csv(out / "monthly_trend.csv", index=False)
    rfm_base = run_sql(conn, "02_rfm_base.sql", {"snapshot": str(snapshot)})
    rfm = score_rfm(rfm_base)
    segment_summary(rfm).to_csv(out / "rfm_summary.csv", index=False)

    print("[4/7] Building training set (features @ cutoff, label = no order in last "
          f"{config.CHURN_WINDOW_DAYS}d) ...")
    f_train = build_features(sales, returns, cutoff)
    f_train["churned"] = label_churn(sales, f_train["CustomerID"], cutoff, snapshot).values

    print("[5/7] Training churn model & driver analysis ...")
    model, metrics, importance = train_model(f_train)
    importance.to_csv(out / "driver_importance.csv", index=False)
    driver_bins(f_train).to_csv(out / "driver_bins.csv", index=False)
    print(f"      ROC-AUC {metrics['roc_auc']:.3f} | churn rate {metrics['baseline_churn_rate']:.1%}")

    print("[6/7] Scoring current customers & scenario analysis ...")
    f_now = build_features(sales, returns, snapshot)
    f_now["churn_prob"] = score_customers(model, f_now).values
    customers = f_now.merge(rfm[["CustomerID", "R", "F", "M", "rfm_segment"]], on="CustomerID")
    customers = add_risk_columns(customers)
    customers.to_csv(out / "customers_scored.csv", index=False)

    scenarios = scenario_table(customers)
    scenarios.to_csv(out / "scenarios.csv", index=False)
    recs = build_recommendations(scenarios, importance)
    recs.to_csv(out / "recommendations.csv", index=False)

    print("[7/7] Writing KPIs ...")
    active = customers[customers["recency_days"] <= config.CHURN_WINDOW_DAYS]
    kpis = {
        "data_source": source, "snapshot_date": str(snapshot.date()),
        "customers_total": int(len(customers)), "customers_active": int(len(active)),
        "historical_churn_rate": metrics["baseline_churn_rate"],
        "revenue_last_12m": float(customers["annual_revenue"].sum()),
        "revenue_at_risk_active": float(active["revenue_at_risk"].sum()),
        "revenue_at_risk_all": float(customers["revenue_at_risk"].sum()),
        "high_risk_customers": int((customers["churn_prob"] >= 0.7).sum()),
        "model": metrics, "cleaning": stats,
        "churn_window_days": config.CHURN_WINDOW_DAYS, "gross_margin": config.GROSS_MARGIN,
    }
    (out / "kpis.json").write_text(json.dumps(kpis, indent=2, default=str))
    print("Done. Outputs in", out)
    return kpis


if __name__ == "__main__":
    run()
