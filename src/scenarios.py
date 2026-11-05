"""Revenue-at-risk and retention-lever scenario analysis."""
import pandas as pd
from src import config


def add_risk_columns(customers: pd.DataFrame) -> pd.DataFrame:
    """expected_loss = P(churn) x annual revenue  ->  'revenue at risk'."""
    df = customers.copy()
    df["revenue_at_risk"] = df["churn_prob"] * df["annual_revenue"]
    return df


def evaluate_lever(customers: pd.DataFrame, segments, save_rate, cost_per_customer,
                   discount_pct, margin=config.GROSS_MARGIN) -> dict:
    """Economics of one lever.

    saved_revenue   = save_rate x revenue_at_risk of targeted customers
    gross_profit    = saved_revenue x (margin - discount_pct)
    program_cost    = n_targeted x cost_per_customer
    net_benefit     = gross_profit - program_cost
    """
    target = customers[customers["rfm_segment"].isin(segments)]
    rar = float(target["revenue_at_risk"].sum())
    saved = save_rate * rar
    profit = saved * (margin - discount_pct)
    cost = len(target) * cost_per_customer
    net = profit - cost
    return {"targeted_customers": int(len(target)), "revenue_at_risk": rar,
            "revenue_saved": saved, "gross_profit_saved": profit,
            "program_cost": cost, "net_benefit": net,
            "roi": (net / cost) if cost > 0 else float("nan")}


def scenario_table(customers: pd.DataFrame, margin=config.GROSS_MARGIN) -> pd.DataFrame:
    rows = []
    for lever, cfg in config.LEVERS.items():
        for name, sr in zip(config.SCENARIO_NAMES, cfg["save_rate"]):
            res = evaluate_lever(customers, cfg["segments"], sr, cfg["cost_per_customer"],
                                 cfg["discount_pct"], margin)
            rows.append({"lever": lever, "scenario": name, "save_rate": sr, **res})
    return pd.DataFrame(rows)
