"""Turn scenario results + drivers into a prioritised action list."""
import pandas as pd

_DRIVER_TEXT = {
    "recency_days": "time since last order", "frequency": "order frequency",
    "monetary": "total spend", "avg_order_value": "average order value",
    "tenure_days": "customer tenure", "unique_products": "product variety",
    "orders_last_90d": "recent order activity", "orders_prev_90d": "previous-quarter activity",
    "spend_last_90d": "recent spend", "return_rate": "return rate",
    "recency_to_gap_ratio": "how overdue they are vs their usual buying rhythm",
    "avg_days_between_orders": "usual gap between orders",
    "avg_items_per_order": "basket size", "is_uk": "country (UK vs. international)",
}


def build_recommendations(scenarios: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    base = scenarios[scenarios["scenario"] == "Base"].sort_values("net_benefit", ascending=False).copy()
    base = base.reset_index(drop=True)
    base["rank"] = base.index + 1

    def priority(row):
        if row["net_benefit"] <= 0: return "Deprioritise"
        return "P1 - Do now" if row["rank"] <= 2 else ("P2 - Next" if row["rank"] <= 4 else "P3 - Later")

    base["priority"] = base.apply(priority, axis=1)
    base["recommendation"] = base.apply(
        lambda r: (f"{r['lever']}: target {r['targeted_customers']:,} customers; "
                   f"base-case net benefit {r['net_benefit']:,.0f} "
                   f"(ROI {r['roi']:.1f}x, range "
                   f"{scenarios[(scenarios.lever == r['lever']) & (scenarios.scenario == 'Conservative')]['net_benefit'].iat[0]:,.0f}"
                   f" to {scenarios[(scenarios.lever == r['lever']) & (scenarios.scenario == 'Optimistic')]['net_benefit'].iat[0]:,.0f})."),
        axis=1)
    top = [(_DRIVER_TEXT.get(f, f)) for f in importance["feature"].head(3)]
    base["top_drivers_context"] = "Top churn drivers: " + ", ".join(top)
    return base[["rank", "priority", "lever", "targeted_customers", "revenue_at_risk",
                 "net_benefit", "roi", "recommendation", "top_drivers_context"]]
