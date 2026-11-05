"""RFM scoring and segmentation."""
import pandas as pd


def _quintile(series: pd.Series, ascending_is_good: bool) -> pd.Series:
    labels = [1, 2, 3, 4, 5] if ascending_is_good else [5, 4, 3, 2, 1]
    return pd.qcut(series.rank(method="first"), 5, labels=labels).astype(int)


def _segment(r, f):
    if r >= 4 and f >= 4: return "Champions"
    if r >= 3 and f >= 3: return "Loyal Customers"
    if r >= 4:            return "New / Promising"
    if r == 3:            return "Needs Attention"
    if f >= 4:            return "Can't Lose Them"
    if f == 3:            return "At Risk"
    if r == 1:            return "Lost"
    return "Hibernating"


def score_rfm(base: pd.DataFrame) -> pd.DataFrame:
    """base: CustomerID, recency_days, frequency, monetary (from SQL)."""
    df = base.copy()
    df["R"] = _quintile(df["recency_days"], ascending_is_good=False)  # recent = 5
    df["F"] = _quintile(df["frequency"], ascending_is_good=True)
    df["M"] = _quintile(df["monetary"], ascending_is_good=True)
    df["rfm_segment"] = [_segment(r, f) for r, f in zip(df["R"], df["F"])]
    return df


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("rfm_segment").agg(
        customers=("CustomerID", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).reset_index()
    g["pct_customers"] = g["customers"] / g["customers"].sum()
    g["pct_revenue"] = g["total_revenue"] / g["total_revenue"].sum()
    return g.sort_values("total_revenue", ascending=False)
