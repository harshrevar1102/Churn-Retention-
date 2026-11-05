"""Customer-level feature engineering, computed strictly from data <= as_of."""
import numpy as np
import pandas as pd

FEATURES = [
    "recency_days", "tenure_days", "frequency", "monetary", "avg_order_value",
    "avg_items_per_order", "unique_products", "avg_days_between_orders",
    "recency_to_gap_ratio", "orders_last_90d", "orders_prev_90d",
    "spend_last_90d", "return_rate", "is_uk",
]


def build_features(sales: pd.DataFrame, returns: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    hist = sales[sales["InvoiceDate"] <= as_of]
    ret = returns[returns["InvoiceDate"] <= as_of]

    orders = (hist.groupby(["CustomerID", "InvoiceNo"])
                  .agg(date=("InvoiceDate", "max"), revenue=("Revenue", "sum"),
                       items=("Quantity", "sum"))
                  .reset_index())
    og = orders.groupby("CustomerID")

    feat = pd.DataFrame({
        "last_order": og["date"].max(),
        "first_order": og["date"].min(),
        "frequency": og.size(),
        "monetary": og["revenue"].sum(),
        "avg_order_value": og["revenue"].mean(),
        "avg_items_per_order": og["items"].mean(),
    })
    feat["recency_days"] = (as_of - feat["last_order"]).dt.days
    feat["tenure_days"] = (as_of - feat["first_order"]).dt.days
    feat["unique_products"] = hist.groupby("CustomerID")["StockCode"].nunique()

    gap = (feat["last_order"] - feat["first_order"]).dt.days / (feat["frequency"] - 1)
    feat["avg_days_between_orders"] = gap.where(feat["frequency"] > 1)
    ratio = feat["recency_days"] / feat["avg_days_between_orders"]
    feat["recency_to_gap_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    d90, d180, d365 = (as_of - pd.Timedelta(days=d) for d in (90, 180, 365))
    recent = orders[orders["date"] > d90].groupby("CustomerID")
    prev = orders[(orders["date"] <= d90) & (orders["date"] > d180)].groupby("CustomerID")
    feat["orders_last_90d"] = recent.size().reindex(feat.index).fillna(0)
    feat["orders_prev_90d"] = prev.size().reindex(feat.index).fillna(0)
    feat["spend_last_90d"] = recent["revenue"].sum().reindex(feat.index).fillna(0)
    feat["annual_revenue"] = (orders[orders["date"] > d365].groupby("CustomerID")["revenue"]
                              .sum().reindex(feat.index).fillna(0))

    ret_val = ret.groupby("CustomerID")["Revenue"].sum().reindex(feat.index).fillna(0)
    feat["return_rate"] = (ret_val / feat["monetary"]).clip(0, 1)

    country = hist.sort_values("InvoiceDate").groupby("CustomerID")["Country"].last()
    feat["country"] = country
    feat["is_uk"] = (country == "United Kingdom").astype(int)

    return feat.drop(columns=["last_order", "first_order"]).reset_index()


def label_churn(sales: pd.DataFrame, customer_ids, cutoff: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """1 = customer placed NO order in (cutoff, end]; 0 = they came back."""
    future = sales[(sales["InvoiceDate"] > cutoff) & (sales["InvoiceDate"] <= end)]
    returned = set(future["CustomerID"].unique())
    return pd.Series([0 if c in returned else 1 for c in customer_ids], index=customer_ids)
