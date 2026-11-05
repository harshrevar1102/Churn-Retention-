"""Churn driver analysis: predictive model + permutation importance + driver bins."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src import config
from src.features import FEATURES

BIN_FEATURES = ["recency_days", "frequency", "avg_order_value", "return_rate",
                "tenure_days", "unique_products", "orders_last_90d"]


def train_model(train_df: pd.DataFrame):
    X, y = train_df[FEATURES], train_df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                           random_state=config.RANDOM_STATE)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {"roc_auc": float(roc_auc_score(y_te, proba)),
               "avg_precision": float(average_precision_score(y_te, proba)),
               "baseline_churn_rate": float(y.mean()),
               "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))}
    perm = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                  random_state=config.RANDOM_STATE)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    return model, metrics, importance


def driver_bins(train_df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate across quintile bins of the key features (easy to explain)."""
    rows = []
    for feat in BIN_FEATURES:
        s = train_df[feat]
        try:
            bins = pd.qcut(s, 5, duplicates="drop")
        except ValueError:
            continue
        g = train_df.groupby(bins, observed=True)["churned"].agg(["count", "mean"]).reset_index()
        for _, r in g.iterrows():
            rows.append({"feature": feat, "bin": str(r[feat]), "customers": int(r["count"]),
                         "churn_rate": float(r["mean"])})
    return pd.DataFrame(rows)


def score_customers(model, feats: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(feats[FEATURES])[:, 1], index=feats.index)
