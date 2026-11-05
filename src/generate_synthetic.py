"""Synthetic generator that mimics the Kaggle 'Online Retail' schema.

Customers get a latent loyalty score that drives order frequency, lifetime
(churn) and return behaviour, so the generated data contains realistic, learnable
churn drivers. Used only when no real dataset is found in data/raw/.
"""
import numpy as np
import pandas as pd


def generate_synthetic_data(n_customers=2500, seed=42,
                            start="2010-12-01", end="2011-12-09") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total_days = (end - start).days

    n_prod = 400
    codes = [str(10000 + i) for i in range(n_prod)]
    prices = np.clip(np.round(np.exp(rng.normal(1.0, 0.8, n_prod)), 2), 0.29, 80)
    descs = [f"PRODUCT {c}" for c in codes]

    countries = ["United Kingdom", "Germany", "France", "EIRE",
                 "Spain", "Netherlands", "Belgium", "Switzerland"]
    cprob = [0.89, 0.03, 0.03, 0.015, 0.01, 0.01, 0.01, 0.005]

    orders = []

    def clip_date(d):
        return min(d, end)

    for i in range(n_customers):
        cid = 12346 + i
        loyalty = rng.beta(2, 2)
        country = rng.choice(countries, p=cprob)
        foreign = country != "United Kingdom"
        first_off = int(rng.beta(1, 1.6) * (total_days - 3))
        rate = np.exp(rng.normal(-3.4 + 1.3 * loyalty - 0.3 * foreign, 0.45))
        mean_life = (30 + 550 * loyalty ** 1.6) * (0.8 if foreign else 1.0)
        life = rng.exponential(mean_life)
        last_off = min(first_off + life, total_days)
        span = max(last_off - first_off, 0)
        n_extra = rng.poisson(rate * span)
        offs = np.concatenate([[first_off], first_off + rng.uniform(0, span, n_extra)])
        fav = rng.choice(n_prod, size=int(rng.integers(8, 40)), replace=False)
        return_p = 0.02 + 0.10 * (1 - loyalty)
        aov_mult = (1 + 2.5 * foreign) * np.exp(rng.normal(0, 0.3))

        for off in offs:
            date = clip_date(start + pd.Timedelta(days=int(off), hours=int(rng.integers(8, 18)),
                                                  minutes=int(rng.integers(0, 60))))
            n_lines = 1 + rng.poisson(3 + 2 * loyalty)
            idx = np.where(rng.random(n_lines) < 0.7,
                           rng.choice(fav, n_lines), rng.integers(0, n_prod, n_lines))
            qty = np.maximum(1, np.round(rng.geometric(0.2, n_lines) * aov_mult)).astype(int)
            orders.append(dict(date=date, cid=cid, country=country, idx=idx, qty=qty, ret=False))
            if rng.random() < return_p:
                k = int(rng.integers(1, 3))
                rdate = date + pd.Timedelta(days=int(rng.integers(1, 11)))
                if rdate <= end:
                    orders.append(dict(date=rdate, cid=cid, country=country,
                                       idx=idx[:k], qty=-qty[:k], ret=True))

    # Guest checkouts without CustomerID (exists in the real data too)
    for _ in range(int(0.08 * len(orders))):
        d = start + pd.Timedelta(days=int(rng.integers(0, total_days)), hours=int(rng.integers(8, 18)))
        n_lines = 1 + rng.poisson(4)
        orders.append(dict(date=d, cid=np.nan, country="United Kingdom",
                           idx=rng.integers(0, n_prod, n_lines),
                           qty=rng.integers(1, 8, n_lines), ret=False))

    orders.sort(key=lambda o: o["date"])
    rows, inv = [], 536365
    for o in orders:
        inv_no = f"C{inv}" if o["ret"] else str(inv)
        inv += 1
        for j, q in zip(o["idx"], o["qty"]):
            rows.append((inv_no, codes[j], descs[j], int(q), o["date"].strftime("%Y-%m-%d %H:%M:%S"),
                         float(prices[j]), o["cid"], o["country"]))
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.shape)
