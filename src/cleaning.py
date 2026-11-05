"""Clean the raw transactions and split them into sales and returns."""
import pandas as pd


def clean(raw: pd.DataFrame):
    df = raw.copy()
    stats = {"raw_rows": len(df)}

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()

    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    stats["after_dropping_missing_customer"] = len(df)
    df["CustomerID"] = df["CustomerID"].astype(float).astype(int)

    df = df.drop_duplicates()
    stats["after_dedup"] = len(df)

    # Keep real products only (drops POST, DOT, M, BANK CHARGES, AMAZONFEE ...)
    df = df[df["StockCode"].str.match(r"^\d")]
    df["Country"] = df["Country"].fillna("Unknown")

    is_cancel = df["InvoiceNo"].str.startswith("C")
    returns = df[is_cancel & (df["Quantity"] < 0)].copy()
    sales = df[~is_cancel & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    returns["Revenue"] = (returns["Quantity"] * returns["UnitPrice"]).abs()

    stats.update(sales_rows=len(sales), return_rows=len(returns),
                 customers=int(sales["CustomerID"].nunique()),
                 date_min=str(sales["InvoiceDate"].min()), date_max=str(sales["InvoiceDate"].max()))
    return sales.reset_index(drop=True), returns.reset_index(drop=True), stats
