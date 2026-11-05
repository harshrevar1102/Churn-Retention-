"""SQL layer: loads cleaned sales into SQLite and runs the .sql files in /sql."""
import sqlite3
import pandas as pd
from src import config


def make_connection(sales: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    cols = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country", "Revenue"]
    out = sales[cols].copy()
    out["InvoiceDate"] = out["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_sql("sales", conn, index=False)
    conn.execute("CREATE INDEX idx_sales_cust ON sales(CustomerID)")
    return conn


def run_sql(conn, filename: str, params: dict | None = None) -> pd.DataFrame:
    query = (config.SQL_DIR / filename).read_text()
    return pd.read_sql_query(query, conn, params=params or {})


def cohort_retention(conn) -> pd.DataFrame:
    raw = run_sql(conn, "01_cohort_retention.sql")
    counts = raw.pivot(index="cohort_month", columns="month_index", values="active_customers")
    retention = counts.div(counts[0], axis=0)
    counts.insert(0, "cohort_size", counts[0])
    return retention, counts
