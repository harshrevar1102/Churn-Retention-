"""Load the real Kaggle file from data/raw/ if present, else use synthetic data."""
import pandas as pd
from src import config
from src.generate_synthetic import generate_synthetic_data

REQUIRED = ["InvoiceNo", "StockCode", "Description", "Quantity",
            "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
RENAME = {"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"}


def _find_real_file():
    files = sorted(list(config.RAW_DIR.glob("*.csv")) + list(config.RAW_DIR.glob("*.xlsx")))
    return files[0] if files else None


def _read(path):
    dtype = {"InvoiceNo": str, "Invoice": str, "StockCode": str}
    if path.suffix == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None, dtype=dtype)
        return pd.concat(sheets.values(), ignore_index=True)
    try:
        return pd.read_csv(path, dtype=dtype, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=dtype, encoding="ISO-8859-1")


def load_raw():
    """Returns (dataframe, source_label)."""
    path = _find_real_file()
    if path is not None:
        df = _read(path).rename(columns=RENAME)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}. "
                             f"Found: {list(df.columns)}")
        return df[REQUIRED], f"Kaggle dataset ({path.name})"

    synth = config.SYNTH_DIR / "synthetic_online_retail.csv"
    if not synth.exists():
        print("No dataset found in data/raw/ -> generating synthetic data ...")
        generate_synthetic_data().to_csv(synth, index=False)
    return pd.read_csv(synth, dtype={"InvoiceNo": str, "StockCode": str}), "Synthetic dataset"
