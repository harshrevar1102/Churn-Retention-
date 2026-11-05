"""Central configuration: paths, business assumptions and retention levers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SYNTH_DIR = ROOT / "data" / "synthetic"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "reports" / "outputs"
SQL_DIR = ROOT / "sql"

RANDOM_STATE = 42

# --- Churn definition -------------------------------------------------------
# A customer has "churned" if they place NO order in the final CHURN_WINDOW_DAYS
# of the data, given that they were active before that window.
CHURN_WINDOW_DAYS = 90

# --- Business assumptions (edit freely; the dashboard also lets you change them)
GROSS_MARGIN = 0.30          # share of revenue that is gross profit

# --- Retention levers -------------------------------------------------------
# save_rate = (conservative, base, optimistic): share of "revenue at risk" saved
# cost_per_customer = campaign cost per targeted customer
# discount_pct = discount given on saved revenue (reduces margin on those sales)
LEVERS = {
    "Proactive outreach: Can't Lose & At Risk": {
        "segments": ["Can't Lose Them", "At Risk"],
        "cost_per_customer": 3.0, "discount_pct": 0.00,
        "save_rate": (0.05, 0.10, 0.15),
    },
    "Targeted discount: At Risk & Needs Attention": {
        "segments": ["At Risk", "Needs Attention"],
        "cost_per_customer": 1.0, "discount_pct": 0.15,
        "save_rate": (0.08, 0.15, 0.22),
    },
    "Loyalty program: Champions & Loyal": {
        "segments": ["Champions", "Loyal Customers"],
        "cost_per_customer": 8.0, "discount_pct": 0.05,
        "save_rate": (0.02, 0.05, 0.08),
    },
    "Win-back campaign: Hibernating & Lost": {
        "segments": ["Hibernating", "Lost"],
        "cost_per_customer": 1.5, "discount_pct": 0.20,
        "save_rate": (0.02, 0.04, 0.07),
    },
    "Onboarding nurture: New / Promising": {
        "segments": ["New / Promising"],
        "cost_per_customer": 2.0, "discount_pct": 0.00,
        "save_rate": (0.05, 0.10, 0.15),
    },
}
SCENARIO_NAMES = ("Conservative", "Base", "Optimistic")
