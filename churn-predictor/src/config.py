"""Shared paths and constants for the churn-predictor project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CSV_PATH = RAW_DIR / "hillstrom.csv"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
PREDICTIONS_TEST_PATH = PROCESSED_DIR / "predictions_test.parquet"

MODELS_DIR = PROJECT_ROOT / "models" / "trained"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_SUMMARY_PATH = REPORTS_DIR / "results_summary.md"

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

RANDOM_SEED = 42

# Fraction of rows held out for validation and test, taken sequentially after
# an initial random shuffle (stratified by treatment, see features.py).
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Raw covariate columns used as model features (everything except the RCT
# arm assignment and the three outcome columns).
CATEGORICAL_FEATURES = ["history_segment", "zip_code", "channel"]
BINARY_FEATURES = ["mens", "womens", "newbie"]
NUMERIC_FEATURES = ["recency", "history"]

TREATMENT_COLUMN = "treatment"
OUTCOME_COLUMN = "spend"
SEGMENT_COLUMN = "segment"
CONTROL_SEGMENT = "No E-Mail"

# Assumed marginal cost of delivering the retention/promotional offer to one
# customer, used only for the $ net-incremental-revenue business translation
# in scripts/08_revenue_simulation.py. This is a documented assumption, not
# a figure recovered from the data -- see reports/results_summary.md for the
# sensitivity sweep across alternative costs.
COST_PER_OFFER_USD = 2.0

# Targeting fractions swept when comparing policies (top-k% by predicted
# uplift vs. blanket vs. random targeting).
TARGETING_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


# Kept modest (rather than econml's larger defaults) so the causal forest
# trains in a reasonable time on a laptop over ~45K training rows.
CAUSAL_FOREST_N_ESTIMATORS = 200
N_QINI_BINS = 10
