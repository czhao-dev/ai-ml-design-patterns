"""Download and load the Hillstrom RCT dataset, and verify randomization held.

The dataset is a genuine three-arm RCT (no email / men's email / women's email).
Before trusting any causal estimate downstream, we check that the two arms used
in this project's binary treatment definition (any email vs. no email) are
actually balanced on observed covariates -- if randomization worked, treated and
control customers should look statistically identical on everything measured
*before* the campaign ran.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from . import config


def download_hillstrom(dest=None, url: str = config.HILLSTROM_URL, force: bool = False):
    """Download the raw Hillstrom CSV if it isn't already present locally."""
    dest = dest or config.RAW_CSV_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV and attach the binary `treatment` column used throughout
    this project (1 = received either promotional email, 0 = control/no email)."""
    path = path or config.RAW_CSV_PATH
    df = pd.read_csv(path)
    df[config.TREATMENT_COLUMN] = (df[config.SEGMENT_COLUMN] != config.CONTROL_SEGMENT).astype(int)
    return df


def standardized_mean_difference(df: pd.DataFrame, column: str) -> float:
    """Absolute standardized mean difference (Cohen's d style) of `column`
    between treated and control rows. Values under ~0.1 are conventionally
    considered "balanced"; larger values would indicate randomization failed
    or the treatment/control groups differ for reasons other than chance."""
    treated = df.loc[df[config.TREATMENT_COLUMN] == 1, column].astype(float)
    control = df.loc[df[config.TREATMENT_COLUMN] == 0, column].astype(float)
    pooled_std = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0
    return float(abs(treated.mean() - control.mean()) / pooled_std)


def covariate_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference for every pre-treatment numeric/binary
    covariate, plus one-hot'd categorical levels, treated vs. control."""
    rows = []
    numeric_and_binary = config.NUMERIC_FEATURES + config.BINARY_FEATURES
    for col in numeric_and_binary:
        rows.append(
            {
                "feature": col,
                "treated_mean": df.loc[df[config.TREATMENT_COLUMN] == 1, col].mean(),
                "control_mean": df.loc[df[config.TREATMENT_COLUMN] == 0, col].mean(),
                "smd": standardized_mean_difference(df, col),
            }
        )
    for cat_col in config.CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
        for level_col in dummies.columns:
            tmp = df[[config.TREATMENT_COLUMN]].copy()
            tmp[level_col] = dummies[level_col].values
            rows.append(
                {
                    "feature": level_col,
                    "treated_mean": tmp.loc[tmp[config.TREATMENT_COLUMN] == 1, level_col].mean(),
                    "control_mean": tmp.loc[tmp[config.TREATMENT_COLUMN] == 0, level_col].mean(),
                    "smd": standardized_mean_difference(tmp, level_col),
                }
            )
    return pd.DataFrame(rows).sort_values("smd", ascending=False).reset_index(drop=True)


def randomization_is_healthy(balance_table: pd.DataFrame, threshold: float = 0.1) -> bool:
    """True if every covariate's SMD is below `threshold` (standard uplift-
    modeling convention for "randomization looks intact")."""
    return bool((balance_table["smd"] < threshold).all())
