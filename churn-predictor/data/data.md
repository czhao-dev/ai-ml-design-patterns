# Data

This project uses Kevin Hillstrom's MineThatData E-Mail Analytics And Data Mining Challenge
dataset — a genuine randomized controlled trial run by a real retailer: 64,000 customers who
purchased within the last twelve months were randomized into three arms (no email, men's
merchandise email, women's merchandise email) and their subsequent visit/conversion/spend over
the following two weeks was recorded.

There is no public RCT dataset for *subscription churn* specifically, so this project uses
Hillstrom's real randomized promotional-email data as the causal-inference benchmark: the
mechanics of "who is influenced by being targeted, and by how much" are identical to a telco/SaaS
retention-offer decision, and using genuine randomization (rather than a simulated treatment
effect) means every $ figure in `reports/results_summary.md` is computed from real counterfactual
data, not an invented ground truth. This tradeoff — real causal validity vs. a marketing rather
than churn narrative — is discussed in the project README.

## Source

```text
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
```

Downloaded automatically by `scripts/01_download_and_validate_rct.py` into `data/raw/hillstrom.csv`
(gitignored). 64,000 rows, columns:

| Column | Type | Description |
|---|---|---|
| `recency` | int | Months since last purchase |
| `history_segment` | categorical | Binned historical spend, e.g. `2) $100 - $200` |
| `history` | float | Actual dollars spent in the past year |
| `mens` | 0/1 | Purchased men's merchandise in the past year |
| `womens` | 0/1 | Purchased women's merchandise in the past year |
| `zip_code` | categorical | `Urban` / `Suburban` / `Rural` |
| `newbie` | 0/1 | New customer in the last twelve months |
| `channel` | categorical | `Phone` / `Web` / `Multichannel` |
| `segment` | categorical | RCT arm: `No E-Mail` / `Mens E-Mail` / `Womens E-Mail` |
| `visit` | 0/1 | Visited the site in the two weeks following the campaign |
| `conversion` | 0/1 | Purchased in the two weeks following the campaign |
| `spend` | float | Dollars spent in the two weeks following the campaign (0 if no purchase) |

## Local Layout

```text
data/raw/hillstrom.csv        # downloaded, gitignored
data/processed/train.parquet  # feature-engineered train/val/test splits, gitignored
data/processed/val.parquet
data/processed/test.parquet
```

Re-run `scripts/01_download_and_validate_rct.py` then `scripts/02_build_features.py` to regenerate
both from scratch.
