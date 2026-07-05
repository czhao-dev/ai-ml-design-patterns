# Data

All four scripts pull their data from the Hugging Face Hub (or a direct download) at runtime — nothing is committed locally.

| Lab | Dataset | Source |
|---|---|---|
| 01 — Instruction Fine-Tuning | CodeAlpaca-20k | Direct JSON download (IBM Skills Network mirror), cached to `data/raw/CodeAlpaca-20k.json` |
| 02 — Reward Modeling | [`Dahoas/synthetic-instruct-gptj-pairwise`](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) | `datasets.load_dataset` |
| 03 — PPO RLHF | [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) | `datasets.load_dataset` |
| 04 — DPO | [`BarraHome/ultrafeedback_binarized`](https://huggingface.co/datasets/BarraHome/ultrafeedback_binarized) | `datasets.load_dataset` |

## Local Layout

```text
data/raw/CodeAlpaca-20k.json   # the only locally cached file; everything else streams from the HF Hub cache (~/.cache/huggingface)
```

`data/raw/` is ignored by Git so the repository stays lightweight. Each script subsamples its dataset down to a size that trains in a few minutes on a laptop CPU (see `src/config.py` for the exact sizes per lab) — see [`reports/results_summary.md`](../reports/results_summary.md) for why those sizes were chosen.
