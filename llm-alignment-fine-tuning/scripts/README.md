# Python Scripts

These files contain the project workflow as cleaned Python source code.

They are useful for:

- Reviewing the project as source code on GitHub
- Searching model, data, and evaluation logic
- Reusing code in future scripts or apps

## Script Order

| Order | Script | Purpose |
|---:|---|---|
| 1 | `01_instruction_fine_tuning.py` | SFT + LoRA instruction fine-tuning of `facebook/opt-350m` on CodeAlpaca-20k; SacreBLEU before/after |
| 2 | `02_reward_modeling.py` | Trains a GPT-2 + LoRA reward model on chosen/rejected pairs; pairwise ranking accuracy |
| 3 | `03_ppo_rlhf.py` | RLHF via PPO on `gpt2-imdb`, steered by a sentiment classifier; trains both a positive- and negative-sentiment policy |
| 4 | `04_dpo_fine_tuning.py` | DPO + LoRA fine-tuning of GPT-2 on preference pairs; before/after generation comparison |
| 5 | `05_summarize_results.py` | Reads each script's saved metrics JSON and renders the cross-technique summary figure |

Each of scripts 01–04 is independently runnable and self-contained (downloads its own data, trains, evaluates, and saves its own artifacts/figures/metrics). Script 05 depends on 01–04 having been run at least once.

## Notes

All four source notebooks this project was built from were written for IBM's CPU-only Skills Network sandbox under a time limit, which shaped how they handled training:

- **Labs 1-1, 1-2, and 2-1** (SFT, reward modeling, PPO) all have their actual training call commented out (`#trainer.train()` / `#stats = ppo_trainer.step(...)`) and instead download a pre-trained checkpoint from IBM's cloud storage to demonstrate the rest of the workflow. This project replaces that shortcut with real, scaled-down local training in all three scripts — every metric in this project's README and [`reports/results_summary.md`](../reports/results_summary.md) comes from an actual training run on a laptop CPU, not a downloaded artifact.
- **Lab 2-2** (DPO) is the one exception — its main body already called `trainer.train()` for real, it just had never been executed end-to-end. `04_dpo_fine_tuning.py` runs it for real and adds a generation comparison scored with a sentiment-classifier proxy reward (reused from the PPO script).
- The DPO notebook's GPU-only, never-exercised 4-bit `bitsandbytes` quantization cell (commented out in the source) was dropped entirely, as was its "Exercises" section's parallel mini-pipeline on a second dataset — the exercise's compare-before-after generation pattern was generalized into `src/visualization.render_before_after_table` and applied directly to the main lab's trained model instead of being duplicated on a separate dataset.
- All four scripts force CPU execution (`torch.backends.mps.is_available()` is monkey-patched to `False` in `src/config.py`). On this project's Apple Silicon dev machine, MPS's per-op dispatch overhead made single-example generation calls (used in SFT and PPO) more than 5x slower than CPU, and some HF/TRL model-loading paths crashed outright on MPS with tied-weight models (`RuntimeError: Placeholder storage has not been allocated on MPS device`). CPU was simply faster and more reliable at this model/batch scale.

Scripts assume they are run with the project's `.venv` activated, from anywhere inside the repository (paths are resolved via `src/config.py`, not the working directory).
