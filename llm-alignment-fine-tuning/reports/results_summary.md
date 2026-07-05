# Results Summary

## Objective

Demonstrate four complementary techniques for aligning a language model with a target behavior — supervised instruction fine-tuning, reward modeling, RLHF via PPO, and Direct Preference Optimization (DPO) — each trained for real on a laptop CPU with LoRA (or, for PPO, full fine-tuning of a small model).

## Datasets

| Lab | Dataset | Examples used |
|---|---|---:|
| SFT | CodeAlpaca-20k | 500 train / 40 eval |
| Reward Modeling | Dahoas/synthetic-instruct-gptj-pairwise | 800 train / 200 eval |
| PPO | stanfordnlp/imdb | 2,000 (sampled per step) |
| DPO | BarraHome/ultrafeedback_binarized | 300 train / 80 eval |

See [`data/data.md`](../data/data.md) for sources.

## Model Results

### Instruction Fine-Tuning (SFT + LoRA) — `scripts/01_instruction_fine_tuning.py`

| Model | SacreBLEU |
|---|---:|
| `facebook/opt-350m` (base) | 2.23 |
| + LoRA SFT (2 epochs, 500 examples) | 1.35 |

Training loss decreased steadily (2.3 → 1.5 over 250 steps; see `reports/figures/sft_loss_curve.png`), but SacreBLEU on the 40-example held-out set went *down* after fine-tuning.

> **Methodology note:** The source notebook never actually trained — `trainer.train()` is commented out, and base/LoRA outputs are both downloaded pre-computed from IBM's cloud storage. This script removes that shortcut and trains for real, which surfaced a genuine finding the original demo (run on only 10 held-out examples) couldn't have shown either way: at this scale, SacreBLEU is a noisy signal for short code completions. Qualitatively, the LoRA model's outputs are more on-topic (e.g. attempting actual Java/Python syntax) than the base model's tendency to echo the instruction template back verbatim — but BLEU's strict n-gram overlap with a single reference doesn't reward that. The original course demo's own BLEU swing (0.0 → 1.2) was similarly small relative to noise.

### Reward Modeling — `scripts/02_reward_modeling.py`

| Metric | Value |
|---|---:|
| Eval accuracy (HF Trainer's internal metric) | 0.90 |
| Pairwise ranking accuracy (held-out, this script's own eval) | 0.96 |

Train loss fell from 0.80 → 0.30 over 150 steps (3 epochs); see `reports/figures/reward_loss_curve.png`.

> **Methodology note:** Like the SFT lab, the source notebook's training call is commented out in favor of a downloaded pre-trained checkpoint. A first real local run (1 epoch, 100 steps) only reached 0.525 pairwise accuracy — barely above chance — which made it clear 100 steps wasn't enough signal for this model/data size. Bumping to 3 epochs / 150 steps with a larger batch size and higher learning rate produced a model that clearly separates chosen from rejected responses.

### PPO RLHF — `scripts/03_ppo_rlhf.py`

| Policy | Mean reward before | Mean reward after | PPO steps |
|---|---:|---:|---:|
| Positive-sentiment | 0.24 | 1.27 | 25 |
| Negative-sentiment | -0.32 | 0.56 | 25 |

Both policies move in their intended direction (see `reports/figures/ppo_positive_reward_curve.png` / `ppo_negative_reward_curve.png` for reward and KL-vs-reference curves — KL rises from ~0 to ~1.7 over training as each policy specializes away from the frozen reference).

> **Methodology note:** The source notebook's PPO step loop is commented out in two places (one per sentiment direction), with both pre-trained policies downloaded from S3 instead. This script runs the real `ppo_trainer.step()` loop for both directions. Two implementation issues surfaced and were fixed during development: (1) `AutoModelForCausalLMWithValueHead`'s `ref_model` isn't moved by TRL's internal `accelerator.prepare()` (only the trainable policy is), so it must be placed on the right device explicitly or generation crashes; (2) the dataset's HF `set_format` must be reset back to `"torch"` (not `None`) after the before/after comparison step, or the second policy's dataloader yields plain Python ints instead of tensors.

### Direct Preference Optimization (DPO) — `scripts/04_dpo_fine_tuning.py`

| Metric | Value |
|---|---:|
| Eval reward accuracy (fraction of held-out pairs where chosen > rejected) | 0.70 |
| Eval reward margin | 0.27 |
| Mean sentiment-proxy reward — base GPT-2 | 1.38 |
| Mean sentiment-proxy reward — DPO-tuned | 1.44 |

Train loss is noisy (single/double-example batches) but trends down across 450 steps / 3 epochs; see `reports/figures/dpo_loss_curve.png`.

> **Methodology note:** Unlike the other three labs, this notebook's main body already called `trainer.train()` — it just had never been run end-to-end. This script runs it for real. The notebook's own numeric eval metric (`eval_rewards/accuracies`, DPO's implicit reward model correctly ranking chosen over rejected) is a more standard signal than a sentiment classifier here, so it's reported alongside a sentiment-proxy reward (reused from the PPO script, per project scope) for a qualitative before/after comparison. With only 300 LoRA-rank-4 training examples, generations stay close to the base model's — DPO's effect is real but modest at this scale.

## Interpretation

Each technique is solving a different sub-problem in the alignment pipeline: SFT teaches a model to follow an instruction/response *format*; reward modeling teaches a separate model to *score* responses by preference; PPO uses a reward signal (here, a sentiment classifier, not the trained reward model — see Future Work) to *optimize* a policy against that score through online RL; and DPO shows the same preference-optimization goal is reachable directly from chosen/rejected pairs, without a separate reward model or an RL loop. The reward model and PPO results are the cleanest wins (0.96 pairwise accuracy; both PPO policies move sharply toward their target reward), because both have a single, well-aligned reward signal and enough training steps to act on it. SFT's BLEU regression and DPO's modest movement both point the same direction: at the very small scale this project trains on (laptop CPU, a few hundred examples, single-digit minutes per script), automatic metrics on free-form generation are noisier and slower to move than a model's internal *score* of a response.

## Key Takeaways

- A GPT-2 + LoRA reward model reaches 0.96 pairwise ranking accuracy on held-out chosen/rejected pairs after just 150 training steps — discriminating preferences is an easier, cleaner learning signal than generating better text outright at this scale.
- PPO reliably steers `gpt2-imdb` toward a target sentiment in only 25 steps per direction, with the expected KL-divergence cost against the frozen reference model.
- Both SFT and DPO show real training-loss improvement that doesn't fully translate into their respective generation-quality metrics (SacreBLEU, sentiment-proxy reward) at this scale — a genuine, reproducible finding rather than an implementation bug (see the per-lab methodology notes above).
- Three of the four source notebooks never actually executed their training step; running them for real (rather than relying on downloaded pre-trained checkpoints) is what surfaced the SFT and reward-modeling findings above.

## Future Work

- Wire Lab 1-2's trained reward model into Lab 2-1's PPO loop in place of the sentiment classifier — the natural next step toward a "real" RLHF pipeline, though it requires reconciling the reward model's chat-style training data with PPO's movie-review policy.
- Add an LLM-judge-based win-rate for DPO instead of (or alongside) the sentiment-proxy reward.
- Scale up subsample sizes and step counts on a GPU box to see whether SFT's BLEU regression and DPO's modest effect size are purely a small-scale artifact.
- Track experiments with a reproducible configuration file instead of hardcoded constants in `src/config.py`.
