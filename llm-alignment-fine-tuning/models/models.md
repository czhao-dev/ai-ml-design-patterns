# Models

Trained artifacts are stored locally under:

```text
models/trained/
```

That folder is ignored by Git. Three of the four labs train LoRA adapters (a few MB each); PPO trains full model + value-head checkpoints (~480 MB each, since TRL's `PPOTrainer` updates all parameters rather than a LoRA adapter). For public sharing, the Hugging Face Hub (`model.push_to_hub(...)`) is the most natural fit here — these are HF-ecosystem artifacts (PEFT adapters and `transformers`/`trl` checkpoints) — ahead of GitHub Releases or Git LFS.

## Local Artifacts

| Path | Contents | Size |
|---|---|---|
| `sft_lora/` | LoRA adapter for `facebook/opt-350m` (instruction fine-tuning) | ~11 MB |
| `reward_model/` | LoRA adapter for the GPT-2 reward model | ~5 MB |
| `ppo_policy/positive/`, `ppo_policy/negative/` | Full `gpt2-imdb` + value-head checkpoints, one per sentiment-steering direction | ~480 MB each |
| `dpo_lora/` | LoRA adapter for the DPO-tuned GPT-2 | ~3 MB |
