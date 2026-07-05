"""Lab 2-2: Direct Preference Optimization (DPO) + LoRA on GPT-2.

Trains GPT-2 on BarraHome/ultrafeedback_binarized preference pairs with TRL's
DPOTrainer, then compares DPO-tuned vs. base GPT-2 generations on a fixed
prompt set, scored with the same sentiment-classifier proxy reward used in
the PPO lab.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peft import LoraConfig
from transformers import AutoModelForCausalLM, GenerationConfig, GPT2Tokenizer, pipeline, set_seed
from trl import DPOConfig, DPOTrainer

from src import config, data_utils, metrics, visualization


def main():
    # CPU throughout, consistent with scripts/01-03 (see their notes on MPS
    # per-op dispatch overhead at this model/batch size).
    device = "cpu"
    print(f"Using device: {device}")
    set_seed(config.SEED)

    tokenizer = GPT2Tokenizer.from_pretrained(config.DPO_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(config.DPO_MODEL_ID)
    base_model.config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(config.DPO_MODEL_ID)
    model.config.use_cache = False

    train_dataset, eval_dataset = data_utils.load_dpo_dataset(config.DPO_TRAIN_SIZE, config.DPO_EVAL_SIZE)
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    peft_config = LoraConfig(**config.DPO_LORA_CONFIG)

    training_args = DPOConfig(
        beta=config.DPO_BETA,
        output_dir=str(config.DPO_OUTPUT_DIR / "checkpoints"),
        num_train_epochs=config.DPO_NUM_EPOCHS,
        per_device_train_batch_size=config.DPO_BATCH_SIZE,
        per_device_eval_batch_size=config.DPO_BATCH_SIZE,
        remove_unused_columns=False,
        logging_steps=5,
        gradient_accumulation_steps=1,
        learning_rate=config.DPO_LEARNING_RATE,
        eval_strategy="epoch",
        warmup_steps=2,
        fp16=False,
        save_strategy="no",
        report_to=[],
        max_length=config.DPO_MAX_LENGTH,
        max_prompt_length=config.DPO_MAX_PROMPT_LENGTH,
        use_cpu=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Training (real local run — the original notebook's main body already did this, but never ran it end to end)...")
    trainer.train()

    eval_entries = [e for e in trainer.state.log_history if "eval_rewards/accuracies" in e]
    final_eval = eval_entries[-1] if eval_entries else {}
    reward_accuracy = final_eval.get("eval_rewards/accuracies")
    reward_margin = final_eval.get("eval_rewards/margins")

    config.DPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.DPO_OUTPUT_DIR))

    visualization.plot_loss_curve(
        trainer.state.log_history, config.FIGURES_DIR / "dpo_loss_curve.png", "Lab 2-2: DPO + LoRA training loss"
    )

    print("Comparing DPO-tuned vs. base GPT-2 generations...")
    generation_config = GenerationConfig(
        do_sample=True, top_k=1, temperature=0.1, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id
    )

    dpo_model = trainer.model
    dpo_model.eval()
    base_model.eval()

    dpo_responses, base_responses = [], []
    for prompt in config.DPO_EVAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        dpo_out = dpo_model.generate(**inputs, generation_config=generation_config)
        base_out = base_model.generate(**inputs, generation_config=generation_config)
        dpo_responses.append(tokenizer.decode(dpo_out[0], skip_special_tokens=True))
        base_responses.append(tokenizer.decode(base_out[0], skip_special_tokens=True))

    sentiment_pipe = pipeline("sentiment-analysis", model=config.PPO_REWARD_MODEL_ID, device=device)
    mean_reward_dpo = metrics.mean_sentiment_reward(dpo_responses, sentiment_pipe, label="POSITIVE")
    mean_reward_base = metrics.mean_sentiment_reward(base_responses, sentiment_pipe, label="POSITIVE")
    print(f"Mean proxy reward — base: {mean_reward_base}, DPO: {mean_reward_dpo}")

    comparison_table = visualization.render_before_after_table(
        config.DPO_EVAL_PROMPTS, base_responses, dpo_responses
    )

    metrics.save_metrics(
        {
            "technique": "Direct Preference Optimization (DPO)",
            "model": config.DPO_MODEL_ID,
            "dataset": config.DPO_DATASET_ID,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "mean_proxy_reward_base": mean_reward_base,
            "mean_proxy_reward_dpo": mean_reward_dpo,
            "eval_reward_accuracy": reward_accuracy,
            "eval_reward_margin": reward_margin,
            "comparison_table": comparison_table,
        },
        config.DPO_METRICS_PATH,
    )
    print(f"Saved metrics to {config.DPO_METRICS_PATH}")


if __name__ == "__main__":
    main()
