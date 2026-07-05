"""Lab 1-2: Reward modeling on GPT-2 + LoRA.

Trains a scalar reward model on Dahoas/synthetic-instruct-gptj-pairwise chosen/
rejected pairs using TRL's RewardTrainer, then evaluates pairwise ranking
accuracy on a held-out split.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from trl import RewardConfig, RewardTrainer

from src import config, data_utils, metrics, visualization


def main():
    # CPU throughout: per-op MPS dispatch overhead at this model/batch size made
    # the SFT script's per-example forward passes >5s/example (see scripts/01).
    device = "cpu"
    print(f"Using device: {device}")

    dataset = load_dataset(config.REWARD_DATASET_ID)
    dataset["train"] = dataset["train"].map(data_utils.add_combined_columns)

    max_length = config.REWARD_MAX_LENGTH
    short_indices = [
        i
        for i, (chosen, rejected) in enumerate(
            zip(dataset["train"]["prompt_chosen"], dataset["train"]["prompt_rejected"])
        )
        if len(chosen) < max_length or len(rejected) < max_length
    ]
    dataset["train"] = dataset["train"].select(short_indices)

    tokenizer = GPT2Tokenizer.from_pretrained(config.REWARD_MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(config.REWARD_MODEL_ID, num_labels=1)
    model.config.pad_token_id = model.config.eos_token_id

    preprocess_function = data_utils.build_reward_preprocess_fn(tokenizer, max_length)
    dataset["train"] = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["prompt", "chosen", "rejected", "prompt_chosen", "prompt_rejected"],
    )

    split = dataset["train"].train_test_split(test_size=0.2, seed=config.SEED)
    train_dataset = data_utils.subsample_dataset(split["train"], config.REWARD_TRAIN_SIZE)
    eval_dataset = data_utils.subsample_dataset(split["test"], config.REWARD_EVAL_SIZE)
    dataset_dict = DatasetDict({"train": train_dataset, "test": eval_dataset})
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    peft_config = LoraConfig(**config.REWARD_LORA_CONFIG, inference_mode=False)

    training_args = RewardConfig(
        output_dir=str(config.REWARD_OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=config.REWARD_BATCH_SIZE,
        per_device_eval_batch_size=config.REWARD_BATCH_SIZE,
        num_train_epochs=config.REWARD_NUM_EPOCHS,
        learning_rate=config.REWARD_LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        max_length=max_length,
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        peft_config=peft_config,
    )

    print("Training (real local run, replacing the original notebook's commented-out trainer.train())...")
    trainer.train()

    config.REWARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.REWARD_OUTPUT_DIR))

    visualization.plot_loss_curve(
        trainer.state.log_history,
        config.FIGURES_DIR / "reward_loss_curve.png",
        "Lab 1-2: Reward model training loss",
    )

    print("Evaluating pairwise ranking accuracy on held-out chosen/rejected pairs...")
    model.eval()
    model.to(device)

    eval_raw = data_utils.subsample_dataset(
        load_dataset(config.REWARD_DATASET_ID, split="train"), config.REWARD_EVAL_SIZE, seed=config.SEED + 1
    )

    def score_batch(texts: list, batch_size: int = 16) -> list:
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(-1)
            scores.extend(logits.tolist())
        return scores

    chosen_texts = ["\n\nHuman: " + row["prompt"] + "\n\nAssistant: " + row["chosen"] for row in eval_raw]
    rejected_texts = ["\n\nHuman: " + row["prompt"] + "\n\nAssistant: " + row["rejected"] for row in eval_raw]
    chosen_scores = score_batch(chosen_texts)
    rejected_scores = score_batch(rejected_texts)

    accuracy = metrics.pairwise_ranking_accuracy(chosen_scores, rejected_scores)
    print(f"Pairwise ranking accuracy: {accuracy}")

    metrics.save_metrics(
        {
            "technique": "Reward Modeling",
            "model": config.REWARD_MODEL_ID,
            "dataset": config.REWARD_DATASET_ID,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "pairwise_ranking_accuracy": accuracy,
            "n_eval_pairs": len(chosen_scores),
        },
        config.REWARD_METRICS_PATH,
    )
    print(f"Saved metrics to {config.REWARD_METRICS_PATH}")


if __name__ == "__main__":
    main()
