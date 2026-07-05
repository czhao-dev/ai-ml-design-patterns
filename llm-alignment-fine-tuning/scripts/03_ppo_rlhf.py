"""Lab 2-1: RLHF via PPO on lvwerra/gpt2-imdb, steered by a sentiment classifier.

Trains two policies with TRL's PPOTrainer against the same frozen reference
model: one rewarded for positive-sentiment continuations, one for negative-
sentiment continuations, then compares mean reward before/after for each.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from src import config, data_utils, metrics, visualization

GEN_KWARGS = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True}
SENT_KWARGS = {"top_k": None, "function_to_apply": "none", "batch_size": 8}


def train_policy(sentiment_label: str, tokenizer, dataset, sentiment_pipe, device, output_length_sampler):
    """Trains one PPO policy rewarded for `sentiment_label` ("POSITIVE"/"NEGATIVE")."""
    ppo_config = PPOConfig(
        model_name=config.PPO_MODEL_ID,
        learning_rate=config.PPO_LEARNING_RATE,
        batch_size=config.PPO_BATCH_SIZE,
        mini_batch_size=config.PPO_MINI_BATCH_SIZE,
        accelerator_kwargs={"cpu": True},
    )
    # ref_model isn't passed through accelerator.prepare() by PPOTrainer, so it
    # must be moved to `device` explicitly or it silently defaults elsewhere.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.PPO_MODEL_ID).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.PPO_MODEL_ID).to(device)

    ppo_trainer = PPOTrainer(
        ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=data_utils.ppo_collator
    )

    gen_kwargs = {**GEN_KWARGS, "pad_token_id": tokenizer.eos_token_id}
    all_stats = []
    data_iter = iter(ppo_trainer.dataloader)

    for step in tqdm(range(config.PPO_NUM_STEPS), desc=f"PPO ({sentiment_label})"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(ppo_trainer.dataloader)
            batch = next(data_iter)

        query_tensors = batch["input_ids"]
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            gen_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **gen_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **SENT_KWARGS)
        scores = [
            item["score"] for output in pipe_outputs for item in output if item["label"] == sentiment_label
        ]
        rewards = [torch.tensor(s) for s in scores]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        all_stats.append(stats)

    return model, ref_model, all_stats


def compare_before_after(model, ref_model, dataset, tokenizer, sentiment_pipe, device, output_length_sampler, sentiment_label, n_samples=16):
    """Generates from both models on the same queries and compares mean sentiment reward."""
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(n_samples, random_state=config.SEED)
    query_tensors = df_batch["input_ids"].tolist()
    queries = df_batch["query"].tolist()
    dataset.set_format(type="torch")

    gen_kwargs = {**GEN_KWARGS, "pad_token_id": tokenizer.eos_token_id}

    def generate_batch(gen_model):
        texts = []
        for q in query_tensors:
            gen_len = output_length_sampler()
            gen_kwargs["max_new_tokens"] = gen_len
            input_ids = torch.tensor(q).unsqueeze(0)
            output = gen_model.generate(input_ids, **gen_kwargs)
            texts.append(tokenizer.decode(output.squeeze()[-gen_len:], skip_special_tokens=True))
        return texts

    before_responses = generate_batch(ref_model)
    after_responses = generate_batch(model)

    before_texts = [q + r for q, r in zip(queries, before_responses)]
    after_texts = [q + r for q, r in zip(queries, after_responses)]

    mean_reward_before = metrics.mean_sentiment_reward(before_texts, sentiment_pipe, label=sentiment_label)
    mean_reward_after = metrics.mean_sentiment_reward(after_texts, sentiment_pipe, label=sentiment_label)

    return mean_reward_before, mean_reward_after, queries, before_responses, after_responses


def main():
    # CPU throughout: PPO generates one query at a time per step (inherent to the
    # algorithm's variable-length sampling), and per-call MPS dispatch overhead
    # at this model/batch size made the SFT script's equivalent loop >5s/example.
    device = "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.PPO_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = data_utils.build_ppo_dataset(
        tokenizer, config.PPO_DATASET_SIZE, config.PPO_INPUT_MIN_LEN, config.PPO_INPUT_MAX_LEN
    )
    print(f"PPO dataset size: {len(dataset)}")

    sentiment_pipe = pipeline("sentiment-analysis", model=config.PPO_REWARD_MODEL_ID, device=device)
    output_length_sampler = LengthSampler(config.PPO_OUTPUT_MIN_LEN, config.PPO_OUTPUT_MAX_LEN)

    results = {}
    for label, save_dir in [("POSITIVE", config.PPO_POSITIVE_DIR), ("NEGATIVE", config.PPO_NEGATIVE_DIR)]:
        print(f"\n=== Training PPO policy steered toward {label} sentiment ===")
        model, ref_model, all_stats = train_policy(
            label, tokenizer, dataset, sentiment_pipe, device, output_length_sampler
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        visualization.plot_ppo_reward_curve(
            all_stats, config.FIGURES_DIR / f"ppo_{label.lower()}_reward_curve.png", f"PPO ({label})"
        )

        mean_before, mean_after, queries, before_resp, after_resp = compare_before_after(
            model, ref_model, dataset, tokenizer, sentiment_pipe, device, output_length_sampler, label
        )
        sample_table = visualization.render_before_after_table(queries[:3], before_resp[:3], after_resp[:3])

        results[label] = {
            "mean_reward_before": mean_before,
            "mean_reward_after": mean_after,
            "n_ppo_steps": config.PPO_NUM_STEPS,
            "sample_before_after_table": sample_table,
        }
        print(f"{label} mean reward: {mean_before} -> {mean_after}")

    metrics.save_metrics(
        {
            "technique": "PPO RLHF",
            "model": config.PPO_MODEL_ID,
            "dataset": config.PPO_DATASET_ID,
            "dataset_size": len(dataset),
            "policies": results,
        },
        config.PPO_METRICS_PATH,
    )
    print(f"Saved metrics to {config.PPO_METRICS_PATH}")


if __name__ == "__main__":
    main()
