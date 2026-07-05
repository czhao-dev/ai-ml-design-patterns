"""Dataset loading and formatting helpers for each alignment lab."""

import multiprocessing
from pathlib import Path

import requests
from datasets import Dataset, load_dataset

from src import config


def subsample_dataset(ds: Dataset, n: int, seed: int = config.SEED) -> Dataset:
    """Shuffle and select at most `n` rows."""
    n = min(n, len(ds))
    return ds.shuffle(seed=seed).select(range(n))


# ---------------------------------------------------------------------------
# Lab 1-1: Instruction fine-tuning (CodeAlpaca-20k)
# ---------------------------------------------------------------------------
def download_codealpaca(
    cache_path: Path = config.SFT_DATASET_CACHE, url: str = config.SFT_DATASET_URL
) -> Path:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return cache_path


def formatting_prompts_func(batch: dict) -> list:
    """Builds '### Instruction / ### Response' training strings (with answer)."""
    return [
        f"### Instruction:\n{instruction}\n\n### Response:\n{output}</s>"
        for instruction, output in zip(batch["instruction"], batch["output"])
    ]


def formatting_prompts_func_no_response(batch: dict) -> list:
    """Same template, but truncated right before the answer (used for generation)."""
    return [
        f"### Instruction:\n{instruction}\n\n### Response:\n" for instruction in batch["instruction"]
    ]


# ---------------------------------------------------------------------------
# Lab 1-2: Reward modeling (Dahoas/synthetic-instruct-gptj-pairwise)
# ---------------------------------------------------------------------------
def add_combined_columns(example: dict) -> dict:
    example["prompt_chosen"] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["chosen"]
    example["prompt_rejected"] = (
        "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["rejected"]
    )
    return example


def build_reward_preprocess_fn(tokenizer, max_length: int):
    def preprocess_function(examples):
        tokenized_chosen = tokenizer(
            examples["prompt_chosen"], truncation=True, max_length=max_length, padding="max_length"
        )
        tokenized_rejected = tokenizer(
            examples["prompt_rejected"], truncation=True, max_length=max_length, padding="max_length"
        )
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    return preprocess_function


# ---------------------------------------------------------------------------
# Lab 2-1: PPO RLHF (IMDB)
# ---------------------------------------------------------------------------
def build_ppo_dataset(tokenizer, dataset_size: int, input_min_len: int, input_max_len: int):
    from trl.core import LengthSampler

    ds = load_dataset(config.PPO_DATASET_ID, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    ds = subsample_dataset(ds, dataset_size)

    input_size = LengthSampler(input_min_len, input_max_len)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def ppo_collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# ---------------------------------------------------------------------------
# Lab 2-2: DPO (BarraHome/ultrafeedback_binarized)
# ---------------------------------------------------------------------------
def process_dpo_row(row: dict) -> dict:
    row["chosen"] = row["chosen"][-1]["content"]
    row["rejected"] = row["rejected"][-1]["content"]
    return row


def load_dpo_dataset(train_size: int, eval_size: int):
    ds = load_dataset(config.DPO_DATASET_ID)
    train_dataset = subsample_dataset(ds["train_prefs"], train_size)
    eval_dataset = subsample_dataset(ds["test_prefs"], eval_size)

    columns_to_drop = ["prompt_id", "messages", "score_chosen", "score_rejected"]
    train_dataset = train_dataset.map(
        process_dpo_row, num_proc=multiprocessing.cpu_count(), remove_columns=columns_to_drop
    )
    eval_dataset = eval_dataset.map(
        process_dpo_row, num_proc=multiprocessing.cpu_count(), remove_columns=columns_to_drop
    )
    return train_dataset, eval_dataset
