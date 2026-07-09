"""Tests for src/data_utils.py.

Functions that hit the network or the Hugging Face Hub (download_codealpaca,
build_ppo_dataset, load_dpo_dataset) are out of scope for these offline
unit tests -- everything else is pure data transformation and is exercised
directly.
"""

from __future__ import annotations

from datasets import Dataset

from src.data_utils import (
    add_combined_columns,
    build_reward_preprocess_fn,
    formatting_prompts_func,
    formatting_prompts_func_no_response,
    ppo_collator,
    process_dpo_row,
    subsample_dataset,
)


def test_subsample_dataset_selects_requested_size():
    ds = Dataset.from_dict({"x": list(range(100))})
    sub = subsample_dataset(ds, n=10, seed=0)
    assert len(sub) == 10


def test_subsample_dataset_clamps_to_dataset_length():
    ds = Dataset.from_dict({"x": list(range(5))})
    sub = subsample_dataset(ds, n=100, seed=0)
    assert len(sub) == 5


def test_subsample_dataset_is_deterministic_for_a_fixed_seed():
    ds = Dataset.from_dict({"x": list(range(50))})
    first = subsample_dataset(ds, n=10, seed=7)
    second = subsample_dataset(ds, n=10, seed=7)
    assert first["x"] == second["x"]


def test_formatting_prompts_func_includes_instruction_and_response():
    batch = {"instruction": ["Write a haiku"], "output": ["old pond / frog jumps in"]}
    formatted = formatting_prompts_func(batch)
    assert formatted == ["### Instruction:\nWrite a haiku\n\n### Response:\nold pond / frog jumps in</s>"]


def test_formatting_prompts_func_no_response_omits_the_answer():
    batch = {"instruction": ["Write a haiku"]}
    formatted = formatting_prompts_func_no_response(batch)
    assert formatted == ["### Instruction:\nWrite a haiku\n\n### Response:\n"]
    assert "old pond" not in formatted[0]


def test_add_combined_columns_builds_human_assistant_transcripts():
    example = {"prompt": "How do I bake bread?", "chosen": "Mix flour and water.", "rejected": "I don't know."}

    result = add_combined_columns(dict(example))

    assert result["prompt_chosen"] == "\n\nHuman: How do I bake bread?\n\nAssistant: Mix flour and water."
    assert result["prompt_rejected"] == "\n\nHuman: How do I bake bread?\n\nAssistant: I don't know."


class _FakeTokenizer:
    """Mimics the small slice of the HF tokenizer API used by the preprocess fn."""

    def __call__(self, texts, truncation=True, max_length=8, padding="max_length"):
        input_ids = [[len(t)] + [0] * (max_length - 1) for t in texts]
        attention_mask = [[1] + [0] * (max_length - 1) for _ in texts]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_build_reward_preprocess_fn_tokenizes_chosen_and_rejected():
    preprocess = build_reward_preprocess_fn(_FakeTokenizer(), max_length=8)
    examples = {"prompt_chosen": ["a chosen reply"], "prompt_rejected": ["a rejected reply"]}

    result = preprocess(examples)

    assert set(result) == {
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
    }
    assert len(result["input_ids_chosen"][0]) == 8


def test_ppo_collator_groups_by_key():
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert ppo_collator(data) == {"a": [1, 2], "b": ["x", "y"]}


def test_process_dpo_row_extracts_last_message_content():
    row = {
        "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "final chosen"}],
        "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "final rejected"}],
    }

    result = process_dpo_row(row)

    assert result["chosen"] == "final chosen"
    assert result["rejected"] == "final rejected"
