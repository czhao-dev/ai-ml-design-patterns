"""Evaluation metrics shared across the alignment scripts."""

import json
from pathlib import Path

import evaluate


def compute_sacrebleu(predictions: list, references: list) -> float:
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references)
    return round(results["score"], 2)


def pairwise_ranking_accuracy(chosen_scores: list, rejected_scores: list) -> float:
    """Fraction of pairs where the reward model scores the chosen response higher."""
    correct = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r)
    return round(correct / len(chosen_scores), 4)


def mean_sentiment_reward(texts: list, sentiment_pipe, label: str = "POSITIVE") -> float:
    """Mean sentiment-classifier score for `label` across a batch of texts."""
    outputs = sentiment_pipe(texts, top_k=None, function_to_apply="none", batch_size=8)
    scores = [item["score"] for output in outputs for item in output if item["label"] == label]
    return round(sum(scores) / len(scores), 4)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
