"""Tests for src/metrics.py.

`compute_sacrebleu` downloads a metric script from the Hugging Face Hub on
first use and is out of scope for these offline unit tests.
"""

from __future__ import annotations

from src.metrics import load_metrics, mean_sentiment_reward, pairwise_ranking_accuracy, save_metrics


def test_pairwise_ranking_accuracy_all_correct():
    assert pairwise_ranking_accuracy([1.0, 2.0, 3.0], [0.5, 1.0, 1.5]) == 1.0


def test_pairwise_ranking_accuracy_all_wrong():
    assert pairwise_ranking_accuracy([0.5, 1.0], [1.0, 2.0]) == 0.0


def test_pairwise_ranking_accuracy_mixed_and_rounded():
    # 2 of 3 pairs have chosen > rejected -> 0.6667, rounded to 4 places.
    result = pairwise_ranking_accuracy([1.0, 2.0, 1.0], [0.5, 3.0, 0.5])
    assert result == 0.6667


class _FakeSentimentPipe:
    """Mimics a HF `pipeline("sentiment-analysis", top_k=None)` callable."""

    def __call__(self, texts, top_k=None, function_to_apply="none", batch_size=8):
        return [
            [{"label": "POSITIVE", "score": 1.0 + i}, {"label": "NEGATIVE", "score": -1.0 - i}]
            for i, _ in enumerate(texts)
        ]


def test_mean_sentiment_reward_averages_requested_label():
    pipe = _FakeSentimentPipe()
    result = mean_sentiment_reward(["a", "b", "c"], pipe, label="POSITIVE")
    # scores are 1.0, 2.0, 3.0 -> mean 2.0
    assert result == 2.0


def test_mean_sentiment_reward_can_target_negative_label():
    pipe = _FakeSentimentPipe()
    result = mean_sentiment_reward(["a", "b"], pipe, label="NEGATIVE")
    # scores are -1.0, -2.0 -> mean -1.5
    assert result == -1.5


def test_save_and_load_metrics_round_trip(tmp_path):
    metrics = {"accuracy": 0.987654321, "loss": 0.1}
    path = tmp_path / "reports" / "metrics.json"

    save_metrics(metrics, path)
    loaded = load_metrics(path)

    assert loaded == metrics
    assert path.exists()
