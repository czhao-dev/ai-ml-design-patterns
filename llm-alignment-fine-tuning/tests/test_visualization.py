"""Tests for src/visualization.py."""

import matplotlib

matplotlib.use("Agg")

from src.visualization import plot_loss_curve, plot_ppo_reward_curve, render_before_after_table


def test_plot_loss_curve_writes_a_file_with_train_and_eval_entries(tmp_path):
    log_history = [
        {"step": 1, "loss": 1.0},
        {"step": 1, "eval_loss": 1.1},
        {"step": 2, "loss": 0.8},
        {"step": 2, "eval_loss": 0.9},
    ]
    output_path = tmp_path / "figures" / "loss.png"

    plot_loss_curve(log_history, output_path, title="Test Loss Curve")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_loss_curve_handles_train_only_history(tmp_path):
    log_history = [{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}]
    output_path = tmp_path / "loss.png"

    plot_loss_curve(log_history, output_path, title="Train Only")

    assert output_path.exists()


def test_plot_ppo_reward_curve_writes_a_file(tmp_path):
    all_stats = [
        {"ppo/mean_scores": 0.1, "objective/kl": 2.0},
        {"ppo/mean_scores": 0.3, "objective/kl": 1.5},
    ]
    output_path = tmp_path / "ppo_reward.png"

    plot_ppo_reward_curve(all_stats, output_path, title="PPO Reward")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_before_after_table_has_header_and_one_row_per_prompt():
    table = render_before_after_table(
        prompts=["Tell me a joke"], before=["old response"], after=["new response"]
    )

    lines = table.splitlines()
    assert lines[0] == "| Prompt | Before | After |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| Tell me a joke | old response | new response |"


def test_render_before_after_table_escapes_pipes_and_newlines():
    table = render_before_after_table(prompts=["a | b\nc"], before=["x"], after=["y"])

    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[2] == "| a \\| b c | x | y |"
