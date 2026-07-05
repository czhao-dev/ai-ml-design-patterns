"""Plotting and reporting helpers shared across the alignment scripts."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(log_history: list, output_path: Path, title: str) -> None:
    """Plots train/eval loss from an HF `Trainer.state.log_history`."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry.get("step", entry.get("epoch")))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", entry.get("epoch")))
            eval_losses.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train_loss", marker="o")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval_loss", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_ppo_reward_curve(all_stats: list, output_path: Path, title: str) -> None:
    """Plots mean reward and KL divergence per PPO step."""
    steps = list(range(len(all_stats)))
    mean_rewards = [stats.get("ppo/mean_scores", 0.0) for stats in all_stats]
    kl = [stats.get("objective/kl", 0.0) for stats in all_stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(steps, mean_rewards, marker="o", color="tab:blue")
    ax1.set_xlabel("PPO step")
    ax1.set_ylabel("Mean reward")
    ax1.set_title(f"{title} — reward")

    ax2.plot(steps, kl, marker="o", color="tab:orange")
    ax2.set_xlabel("PPO step")
    ax2.set_ylabel("KL vs. reference")
    ax2.set_title(f"{title} — KL divergence")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def render_before_after_table(prompts: list, before: list, after: list) -> str:
    """Renders a markdown table comparing before/after generations for a fixed prompt set."""
    lines = ["| Prompt | Before | After |", "|---|---|---|"]
    for prompt, b, a in zip(prompts, before, after):
        clean = lambda s: s.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {clean(prompt)} | {clean(b)} | {clean(a)} |")
    return "\n".join(lines)
