"""Collects the metrics JSON from each of the 4 alignment scripts and renders
a single cross-technique summary figure + console report.

Run this after scripts 01-04 have each completed at least once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src import config, metrics


def main():
    sft = metrics.load_metrics(config.SFT_METRICS_PATH)
    reward = metrics.load_metrics(config.REWARD_METRICS_PATH)
    ppo = metrics.load_metrics(config.PPO_METRICS_PATH)
    dpo = metrics.load_metrics(config.DPO_METRICS_PATH)

    print("=== Lab 1-1: Instruction Fine-Tuning (SFT + LoRA) ===")
    print(f"SacreBLEU: {sft['sacrebleu_base']} -> {sft['sacrebleu_lora']}")

    print("\n=== Lab 1-2: Reward Modeling ===")
    print(f"Pairwise ranking accuracy: {reward['pairwise_ranking_accuracy']} (n={reward['n_eval_pairs']})")

    print("\n=== Lab 2-1: PPO RLHF ===")
    for label, result in ppo["policies"].items():
        print(f"{label}: mean reward {result['mean_reward_before']} -> {result['mean_reward_after']}")

    print("\n=== Lab 2-2: Direct Preference Optimization (DPO) ===")
    print(f"Mean proxy reward: {dpo['mean_proxy_reward_base']} -> {dpo['mean_proxy_reward_dpo']}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.bar(["Base", "SFT+LoRA"], [sft["sacrebleu_base"], sft["sacrebleu_lora"]], color=["gray", "tab:blue"])
    ax.set_title("SFT: SacreBLEU")
    ax.set_ylabel("SacreBLEU")

    ax = axes[0, 1]
    ax.bar(["Reward Model"], [reward["pairwise_ranking_accuracy"]], color="tab:green")
    ax.set_ylim(0, 1)
    ax.set_title("Reward Modeling: Pairwise Accuracy")
    ax.set_ylabel("Accuracy")

    ax = axes[1, 0]
    labels = list(ppo["policies"].keys())
    before = [ppo["policies"][l]["mean_reward_before"] for l in labels]
    after = [ppo["policies"][l]["mean_reward_after"] for l in labels]
    x = range(len(labels))
    ax.bar([i - 0.15 for i in x], before, width=0.3, label="Before", color="gray")
    ax.bar([i + 0.15 for i in x], after, width=0.3, label="After", color="tab:orange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("PPO: Mean Sentiment Reward")
    ax.set_ylabel("Mean reward")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(
        ["Base GPT-2", "DPO"],
        [dpo["mean_proxy_reward_base"], dpo["mean_proxy_reward_dpo"]],
        color=["gray", "tab:purple"],
    )
    ax.set_title("DPO: Mean Proxy Reward")
    ax.set_ylabel("Mean reward")

    plt.tight_layout()
    output_path = config.FIGURES_DIR / "alignment_techniques_summary.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved summary figure to {output_path}")


if __name__ == "__main__":
    main()
