import argparse
import json
import math
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = []
    with open(args.log, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    steps = [r["step"] for r in records]
    train_losses = [r["train_loss"] for r in records]
    val_losses = [r["val_loss"] for r in records]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="train_loss")
    plt.plot(steps, val_losses, label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training / Validation Loss")

    run_name = os.path.basename(os.path.dirname(os.path.dirname(args.log)))
    out_path = args.output or os.path.join("experiments", "results", f"{run_name}_loss.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)

    best_idx = min(range(len(records)), key=lambda i: val_losses[i])
    print(f"Final train loss: {train_losses[-1]:.2f}")
    print(f"Final validation loss: {val_losses[-1]:.2f}")
    print(f"Final perplexity: {math.exp(val_losses[-1]):.2f}")
    print(f"Best checkpoint: step {steps[best_idx]} (val_loss {val_losses[best_idx]:.2f})")
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
