import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.dataset import get_batch
from tinyllm.model import GPT
from tinyllm.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed


def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return peak_lr * 0.1
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * 0.1 + coeff * peak_lr * 0.9


def build_optimizer(model, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, data_path, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data_path, batch_size, context_length, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps-override", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    m, t, d, o = cfg["model"], cfg["training"], cfg["data"], cfg["output"]

    set_seed(t["seed"])
    device = get_device(args.device)
    print(f"device: {device}")

    model = GPT(**m).to(device)
    optimizer = build_optimizer(model, t["learning_rate"], t["weight_decay"])

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=device)
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    os.makedirs(o["checkpoint_dir"], exist_ok=True)
    os.makedirs(o["log_dir"], exist_ok=True)
    log_path = os.path.join(o["log_dir"], "train_log.jsonl")
    log_file = open(log_path, "a")

    max_steps = args.max_steps_override or t["max_steps"]
    best_val_loss = float("inf")

    t0 = time.time()
    tokens_processed = 0
    for step in range(start_step, max_steps):
        lr = get_lr(step, t["warmup_steps"], max_steps, t["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(d["train_path"], t["batch_size"], m["context_length"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()
        tokens_processed += t["batch_size"] * m["context_length"]

        if step % t["eval_interval"] == 0 or step == max_steps - 1:
            val_loss = estimate_loss(
                model, d["valid_path"], t["batch_size"], m["context_length"], device, t["eval_iters"]
            )
            elapsed = time.time() - t0
            tps = tokens_processed / elapsed if elapsed > 0 else 0.0
            record = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "lr": lr,
                "tokens_per_sec": tps,
                "timestamp": time.time(),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(f"step {step}: train_loss {loss.item():.4f}, val_loss {val_loss:.4f}, lr {lr:.2e}, tok/s {tps:.0f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(o["checkpoint_dir"], "best.pt"), model, optimizer, step, cfg)

        if step % t["checkpoint_interval"] == 0 or step == max_steps - 1:
            save_checkpoint(os.path.join(o["checkpoint_dir"], "latest.pt"), model, optimizer, step, cfg)

    log_file.close()
    print("training complete")


if __name__ == "__main__":
    main()
