import argparse
import os
import resource
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.model import GPT
from tinyllm.utils import load_checkpoint

N_WARMUP = 5
N_ITERS = 20


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    # ru_maxrss is bytes on macOS, KB on Linux.
    divisor = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def bench_device(model_cfg, checkpoint, device_name, batch_size, context_length):
    device = torch.device(device_name)
    model = GPT(**model_cfg).to(device)
    if checkpoint:
        load_checkpoint(checkpoint, model, device=device)
    model.eval()

    x = torch.randint(0, model_cfg["vocab_size"], (batch_size, context_length), device=device)
    y = torch.randint(0, model_cfg["vocab_size"], (batch_size, context_length), device=device)

    for _ in range(N_WARMUP):
        logits, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        with torch.no_grad():
            model(x)
    sync(device)
    infer_elapsed = time.perf_counter() - t0
    infer_tps = (batch_size * context_length * N_ITERS) / infer_elapsed

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    sync(device)
    train_elapsed = time.perf_counter() - t0
    train_tps = (batch_size * context_length * N_ITERS) / train_elapsed

    mem = peak_memory_mb(device)
    return infer_tps, train_tps, mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()

    model_cfg = yaml.safe_load(open(args.config))["model"]
    model_cfg["context_length"] = args.context_length

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    print(f"| Device  | Context Length | Inference Tok/s | Train Tok/s | Peak Memory (MB) |")
    print(f"| ------- | --------------: | ---------------: | -----------: | ----------------: |")
    for dev in devices:
        infer_tps, train_tps, mem = bench_device(
            model_cfg, args.checkpoint, dev, args.batch_size, args.context_length
        )
        print(f"| {dev:7s} | {args.context_length:14d} | {infer_tps:16.0f} | {train_tps:11.0f} | {mem:17.1f} |")


if __name__ == "__main__":
    main()
