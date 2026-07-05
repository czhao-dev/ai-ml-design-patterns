import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.dataset import get_batch
from tinyllm.model import GPT
from tinyllm.utils import get_device, load_checkpoint


@torch.no_grad()
def evaluate(model, valid_path, batch_size, context_length, device, n_batches):
    model.eval()
    losses = torch.zeros(n_batches)
    for i in range(n_batches):
        x, y = get_batch(valid_path, batch_size, context_length, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(**ckpt["config"]["model"]).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    val_loss = evaluate(
        model, args.valid, args.batch_size, ckpt["config"]["model"]["context_length"], device, args.n_batches
    )
    perplexity = math.exp(val_loss)

    print(f"Validation loss: {val_loss:.2f}")
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
