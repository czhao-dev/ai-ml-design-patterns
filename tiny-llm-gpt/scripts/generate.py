import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.generation import generate
from tinyllm.model import GPT
from tinyllm.tokenizer import Tokenizer
from tinyllm.utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    tokenizer = Tokenizer.load(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(**ckpt["config"]["model"]).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        context_length=ckpt["config"]["model"]["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tokenizer.eos_id,
    )

    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
