import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.tokenizer import Tokenizer

CHUNK_LINES = 10000


def encode_file_to_bin(tokenizer: Tokenizer, text_path: str, out_path: str) -> int:
    eos_id = tokenizer.eos_id
    token_chunks = []
    total_tokens = 0

    with open(text_path, "r", encoding="utf-8") as f:
        buffer_lines = []
        for line in tqdm(f, desc=f"encoding {os.path.basename(text_path)}"):
            buffer_lines.append(line)
            if len(buffer_lines) >= CHUNK_LINES:
                ids = tokenizer.encode("".join(buffer_lines))
                token_chunks.append(np.array(ids, dtype=np.uint16))
                total_tokens += len(ids)
                buffer_lines = []
        if buffer_lines:
            ids = tokenizer.encode("".join(buffer_lines))
            token_chunks.append(np.array(ids, dtype=np.uint16))
            total_tokens += len(ids)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    offset = 0
    for chunk in token_chunks:
        arr[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    arr.flush()
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.tokenizer)

    train_out = os.path.join(args.output, "train.bin")
    valid_out = os.path.join(args.output, "valid.bin")

    n_train = encode_file_to_bin(tokenizer, args.train, train_out)
    n_valid = encode_file_to_bin(tokenizer, args.valid, valid_out)

    print(f"train.bin: {n_train} tokens -> {train_out}")
    print(f"valid.bin: {n_valid} tokens -> {valid_out}")


if __name__ == "__main__":
    main()
