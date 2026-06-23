import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tinyllm.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.train(args.input, args.vocab_size)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)

    print(f"Tokenizer saved to {args.output}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
