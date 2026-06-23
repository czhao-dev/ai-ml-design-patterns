"""Download the TinyStories dataset (roneneldan/TinyStories) into data/raw/.

By default downloads a truncated subset of train.txt (fast iteration). Pass
--full to download the complete dataset instead.
"""
import argparse
import os

from huggingface_hub import hf_hub_download

REPO_ID = "roneneldan/TinyStories"
DELIMITER = "<|endoftext|>"


def truncate_stories(text: str, max_stories: int) -> str:
    stories = text.split(DELIMITER)
    # split() leaves a trailing empty string after the final delimiter; drop it.
    stories = [s for s in stories if s.strip()]
    kept = stories[:max_stories]
    return DELIMITER.join(kept) + DELIMITER


def fetch_split(filename: str, out_path: str, max_stories: int | None) -> None:
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    with open(downloaded_path, "r", encoding="utf-8") as f:
        text = f.read()
    if max_stories is not None:
        text = truncate_stories(text, max_stories)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    n_stories = text.count(DELIMITER)
    print(f"Wrote {out_path} ({len(text) / 1e6:.1f} MB, {n_stories} stories)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-stories", type=int, default=75000)
    parser.add_argument("--max-valid-stories", type=int, default=3000)
    parser.add_argument("--full", action="store_true", help="Download the complete dataset, no truncation.")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    max_train = None if args.full else args.max_train_stories
    max_valid = None if args.full else args.max_valid_stories

    fetch_split("TinyStories-train.txt", os.path.join(args.output_dir, "train.txt"), max_train)
    fetch_split("TinyStories-valid.txt", os.path.join(args.output_dir, "valid.txt"), max_valid)


if __name__ == "__main__":
    main()
