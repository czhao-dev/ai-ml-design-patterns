"""Download and unzip the MovieLens ml-latest-small dataset.

Not committed to the repo (mirrors this project's existing policy of not
redistributing third-party datasets, even though ml-latest-small's license
permits it with attribution -- see README License section). Downloads into
data/movielens/, which is gitignored.

Usage: python scripts/download_movielens.py [--dest data/movielens]
"""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/movielens")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "ml-latest-small" / "ratings.csv").exists():
        print(f"Already downloaded: {dest / 'ml-latest-small'}")
        return

    print(f"Downloading {URL} ...")
    with urlopen(URL) as response:
        data = response.read()

    print(f"Extracting into {dest} ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)

    print(f"Done. Dataset at {dest / 'ml-latest-small'}")
    print("Citation (required by the GroupLens license): F. Maxwell Harper and "
          "Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. "
          "ACM Transactions on Interactive Intelligent Systems 5, 4: 19:1-19:19.")


if __name__ == "__main__":
    main()
