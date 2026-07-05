"""Download IMDb's official "Non-Commercial Datasets" TSVs.

Source: https://developer.imdb.com/non-commercial-datasets/ (updated daily,
free for personal/non-commercial use). Not committed to the repo -- mirrors
this project's existing policy of not redistributing IMDb/MovieLens data
(see README License section). Downloads into data/full_raw/, which is
gitignored.

This is the raw-data step only. Run scripts/build_imdb_full_dataset.py
afterwards to convert these TSVs into the data/full/ layout the igraph
stage (run_all.sh) expects.

Usage: python scripts/download_imdb_full.py [--dest data/full_raw]
"""

import argparse
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://datasets.imdbws.com"
FILES = ["title.basics.tsv.gz", "title.ratings.tsv.gz", "name.basics.tsv.gz", "title.principals.tsv.gz"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/full_raw")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        out_path = dest / filename
        if out_path.exists():
            print(f"Already downloaded: {out_path}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url} ...")
        with urlopen(url) as response, open(out_path, "wb") as out_file:
            while chunk := response.read(1 << 20):
                out_file.write(chunk)
        print(f"  -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("Done.")
    print("Citation/attribution: IMDb Non-Commercial Datasets, "
          "https://developer.imdb.com/non-commercial-datasets/ -- for personal, "
          "non-commercial use only. Do not commit these files or redistribute them.")


if __name__ == "__main__":
    main()
