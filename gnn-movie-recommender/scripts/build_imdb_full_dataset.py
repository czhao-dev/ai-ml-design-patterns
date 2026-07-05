"""Convert IMDb's Non-Commercial Datasets (raw TSVs) into the actor/movie/rating
plain-text format expected by src/python/build_actor_movie_dicts.py and friends
(same shape as data/sample/, just at real scale).

Run scripts/download_imdb_full.py first to fetch the four TSVs into
--raw-dir (default data/full_raw/). Output goes to --dest (default data/full/),
which run_all.sh / configs/imdb_full.yaml already expect
(DATA_DIR=data/full RESULTS_DIR=results_full ./run_all.sh).

Filtering (documented, not the original curated sample -- this is real IMDb data
at scale, so the selection criteria matter for reproducibility):
    - titleType == "movie", isAdult == "0", startYear present.
    - numVotes >= --min-votes (default 25) -- keeps ~244K movies out of ~751K
      raw movie-type titles (see the module docstring below for the vote/count
      tradeoff curve); this is the concrete instantiation of the "~254K movies"
      scale the README's configs/imdb_full.yaml already documents.
    - Movies whose "Title (Year)" string collides with another kept movie
      (~1.6% of raw titles do -- IMDb has plenty of same-name-same-year films)
      keep only the higher-vote-count title, since the existing pipeline keys
      everything by that string, not by IMDb's tconst.
    - title.principals only includes "principal" cast/crew (usually the top
      handful of actor/actress credits per title), which is what keeps the
      O(cast^2) actor-actor / movie-movie edge construction in
      build_actor_movie_dicts.py / build_movie_network.py tractable at this
      scale -- there is no separate cast-size cap applied here.
    - Actors/actresses are only written out if they have >=5 kept-movie
      credits, mirroring build_actor_movie_dicts.py's own
      `len(line_split) <= 5` filter -- this just keeps data/full/ smaller;
      the pipeline would silently drop them anyway.
    - director_top100.txt has no equivalent official "top 100" list at this
      scale (the original was hand-curated), so it's approximated here as the
      100 directors with the highest combined numVotes across their kept
      movies -- a popularity/reach proxy, not a quality ranking. Documented as
      a proxy, not a faithful reconstruction of the original list.

Movie "keys" are cleaned with the exact same regex
(`build_actor_movie_dicts.read_movies_file`'s special-char strip +
parenthetical-removal) used internally by the igraph stage, and that cleaned
string is what's written to *every* output file (movie_genre.txt and
movie_rating.txt are read verbatim, with no cleaning applied downstream -- see
src/python/build_movie_genre_index.py / build_movie_rating_index.py -- so they
must already contain the post-cleaning key or genre/rating lookups will miss).

Usage:
    python scripts/build_imdb_full_dataset.py \\
        --raw-dir data/full_raw --dest data/full --min-votes 25 --top-directors 100
"""

import argparse
import csv
import gzip
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

CATEGORIES = {"actor", "actress", "director"}


def clean_movie_field(raw):
    """Mirrors build_actor_movie_dicts.read_movies_file's cleaning, so the
    canonical movie key is identical everywhere it's written."""
    s = re.sub(r"[!@#$&.*']", "", raw)
    s = re.sub(r"\([^0-9)]*\)", "", s)
    s = re.sub(r"{{.*?}}", "", s)
    return s.strip()


def open_tsv(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def load_movies(basics_path):
    """tconst -> (title, year, genre) for real, non-adult movie-type titles."""
    print("(Step 1 of 7) Reading title.basics.tsv.gz (titleType == movie)...")
    movies = {}
    with open_tsv(basics_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            tconst, title_type, primary_title, _orig, is_adult, start_year, _end, _runtime, genres = row[:9]
            if title_type != "movie" or is_adult == "1" or start_year == "\\N":
                continue
            genre = genres.split(",")[0] if genres != "\\N" else "unknown"
            movies[tconst] = (primary_title, start_year, genre)
    print(f"  {len(movies):,} movie-type titles found.")
    return movies


def load_ratings_and_dedupe(ratings_path, movies, min_votes):
    """Filter to numVotes >= min_votes, then dedupe by cleaned "Title (Year)"
    key (keeping the higher-vote title per collision).

    Returns: tconst_to_key (winners only), key_to_info {key: (genre, rating)}.
    """
    print(f"(Step 2 of 7) Reading title.ratings.tsv.gz (numVotes >= {min_votes})...")
    candidates = {}  # tconst -> (key, genre, rating, votes)
    with open_tsv(ratings_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            tconst, avg_rating, num_votes = row[:3]
            info = movies.get(tconst)
            if info is None or int(num_votes) < min_votes:
                continue
            title, year, genre = info
            key = clean_movie_field(f"{title} ({year})")
            candidates[tconst] = (key, genre, float(avg_rating), int(num_votes))
    print(f"  {len(candidates):,} movies pass the vote threshold.")

    print("(Step 3 of 7) Deduping collided \"Title (Year)\" keys (keeping highest-vote title)...")
    best_by_key = {}  # key -> (tconst, votes)
    for tconst, (key, _genre, _rating, votes) in candidates.items():
        current = best_by_key.get(key)
        if current is None or votes > current[1]:
            best_by_key[key] = (tconst, votes)

    tconst_to_key = {tconst: key for key, (tconst, _votes) in best_by_key.items()}
    key_to_info = {}
    for key, (tconst, votes) in best_by_key.items():
        _key, genre, rating, _votes = candidates[tconst]
        key_to_info[key] = (genre, rating, votes)
    dropped = len(candidates) - len(tconst_to_key)
    print(f"  {len(tconst_to_key):,} unique movies kept ({dropped:,} lower-vote duplicates dropped).")
    return tconst_to_key, key_to_info


def load_credits(principals_path, tconst_to_key):
    """nconst -> [movie keys] per category, deduped, movie-set filtered."""
    print("(Step 4 of 7) Streaming title.principals.tsv.gz (actor/actress/director credits)...")
    credits = {"actor": defaultdict(list), "actress": defaultdict(list), "director": defaultdict(list)}
    seen = defaultdict(set)  # (category, nconst) -> set of keys already recorded
    n_rows = 0
    with open_tsv(principals_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            n_rows += 1
            if n_rows % 20_000_000 == 0:
                print(f"  ...{n_rows:,} principal rows scanned")
            if len(row) < 4:
                continue
            tconst, _ordering, nconst, category = row[0], row[1], row[2], row[3]
            if category not in CATEGORIES:
                continue
            key = tconst_to_key.get(tconst)
            if key is None:
                continue
            seen_set = seen[(category, nconst)]
            if key in seen_set:
                continue
            seen_set.add(key)
            credits[category][nconst].append(key)
    for category in CATEGORIES:
        print(f"  {category}: {len(credits[category]):,} distinct people credited.")
    return credits["actor"], credits["actress"], credits["director"]


def resolve_names(name_basics_path, needed_nconsts):
    print(f"(Step 5 of 7) Resolving {len(needed_nconsts):,} person names from name.basics.tsv.gz...")
    names = {}
    with open_tsv(name_basics_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            nconst, primary_name = row[0], row[1]
            if nconst in needed_nconsts:
                names[nconst] = primary_name
    print(f"  {len(names):,} names resolved.")
    return names


def write_person_movies_file(path, credits, names, min_movies=1):
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for nconst, movie_keys in credits.items():
            if len(movie_keys) < min_movies:
                continue
            name = names.get(nconst)
            if name is None:
                continue
            f.write("\t".join([name, *movie_keys]) + "\n")
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/full_raw")
    parser.add_argument("--dest", default="data/full")
    parser.add_argument("--min-votes", type=int, default=25)
    parser.add_argument("--top-directors", type=int, default=100)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name in ["title.basics.tsv.gz", "title.ratings.tsv.gz", "name.basics.tsv.gz", "title.principals.tsv.gz"]:
        if not (raw_dir / name).exists():
            sys.exit(f"Missing {raw_dir / name} -- run scripts/download_imdb_full.py first.")

    start = time.perf_counter()

    movies = load_movies(raw_dir / "title.basics.tsv.gz")
    tconst_to_key, key_to_info = load_ratings_and_dedupe(raw_dir / "title.ratings.tsv.gz", movies, args.min_votes)
    del movies

    actor_credits, actress_credits, director_credits = load_credits(raw_dir / "title.principals.tsv.gz", tconst_to_key)

    # Mirror build_actor_movie_dicts.py's own >=5-movie filter for actors/actresses
    # (directors are not filtered this way anywhere in the pipeline).
    actor_credits = {n: m for n, m in actor_credits.items() if len(m) >= 5}
    actress_credits = {n: m for n, m in actress_credits.items() if len(m) >= 5}

    needed_nconsts = set(actor_credits) | set(actress_credits) | set(director_credits)
    names = resolve_names(raw_dir / "name.basics.tsv.gz", needed_nconsts)

    print("(Step 6 of 7) Writing actor_movies.txt / actress_movies.txt / director_movies.txt / director_top100.txt...")
    n_actors = write_person_movies_file(dest / "actor_movies.txt", actor_credits, names)
    n_actresses = write_person_movies_file(dest / "actress_movies.txt", actress_credits, names)
    n_directors = write_person_movies_file(dest / "director_movies.txt", director_credits, names)

    # director_top100.txt: proxy popularity ranking (see module docstring).
    director_votes = {}
    for nconst, movie_keys in director_credits.items():
        if nconst not in names:
            continue
        director_votes[nconst] = sum(key_to_info[k][2] for k in movie_keys)
    top_directors = sorted(director_votes, key=director_votes.get, reverse=True)[: args.top_directors]
    with open(dest / "director_top100.txt", "w", encoding="utf-8") as f:
        for nconst in top_directors:
            f.write(names[nconst] + "\n")

    print(f"  actor_movies.txt: {n_actors:,} actors, actress_movies.txt: {n_actresses:,} actresses, "
          f"director_movies.txt: {n_directors:,} directors, director_top100.txt: {len(top_directors)} directors.")

    print("(Step 7 of 7) Writing movie_genre.txt / movie_rating.txt...")
    with open(dest / "movie_genre.txt", "w", encoding="utf-8") as genre_f, \
            open(dest / "movie_rating.txt", "w", encoding="utf-8") as rating_f:
        for key, (genre, rating, _votes) in key_to_info.items():
            genre_f.write(f"{key}\t{genre}\n")
            rating_f.write(f"{key}\t\t{rating}\n")

    elapsed = time.perf_counter() - start
    print(f"\nDone in {elapsed:.1f}s. Wrote {dest}/ "
          f"({len(key_to_info):,} labeled movies, {n_actors:,} actors, {n_actresses:,} actresses, "
          f"{n_directors:,} directors).")
    print("Citation/license: derived from IMDb Non-Commercial Datasets "
          "(https://developer.imdb.com/non-commercial-datasets/), personal/non-commercial use only. "
          "Do not commit data/full/ or redistribute it.")


if __name__ == "__main__":
    main()
