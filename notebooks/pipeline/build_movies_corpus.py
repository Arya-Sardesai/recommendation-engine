"""
build_movies_corpus.py

Reads the TMDB 930k Kaggle dump (TMDB_movie_dataset_v11.csv), filters to
embeddable, released films with real overviews, and writes movies.parquet.

Run from REPO ROOT:
    python notebooks/pipeline/build_movies_corpus.py

Mirrors the book pipeline: produces a clean corpus parquet ready for embedding.
The 637MB CSV is read in chunks to stay within RAM.
"""

import sys
from pathlib import Path
import pandas as pd

# --- paths (repo-root relative) ---
ROOT = Path(__file__).parent.parent.parent
RAW = ROOT / "data" / "raw" / "movies" / "TMDB_movie_dataset_v11.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "movies.parquet"

# --- filter thresholds ---
# English films need more votes to clear noise; non-English get a lower bar
# so regional cinema (Bollywood, Korean, French) survives.
VOTE_FLOOR_EN = 20
VOTE_FLOOR_OTHER = 8
MIN_OVERVIEW_CHARS = 30  # drop stub/one-word overviews

# Columns we actually keep. Everything else (backdrop, budget, revenue,
# homepage, production_*) is dropped to keep the parquet lean.
KEEP = [
    "id", "title", "original_title", "original_language",
    "release_date", "vote_average", "vote_count", "runtime",
    "overview", "tagline", "genres", "keywords", "poster_path",
]

USECOLS = [
    "id", "title", "original_title", "original_language", "status",
    "release_date", "vote_average", "vote_count", "runtime", "adult",
    "overview", "tagline", "genres", "keywords", "poster_path",
]


def passes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all row filters to a chunk and return the surviving rows."""
    # released only
    df = df[df["status"] == "Released"]
    # not adult
    df = df[df["adult"] == False]  # noqa: E712  (pandas bool col)
    # real overview
    df = df[df["overview"].notna()]
    df = df[df["overview"].str.len() >= MIN_OVERVIEW_CHARS]
    # vote floor, language-dependent
    is_en = df["original_language"] == "en"
    keep_en = is_en & (df["vote_count"] >= VOTE_FLOOR_EN)
    keep_other = (~is_en) & (df["vote_count"] >= VOTE_FLOOR_OTHER)
    df = df[keep_en | keep_other]
    return df


def main():
    if not RAW.exists():
        sys.exit(f"ERROR: {RAW} not found. Run from repo root after extracting the TMDB zip.")

    print(f"Reading {RAW.name} in chunks ...")
    chunks = []
    total_in = 0
    reader = pd.read_csv(
        RAW,
        usecols=USECOLS,
        chunksize=100_000,
        dtype={"id": "int64", "vote_count": "float64", "vote_average": "float64"},
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        total_in += len(chunk)
        kept = passes(chunk)
        chunks.append(kept)
        print(f"  chunk {i:>2}: read {len(chunk):>7,}  kept {len(kept):>6,}  (running in: {total_in:,})")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal read: {total_in:,}  ->  after filter: {len(df):,}")

    # de-dupe on TMDB id (dumps occasionally have dupes)
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    if before != len(df):
        print(f"Dropped {before - len(df):,} duplicate ids")

    # derive release_year for agent year-filters
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.astype("Int64")

    # final column selection
    df = df[KEEP + ["release_year"]]

    # vote_count back to int now that NaNs are filtered out
    df["vote_count"] = df["vote_count"].astype("int64")

    df.to_parquet(OUT, index=False)
    print(f"\nWrote {len(df):,} films -> {OUT}")

    # quick sanity readout
    print("\n--- sanity ---")
    print("Top languages:")
    print(df["original_language"].value_counts().head(10).to_string())
    print(f"\nFilms with non-null genres: {df['genres'].notna().sum():,}")
    print(f"Median vote_count: {df['vote_count'].median():.0f}")
    print(f"Year range: {df['release_year'].min()} - {df['release_year'].max()}")


if __name__ == "__main__":
    main()