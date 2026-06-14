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
import numpy as np
import pandas as pd

# --- paths (repo-root relative) ---
ROOT = Path(__file__).parent.parent.parent
RAW = ROOT / "data" / "raw" / "movies" / "TMDB_movie_dataset_v11.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "movies.parquet"

# --- filter thresholds (tiered) ---
# Clean tier: kept unconditionally (high vote count is its own quality signal).
VOTE_CLEAN_EN = 20
VOTE_CLEAN_OTHER = 8
# Probationary tier: lower-vote films, kept ONLY if they clear the quality
# gates below. This is how we add coverage without adding the noise.
VOTE_FLOOR_EN = 8
VOTE_FLOOR_OTHER = 3

MIN_OVERVIEW_CHARS = 30          # base: any kept film needs at least this
PROBATION_OVERVIEW_CHARS = 100   # probationary films need a fuller overview
PROBATION_MIN_RUNTIME = 40       # feature-length floor (mins); kills shorts/episodes

# Columns we actually keep. Everything else (backdrop, budget, revenue,
# homepage, production_*) is dropped to keep the parquet lean.
KEEP = [
    "id", "imdb_id", "title", "original_title", "original_language",
    "release_date", "vote_average", "vote_count", "runtime",
    "overview", "tagline", "genres", "keywords", "poster_path", "studio",
]

USECOLS = [
    "id", "imdb_id", "title", "original_title", "original_language", "status",
    "release_date", "vote_average", "vote_count", "runtime", "adult",
    "overview", "tagline", "genres", "keywords", "poster_path",
    "production_companies",
]


def passes(df: pd.DataFrame):
    """Tiered filter. Returns (kept_df, n_clean, n_prob_kept, n_prob_rejected).

    Base gates apply to everyone. Then films split into:
      - clean tier (high vote count): kept unconditionally
      - probationary tier (low vote count): kept only if quality gates pass
    All columns are coerced to safe types first (per-chunk dtype drift safety).
    """
    # --- base gates (everyone) ---
    status = df["status"].astype(str).str.strip()
    df = df[status == "Released"]

    adult = df["adult"].astype(str).str.strip().str.lower()
    df = df[~adult.isin(["true", "1"])]

    ov = df["overview"].astype(str).str.strip()
    df = df[(ov.str.lower() != "nan") & (ov.str.len() >= MIN_OVERVIEW_CHARS)]

    if df.empty:
        return df, 0, 0, 0

    # --- signals (recomputed on the filtered frame) ---
    ov_len = df["overview"].astype(str).str.strip().str.len()
    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    rt = pd.to_numeric(df["runtime"], errors="coerce").fillna(0)
    genres = df["genres"].astype(str).str.strip().str.lower()
    poster = df["poster_path"].astype(str).str.strip().str.lower()
    is_en = (df["original_language"].astype(str) == "en").to_numpy()

    clean_floor = np.where(is_en, VOTE_CLEAN_EN, VOTE_CLEAN_OTHER)
    prob_floor = np.where(is_en, VOTE_FLOOR_EN, VOTE_FLOOR_OTHER)

    is_clean = vc.to_numpy() >= clean_floor
    is_probation = (vc.to_numpy() >= prob_floor) & (vc.to_numpy() < clean_floor)

    # quality gates for the probationary tier
    has_genres = (genres != "nan") & (genres.str.len() > 0)
    has_poster = (poster != "nan") & (poster.str.len() > 0)
    quality = (
        (ov_len >= PROBATION_OVERVIEW_CHARS).to_numpy()
        & (rt.to_numpy() >= PROBATION_MIN_RUNTIME)
        & has_genres.to_numpy()
        & has_poster.to_numpy()
    )

    prob_kept = is_probation & quality
    prob_rejected = is_probation & ~quality
    keep = is_clean | prob_kept

    return (
        df[keep],
        int(is_clean.sum()),
        int(prob_kept.sum()),
        int(prob_rejected.sum()),
    )


def main():
    if not RAW.exists():
        sys.exit(f"ERROR: {RAW} not found. Run from repo root after extracting the TMDB zip.")

    print(f"Reading {RAW.name} in chunks ...")
    chunks = []
    total_in = 0
    tot_clean = tot_prob_kept = tot_prob_rej = 0
    reader = pd.read_csv(
        RAW,
        usecols=USECOLS,
        chunksize=100_000,
        dtype=str,            # read everything as str; coerce per-filter (robust to dump quirks)
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        total_in += len(chunk)
        kept, n_clean, n_prob_kept, n_prob_rej = passes(chunk)
        tot_clean += n_clean
        tot_prob_kept += n_prob_kept
        tot_prob_rej += n_prob_rej
        chunks.append(kept)
        print(f"  chunk {i:>2}: read {len(chunk):>7,}  kept {len(kept):>6,}  "
              f"(clean {n_clean:,} / probation +{n_prob_kept:,} -{n_prob_rej:,})")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal read: {total_in:,}  ->  after filter: {len(df):,}")
    print(f"  clean tier:            {tot_clean:,}")
    print(f"  probationary kept:     {tot_prob_kept:,}")
    print(f"  probationary rejected: {tot_prob_rej:,}  "
          f"({tot_prob_rej / max(tot_prob_kept + tot_prob_rej, 1) * 100:.0f}% of marginal band cut as noise)")

    # de-dupe on TMDB id (dumps occasionally have dupes)
    before = len(df)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df[df["id"].notna()]
    df = df.drop_duplicates(subset="id", keep="first")
    if before != len(df):
        print(f"Dropped {before - len(df):,} duplicate/invalid ids")

    # numeric coercions (were read as str for robustness)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype("int64")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").astype("Int64")
    df["id"] = df["id"].astype("int64")

    # derive release_year for agent year-filters
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.astype("Int64")

    # studio = primary production company. TMDB's production_companies is a
    # comma-joined string (e.g. "A24, Plan B Entertainment"); take the first
    # as the headline studio, which is the taste-relevant one for our purposes.
    def primary_studio(val: str) -> str:
        if not isinstance(val, str):
            return ""
        s = val.strip()
        if not s or s.lower() == "nan":
            return ""
        return s.split(",")[0].strip()

    df["studio"] = df["production_companies"].apply(primary_studio)

    # normalize missing imdb_id ('nan' string -> empty) for clean joins later
    df["imdb_id"] = df["imdb_id"].astype(str).str.strip()
    df.loc[df["imdb_id"].str.lower() == "nan", "imdb_id"] = ""

    # final column selection
    df = df[KEEP + ["release_year"]]

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