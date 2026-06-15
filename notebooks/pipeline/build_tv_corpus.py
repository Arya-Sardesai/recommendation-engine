"""
build_tv_corpus.py  —  TV chapter, corpus build (mirror of build_movies_corpus.py)

Reads the asaniczka 150K TMDB TV dump, filters to embeddable series, assembles
the embed text, and writes data/processed/tv_corpus.parquet.

Key differences from the movies builder (forced by the TV schema):
  - overview is REQUIRED (44.7% null in raw) -> hard pre-filter, not a tier gate
  - no imdb_id  -> IMDb credit join dropped; showrunner/network come from the CSV
  - no keywords -> no tag layer in v1
  - runtime gate dropped (TV episode runtimes are not a quality signal)
  - vote floors gentler than movies (TV vote counts run far lower)

Run from the repo root:  python notebooks/pipeline/build_tv_corpus.py
"""

import zipfile
from pathlib import Path

import pandas as pd

# ---- paths (robust to cwd: script lives in notebooks/pipeline/) -------------
ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data" / "raw" / "archive (1).zip"
OUT = ROOT / "data" / "processed" / "tv_corpus.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- tunable thresholds (adjust from the printed report) --------------------
MIN_OVERVIEW_CHARS = 80      # drop overview stubs

CLEAN_EN_VOTES = 10          # EN: kept unconditionally at/above this
CLEAN_OTHER_VOTES = 5        # non-EN: kept unconditionally at/above this

PROB_EN_VOTES = 4            # EN: probationary band [PROB, CLEAN)
PROB_OTHER_VOTES = 2         # non-EN: probationary band [PROB, CLEAN)
# probationary rows kept ONLY if they pass the quality gates below
# (overview already guaranteed; require genres AND poster)


def load_raw() -> pd.DataFrame:
    with zipfile.ZipFile(RAW_ZIP) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    print(f"raw: {df.shape[0]:,} rows x {df.shape[1]} cols  ({csv_name})")
    return df


def make_embed_text(row) -> str:
    name = str(row["name"]).strip()
    yr = row["start_year"]
    yr_part = f" ({int(yr)})" if pd.notna(yr) else ""
    genres = str(row["genres"]).strip() if pd.notna(row["genres"]) else ""
    g_part = f" {genres}." if genres else ""
    ov = str(row["overview"]).strip()
    return f"{name}{yr_part}.{g_part} {ov}".strip()


def main() -> None:
    df = load_raw()

    # --- adult drop ---------------------------------------------------------
    if "adult" in df.columns:
        adult = df["adult"].astype(str).str.lower().isin(["true", "1"])
        df = df[~adult]
        print(f"after adult drop: {len(df):,}")

    # --- hard overview requirement -----------------------------------------
    ov = df["overview"].astype(str).str.strip()
    df = df[df["overview"].notna() & (ov.str.len() >= MIN_OVERVIEW_CHARS)].copy()
    print(f"after overview gate (>= {MIN_OVERVIEW_CHARS} chars): {len(df):,}")

    # --- derived fields -----------------------------------------------------
    df["start_year"] = pd.to_datetime(
        df["first_air_date"], errors="coerce"
    ).dt.year
    df["end_year"] = pd.to_datetime(
        df["last_air_date"], errors="coerce"
    ).dt.year

    is_en = df["original_language"].eq("en")
    has_genres = df["genres"].notna() & df["genres"].astype(str).str.strip().ne("")
    has_poster = df["poster_path"].notna() & df["poster_path"].astype(str).str.strip().ne("")
    vc = df["vote_count"].fillna(0)

    # --- vote distribution report (so thresholds are data-driven) ----------
    print("\nvote_count quantiles:")
    for label, mask in [("all", slice(None)), ("EN", is_en), ("non-EN", ~is_en)]:
        q = vc[mask].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
        print(f"  {label:6s}  " + "  ".join(f"p{int(p*100)}={v:.0f}"
                                            for p, v in q.items()))

    # --- tiered keep --------------------------------------------------------
    clean = ((is_en & (vc >= CLEAN_EN_VOTES)) |
             (~is_en & (vc >= CLEAN_OTHER_VOTES)))

    prob_band = ((is_en & vc.between(PROB_EN_VOTES, CLEAN_EN_VOTES - 1)) |
                 (~is_en & vc.between(PROB_OTHER_VOTES, CLEAN_OTHER_VOTES - 1)))
    prob_kept = prob_band & has_genres & has_poster

    keep = clean | prob_kept
    print(f"\nclean tier kept:        {int(clean.sum()):,}")
    print(f"probationary in band:   {int(prob_band.sum()):,}")
    print(f"probationary kept:      {int(prob_kept.sum()):,} "
          f"(of band, gated on genres+poster)")

    df = df[keep].copy()

    # --- dedupe + embed text -----------------------------------------------
    df = df.drop_duplicates(subset="id")
    df["embed_text"] = df.apply(make_embed_text, axis=1)

    # --- select / rename columns -------------------------------------------
    df = df.rename(columns={"id": "tmdb_id"})
    cols = [
        "tmdb_id", "name", "original_name", "start_year", "end_year",
        "original_language", "vote_count", "vote_average", "popularity",
        "genres", "created_by", "networks", "number_of_seasons",
        "number_of_episodes", "episode_run_time", "type", "status",
        "origin_country", "tagline", "poster_path", "overview", "embed_text",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # --- write + final report ----------------------------------------------
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  ->  {len(df):,} series")

    print("\nlanguage breakdown (top 12):")
    print(df["original_language"].value_counts().head(12).to_string())

    print("\nfacet coverage:")
    for c in ["created_by", "networks", "genres", "poster_path", "start_year"]:
        if c in df.columns:
            cov = df[c].notna().mean() * 100
            print(f"  {c:14s} {cov:5.1f}%")


if __name__ == "__main__":
    main()