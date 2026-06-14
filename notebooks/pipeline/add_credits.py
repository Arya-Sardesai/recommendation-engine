"""
add_credits.py

Adds `director`, `directors_all`, and `cast` (top-billed actors) columns to
movies.parquet by joining IMDb non-commercial data on imdb_id. Standalone and
idempotent: reads the existing corpus, writes it back enriched.

Supersedes add_directors.py (does directors AND cast in one pass, sharing the
single expensive read of name.basics).

Join chain:
    movies.imdb_id (tconst)
      -> title.crew.tsv       (tconst -> director nconsts)
      -> title.principals.tsv (tconst -> actor nconsts, with billing order)
      -> name.basics.tsv      (nconst -> primaryName)   [read ONCE for both]

`director`      = first credited director (headline auteur signal)
`directors_all` = full director list (for the agent)
`cast`          = top-N actors by billing order (list)

IMDb files (download once, no account, works in India):
    https://datasets.imdbws.com/title.crew.tsv.gz        -> data/raw/movies/imdb/
    https://datasets.imdbws.com/title.principals.tsv.gz  -> data/raw/movies/imdb/
    https://datasets.imdbws.com/name.basics.tsv.gz       -> data/raw/movies/imdb/

NOTE: title.principals is the largest IMDb file (~700MB gz). It is streamed
and filtered to corpus tconsts + acting roles, so RAM stays bounded, but this
is the slowest step (a few minutes).

Run from REPO ROOT:
    python notebooks/pipeline/add_credits.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"
IMDB = ROOT / "data" / "raw" / "movies" / "imdb"
CORPUS = PROC / "movies.parquet"
CREW = IMDB / "title.crew.tsv.gz"
PRINCIPALS = IMDB / "title.principals.tsv.gz"
NAMES = IMDB / "name.basics.tsv.gz"

TOP_CAST_N = 5   # number of top-billed actors to keep per film


def main():
    for p in (CORPUS, CREW, PRINCIPALS, NAMES):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")

    df = pd.read_parquet(CORPUS)
    if "imdb_id" not in df.columns:
        sys.exit("ERROR: movies.parquet has no imdb_id column. "
                 "Re-run build_movies_corpus.py (updated to keep imdb_id) first.")

    corpus_tconsts = set(df.loc[df["imdb_id"].astype(str).str.startswith("tt"), "imdb_id"])
    print(f"Corpus films with an imdb_id: {len(corpus_tconsts):,} / {len(df):,}")

    # 1. title.crew -> directors
    print("\nReading title.crew ...")
    crew = pd.read_csv(
        CREW, sep="\t", usecols=["tconst", "directors"],
        dtype=str, na_values="\\N", compression="gzip",
    )
    crew = crew[crew["tconst"].isin(corpus_tconsts) & crew["directors"].notna()]
    print(f"  matched crew rows: {len(crew):,}")

    needed_nconsts = set()
    crew_director_lists = {}
    for tconst, directors in zip(crew["tconst"], crew["directors"]):
        ids = [d for d in directors.split(",") if d.startswith("nm")]
        if ids:
            crew_director_lists[tconst] = ids
            needed_nconsts.update(ids)

    # 2. title.principals -> top-billed actors (streamed)
    print("Streaming title.principals (largest file; a few minutes) ...")
    cast_accum = {}
    reader = pd.read_csv(
        PRINCIPALS, sep="\t",
        usecols=["tconst", "ordering", "nconst", "category"],
        dtype=str, na_values="\\N", compression="gzip", chunksize=2_000_000,
    )
    for i, chunk in enumerate(reader):
        c = chunk[chunk["tconst"].isin(corpus_tconsts)
                  & chunk["category"].isin(["actor", "actress"])]
        for tconst, ordering, nconst in zip(c["tconst"], c["ordering"], c["nconst"]):
            try:
                o = int(ordering)
            except (ValueError, TypeError):
                o = 999
            cast_accum.setdefault(tconst, []).append((o, nconst))
        if i % 3 == 0:
            print(f"  chunk {i}: films with cast so far {len(cast_accum):,}")

    cast_nconst_lists = {}
    for tconst, pairs in cast_accum.items():
        pairs.sort(key=lambda x: x[0])
        # dedupe by nconst, preserving billing order (a person can appear in
        # multiple principals rows — e.g. two roles — which would otherwise
        # produce "Nicolas Cage, Nicolas Cage")
        seen = set()
        top = []
        for _, nc in pairs:
            if nc in seen:
                continue
            seen.add(nc)
            top.append(nc)
            if len(top) >= TOP_CAST_N:
                break
        cast_nconst_lists[tconst] = top
        needed_nconsts.update(top)

    print(f"  films with cast: {len(cast_nconst_lists):,}")
    print(f"  unique people (directors+cast) to resolve: {len(needed_nconsts):,}")

    # 3. name.basics -> primaryName (single stream for ALL needed nconsts)
    print("Streaming name.basics ...")
    nconst_to_name = {}
    reader = pd.read_csv(
        NAMES, sep="\t", usecols=["nconst", "primaryName"],
        dtype=str, na_values="\\N", compression="gzip", chunksize=500_000,
    )
    for i, chunk in enumerate(reader):
        hit = chunk[chunk["nconst"].isin(needed_nconsts)]
        for n, nm in zip(hit["nconst"], hit["primaryName"]):
            nconst_to_name[n] = nm
        if i % 5 == 0:
            print(f"  chunk {i}: resolved {len(nconst_to_name):,}/{len(needed_nconsts):,}")
        if len(nconst_to_name) == len(needed_nconsts):
            break
    print(f"  resolved {len(nconst_to_name):,}/{len(needed_nconsts):,}")

    # 4. assemble columns
    def names_for(ids):
        out = [nconst_to_name.get(i) for i in ids]
        return [n for n in out if n]

    tconst_to_primary_dir = {}
    tconst_to_all_dir = {}
    for tconst, ids in crew_director_lists.items():
        names = names_for(ids)
        if names:
            tconst_to_primary_dir[tconst] = names[0]
            tconst_to_all_dir[tconst] = names

    tconst_to_cast = {t: names_for(ids) for t, ids in cast_nconst_lists.items()}

    df["director"] = df["imdb_id"].map(tconst_to_primary_dir).fillna("")
    df["directors_all"] = df["imdb_id"].map(lambda t: tconst_to_all_dir.get(t, []))
    df["cast"] = df["imdb_id"].map(lambda t: tconst_to_cast.get(t, []))

    df.to_parquet(CORPUS, index=False)

    # sanity
    n_dir = (df["director"] != "").sum()
    n_cast = df["cast"].apply(lambda c: len(c) > 0).sum()
    print(f"\nWrote credits -> {CORPUS}")
    print(f"Films with a director: {n_dir:,} / {len(df):,}  ({n_dir/len(df)*100:.0f}%)")
    print(f"Films with cast:       {n_cast:,} / {len(df):,}  ({n_cast/len(df)*100:.0f}%)")

    print("\n--- sanity ---")
    print("Most prolific directors:")
    print(df.loc[df["director"] != "", "director"].value_counts().head(10).to_string())
    from collections import Counter
    actor_counts = Counter()
    for c in df["cast"]:
        actor_counts.update(c)
    print("\nMost-cast actors:")
    for name, cnt in actor_counts.most_common(10):
        print(f"  {cnt:>4}  {name}")
    ex = df[df["cast"].apply(lambda c: len(c) >= 3)].head(1)
    if len(ex):
        row = ex.iloc[0]
        print(f"\nExample - {row['title']}:")
        print(f"  director: {row['director']}")
        print(f"  cast: {', '.join(row['cast'])}")
        if "studio" in df.columns:
            print(f"  studio: {row['studio']}")


if __name__ == "__main__":
    main()