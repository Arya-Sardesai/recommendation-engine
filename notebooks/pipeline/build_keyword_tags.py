"""
build_keyword_tags.py

Applies keyword_map.py to the movie and TV corpora, unions the movie result with
the existing genome-derived movie_tags.parquet, and reports what it would do.

AUDIT FIRST. With AUDIT_ONLY=True (default) this writes NOTHING -- it prints
coverage, union stats, and sample titles per tag so precision can be eyeballed
before anything lands on disk. Pilots before full runs.

TUNING LOOP: if a tag's samples look wrong, delete the offending SUPPORTING
source from keyword_map.py and re-run. Costs a minute per iteration. Revision 2
of the map already removed 16 sources this way -- see that file's docstring for
the audited evidence behind each deletion.

TIER ROUTING
------------
    movies -> ALL_MOVIE_TAGS = UNIVERSAL | SCREEN | MOVIE_ONLY
    tv     -> ALL_TV_TAGS    = UNIVERSAL | SCREEN | TV_ONLY
So `miniseries` can never land on a film and `long-take` can never land on a
series, without the map needing to know which modality it is being applied to.

UNION RULE (movies only)
------------------------
Genome WINS where both sources produce the same (movie_id, tag). Genome is
model-derived relevance over a curated vocabulary -- better evidence than a
keyword lookup. Keywords fill the ~88% of films genome never reached.

MAX_TAGS_PER_ITEM raised 7 -> 10. At 7 the first audit evicted 4,570
genome-derived rows to make room for keyword tags, which threw away the better
evidence. There is legitimately more evidence per film now that two sources
feed the union, so the cap should reflect that. The audit still reports genome
eviction; if it stays high, raise it again.

TV has no genome, so TV output is keyword-only.

OUTPUTS (only when AUDIT_ONLY=False)
    data/processed/movie_tags_v2.parquet   (movie_id, tag, score)
    data/processed/tv_tags.parquet         (tv_id,    tag, score)
Writes to _v2 rather than overwriting movie_tags.parquet -- same convention as
book_tags_v2.parquet. Repoint the app or rename once validated.

Run from REPO ROOT:
    python notebooks/pipeline/build_keyword_tags.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from keyword_map import KEYWORD_MAP, tags_for  # noqa: E402
from tag_taxonomy import (  # noqa: E402
    ALL_MOVIE_TAGS, ALL_TV_TAGS, UNIVERSAL_TAGS, SCREEN_TAGS,
    MOVIE_ONLY_TAGS, TV_ONLY_TAGS,
)

# ---- config -----------------------------------------------------------------
AUDIT_ONLY = True          # False = actually write the parquets
MAX_TAGS_PER_ITEM = 10     # was 7; see docstring
N_SAMPLES = 8              # sample titles printed per tag

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"
MOVIES = PROC / "movies.parquet"
TV = PROC / "tv_corpus.parquet"
GENOME_TAGS = PROC / "movie_tags.parquet"
OUT_MOVIES = PROC / "movie_tags_v2.parquet"
OUT_TV = PROC / "tv_tags.parquet"


def check_map():
    """Every map key must be declared in some screen-facing tier. Fail loud."""
    declared = UNIVERSAL_TAGS | SCREEN_TAGS | MOVIE_ONLY_TAGS | TV_ONLY_TAGS
    unknown = set(KEYWORD_MAP) - declared
    if unknown:
        sys.exit(f"ERROR: keyword_map keys not declared in taxonomy: {sorted(unknown)}")
    unmapped = declared - set(KEYWORD_MAP)
    if unmapped:
        print(f"NOTE: {len(unmapped)} declared screen tags have no keyword source "
              f"(genome-only or intentionally dropped): {sorted(unmapped)}\n")


def apply_map(df, id_col, title_col, allowed, label):
    """-> (long DataFrame [id, tag, score], dict tag -> [sample titles])"""
    if "keywords_tmdb" not in df.columns:
        sys.exit(f"ERROR: {label} corpus has no 'keywords_tmdb' column. Run the TMDB merge.")
    rows, samples, n_with_any = [], {}, 0
    for _id, title, kws in zip(df[id_col], df[title_col], df["keywords_tmdb"]):
        got = tags_for(kws, allowed=allowed)
        if got:
            n_with_any += 1
        for tag, score in got:
            rows.append((_id, tag, score))
            samples.setdefault(tag, [])
            if len(samples[tag]) < N_SAMPLES:
                samples[tag].append(str(title))
    out = pd.DataFrame(rows, columns=[id_col, "tag", "score"])
    print(f"[{label}] {n_with_any:,}/{len(df):,} items received >=1 keyword tag "
          f"({n_with_any/len(df):.1%})  |  {len(out):,} tag rows")
    return out, samples


def cap(df, id_col):
    return (df.sort_values([id_col, "score"], ascending=[True, False])
              .groupby(id_col).head(MAX_TAGS_PER_ITEM))


def report_tags(df, id_col, samples, label):
    print(f"\n{'='*74}\n{label}: per-tag counts + samples (eyeball for sweep-in)\n{'='*74}")
    counts = df.groupby("tag")[id_col].nunique().sort_values(ascending=False)
    for tag, n in counts.items():
        scores = df.loc[df.tag == tag, "score"].unique()
        tier = "P" if any(s == 0.7 for s in scores) else "s"
        print(f"\n  {tag}  [{tier}]  {n:,} items")
        print(f"    {' | '.join(samples.get(tag, [])[:N_SAMPLES])}")


def main():
    check_map()
    for p in (MOVIES, TV):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")

    # ---------------- movies ----------------
    mv = pd.read_parquet(MOVIES, columns=["id", "title", "keywords_tmdb"])
    kw_mv, mv_samples = apply_map(mv, "id", "title", ALL_MOVIE_TAGS, "movies")
    kw_mv = kw_mv.rename(columns={"id": "movie_id"})
    kw_pairs = kw_mv[["movie_id", "tag"]].drop_duplicates()

    if GENOME_TAGS.exists():
        gen = pd.read_parquet(GENOME_TAGS)
        print(f"[movies] genome tags: {len(gen):,} rows across {gen.movie_id.nunique():,} films")
        gen["_src"] = "genome"
        kw_mv["_src"] = "keyword"
        both = pd.concat([gen, kw_mv], ignore_index=True)
        before = len(both)
        # "genome" < "keyword" alphabetically, so keep='first' = genome wins
        both = both.sort_values("_src").drop_duplicates(subset=["movie_id", "tag"], keep="first")
        print(f"[movies] union: {before:,} -> {len(both):,} rows after genome-wins dedup")
        capped = cap(both, "movie_id")
        evicted = both[~both.index.isin(capped.index)]
        print(f"[movies] cap at {MAX_TAGS_PER_ITEM}: dropped {len(both)-len(capped):,} rows "
              f"({(evicted['_src']=='genome').sum():,} genome-derived)")
        mv_final = capped[["movie_id", "tag", "score"]]
    else:
        print("[movies] WARNING: no genome movie_tags.parquet; keyword-only output")
        mv_final = cap(kw_mv, "movie_id")[["movie_id", "tag", "score"]]

    print(f"[movies] FINAL: {len(mv_final):,} rows, {mv_final.movie_id.nunique():,} films "
          f"({mv_final.movie_id.nunique()/len(mv):.1%} of corpus)")

    # ---------------- tv ----------------
    tv = pd.read_parquet(TV, columns=["tmdb_id", "name", "keywords_tmdb"])
    kw_tv, tv_samples = apply_map(tv, "tmdb_id", "name", ALL_TV_TAGS, "tv")
    kw_tv = kw_tv.rename(columns={"tmdb_id": "tv_id"})
    tv_final = cap(kw_tv, "tv_id")[["tv_id", "tag", "score"]]
    print(f"[tv] FINAL: {len(tv_final):,} rows, {tv_final.tv_id.nunique():,} series "
          f"({tv_final.tv_id.nunique()/len(tv):.1%} of corpus)")

    # cross-modal bridge check -- how many screen titles can reach books at all
    for name, d, idc in (("movies", mv_final, "movie_id"), ("tv", tv_final, "tv_id")):
        uni = d[d.tag.isin(UNIVERSAL_TAGS)][idc].nunique()
        adapt = d[d.tag.isin({"book-adaptation", "manga-adaptation", "comic-adaptation"})][idc].nunique()
        print(f"[{name}] bridgeable (>=1 universal tag): {uni:,}  |  "
              f"source-medium tagged: {adapt:,}")

    report_tags(mv_final.merge(kw_pairs, on=["movie_id", "tag"], how="inner"),
                "movie_id", mv_samples, "MOVIES (keyword-derived rows only)")
    report_tags(tv_final, "tv_id", tv_samples, "TV")

    if AUDIT_ONLY:
        print(f"\n{'='*74}\nAUDIT ONLY -- nothing written. Set AUDIT_ONLY=False to write:")
        print(f"  {OUT_MOVIES}\n  {OUT_TV}")
        return

    mv_final.to_parquet(OUT_MOVIES, index=False)
    tv_final.to_parquet(OUT_TV, index=False)
    print(f"\nWrote:\n  {OUT_MOVIES}\n  {OUT_TV}")


if __name__ == "__main__":
    main()
