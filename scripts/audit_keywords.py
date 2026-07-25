"""
audit_keywords.py  (one-shot, pre-taxonomy)

Regenerates keyword frequency tables from the ENRICHED corpora (post TMDB bulk
merge) and measures the kaggle-vs-tmdb keyword column overlap for movies.

Answers, in one run:
  1. Coverage + density for keywords_tmdb, movies and TV.
  2. How many movies have EMPTY keywords_tmdb but NON-EMPTY legacy kaggle
     keywords -> i.e. whether a union recovers coverage on the 33% tail.
  3. Full frequency tables, written to disk as CSV so they can be UPLOADED
     (paperclip) rather than pasted -- avoids the empty-attachment-chip quirk.
  4. Where the CURRENT embed_movies.py KW_STOP entries rank, so the stoplist
     decision is made against real mass, not vibes.

Run from anywhere in the repo:
    python scripts/oneshot/audit_keywords.py
"""

from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

# --- locate repo root by walking up until we find data/processed --------------
def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "processed").is_dir():
            return p
    raise SystemExit("ERROR: could not find data/processed above this script.")

ROOT = find_root(Path(__file__).resolve().parent)
PROCESSED = ROOT / "data" / "processed"
OUTDIR = ROOT / "data" / "tmdb_raw"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Mirror of the CURRENT embed_movies.py KW_STOP, so we can see what it costs.
CURRENT_KW_STOP = {
    "aftercreditsstinger", "duringcreditsstinger", "based on novel or book",
    "based on comic", "based on young adult novel", "based on true story",
    "woman director", "sequel", "prequel", "remake", "reboot",
}


def to_terms(val) -> list[str]:
    """Coerce list/ndarray OR comma-string into clean lowercase terms."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        terms = [str(x) for x in val]
    else:
        s = str(val)
        if s.strip().lower() in ("", "nan", "none", "[]"):
            return []
        terms = s.split(",")
    out = []
    for t in terms:
        t = t.strip().lower()
        if t and t not in ("nan", "none"):
            out.append(t)
    return out


def profile(df: pd.DataFrame, col: str, label: str) -> Counter:
    if col not in df.columns:
        print(f"  [{label}] column {col!r} NOT PRESENT")
        return Counter()
    lists = [to_terms(v) for v in df[col]]
    counts = np.array([len(x) for x in lists])
    nonzero = counts[counts > 0]
    print(f"  [{label}] {col}")
    print(f"      rows            : {len(df):,}")
    print(f"      with >=1 keyword: {len(nonzero):,} ({100*len(nonzero)/len(df):.1f}%)")
    print(f"      ZERO keywords   : {len(df)-len(nonzero):,} "
          f"({100*(len(df)-len(nonzero))/len(df):.1f}%)")
    if len(nonzero):
        print(f"      density (nonzero): median={int(np.median(nonzero))} "
              f"mean={nonzero.mean():.1f} p90={int(np.percentile(nonzero,90))} "
              f"max={nonzero.max()}")
    print(f"      distinct terms  : {len({t for x in lists for t in x}):,}")
    return Counter(t for x in lists for t in x)


def dump_table(counter: Counter, n_rows: int, path: Path):
    rows = [{"keyword": k, "count": c, "pct_of_corpus": round(100 * c / n_rows, 3)}
            for k, c in counter.most_common()]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"      -> wrote {len(rows):,} terms to {path}")


def show_stop_ranks(counter: Counter, n_rows: int, label: str):
    ranks = {k: i + 1 for i, (k, _) in enumerate(counter.most_common())}
    print(f"\n  [{label}] where CURRENT KW_STOP entries actually sit:")
    for term in sorted(CURRENT_KW_STOP):
        if term in ranks:
            c = counter[term]
            print(f"      rank {ranks[term]:>6,}  {c:>7,} films "
                  f"({100*c/n_rows:5.2f}%)  {term}")
        else:
            print(f"      {'absent':>13}                          {term}")


def main():
    print("=" * 72)
    print("MOVIES")
    print("=" * 72)
    mv = pd.read_parquet(PROCESSED / "movies.parquet")
    print(f"  columns: {list(mv.columns)}\n")

    c_tmdb = profile(mv, "keywords_tmdb", "movies")
    print()
    c_kaggle = profile(mv, "keywords", "movies-legacy")

    # --- union recovery question ---------------------------------------------
    if "keywords" in mv.columns and "keywords_tmdb" in mv.columns:
        tmdb_l = [to_terms(v) for v in mv["keywords_tmdb"]]
        kag_l = [to_terms(v) for v in mv["keywords"]]
        empty_tmdb = [i for i, x in enumerate(tmdb_l) if not x]
        recover = [i for i in empty_tmdb if kag_l[i]]
        print("\n  --- UNION RECOVERY (the 33% tail) ---")
        print(f"      movies with EMPTY keywords_tmdb : {len(empty_tmdb):,}")
        print(f"      ...of which kaggle has keywords : {len(recover):,} "
              f"({100*len(recover)/max(len(empty_tmdb),1):.1f}% of the empties, "
              f"{100*len(recover)/len(mv):.1f}% of corpus)")
        if recover:
            print("      sample recoverable rows:")
            tcol = "title" if "title" in mv.columns else mv.columns[0]
            for i in recover[:8]:
                print(f"        {mv.iloc[i][tcol]!r:<45} -> {kag_l[i][:6]}")
        # vocabulary drift: does kaggle use terms tmdb never uses?
        only_kaggle = set(c_kaggle) - set(c_tmdb)
        print(f"      terms present in kaggle but NEVER in tmdb: {len(only_kaggle):,}")
        if only_kaggle:
            top_only = sorted(only_kaggle, key=lambda t: -c_kaggle[t])[:15]
            print(f"        top by kaggle count: {top_only}")

    print("\n" + "=" * 72)
    print("TV")
    print("=" * 72)
    tv = pd.read_parquet(PROCESSED / "tv_corpus.parquet")
    print(f"  columns: {list(tv.columns)}\n")
    c_tv = profile(tv, "keywords_tmdb", "tv")

    # --- write tables ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("FREQUENCY TABLES (upload these files, don't paste)")
    print("=" * 72)
    dump_table(c_tmdb, len(mv), OUTDIR / "kw_freq_movies.csv")
    dump_table(c_tv, len(tv), OUTDIR / "kw_freq_tv.csv")

    # --- console preview: enough to eyeball, small enough to paste ------------
    for label, counter, n in (("MOVIES", c_tmdb, len(mv)), ("TV", c_tv, len(tv))):
        print(f"\n  --- {label}: top 40 ---")
        for i, (k, c) in enumerate(counter.most_common(40), 1):
            print(f"      {i:>3}. {c:>7,} ({100*c/n:5.2f}%)  {k}")

    show_stop_ranks(c_tmdb, len(mv), "movies")

    # --- cult vocab check (agenda item 3) ------------------------------------
    print("\n  --- CULT VOCAB counts in enriched corpus ---")
    for term in ("cult film", "cult", "b movie", "exploitation", "grindhouse",
                 "psychotronic", "psychotronic film", "video nasty", "camp"):
        print(f"      movies {c_tmdb.get(term,0):>6,}   tv {c_tv.get(term,0):>5,}   {term}")

    summary = {
        "movies_rows": len(mv), "tv_rows": len(tv),
        "movies_distinct_kw": len(c_tmdb), "tv_distinct_kw": len(c_tv),
    }
    (OUTDIR / "kw_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTDIR / 'kw_audit_summary.json'}")


if __name__ == "__main__":
    main()