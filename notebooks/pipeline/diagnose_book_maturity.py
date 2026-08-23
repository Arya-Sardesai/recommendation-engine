"""
diagnose_book_maturity.py — is there enough signal to filter YA out of adult
book recommendations?

THE COMPLAINT
An adult profile (Bryson, Bourdain, Lahiri, Fight Club, Eggers, Gaddis) gets
And They Were Roommates, They Hate Each Other and The Gentleman's Guide to
Getting Lucky in its top 20. Same class as the TV kids problem, one level up:
not children's content, but content written for a different age bracket.

WHY THIS NEEDS MEASURING FIRST
The TV fix worked because TMDB content ratings give a 6-point ordinal with 99%
coverage. Books have NO certification of any kind. The only available signal is
the Hardcover genre folksonomy, which is:
  - community-assigned, so coverage varies
  - compressed: at best children's / middle grade / YA / adult, four levels
    against television's six
  - unreliable at the boundary, because "Young Adult" is applied both to books
    written for teenagers AND to books merely featuring them

A 1-step drop is the whole distance from adult to YA, so a threshold that
catches the complaint may also suppress everything a profile is adjacent to.
That is the number this script exists to check BEFORE any filter is written.

READ-ONLY. No writes, no API.

USAGE
    python notebooks\\pipeline\\diagnose_book_maturity.py
    python notebooks\\pipeline\\diagnose_book_maturity.py --ratings tests\\movies_ratings.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
DEFAULT_PARQUET = PROCESSED / "books_v2_hc.parquet"
MIN_RATINGS = 10          # app.py's serving floor

# Ordinal audience level. Deliberately coarse — the folksonomy cannot support
# finer distinctions, and pretending otherwise would invent precision.
LEVELS = {
    0: ("children", {"children's", "childrens", "picture books", "juvenile",
                     "early readers", "chapter books", "kids"}),
    1: ("middle grade", {"middle grade", "middle-grade", "juvenile fiction"}),
    2: ("young adult", {"young adult", "young adult fiction", "ya", "teen",
                        "teen fiction", "romantasy"}),
}
ADULT = 3


def level_of(genres):
    """Lowest (youngest) matching level, or ADULT when nothing matches.

    Lowest wins: a book tagged both "Middle Grade" and "Fantasy" is middle
    grade. Absence of any label means ADULT by default, which is the only safe
    reading -- most adult books simply are not tagged "adult".
    """
    if genres is None:
        return ADULT
    if isinstance(genres, str):
        gs = {g.strip().lower() for g in genres.split(",") if g.strip()}
    else:
        try:
            gs = {str(g).strip().lower() for g in genres}
        except TypeError:
            return ADULT
    for lvl in sorted(LEVELS):
        if gs & LEVELS[lvl][1]:
            return lvl
    return ADULT


def load_book_ratings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        v = raw.get("re_books")
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return v
    if isinstance(raw, list):
        return raw
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--ratings", default=None,
                    help="localStorage dump; adds a profile-side read")
    ap.add_argument("--min-ratings", type=int, default=MIN_RATINGS)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"corpus: {len(df):,} rows\n")

    df = df.copy()
    df["_level"] = [level_of(g) for g in df["genres"]]
    rc = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    serv = df[rc >= args.min_ratings]

    print("=" * 68)
    print(f"AUDIENCE LEVEL over servable books (ratings_count >= {args.min_ratings})")
    print("=" * 68)
    print(f"  servable: {len(serv):,} of {len(df):,}")
    names = {0: "children", 1: "middle grade", 2: "young adult", 3: "adult/unlabelled"}
    vc = serv["_level"].value_counts().sort_index()
    for lvl, c in vc.items():
        print(f"    {names[lvl]:18s} {c:7,}  {c/max(len(serv),1):6.1%}")
    print()
    print("  NOTE: level 3 is 'adult OR not labelled'. The folksonomy has no")
    print("  positive 'adult' tag, so this bucket is not evidence of anything —")
    print("  it is the absence of a youth label.")
    print()

    # How reliable is the YA label? Sample the most-read of each level.
    print("  most-read at each level (sanity-check by eye):")
    for lvl in sorted(names):
        sub = serv[serv["_level"] == lvl]
        if len(sub):
            top = sub.nlargest(5, "ratings_count")[["title", "primary_author"]]
            for r in top.itertuples():
                print(f"    {names[lvl]:14s} {str(r.title)[:44]:46s} {str(r.primary_author)[:22]}")
            print()

    # Which genre labels actually drive the classification?
    hits = Counter()
    for g in serv["genres"]:
        if isinstance(g, str):
            gs = [x.strip().lower() for x in g.split(",")]
        else:
            try:
                gs = [str(x).strip().lower() for x in g]
            except TypeError:
                continue
        for lvl in LEVELS:
            for tag in LEVELS[lvl][1]:
                if tag in gs:
                    hits[tag] += 1
    print("  labels doing the work:")
    for tag, c in hits.most_common(12):
        print(f"    {tag:24s} {c:6,}")
    print()

    if args.ratings:
        rts = load_book_ratings(args.ratings)
        if not rts:
            print("  (no re_books ratings found in that dump)")
            return
        idx_ok = [r for r in rts
                  if r.get("corpus_index") is not None
                  and 0 <= int(r["corpus_index"]) < len(df)]
        liked = [r for r in idx_ok if r.get("rating", 0) >= 3.5]
        lv = [int(df.iloc[int(r["corpus_index"])]["_level"]) for r in liked]
        print("=" * 68)
        print(f"PROFILE ({len(rts)} rated, {len(liked)} positively)")
        print("=" * 68)
        c = Counter(lv)
        for lvl in sorted(names):
            if c.get(lvl):
                print(f"    {names[lvl]:18s} {c[lvl]:4d}")
        if lv:
            med = pd.Series(lv).median()
            print(f"\n  median level: {med}")
            for thr in (1, 2, 3):
                n = int((serv["_level"] <= med - thr).sum())
                print(f"    a drop-of-{thr} rule would suppress {n:,} servable books"
                      f"  ({n/max(len(serv),1):.1%})")
            print()
            print("  A drop-of-1 is the entire distance from adult to YA, so it")
            print("  suppresses the whole YA shelf. Check the titles it would")
            print("  remove before deciding that is what you want.")

    print("\n" + "=" * 68)
    print("READ THIS AS")
    print("=" * 68)
    print("  If the YA bucket is large AND the samples look right, a")
    print("  profile-conditional rule is viable. If the labels are noisy, or")
    print("  the only workable threshold suppresses an entire shelf, this is a")
    print("  stated limitation rather than a filter.")


if __name__ == "__main__":
    main()
