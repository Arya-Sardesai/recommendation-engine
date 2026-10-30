"""
build_book_awards.py -- award_books.jsonl -> data/processed/book_awards.parquet

Gives every catalogue book that won or was shortlisted for a prize a row per
award, in the shape the app's book trophy badge and award bonus read:

    book_id   str   Hardcover id, as in books_v2_hc3.parquet
    award     str   "Booker Prize", "Hugo Award for Best Novel", ...
    result    str   won | nominated | laureate (Nobel: given to the author)
    tier      str   major | minor   (only MAJOR awards move the ranking;
                                     minor ones still show on the badge)

MATCHING
Same title + author key build_award_list.py used for its in_catalogue flag
(imported, not copied, so the two can't disagree). A key that hits several
corpus rows (two editions) badges all of them.

TIERS (a judgment call -- the sweep's "wins only" row tests whether it matters)
  minor  Sahitya Akademi (one prize per Indian language per year, so the
         English winners are a thin and uneven slice), canon lists
         ("100 best ..." CSVs passed with --extra), and NOMINATIONS for the
         fan-voted crime/SFF prizes with long ballots (Locus, Anthony, Agatha,
         Macavity). A win of any of those is still major.
  major  everything else: Booker, Pulitzer, Hugo, Nebula, Edgar, Goncourt ...
  Wikipedia shortlist rows come in as "listed" and are read as nominated: the
  tables mix winners and shortlist, and a real win also arrives from Wikidata.

Run again after the November merge so the ~5,700 award books added then get
their badges too.

USAGE (repo root):
    python notebooks\\pipeline\\build_book_awards.py
    python notebooks\\pipeline\\build_book_awards.py --corpus data\\processed\\books_v2_hc4.parquet
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_award_list import book_key  # noqa: E402  same key as in_catalogue

AWARD_BOOKS = Path("data/raw/books/awards/award_books.jsonl")
DEFAULT_CORPUS = Path("data/processed/books_v2_hc3.parquet")
OUT = Path("data/processed/book_awards.parquet")

MINOR_ANY = ("sahitya akademi",)
MINOR_NOMS = ("locus award", "anthony award", "agatha award", "macavity award")
LIST_WORDS = ("list", "best books", "100", "canon", "must read", "must-read")

_AW = re.compile(r"^(.*) \((won|nominated|laureate|listed)\)$")


def tier_of(award, result):
    a = award.lower()
    if any(m in a for m in MINOR_ANY):
        return "minor"
    if result == "listed" and any(w in a for w in LIST_WORDS):
        return "minor"
    if result != "won" and any(m in a for m in MINOR_NOMS):
        return "minor"
    return "major"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not AWARD_BOOKS.exists():
        sys.exit(f"{AWARD_BOOKS} not found -- run build_award_list.py --stage match first")
    corpus = pd.read_parquet(args.corpus, columns=["book_id", "title", "primary_author"])
    print(f"corpus {Path(args.corpus).name}: {len(corpus):,} books")

    by_key = {}
    for bid, t, a in zip(corpus["book_id"].astype(str), corpus["title"],
                         corpus["primary_author"]):
        k = book_key(t, a)
        if k:
            by_key.setdefault(k, []).append(bid)

    rows, n_books, n_hit, unparsed = [], 0, 0, 0
    for line in AWARD_BOOKS.open(encoding="utf-8"):
        if not line.strip():
            continue
        b = json.loads(line)
        n_books += 1
        ids = by_key.get(book_key(b["title"], b["author"]))
        if not ids:
            continue
        n_hit += 1
        seen = set()
        for s in b["awards"]:
            m = _AW.match(s.strip())
            if not m:
                unparsed += 1
                continue
            award, res = m.group(1).strip(), m.group(2)
            tier = tier_of(award, res)
            res = "nominated" if res == "listed" else res
            for bid in ids:
                if (bid, award, res) in seen:
                    continue
                seen.add((bid, award, res))
                rows.append({"book_id": bid, "award": award, "result": res, "tier": tier})

    df = pd.DataFrame(rows, columns=["book_id", "award", "result", "tier"])
    # a book listed as both won and nominated for one prize keeps the win
    df["_r"] = df["result"].map({"won": 0, "laureate": 1, "nominated": 2})
    df = (df.sort_values("_r").drop_duplicates(["book_id", "award"])
            .drop(columns="_r").reset_index(drop=True))

    nonnum = int((~df["book_id"].str.fullmatch(r"\d+")).sum())
    if nonnum:
        print(f"WARNING: {nonnum} rows have a non-numeric book_id; the app keys "
              f"awards by int and will skip them")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    per_book = df.groupby("book_id")
    best = per_book.apply(lambda g: (
        "won" if ((g["tier"] == "major") & (g["result"] == "won")).any() else
        "nominated" if (g["tier"] == "major").any() else "minor only"))
    print(f"award books: {n_books:,}, in this corpus: {n_hit:,} "
          f"-> {df['book_id'].nunique():,} corpus rows, {len(df):,} award rows")
    if unparsed:
        print(f"  {unparsed} award strings did not parse and were skipped")
    print("  best MAJOR award per book (this is what the bonus uses):")
    for k, v in best.value_counts().items():
        print(f"    {k:<12}{v:>7,}  ({v / len(corpus):.1%} of the catalogue)")
    print("\n  rows by award (top 25):")
    c = Counter(zip(df["award"], df["result"], df["tier"]))
    for (a, r, t), n in sorted(c.items(), key=lambda x: -x[1])[:25]:
        print(f"    {n:>6,}  {a[:46]:<48}{r:<11}{t}")
    print(f"\nwrote {args.out}  -- upload it to the data repo before flipping "
          f"BOOK_AWARD_WIN on the Space")


if __name__ == "__main__":
    main()
