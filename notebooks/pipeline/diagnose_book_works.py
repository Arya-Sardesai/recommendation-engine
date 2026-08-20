"""
diagnose_book_works.py — is edition dedup doing anything, and what would it cost
to make it work?

THE FINDING THAT PROMPTED THIS
books_v2_hc.parquet has work_id == book_id on 100% of 74,054 rows. The column
exists only because recommend_books does `book["work_id"]` unguarded, so its
absence is a live KeyError. But identical values mean:

    if work_id and work_id in seen_works:   ->  never fires

Edition dedup has never run. _is_title_dup has been carrying that job alone,
and it compares TITLE TOKENS -- so it catches "Dune" vs "Dune", but not
"Dune" vs "Dune: 50th Anniversary Edition" if tokenisation keeps the subtitle,
and not two records of the same work under differently-punctuated titles.

The harvest paginates users_count desc over BOOK RECORDS, not works, and
Hardcover demonstrably stores several records per work -- that is the entire
premise of fetch_hardcover_targeted_v2.py (Maltese Falcon: uc=680 desc=n and a
described sibling further down). So duplicates are expected in the corpus. This
script measures how many, and how many could actually reach a user.

WHAT IT DOES  (read-only, no API, no writes)
  1. groups rows by a work key: normalised core title + normalised author
  2. counts groups with >1 record, and how many of those have 2+ records ABOVE
     the serving floor (only those can collide in a recommendation list)
  3. shows the worst offenders
  4. reports what assigning real work_ids would change

WHY THIS IS CHEAP TO FIX
Assigning work_id from the grouping rewrites COLUMN VALUES ONLY. Row count and
row order are untouched, so the .npy and FAISS index stay valid: parquet-only
upload, no re-embed. Same class of change as the movie certification merge.

USAGE
    python notebooks/pipeline/diagnose_book_works.py
    python notebooks/pipeline/diagnose_book_works.py --write-preview
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
IN_PARQUET = PROCESSED / "books_v2_hc.parquet"
PREVIEW_PARQUET = PROCESSED / "books_v2_hc_workids.parquet"

MIN_RATINGS = 10          # app.py recommend_books serving floor


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


# Subtitles that carry NO distinguishing information -- dropping them groups
# "The Maltese Falcon" with "The Maltese Falcon: A Novel". Anything not on this
# list is treated as substantive.
_GENERIC_SUBTITLE = re.compile(
    r"^(a |an |the )?("
    r"novel|memoir|novella|story|stories|a story|true story|thriller|romance|"
    r"mystery|biography|autobiography|graphic novel|light novel|the manga|"
    r"manga|comic|omnibus|boxed set|box set|"
    r"(\d+(st|nd|rd|th) )?anniversary edition|deluxe edition|special edition|"
    r"collector s edition|complete edition|revised edition|expanded edition|"
    r"illustrated edition|movie tie in edition|reissue|unabridged"
    r")$")

# Volume/part markers anywhere in the title. Vol. 1 and Vol. 2 are DIFFERENT
# BOOKS and must never share a work key.
_VOLUME = re.compile(r"\b(?:vol|volume|book|part|no|number)\s*\.?\s*(\d+)\b")
_TRAILING_NUM = re.compile(r",\s*(\d+)\s*$")


def _core_title(title):
    """Conservative work key for CORPUS-WIDE use.

    NOT the same rule as fetch_hardcover_targeted_v2._core, and deliberately
    so. That function splits on the first ':' and keeps the head, which is
    right for a hand-curated single-author sweep but catastrophic here: it
    collapses every volume of a manga series into one work ("Demon Slayer:
    Kimetsu no Yaiba, Vol. 1..23" -> "demon slayer") and every named sequel
    into its parent ("Dune: The Butlerian Jihad" -> "dune"). Assigning
    work_ids from that key would suppress all but one volume of every series
    in the corpus -- a far worse defect than the duplicates it fixes.

    Here a subtitle is dropped ONLY when it is generic edition/format wording,
    and any volume number is preserved in the key.
    """
    raw = _norm(title)
    vol = None
    m = _VOLUME.search(raw) or _TRAILING_NUM.search(raw)
    if m:
        vol = m.group(1)

    t = re.sub(r"\([^)]*\)", " ", str(title))
    parts = re.split(r":| - ", t, maxsplit=1)
    head = _norm(parts[0])
    tail = _norm(parts[1]) if len(parts) > 1 else ""

    if tail and _GENERIC_SUBTITLE.match(tail):
        core = head
    elif tail:
        core = f"{head} {tail}"
    else:
        core = head

    core = re.sub(r"^(the|a|an) ", "", core).strip()
    if not core:
        return ""
    return f"{core}#{vol}" if vol else core


def work_key(title, author):
    """Title alone is too loose -- 'Ulysses' by Joyce and by Tennyson are
    different works, and generic titles ('Poems', 'Selected Stories') collide
    across dozens of authors. Author is part of the key."""
    ct = _core_title(title)
    ca = _norm(author)
    if not ct:
        return None
    return f"{ct}||{ca}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(IN_PARQUET))
    ap.add_argument("--min-ratings", type=int, default=MIN_RATINGS)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--write-preview", action="store_true",
                    help="write a copy with real work_ids (does NOT touch the live parquet)")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        sys.exit(f"{path} not found — run from the repo root")

    df = pd.read_parquet(path)
    print(f"corpus: {len(df):,} rows")

    if "work_id" not in df.columns:
        print("  work_id column ABSENT — recommend_books would raise KeyError.")
    else:
        deg = (df["work_id"].astype(str) == df["book_id"].astype(str)).mean()
        print(f"  work_id == book_id on {deg:.1%} of rows"
              f"{'  -> edition dedup is a NO-OP' if deg > 0.99 else ''}")

    keys = [work_key(t, a) for t, a in zip(df["title"], df["primary_author"])]
    df = df.assign(_wkey=keys)
    unkeyed = df["_wkey"].isna().sum()
    if unkeyed:
        print(f"  {unkeyed:,} rows produced no usable key (empty title after normalising)")

    kdf = df[df["_wkey"].notna()]
    sizes = kdf["_wkey"].value_counts()
    multi = sizes[sizes > 1]

    print(f"\ndistinct works        : {len(sizes):,}")
    print(f"works with >1 record  : {len(multi):,}")
    print(f"redundant rows        : {int(multi.sum() - len(multi)):,} "
          f"({(multi.sum() - len(multi)) / max(len(df), 1):.1%} of the corpus)")

    # --- the number that matters ------------------------------------------
    # A duplicate only harms a user if TWO records of the same work can both
    # be recommended. Records below the serving floor are unreachable, so a
    # work with one servable record and four dead ones is invisible.
    serv = kdf[kdf["ratings_count"] >= args.min_ratings]
    ssizes = serv["_wkey"].value_counts()
    smulti = ssizes[ssizes > 1]
    print(f"\nabove the serving floor (ratings_count >= {args.min_ratings}): {len(serv):,} rows")
    print(f"  works with 2+ SERVABLE records: {len(smulti):,}")
    print(f"  redundant SERVABLE rows       : {int(smulti.sum() - len(smulti)):,}")
    print("  ^ this is the set that can put the same book in a list twice.")
    if len(smulti) == 0:
        print("    zero here means dedup would change nothing user-visible,")
        print("    and the work_id column is only a crash guard.")

    if len(smulti):
        print(f"\nworst offenders (by servable record count):")
        for k, c in smulti.head(args.show).items():
            rows = serv[serv["_wkey"] == k]
            title = rows.iloc[0]["title"]
            auth = rows.iloc[0]["primary_author"]
            rc = ", ".join(str(int(x)) for x in rows["ratings_count"].head(6))
            print(f"  {c:2d}x  {str(title)[:44]:46s} {str(auth)[:22]:24s} ratings=[{rc}]")

    # --- title-token overlap: what _is_title_dup already catches -----------
    # If the duplicate records share title tokens, the existing check already
    # suppresses them and real work_ids buy little. If they differ (subtitle,
    # edition wording), work_ids are the only thing that would catch them.
    if len(smulti):
        # One grouped pass. The obvious version -- serv[serv["_wkey"] == k]
        # inside a loop over smulti.index -- rescans all servable rows once per
        # duplicate work, which at corpus scale is minutes of apparent hang.
        _dupe = serv[serv["_wkey"].isin(set(smulti.index))]
        _ntitles = (_dupe.assign(_nt=[_norm(t) for t in _dupe["title"]])
                         .groupby("_wkey")["_nt"].nunique())
        both_same_title = int((_ntitles == 1).sum())
        print(f"\n  of {len(smulti):,} servable duplicate works, {both_same_title:,} have "
              f"IDENTICAL normalised titles")
        print("    -> those are likely already caught by _is_title_dup;")
        print(f"    -> the remaining {len(smulti) - both_same_title:,} differ only by")
        print("       subtitle/edition wording and are NOT caught today.")

    if args.write_preview:
        out = df.drop(columns="_wkey").copy()
        # stable id per work: the book_id of the most-rated record in the group
        rep = (kdf.sort_values("ratings_count", ascending=False)
                  .groupby("_wkey")["book_id"].first().to_dict())
        out["work_id"] = [rep.get(k, b) if k else b
                          for k, b in zip(keys, df["book_id"])]
        assert len(out) == len(df), "row count changed"
        assert out["book_id"].tolist() == df["book_id"].tolist(), "ROW ORDER CHANGED"
        out.to_parquet(PREVIEW_PARQUET, index=False)
        changed = (out["work_id"].astype(str) != df["book_id"].astype(str)).sum()
        print(f"\nwrote {PREVIEW_PARQUET}")
        print(f"  {changed:,} rows now carry a work_id different from their book_id")
        print("  row order preserved -> .npy and FAISS stay valid, parquet-only upload")
        print("  NOT live: app.py still names books_v2_hc.parquet")


if __name__ == "__main__":
    main()
