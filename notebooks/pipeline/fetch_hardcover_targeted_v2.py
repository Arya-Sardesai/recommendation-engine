"""
fetch_hardcover_targeted_v2.py — add specific books the bulk harvest missed.

WHY THIS EXISTS
Hardcover stores multiple records per work, and description availability varies
BETWEEN them. fetch_hardcover_v2.py paginates by users_count desc, so for a
given work it meets the most-read record first -- and if that record has no
description it is rejected on MIN_DESC_CHARS and the work is gone, even though
a less-read duplicate carrying a full description sits further down the list.

Measured examples:
    uc=680  desc=n   The Maltese Falcon                  <- rejected, work lost
    uc=25   desc=n   Dongri to Dubai                     <- rejected, work lost
    uc=1    desc=y   Dongri to Dubai - Six Decades ...   <- never reached

This script inverts the selection: gather every record for an author, group by
core title, and keep the record that HAS a description (highest users_count
among those), not the most popular record overall.

RATINGS FLOOR IS RELAXED HERE ON PURPOSE. MIN_RATINGS=5 in the bulk pipeline
keeps junk out of a 120K crawl; it is the wrong gate for a hand-curated list,
where the described record of a real book often has ratings_count 0-1. These
books sit below app.py's serving floor (min_ratings=10) so they will not be
RECOMMENDED -- but they can be SEARCHED and RATED, which is the entire point:
a book only needs an embedding to work as a profile anchor. No ratings gate is
applied in this script at all; that is deliberate, not an omission.

APPEND-ONLY. New rows go at the end of the parquet, so existing row positions
(and therefore every corpus_index in every user's localStorage) stay valid.
No migration, no 'key < ntotal'.

USAGE
    $env:HARDCOVER_TOKEN = "..."
    python notebooks\\pipeline\\fetch_hardcover_targeted_v2.py --stage fetch
    python notebooks\\pipeline\\fetch_hardcover_targeted_v2.py --stage merge
    python notebooks\\pipeline\\fetch_hardcover_targeted_v2.py --stage merge --write

Run from the REPO ROOT — the paths below are relative to it.

THEN (order matters -- alignment law):
    embed ONLY rows [start_index:] and concatenate to the existing .npy
    rebuild FAISS from the concatenated array
    re-run the tag build over the new parquet
    verify rows == npy.shape[0] == index.ntotal, self-match probe an appended row
    upload embeddings + index FIRST, parquet SECOND, flip B_* constants LAST
"""
import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# CURATE THIS LIST. Authors whose catalogue should be swept for described
# records, plus optional explicit titles to restrict to.
# titles=None  -> take every described work by that author
# titles=[...] -> only works whose core title matches one of these
# --------------------------------------------------------------------------
TARGETS = [
    {"author": "S. Hussain Zaidi", "titles": None},
    {"author": "Mahasweta Devi", "titles": ["Imaginary Maps", "Truth/Untruth"]},
    {"author": "Lee Maracle", "titles": ["Ravensong", "Celia's Song"]},
    {"author": "Dashiell Hammett", "titles": ["The Maltese Falcon"]},
    {"author": "Devdutt Pattanaik",
     "titles": ["Shikhandi", "My Gita", "Myth = Mithya", "The Pregnant King",
                "7 Secrets Of Shiva", "The Book of Ram"]},
]

ENDPOINT = "https://api.hardcover.app/v1/graphql"
RAW_DIR = Path("data/raw/books/hardcover_v2")
PROCESSED = Path("data/processed")
TARGET_RAW = RAW_DIR / "targeted_raw.jsonl"
TARGET_REJECTS = RAW_DIR / "targeted_rejects.jsonl"
BASE_PARQUET = PROCESSED / "books_v2_hc.parquet"
OUT_PARQUET = PROCESSED / "books_v2_hc2.parquet"

MIN_DESC_CHARS = 100     # same as bulk: below this there is nothing to embed
SLEEP_BETWEEN = 0.35

# Author lookup is an exact string match server-side. If a name is stored
# differently ("Hussain Zaidi" vs "S. Hussain Zaidi") the query returns nothing
# and the script prints NOT FOUND -- which means "no author by that exact
# name", NOT "this author has no books". Set FUZZY_AUTHOR to retry with a
# case-insensitive contains match before giving up.
FUZZY_AUTHOR = True

AUTHOR_QUERY = """
query Targeted($name: String!) {
  authors(where: {name: {_eq: $name}}, limit: 3) {
    name
    contributions(limit: 200) {
      book {
        id
        slug
        title
        subtitle
        description
        release_date
        pages
        rating
        ratings_count
        users_count
        cached_tags
        cached_image
        contributions { author { name } }
      }
    }
  }
}
"""

AUTHOR_QUERY_FUZZY = AUTHOR_QUERY.replace(
    'where: {name: {_eq: $name}}', 'where: {name: {_ilike: $name}}')


def headers():
    token = os.environ.get("HARDCOVER_TOKEN")
    if not token:
        raise SystemExit("Set HARDCOVER_TOKEN first: $env:HARDCOVER_TOKEN = 'your_token'")
    return {"authorization": token, "Content-Type": "application/json"}


def run_query(query, variables=None, retries=4):
    for attempt in range(retries):
        try:
            resp = requests.post(ENDPOINT, headers=headers(),
                                 json={"query": query, "variables": variables or {}},
                                 timeout=60)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                print("    GraphQL errors:", json.dumps(data["errors"])[:300])
                return None
            return data.get("data")
        except Exception as e:
            wait = 3 * (attempt + 1)
            print(f"    request failed ({type(e).__name__}: {e}); retry in {wait}s")
            time.sleep(wait)
    return None


def fetch_author(name):
    """Exact name first, then a contains match if that returns nothing."""
    data = run_query(AUTHOR_QUERY, {"name": name})
    time.sleep(SLEEP_BETWEEN)
    if data and data.get("authors"):
        return data
    if not FUZZY_AUTHOR:
        return data
    # last token is the most stable part of a name across catalogues
    stem = name.split()[-1]
    print(f"   exact match failed; retrying on %{stem}%")
    data = run_query(AUTHOR_QUERY_FUZZY, {"name": f"%{stem}%"})
    time.sleep(SLEEP_BETWEEN)
    if data and data.get("authors"):
        found = [a.get("name") for a in data["authors"]]
        print(f"   fuzzy matched: {found}")
    return data


# --- verbatim from fetch_hardcover_v2.py so both paths parse identically ----
def _tag_list(cached_tags, category):
    if not isinstance(cached_tags, dict):
        return []
    items = cached_tags.get(category)
    if not isinstance(items, list):
        return []
    out = []
    for t in items:
        if isinstance(t, dict):
            name = t.get("tag") or t.get("name") or ""
            count = t.get("count") or t.get("tagged_count") or 0
            if name:
                out.append({"tag": str(name), "count": int(count) if count else 0})
        elif isinstance(t, str) and t:
            out.append({"tag": t, "count": 0})
    return out


def cover_url(cached_image):
    if isinstance(cached_image, dict):
        return cached_image.get("url") or ""
    if isinstance(cached_image, str):
        return cached_image
    return ""


def primary_author(contributions):
    if not isinstance(contributions, list):
        return ""
    for c in contributions:
        if isinstance(c, dict):
            a = c.get("author")
            if isinstance(a, dict) and a.get("name"):
                return str(a["name"])
    return ""
# ---------------------------------------------------------------------------


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(title):
    """Group key: subtitle, parentheticals, and a leading article removed, so
    the description-less record and its described duplicate land in the same
    bucket ('The Pregnant King' / 'Pregnant King : A Novel')."""
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    t = _norm(t)
    return re.sub(r"^(the|a|an) ", "", t).strip()


def stage_fetch():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    picked, rejected = [], []

    for target in TARGETS:
        name = target["author"]
        wanted = {_core(t) for t in target["titles"]} if target.get("titles") else None

        data = fetch_author(name)
        if not data or not data.get("authors"):
            print(f"{name}: NOT FOUND "
                  f"(no author under this name — not proof the books are absent)")
            continue

        books, seen = [], set()
        for a in data["authors"]:
            for c in (a.get("contributions") or []):
                b = c.get("book")
                if b and b["id"] not in seen:
                    seen.add(b["id"])
                    books.append(b)

        # group by core title, then prefer a record that HAS a description
        groups = {}
        for b in books:
            groups.setdefault(_core(b.get("title") or ""), []).append(b)

        print(f"\n=== {name}: {len(books)} records, {len(groups)} distinct works")

        if wanted is not None:
            unseen = wanted - set(groups)
            if unseen:
                print(f"   requested titles with NO record at all: {sorted(unseen)}")

        for core, recs in groups.items():
            if wanted is not None and core not in wanted:
                continue
            described = [b for b in recs
                         if len((b.get("description") or "").strip()) >= MIN_DESC_CHARS]
            if not described:
                best = max(recs, key=lambda b: b.get("users_count") or 0)
                rejected.append({"id": best.get("id"), "title": best.get("title"),
                                 "author": name, "reason": "no_described_record",
                                 "records_checked": len(recs)})
                print(f"   SKIP  {best.get('title','')[:52]:54s} "
                      f"({len(recs)} records, none described)")
                continue
            best = max(described, key=lambda b: b.get("users_count") or 0)
            if not primary_author(best.get("contributions")):
                rejected.append({"id": best.get("id"), "title": best.get("title"),
                                 "author": name, "reason": "no_author"})
                continue
            picked.append(best)
            others = len(recs) - 1
            print(f"   KEEP  {best.get('title','')[:52]:54s} "
                  f"uc={best.get('users_count') or 0:<5} "
                  f"ratings={best.get('ratings_count') or 0:<4}"
                  f"{f'  (chosen over {others} sibling record(s))' if others else ''}")

    with open(TARGET_RAW, "w", encoding="utf-8") as fh:
        for b in picked:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(TARGET_REJECTS, "w", encoding="utf-8") as fh:
        for r in rejected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nfetch: kept {len(picked)}, skipped {len(rejected)}")
    print(f"  -> {TARGET_RAW}")
    print(f"  -> {TARGET_REJECTS}")
    print("NOTE: this stage OVERWRITES targeted_raw.jsonl. It is not resumable;")
    print("      re-running repeats the whole sweep (~30s), which is fine.")


def stage_merge(write=False):
    import pandas as pd

    if not TARGET_RAW.exists():
        raise SystemExit(f"{TARGET_RAW} not found — run --stage fetch first")
    if not BASE_PARQUET.exists():
        raise SystemExit(f"{BASE_PARQUET} not found")

    base = pd.read_parquet(BASE_PARQUET)
    existing_ids = set(base["book_id"].astype(str))
    print(f"base corpus: {len(base):,} rows")

    rows, dupes = [], 0
    with open(TARGET_RAW, encoding="utf-8") as fh:
        for line in fh:
            b = json.loads(line)
            bid = str(b["id"])
            if bid in existing_ids:
                dupes += 1
                continue
            rd = b.get("release_date") or ""
            year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
            ct = b.get("cached_tags")
            rows.append({
                "book_id": bid,
                "slug": b.get("slug") or "",
                "title": b.get("title") or "",
                "subtitle": b.get("subtitle") or "",
                "description": (b.get("description") or "").strip(),
                "primary_author": primary_author(b.get("contributions")),
                "genres": [t["tag"] for t in _tag_list(ct, "Genre")],
                "moods": [t["tag"] for t in _tag_list(ct, "Mood")],
                "tags": [t["tag"] for t in _tag_list(ct, "Tag")],
                "publication_year": year,
                "release_date": rd,
                "num_pages": b.get("pages"),
                "average_rating": b.get("rating"),
                "ratings_count": b.get("ratings_count") or 0,
                "users_count": b.get("users_count") or 0,
                "image_url": cover_url(b.get("cached_image")),
                "source": "hardcover",
                # missing from the bulk merge block; add it there too or the
                # next full run regenerates a parquet without it and
                # recommend_books raises KeyError.
                "work_id": bid,
            })

    if not rows:
        print(f"nothing new to append ({dupes} already in corpus).")
        return

    new = pd.DataFrame(rows).drop_duplicates(subset="book_id").reset_index(drop=True)

    # --- schema reconciliation, BEFORE the reindex silently applies it -------
    # reindex(columns=base.columns) drops any field the base lacks and NaN-fills
    # any column the row dict does not build. Both are silent. If work_id is in
    # the dropped list, the KeyError guard this script exists to carry is being
    # thrown away here, and the failure resurfaces after the upload ritual.
    built = set(rows[0])
    base_cols = set(base.columns)
    dropped = sorted(built - base_cols)
    filled = sorted(base_cols - built)

    if dropped:
        raise SystemExit(
            f"\nthese fields would be DISCARDED by the column reindex: {dropped}\n"
            f"the base parquet has no such column.\n"
            f"  - if 'work_id' is listed: add it to the base parquet (and to\n"
            f"    fetch_hardcover_v2.py's merge rows block) FIRST, or this\n"
            f"    script's crash fix does nothing.\n"
            f"  - otherwise drop the field from the row dict above.\n"
            f"base columns are: {sorted(base_cols)}"
        )
    if filled:
        print(f"\n  WARNING: appended rows will be NaN in {filled}")
        print("  Check each against the embed step and recommend_books before")
        print("  --write. A null embed_text on exactly the rows being embedded")
        print("  produces empty vectors that still pass the alignment asserts.")

    new = new.reindex(columns=list(base.columns))   # exact column order

    print(f"\nappending  : {len(new):,} rows  ({dupes} skipped as already present)")
    print(f"  with genres  {(new['genres'].str.len() > 0).mean():.0%}")
    print(f"  with moods   {(new['moods'].str.len() > 0).mean():.0%}")
    print(f"  with cover   {(new['image_url'] != '').mean():.0%}")
    print(f"  median ratings {new['ratings_count'].median():.0f} "
          f"(below app.py min_ratings=10 -> ratable as anchors, not recommended)")
    print("\n  appended titles:")
    for _, r in new.iterrows():
        print(f"    {r['title'][:58]:60s} {str(r['primary_author'])[:26]}")

    out = pd.concat([base, new], ignore_index=True)

    # APPEND-ONLY PROOF: every pre-existing row must keep its exact position,
    # or stored corpus_index values silently point at the wrong book.
    assert len(out) == len(base) + len(new), "row count mismatch"
    assert out["book_id"].iloc[:len(base)].tolist() == base["book_id"].tolist(), \
        "existing rows moved — refusing to write"
    assert out["book_id"].is_unique, "duplicate book_id after concat"
    assert list(out.columns) == list(base.columns), "column set changed"
    print("\n  append-only verified: rows 0..%d unchanged" % (len(base) - 1))
    print(f"  new rows occupy positions {len(base):,}..{len(out) - 1:,}")
    print(f"  EMBED ONLY FROM INDEX {len(base):,} ONWARD")

    if not write:
        print("\nDRY RUN — nothing written. Re-run with --write.")
        return

    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET}  ({len(out):,} rows)")
    print("Live corpus untouched. Next: embed rows "
          f"[{len(base):,}:] and concatenate to the existing .npy.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fetch", "merge", "all"], default="all")
    ap.add_argument("--write", action="store_true", help="merge: actually write the parquet")
    args = ap.parse_args()

    if args.stage in ("fetch", "all"):
        stage_fetch()
    if args.stage in ("merge", "all"):
        stage_merge(write=args.write)


if __name__ == "__main__":
    main()
