"""
fetch_hardcover_v2.py — harvest the Hardcover catalogue as the v2 book corpus.

Replaces the frozen 2017 Goodreads dump with a live, API-backed source that
carries real community genre/mood tags, covers, and current release dates.

Structured as a reusable pipeline (like expand_movies.py / expand_tv.py), not a
one-off: stages are independently runnable and resumable, and the same script
becomes the monthly book refresh by ordering on release_date instead.

WHY 120K: the coverage pilot measured field presence by popularity rank.
description>=100 chars and genre-tag presence both hold near 1.0 to rank 10K,
~0.75 at 50K, ~0.66 at 100K, then fall to ~0.45 by 200K. Harvesting 120K yields
roughly 90K usable rows; going deeper spends requests on rows that mostly fail
the description filter.

WORK_ID
recommend_books dedups editions on df["work_id"] and raises KeyError if the
column is absent -- a confirmed live crash that took the books tab down. This
script did not emit it, so re-running the merge regenerated a parquet without
it. The merge stage now CARRIES FORWARD work_id from the existing parquet by
book_id rather than inventing values, because the live column may hold real
work-level ids that a book_id fallback would silently degrade into a no-op.

STAGES
  harvest : page through books by users_count desc -> raw.jsonl (resumable)
  filter  : apply quality gates -> filtered.jsonl + rejects.jsonl (with reasons)
  merge   : build books_v2_hc.parquet (dry-run by default)

USAGE
    $env:HARDCOVER_TOKEN = "..."
    python notebooks\\pipeline\\fetch_hardcover_v2.py --stage harvest --limit 500
    python notebooks\\pipeline\\fetch_hardcover_v2.py --stage harvest
    python notebooks\\pipeline\\fetch_hardcover_v2.py --stage filter
    python notebooks\\pipeline\\fetch_hardcover_v2.py --stage merge
    python notebooks\\pipeline\\fetch_hardcover_v2.py --stage merge --write
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ENDPOINT = "https://api.hardcover.app/v1/graphql"
RAW_DIR = Path("data/raw/books/hardcover_v2")
PROCESSED = Path("data/processed")
RAW_PATH = RAW_DIR / "raw.jsonl"
FILTERED_PATH = RAW_DIR / "filtered.jsonl"
REJECTS_PATH = RAW_DIR / "rejects.jsonl"
STATE_PATH = RAW_DIR / "state.json"
OUT_PARQUET = PROCESSED / "books_v2_hc.parquet"

PAGE_SIZE = 100
TARGET_ROWS = 120_000
SLEEP_BETWEEN = 0.35     # ~1,200 requests total

# Quality gates (filter stage)
MIN_DESC_CHARS = 100
MIN_RATINGS = 5
REQUIRE_GENRE = False    # genre-less rows are still embeddable; keep them

# UNITS WARNING: Hardcover ratings_count is far smaller than Goodreads'.
# Pilot medians by rank: 3,255 (top), 61 (10K), 9 (50K), 5 (100K), 2 (200K).
# The serving floor in app.py (min_ratings=2000, calibrated for Goodreads)
# would wipe out everything past the first few thousand books. It MUST be
# recalibrated against this distribution, not carried over.

BOOKS_QUERY = """
query Harvest($limit: Int!, $offset: Int!) {
  books(
    order_by: {users_count: desc}
    limit: $limit
    offset: $offset
  ) {
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
"""


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


# --------------------------------------------------------------------------
# cached_tags categories: Tag, Mood, Genre, Content Warning
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Stage: harvest
# --------------------------------------------------------------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"next_offset": 0, "rows_written": 0, "seen_ids": []}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def stage_harvest(limit=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    seen = set(state.get("seen_ids", []))
    target = limit or TARGET_ROWS
    offset = state["next_offset"]

    print(f"harvest: target {target:,} rows, resuming at offset {offset:,} "
          f"({len(seen):,} ids already seen)")

    written = state.get("rows_written", 0)
    empty_pages = 0

    with open(RAW_PATH, "a", encoding="utf-8") as fh:
        while written < target:
            data = run_query(BOOKS_QUERY, {"limit": PAGE_SIZE, "offset": offset})
            if data is None:
                print("  query failed after retries; stopping. Re-run to resume.")
                break
            books = data.get("books") or []
            if not books:
                empty_pages += 1
                print(f"  empty page at offset {offset:,}")
                if empty_pages >= 2:
                    print("  catalogue exhausted.")
                    break
                offset += PAGE_SIZE
                continue
            empty_pages = 0

            new = 0
            for b in books:
                bid = b.get("id")
                if bid is None or bid in seen:
                    continue
                seen.add(bid)
                fh.write(json.dumps(b, ensure_ascii=False) + "\n")
                new += 1
            written += new
            offset += PAGE_SIZE

            state.update({"next_offset": offset, "rows_written": written,
                          "seen_ids": list(seen)})
            if (offset // PAGE_SIZE) % 25 == 0:
                save_state(state)
                fh.flush()
                print(f"  offset {offset:,}  rows {written:,}  (+{new} this page)")
            time.sleep(SLEEP_BETWEEN)

    save_state(state)
    print(f"harvest done: {written:,} rows in {RAW_PATH}")


# --------------------------------------------------------------------------
# Stage: filter
# --------------------------------------------------------------------------
def stage_filter():
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} not found — run --stage harvest first")

    kept = 0
    reasons = Counter()
    with open(RAW_PATH, encoding="utf-8") as fin, \
         open(FILTERED_PATH, "w", encoding="utf-8") as fout, \
         open(REJECTS_PATH, "w", encoding="utf-8") as frej:
        for line in fin:
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                reasons["malformed_json"] += 1
                continue

            desc = (b.get("description") or "").strip()
            genres = _tag_list(b.get("cached_tags"), "Genre")
            author = primary_author(b.get("contributions"))

            why = None
            if not b.get("title"):
                why = "no_title"
            elif len(desc) < MIN_DESC_CHARS:
                why = "short_description"
            elif not author:
                why = "no_author"
            elif (b.get("ratings_count") or 0) < MIN_RATINGS:
                why = "below_min_ratings"
            elif REQUIRE_GENRE and not genres:
                why = "no_genre_tag"

            if why:
                reasons[why] += 1
                # rejects logged WITH reason, so a future threshold change can
                # be made from data rather than guesswork — the thing the TV
                # pipeline couldn't do when Pluribus was silently dropped.
                frej.write(json.dumps({"id": b.get("id"), "title": b.get("title"),
                                       "reason": why}, ensure_ascii=False) + "\n")
                continue

            fout.write(json.dumps(b, ensure_ascii=False) + "\n")
            kept += 1

    total = kept + sum(reasons.values())
    print(f"filter: kept {kept:,} of {total:,} ({kept / max(total, 1):.1%})")
    for r, c in reasons.most_common():
        print(f"  rejected {r:<22} {c:,}")
    print(f"  -> {FILTERED_PATH}")
    print(f"  -> {REJECTS_PATH} (auditable; re-runnable after a threshold change)")


# --------------------------------------------------------------------------
# Stage: merge
# --------------------------------------------------------------------------
def stage_merge(write=False):
    import pandas as pd

    if not FILTERED_PATH.exists():
        raise SystemExit(f"{FILTERED_PATH} not found — run --stage filter first")

    rows = []
    with open(FILTERED_PATH, encoding="utf-8") as fh:
        for line in fh:
            b = json.loads(line)
            ct = b.get("cached_tags")
            rd = b.get("release_date") or ""
            year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
            rows.append({
                "book_id": str(b["id"]),
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
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="book_id").reset_index(drop=True)

    # --- work_id: carry forward, never invent --------------------------------
    # recommend_books does `book["work_id"]` unguarded. Regenerating this
    # parquet without the column is a live crash path. The existing parquet is
    # the only source of truth for what the values mean, so join on book_id and
    # report how much was recovered instead of defaulting everything to
    # book_id, which would turn edition dedup into a silent no-op.
    prior_wid = {}
    if OUT_PARQUET.exists():
        try:
            _prev = pd.read_parquet(OUT_PARQUET, columns=["book_id", "work_id"])
            prior_wid = dict(zip(_prev["book_id"].astype(str), _prev["work_id"]))
            print(f"  carried work_id from existing parquet for {len(prior_wid):,} book_ids")
        except (KeyError, ValueError, OSError) as e:
            print(f"  existing parquet has no usable work_id ({type(e).__name__}); "
                  f"falling back to book_id")
    else:
        print("  no existing parquet — work_id falls back to book_id")

    df["work_id"] = [prior_wid.get(b, b) for b in df["book_id"]]
    _fellback = sum(1 for b in df["book_id"] if b not in prior_wid)
    if _fellback:
        print(f"  work_id fallback (== book_id) on {_fellback:,} rows "
              f"({_fellback / max(len(df), 1):.1%})")
        if prior_wid:
            print("    these dedup against nothing. Fine for genuinely new books;")
            print("    a large number here means the join key drifted — check before --write.")
    _degenerate = (df["work_id"].astype(str) == df["book_id"].astype(str)).mean()
    if _degenerate > 0.99:
        print(f"  NOTE: work_id == book_id on {_degenerate:.0%} of rows, so edition")
        print("        dedup is a no-op. The column exists to stop the KeyError;")
        print("        real edition grouping needs a work-level key (normalised")
        print("        title + author) and is a separate, behaviour-changing job.")

    # --- row order vs the live parquet ---------------------------------------
    # If filtered.jsonl is unchanged, this regenerates the SAME rows in the SAME
    # order, which means .npy and FAISS stay valid and this becomes a
    # parquet-only upload with no re-embed. Worth knowing which case you are in
    # BEFORE uploading, not after.
    if OUT_PARQUET.exists():
        try:
            _prev_ids = pd.read_parquet(OUT_PARQUET, columns=["book_id"])["book_id"].astype(str).tolist()
            _now_ids = df["book_id"].astype(str).tolist()
            if _prev_ids == _now_ids:
                print(f"\n  ROW ORDER IDENTICAL to the live parquet ({len(_now_ids):,} rows).")
                print("  Embeddings and FAISS stay valid -> PARQUET-ONLY upload, no re-embed.")
            else:
                print(f"\n  ROW ORDER CHANGED: live {len(_prev_ids):,} rows, now {len(_now_ids):,}.")
                print("  Every stored corpus_index is invalidated and a FULL re-embed +")
                print("  index rebuild is required. Do NOT treat this as a cheap upload.")
                _common = len(set(_prev_ids) & set(_now_ids))
                print(f"  ({_common:,} book_ids in common; "
                      f"{len(set(_now_ids) - set(_prev_ids)):,} new, "
                      f"{len(set(_prev_ids) - set(_now_ids)):,} dropped)")
        except (KeyError, ValueError, OSError):
            pass

    print(f"\nmerge: {len(df):,} rows")
    print(f"  with work_id     {df['work_id'].notna().mean():.1%}")
    print(f"  with genres      {(df['genres'].str.len() > 0).mean():.1%}")
    print(f"  with moods       {(df['moods'].str.len() > 0).mean():.1%}")
    print(f"  with cover       {(df['image_url'] != '').mean():.1%}")
    print(f"  with year        {df['publication_year'].notna().mean():.1%}")
    print(f"  median ratings   {df['ratings_count'].median():.0f}")
    print(f"  median users     {df['users_count'].median():.0f}")

    yh = df[df["publication_year"].notna()]
    if len(yh):
        print("\n  year histogram (2020+):")
        for y in sorted({int(x) for x in yh["publication_year"] if x and x >= 2020}):
            print(f"    {y}  {(yh['publication_year'] == y).sum():,}")

    gc = Counter(g for gs in df["genres"] for g in gs)
    print("\n  top genres:")
    for g, c in gc.most_common(15):
        print(f"    {g:<28} {c:,}")

    mc = Counter(m for ms in df["moods"] for m in ms)
    print("\n  top moods:")
    for m, c in mc.most_common(10):
        print(f"    {m:<28} {c:,}")

    if not write:
        print("\nDRY RUN — nothing written. Re-run with --write once this looks right.")
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)
    if OUT_PARQUET.exists():
        backup = OUT_PARQUET.with_suffix(".parquet.bak")
        if not backup.exists():
            OUT_PARQUET.replace(backup)
            print(f"  existing parquet backed up to {backup}")
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET}  ({len(df):,} rows)")
    print("NOTE: this does NOT touch the live corpus. books_v1.parquet and the")
    print("current embeddings stay as they are until app.py names new files —")
    print("embeddings + FAISS uploaded FIRST, parquet second, constants last.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["harvest", "filter", "merge", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows in harvest (smoke test, e.g. 500)")
    ap.add_argument("--write", action="store_true", help="merge: actually write the parquet")
    args = ap.parse_args()

    if args.stage in ("harvest", "all"):
        stage_harvest(limit=args.limit)
    if args.stage in ("filter", "all"):
        stage_filter()
    if args.stage in ("merge", "all"):
        stage_merge(write=args.write)


if __name__ == "__main__":
    main()
