"""Hardcover coverage pilot — the go/no-go before harvesting 250K books.

The question this answers: does Hardcover carry genres, descriptions, covers
and ratings ALL THE WAY DOWN, or only for the popular head? The existing 23,673
Hardcover rows in books_v1 are the head of their catalogue and are 0.24% tagged
/ 0% genre'd, which tells us nothing about depth.

Samples across popularity strata (by users_count rank) and reports per-stratum
field coverage. If coverage collapses past ~rank 50K, harvesting 250K yields
mostly unusable rows and the plan needs to change.

Usage:
    $env:HARDCOVER_TOKEN = "..."
    python pilot_hardcover.py

Writes data/raw/books/hardcover_v2/pilot.json for later reference.
"""
import json
import os
import time
from pathlib import Path

import requests

TOKEN = os.environ.get("HARDCOVER_TOKEN")
if not TOKEN:
    raise SystemExit("Set HARDCOVER_TOKEN first: $env:HARDCOVER_TOKEN = 'your_token'")

ENDPOINT = "https://api.hardcover.app/v1/graphql"
HEADERS = {"authorization": TOKEN, "Content-Type": "application/json"}

OUT = Path("data/raw/books/hardcover_v2")
OUT.mkdir(parents=True, exist_ok=True)

# Sample 100 books at each of these offsets in the users_count-desc ordering.
# If the corpus is ~250K deep, these probe the head, the middle, and the tail.
STRATA = [0, 10_000, 50_000, 100_000, 200_000]
PER_STRATUM = 100

QUERY = """
query Probe($limit: Int!, $offset: Int!) {
  books(
    order_by: {users_count: desc}
    limit: $limit
    offset: $offset
  ) {
    id
    title
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


def run_query(query, variables=None, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(ENDPOINT, headers=HEADERS,
                                 json={"query": query, "variables": variables or {}},
                                 timeout=45)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                print("  GraphQL errors:", json.dumps(data["errors"])[:400])
                return None
            return data.get("data")
        except Exception as e:
            print(f"  request failed ({e}); retry {attempt + 1}/{retries}")
            time.sleep(2 * (attempt + 1))
    return None


def genre_tags(cached_tags):
    """cached_tags is a dict of category -> list of tag objects.
    Genre-specific tags live under the 'Genre' key."""
    if not isinstance(cached_tags, dict):
        return []
    for key in ("Genre", "genre", "Genres"):
        if key in cached_tags and isinstance(cached_tags[key], list):
            out = []
            for t in cached_tags[key]:
                if isinstance(t, dict):
                    out.append(t.get("tag") or t.get("name") or "")
                elif isinstance(t, str):
                    out.append(t)
            return [x for x in out if x]
    return []


def all_tags(cached_tags):
    if not isinstance(cached_tags, dict):
        return []
    out = []
    for v in cached_tags.values():
        if isinstance(v, list):
            for t in v:
                if isinstance(t, dict):
                    out.append(t.get("tag") or t.get("name") or "")
                elif isinstance(t, str):
                    out.append(t)
    return [x for x in out if x]


def has_cover(book):
    ci = book.get("cached_image")
    if isinstance(ci, dict):
        return bool(ci.get("url"))
    return bool(ci)


print("=" * 78)
print("HARDCOVER COVERAGE PILOT")
print("=" * 78)

# quick auth check
me = run_query("query { me { username } }")
print(f"auth: {me}\n")

results = {}
raw_samples = {}

for offset in STRATA:
    print(f"--- rank {offset:,} .. {offset + PER_STRATUM:,} ---")
    data = run_query(QUERY, {"limit": PER_STRATUM, "offset": offset})
    if not data or not data.get("books"):
        print("  no rows returned — corpus may be shallower than this offset\n")
        results[offset] = None
        continue
    books = data["books"]
    raw_samples[offset] = books[:3]

    n = len(books)
    desc_ok = sum(1 for b in books if (b.get("description") or "") and len(b["description"]) >= 100)
    genres_ok = sum(1 for b in books if genre_tags(b.get("cached_tags")))
    anytag_ok = sum(1 for b in books if all_tags(b.get("cached_tags")))
    cover_ok = sum(1 for b in books if has_cover(b))
    rating_ok = sum(1 for b in books if b.get("rating"))
    date_ok = sum(1 for b in books if b.get("release_date"))
    author_ok = sum(1 for b in books if b.get("contributions"))
    uc = [b.get("users_count") or 0 for b in books]
    rc = [b.get("ratings_count") or 0 for b in books]

    genre_counts = [len(genre_tags(b.get("cached_tags"))) for b in books]
    tag_counts = [len(all_tags(b.get("cached_tags"))) for b in books]

    results[offset] = {
        "n": n,
        "description>=100": round(desc_ok / n, 3),
        "has_genre_tag": round(genres_ok / n, 3),
        "has_any_tag": round(anytag_ok / n, 3),
        "has_cover": round(cover_ok / n, 3),
        "has_rating": round(rating_ok / n, 3),
        "has_release_date": round(date_ok / n, 3),
        "has_author": round(author_ok / n, 3),
        "users_count_median": sorted(uc)[n // 2],
        "ratings_count_median": sorted(rc)[n // 2],
        "median_genre_tags": sorted(genre_counts)[n // 2],
        "median_all_tags": sorted(tag_counts)[n // 2],
    }
    for k, v in results[offset].items():
        print(f"  {k:<22} {v}")
    print()
    time.sleep(1)

# ---------------------------------------------------------------------------
print("=" * 78)
print("SUMMARY — usable = description>=100 AND has_genre_tag AND has_rating")
print("=" * 78)
print(f"{'rank':>10} {'n':>5} {'desc':>7} {'genre':>7} {'cover':>7} {'rating':>7} {'users_med':>11}")
print("-" * 78)
for offset in STRATA:
    r = results.get(offset)
    if r is None:
        print(f"{offset:>10,} {'-':>5} {'(no rows — corpus ends before here)':>50}")
        continue
    print(f"{offset:>10,} {r['n']:>5} {r['description>=100']:>7} {r['has_genre_tag']:>7} "
          f"{r['has_cover']:>7} {r['has_rating']:>7} {r['users_count_median']:>11,}")

print()
print("READ THIS:")
print("  If has_genre_tag stays high (>0.6) through rank 100K+, the full swap is on.")
print("  If it collapses past rank ~50K, harvest only as deep as coverage holds.")
print("  If descriptions thin out, that caps the embeddable corpus regardless of tags.")

with open(OUT / "pilot.json", "w", encoding="utf-8") as f:
    json.dump({"results": results, "samples": raw_samples}, f, indent=2)
print(f"\nwrote {OUT / 'pilot.json'}")

# show one raw record so we can see the real shape of cached_tags
if raw_samples:
    first = raw_samples[STRATA[0]][0]
    print("\n--- one raw record (field shape check) ---")
    print(json.dumps({k: first.get(k) for k in
                      ("id", "title", "release_date", "rating", "ratings_count",
                       "users_count", "cached_image")}, indent=2)[:800])
    print("cached_tags keys:", list(first.get("cached_tags", {}).keys())
          if isinstance(first.get("cached_tags"), dict) else type(first.get("cached_tags")))
