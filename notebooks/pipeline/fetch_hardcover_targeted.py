# notebooks/pipeline/fetch_hardcover_targeted.py
"""
Manual additions tool — for books that slip through the main Hardcover fetch.

Usage:
  1. Add titles to TARGETS list below
  2. Run from repo root: python notebooks/pipeline/fetch_hardcover_targeted.py
  3. Re-run merge + embed pipeline to include new books

Re-run safe: already-fetched books are skipped automatically.
"""
import os
import json
import time
from pathlib import Path
import requests

TOKEN = os.environ.get("HARDCOVER_TOKEN")
if not TOKEN:
    raise SystemExit("Set HARDCOVER_TOKEN env var first.")

ENDPOINT = "https://api.hardcover.app/v1/graphql"
HEADERS = {"authorization": TOKEN, "Content-Type": "application/json"}

OUT_FILE = Path("data/raw/hardcover_recent.jsonl")

# ---------------------------------------------------------------
# ADD TITLES HERE — exact title preferred, close enough also works
# ---------------------------------------------------------------
TARGETS = [
    "Malibu Rising",
    "Daisy Jones & The Six",
    "The Seven Husbands of Evelyn Hugo",
    "People We Meet on Vacation",
    "Beach Read",
]
# ---------------------------------------------------------------

FIELDS = "id title description release_date pages rating ratings_count users_count contributions { author { name } }"

QUERY_EXACT = """
query SearchBook($title: String!) {
  books(where: {title: {_eq: $title}, description: {_is_null: false}}, limit: 3) {
    %s
  }
}
""" % FIELDS

QUERY_FUZZY = """
query SearchBook($title: String!) {
  books(where: {title: {_ilike: $title}, description: {_is_null: false}}, limit: 3) {
    %s
  }
}
""" % FIELDS


def fetch(query, title):
    r = requests.post(
        ENDPOINT, headers=HEADERS,
        json={"query": query, "variables": {"title": title}},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for '{title}'")
        return []
    return r.json().get("data", {}).get("books", [])


def best_edition(books):
    """Prefer most recent edition with a real release date."""
    dated = [b for b in books if b.get("release_date")]
    return max(dated, key=lambda b: b["release_date"]) if dated else books[0]


def main():
    existing_ids = set()
    if OUT_FILE.exists():
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    print(f"Existing books: {len(existing_ids):,} (dedup check active)\n")

    added = 0
    with open(OUT_FILE, "a", encoding="utf-8") as out:
        for title in TARGETS:
            # Try exact first
            books = fetch(QUERY_EXACT, title)

            # Fall back to fuzzy
            if not books:
                print(f"  Exact match failed for '{title}', trying fuzzy...")
                books = fetch(QUERY_FUZZY, f"%{title}%")

            if not books:
                print(f"  NOT FOUND: {title}")
                continue

            b = best_edition(books)

            if b["id"] in existing_ids:
                print(f"  SKIP (already exists): {b['title']}")
                continue

            out.write(json.dumps(b, ensure_ascii=False) + "\n")
            existing_ids.add(b["id"])
            added += 1
            print(f"  ADDED: {b['title']} ({b.get('release_date', '?')})")
            time.sleep(0.3)

    print(f"\nDone. Added {added} new books.")
    if added > 0:
        print("Remember to re-run the pipeline: merge -> rescale -> embed -> rematch -> upload")


if __name__ == "__main__":
    main()