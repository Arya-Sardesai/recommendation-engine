"""
What does Hardcover actually have for a given AUTHOR, and is it usable?

v1 of this script queried by title with _eq, which silently returned the wrong
book when a title is shared ("Black Friday" -> Jan Stryvant, "Ravensong" ->
TJ Klune). Querying by author avoids that entirely.

The decisive column is DESC. The corpus is description-embedded, so a book with
no description cannot be embedded meaningfully -- it is unusable even if
harvested. HARVEST FLOOR for books_v2_hc is users_count >= 13.

Buckets printed at the end:
  IN CORPUS      - already there, nothing to do
  FETCHABLE      - has a description, below the floor -> targeted fetch
  NO DESCRIPTION - needs a second metadata source (Open Library / Google Books)

Run from the main repo root:
    python scripts/diagnose_authors.py
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

# --- edit: authors whose books you know are missing --------------------------
AUTHORS = [
    "S. Hussain Zaidi",
    "Devdutt Pattanaik",
    "Mahasweta Devi",
    "Lee Maracle",
    "Dashiell Hammett",
]
# -----------------------------------------------------------------------------

HARVEST_FLOOR = 13          # lowest users_count that survived the v2 harvest
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "data" / "processed" / "books_v2_hc.parquet"

API = "https://api.hardcover.app/v1/graphql"
TOKEN = os.environ.get("HARDCOVER_TOKEN", "")
if not TOKEN:
    raise SystemExit("Set HARDCOVER_TOKEN first (full 'Bearer ...' string).")
HEADERS = {"authorization": TOKEN, "content-type": "application/json"}

Q = """
query($name: String!) {
  authors(where: {name: {_eq: $name}}, limit: 3) {
    name
    contributions(limit: 60) {
      book {
        id
        title
        users_count
        ratings_count
        description
      }
    }
  }
}
"""


def main():
    corpus_titles = set()
    if CORPUS.exists():
        c = pd.read_parquet(CORPUS, columns=["title"])
        corpus_titles = set(c["title"].str.lower().str.strip())
        print(f"corpus: {len(corpus_titles):,} titles   floor: users_count >= {HARVEST_FLOOR}\n")

    buckets = {"IN CORPUS": [], "FETCHABLE": [], "NO DESCRIPTION": [], "ABOVE FLOOR?": []}

    for name in AUTHORS:
        r = requests.post(API, headers=HEADERS,
                          json={"query": Q, "variables": {"name": name}}, timeout=30)
        time.sleep(0.35)
        if r.status_code != 200:
            print(f"{name}: HTTP {r.status_code}")
            continue
        data = r.json()
        if "errors" in data:
            print(f"{name}: {str(data['errors'])[:120]}")
            continue

        authors = data["data"]["authors"]
        if not authors:
            print(f"{name}: NOT FOUND in Hardcover\n")
            continue

        books, seen = [], set()
        for a in authors:
            for c_ in (a.get("contributions") or []):
                b = c_.get("book")
                if b and b["id"] not in seen:
                    seen.add(b["id"])
                    books.append(b)

        print(f"=== {name}  ({len(books)} books in Hardcover)")
        for b in sorted(books, key=lambda x: -(x.get("users_count") or 0)):
            t = b["title"]
            uc = b.get("users_count") or 0
            has_desc = len((b.get("description") or "").strip()) >= 100
            in_corpus = t.lower().strip() in corpus_titles

            if in_corpus:
                tag = "IN CORPUS"
            elif not has_desc:
                tag = "NO DESCRIPTION"
            elif uc >= HARVEST_FLOOR:
                tag = "ABOVE FLOOR?"      # should have been harvested; investigate
            else:
                tag = "FETCHABLE"
            buckets[tag].append(f"{t} ({name})")
            print(f"   {tag:15s} uc={uc:<6d} desc={'y' if has_desc else 'n'}  {t[:52]}")
        print()

    print("=" * 78)
    for k, v in buckets.items():
        print(f"{k:15s} {len(v)}")
    if buckets["FETCHABLE"]:
        print("\nFETCHABLE -> add these via fetch_hardcover_targeted.py:")
        for t in buckets["FETCHABLE"]:
            print("   ", t)
    if buckets["NO DESCRIPTION"]:
        print("\nNO DESCRIPTION -> unusable without a second metadata source:")
        for t in buckets["NO DESCRIPTION"][:15]:
            print("   ", t)


if __name__ == "__main__":
    main()
