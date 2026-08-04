"""
Fetch ~10K popular books released >= 2018 from Hardcover, for backfilling the
recency gap in the Goodreads corpus.

- Paginates with limit/offset, polite delay between requests
- Backs off and retries on rate-limit / transient errors
- Saves incrementally to JSONL so a mid-run failure doesn't lose progress
- Resumable: re-run and it continues from where it stopped

Set token first:  $env:HARDCOVER_TOKEN = "your_token"
Run:              python notebooks/fetch_hardcover.py
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

# config
TARGET = 25_000            # how many books to fetch
PAGE_SIZE = 50             # books per request
DELAY = 0.6               # seconds between requests (polite)
MIN_DELAY_ON_ERROR = 5     # backoff base on error
RELEASE_AFTER = "2018-01-01"

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "hardcover_recent.jsonl"

QUERY = """
query FetchBooks($limit: Int!, $offset: Int!) {
  books(
    where: {release_date: {_gte: "%s"}, description: {_is_null: false}}
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
    contributions {
      author { name }
    }
  }
}
""" % RELEASE_AFTER


def run_query(limit, offset, max_retries=4):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                ENDPOINT, headers=HEADERS,
                json={"query": QUERY, "variables": {"limit": limit, "offset": offset}},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "errors" in data:
                    print(f"  GraphQL error at offset {offset}: {data['errors']}")
                    return None
                return data["data"]["books"]
            else:
                wait = MIN_DELAY_ON_ERROR * (attempt + 1)
                print(f"  HTTP {resp.status_code} at offset {offset}, retrying in {wait}s...")
                time.sleep(wait)
        except requests.RequestException as e:
            wait = MIN_DELAY_ON_ERROR * (attempt + 1)
            print(f"  Request error at offset {offset}: {e}, retrying in {wait}s...")
            time.sleep(wait)
    print(f"  FAILED at offset {offset} after {max_retries} retries")
    return None


def main():
    # resume support: count how many already saved
    start_offset = 0
    if OUT_FILE.exists():
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        start_offset = existing
        print(f"Resuming: {existing} books already saved, continuing from offset {start_offset}")

    fetched = start_offset
    offset = start_offset

    with open(OUT_FILE, "a", encoding="utf-8") as out:
        while fetched < TARGET:
            books = run_query(PAGE_SIZE, offset)
            if books is None:
                print("Stopping due to repeated errors. Re-run to resume.")
                break
            if len(books) == 0:
                print(f"No more books returned at offset {offset}. Reached end of results.")
                break
            for b in books:
                out.write(json.dumps(b, ensure_ascii=False) + "\n")
            out.flush()
            fetched += len(books)
            offset += len(books)
            print(f"  fetched {fetched:,} / {TARGET:,} (last: {books[-1]['title'][:40]})")
            time.sleep(DELAY)

    print(f"\nDone. Total books in {OUT_FILE.name}: {fetched:,}")


if __name__ == "__main__":
    main()