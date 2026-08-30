"""
Diagnose WHY specific books are missing from books_v2_hc.parquet.

Three possible causes, and they need different fixes:
  A. below the harvest depth  -> deeper harvest would catch them
  B. in Hardcover but unpopular -> targeted fetch (fetch_hardcover_targeted.py)
  C. not in Hardcover at all    -> nothing to do

Run from the main repo root:
    python scripts/diagnose_missing_books.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

# --- edit this list: the books you know are missing -------------------------
MISSING = [
    ("Dongri to Dubai", "Hussain Zaidi"),
    ("Black Friday", "Hussain Zaidi"),
    ("Mafia Queens of Mumbai", "Hussain Zaidi"),
    ("Byculla to Bangkok", "Hussain Zaidi"),
    ("Imaginary Maps", "Mahasweta Devi"),
    ("Ravensong", "Lee Maracle"),
]
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "data" / "processed" / "books_v2_hc.parquet"
RAW = REPO_ROOT / "data" / "raw" / "hardcover_v2" / "raw.jsonl"

API = "https://api.hardcover.app/v1/graphql"
TOKEN = os.environ.get("HARDCOVER_TOKEN", "")
if not TOKEN:
    raise SystemExit("Set HARDCOVER_TOKEN first (full 'Bearer ...' string).")
HEADERS = {"authorization": TOKEN, "content-type": "application/json"}

# _eq not _ilike: _ilike returns 403 on this API (learned in v1).
Q = """
query($t: String!) {
  books(where: {title: {_eq: $t}}, order_by: {users_count: desc}, limit: 5) {
    id
    title
    users_count
    ratings_count
    description
    contributions { author { name } }
  }
}
"""


def hc_lookup(title):
    r = requests.post(API, headers=HEADERS,
                      json={"query": Q, "variables": {"t": title}}, timeout=30)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    payload = r.json()
    if "errors" in payload:
        return None, str(payload["errors"])[:120]
    return payload["data"]["books"], None


def harvest_floor():
    """users_count of the least popular row we actually harvested."""
    if not RAW.exists():
        return None
    last = None
    with RAW.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if not last:
        return None
    try:
        return json.loads(last).get("users_count")
    except json.JSONDecodeError:
        return None


def main():
    floor = harvest_floor()
    print(f"harvest floor (users_count of last harvested row): {floor}")

    corpus = None
    if CORPUS.exists():
        corpus = pd.read_parquet(CORPUS, columns=["title", "primary_author"])
        corpus["_t"] = corpus["title"].str.lower().str.strip()
        print(f"corpus: {len(corpus):,} rows\n")

    verdicts = []
    for title, author in MISSING:
        in_corpus = False
        if corpus is not None:
            in_corpus = (corpus["_t"] == title.lower().strip()).any()

        books, err = hc_lookup(title)
        time.sleep(0.35)

        if err:
            print(f"{title:32s} ERROR {err}")
            continue

        if not books:
            verdict = "C: not in Hardcover"
            print(f"{title:32s} {verdict}")
        else:
            b = books[0]
            names = [c["author"]["name"] for c in (b.get("contributions") or [])
                     if c.get("author")]
            uc = b.get("users_count") or 0
            has_desc = bool((b.get("description") or "").strip())
            if in_corpus:
                verdict = "already in corpus"
            elif floor is not None and uc < floor:
                verdict = f"B: in HC but below harvest floor (users_count={uc})"
            else:
                verdict = f"A: should have been harvested (users_count={uc}) - check filter"
            print(f"{title:32s} {verdict}")
            print(f"{'':32s}   hc_title={b['title']!r} authors={names}")
            print(f"{'':32s}   ratings={b.get('ratings_count')} desc={'yes' if has_desc else 'NO'}")
        verdicts.append((title, verdict))

    print("\n--- summary ---")
    for t, v in verdicts:
        print(f"  {t:32s} {v}")
    print("\nA -> deeper harvest.  B -> fetch_hardcover_targeted.py.  C -> unavailable.")
    print("NOTE: desc=NO means the book would be dropped by the description")
    print("filter even if harvested (29,420 rows were rejected that way).")


if __name__ == "__main__":
    main()
