"""
probe_targeted_harvest.py — can the corpus be harvested by CATEGORY instead of
by popularity?

THE PROBLEM THIS EXISTS TO SOLVE
books_v2_hc was paginated `order_by: {users_count: desc}` down to a floor of 13.
That returns, by construction, the most-read books on a Western reading app.
Amish Tripathi sells in the millions and still falls below the cut, because
Hardcover's users are not Indian readers. The corpus is a picture of a
READERSHIP, not of publishing -- which is the same class of error as Hardcover's
"Young Adult" meaning "assigned in school".

Going deeper does not fix it. users_count >= 5 returns more of the same
population, further down. What is needed is a different QUERY SHAPE: most-read
WITHIN a category, not most-read overall.

WHAT THIS ANSWERS, BEFORE ANY HARVEST IS WRITTEN
  1. Does Hardcover's GraphQL expose publisher / language / country filters at
     all? If yes, a category sweep is a weekend. If no, it is one author at a
     time and Google Books becomes the primary source.
  2. For a list of known-missing authors, how many of their books does Hardcover
     actually hold, and how many carry a description? (diagnose_authors.py
     answers this per author; this batches it.)
  3. How many are already in the corpus, so the harvest is sized against the
     real gap rather than the whole bibliography.

Schema introspection first, because guessing at field names and reading the
error is slower than asking the API what it supports.

READ-ONLY. No writes.

USAGE
    $env:HARDCOVER_TOKEN = "..."
    python notebooks\\pipeline\\probe_targeted_harvest.py --stage schema
    python notebooks\\pipeline\\probe_targeted_harvest.py --stage authors
    python notebooks\\pipeline\\probe_targeted_harvest.py --stage publishers
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

ENDPOINT = "https://api.hardcover.app/v1/graphql"
CORPUS = Path("data/processed/books_v2_hc2.parquet")
MIN_DESC = 100
SLEEP = 0.4

# Authors whose readership is real but whose Hardcover users_count is low --
# the exact population the popularity harvest cannot reach.
AUTHORS = [
    "Amish Tripathi",
    "Chetan Bhagat",
    "Devdutt Pattanaik",
    "S. Hussain Zaidi",
    "Mahasweta Devi",
    "Ruskin Bond",
    "R.K. Narayan",
    "Jhumpa Lahiri",
    "Arundhati Roy",
    "Perumal Murugan",
    "Lee Maracle",
    "Thomas King",
]

# If a publisher filter exists, these reach an entire catalogue in one query
# instead of one author at a time.
PUBLISHERS = [
    "Penguin India", "HarperCollins India", "Aleph Book Company",
    "Seagull Books", "Westland", "Rupa Publications", "Speaking Tiger",
]


def headers():
    tok = os.environ.get("HARDCOVER_TOKEN")
    if not tok:
        raise SystemExit("Set HARDCOVER_TOKEN first")
    return {"authorization": tok, "content-type": "application/json"}


def run(query, variables=None, quiet=False):
    try:
        r = requests.post(ENDPOINT, headers=headers(),
                          json={"query": query, "variables": variables or {}},
                          timeout=45)
    except Exception as e:
        if not quiet:
            print(f"    request failed: {type(e).__name__}")
        return None, str(e)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    d = r.json()
    if "errors" in d:
        return None, json.dumps(d["errors"])[:300]
    return d.get("data"), None


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(t):
    t = re.sub(r"\([^)]*\)", " ", str(t))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def corpus_keys():
    if not CORPUS.exists():
        print(f"  NOTE: {CORPUS} not found -- cannot report what is already in")
        return set()
    df = pd.read_parquet(CORPUS, columns=["title"])
    return {_core(t) for t in df["title"]}


# --------------------------------------------------------------------------
def stage_schema():
    """Ask the API what books can be filtered on, instead of guessing."""
    print("introspecting books_bool_exp (the where: filter) ...\n")
    q = """
    query {
      __type(name: "books_bool_exp") {
        inputFields { name type { name kind ofType { name } } }
      }
    }
    """
    data, err = run(q)
    if err or not data or not data.get("__type"):
        print(f"  introspection unavailable: {err}")
        print("  Many production GraphQL servers disable it. Falling back to")
        print("  probing candidate filters directly.\n")
        return probe_filters_directly()

    fields = [f["name"] for f in data["__type"]["inputFields"]]
    print(f"  {len(fields)} filterable fields on books:\n")
    for f in sorted(fields):
        print(f"    {f}")

    interesting = [f for f in fields
                   if any(k in f.lower() for k in
                          ("lang", "publish", "country", "edition", "region",
                           "tag", "genre", "contrib", "author"))]
    print("\n  the ones that could drive a category harvest:")
    for f in interesting or ["(none obvious)"]:
        print(f"    {f}")


def probe_filters_directly():
    """Try each candidate filter and see which the server accepts."""
    candidates = [
        ("language", '{language: {language: {_eq: "Hindi"}}}'),
        ("publisher (nested)", '{editions: {publisher: {name: {_eq: "Penguin"}}}}'),
        ("country", '{country: {name: {_eq: "India"}}}'),
        ("cached_tags", '{cached_tags: {_contains: {Genre: ["Indian Literature"]}}}'),
        ("release_date", '{release_date: {_gte: "2020-01-01"}}'),
    ]
    print("  probing candidate filters one at a time:\n")
    for label, where in candidates:
        q = "query { books(where: %s, limit: 2) { id title } }" % where
        data, err = run(q, quiet=True)
        if err:
            print(f"    {label:<22} REJECTED  {err[:90]}")
        else:
            n = len(data.get("books") or [])
            print(f"    {label:<22} OK        returned {n}")
    print("\n  An accepted filter is a category harvest you can write today.")
    print("  All rejected means author-by-author, or Google Books instead.")


# --------------------------------------------------------------------------
AUTHOR_Q = """
query($name: String!) {
  authors(where: {name: {_eq: $name}}, limit: 3) {
    name
    contributions(limit: 100) {
      book { id title description users_count ratings_count release_date }
    }
  }
}
"""


def stage_authors():
    have = corpus_keys()
    print(f"corpus: {len(have):,} title keys\n")
    print(f"{'author':<24}{'in HC':>7}{'described':>11}{'in corpus':>11}{'GAP':>6}")
    print("-" * 60)
    total_gap = 0
    gap_titles = []
    errors = 0
    for name in AUTHORS:
        # _eq, not _ilike: this API returns 403 on _ilike (learned in v1 and
        # documented in diagnose_missing_books.py). The cost is that the name
        # must match Hardcover's spelling exactly -- "R.K. Narayan" and
        # "R. K. Narayan" are different keys, and a zero here may mean a
        # spelling mismatch rather than an absent author.
        data, err = run(AUTHOR_Q, {"name": name})
        time.sleep(SLEEP)
        if err:
            errors += 1
            print(f"{name:<24}  ERROR {err[:30]}")
            continue
        books, seen = [], set()
        for a in (data.get("authors") or []):
            for c in (a.get("contributions") or []):
                b = c.get("book")
                if b and b["id"] not in seen:
                    seen.add(b["id"])
                    books.append(b)
        if not books:
            print(f"{name:<24}{0:>7}{'':>11}{'':>11}{'':>6}   <- name not matched exactly?")
            continue
        described = [b for b in books
                     if len((b.get("description") or "").strip()) >= MIN_DESC]
        missing = [b for b in described if _core(b["title"]) not in have]
        total_gap += len(missing)
        gap_titles += [(name, b["title"], b.get("users_count") or 0) for b in missing]
        print(f"{name:<24}{len(books):>7}{len(described):>11}"
              f"{len(described)-len(missing):>11}{len(missing):>6}")

    if errors == len(AUTHORS):
        print("\n  EVERY query failed. This is an API problem, not a finding --")
        print("  do not read a zero gap as 'nothing to recover'.")
        return
    if errors:
        print(f"\n  NOTE: {errors} of {len(AUTHORS)} queries errored; the totals")
        print("  below cover only the ones that returned.")

    print(f"\n  recoverable now (described, not in corpus): {total_gap}")
    if gap_titles:
        print("  a sample, with the users_count that kept them out:")
        for a, t, uc in sorted(gap_titles, key=lambda x: -x[2])[:20]:
            print(f"    uc={uc:<6} {t[:46]:48s} {a}")
        print("\n  These need no new source. They are in Hardcover, they have")
        print("  descriptions, and the popularity harvest simply never reached them.")


# --------------------------------------------------------------------------
def stage_publishers():
    print("testing whether a publisher filter exists at all ...\n")
    for pub in PUBLISHERS[:3]:
        q = """
        query($p: String!) {
          books(where: {editions: {publisher: {name: {_eq: $p}}}}, limit: 5) {
            id title users_count
          }
        }
        """
        data, err = run(q, {"p": pub}, quiet=True)
        if err:
            print(f"  {pub:<26} REJECTED  {err[:80]}")
        else:
            books = data.get("books") or []
            print(f"  {pub:<26} OK  {len(books)} returned")
            for b in books[:3]:
                print(f"      {str(b['title'])[:50]}")
        time.sleep(SLEEP)
    print("\n  If these are rejected, publisher is not a filterable field and the")
    print("  category harvest has to run author-by-author -- slower, but the")
    print("  author stage above already shows it works.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["schema", "authors", "publishers"],
                    default="schema")
    args = ap.parse_args()
    {"schema": stage_schema, "authors": stage_authors,
     "publishers": stage_publishers}[args.stage]()


if __name__ == "__main__":
    main()
