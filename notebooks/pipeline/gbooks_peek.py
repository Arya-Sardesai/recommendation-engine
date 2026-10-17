"""
gbooks_peek.py -- what is actually in the Google Books queue?

Prints how many books sit in each popularity band, and a random sample of
titles from each, so the queue can be judged by eye. Read-only, no API calls.

    python notebooks\\pipeline\\gbooks_peek.py
    python notebooks\\pipeline\\gbooks_peek.py --n 15
"""
import argparse
import json
import random
from pathlib import Path

TARGETS = Path("data/raw/books/hardcover_v2/gbooks_targets.jsonl")
BANDS = [(1000, 10 ** 9), (200, 1000), (50, 200), (25, 50), (0, 25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="sample size per band")
    args = ap.parse_args()
    rows = [json.loads(l) for l in TARGETS.open(encoding="utf-8") if l.strip()]
    random.seed(1)
    print(f"{len(rows):,} books in the queue (most-read first)\n")
    for lo, hi in BANDS:
        band = [r for r in rows if lo <= r.get("users_count", 0) < hi]
        if not band:
            continue
        label = f"{lo}+ readers" if hi >= 10 ** 9 else f"{lo}-{hi - 1} readers"
        print(f"== {label}: {len(band):,} books ==")
        for r in random.sample(band, min(args.n, len(band))):
            print(f"   {r['users_count']:>5}  {r['title'][:55]:57s} {r['author'][:28]}")
        print()


if __name__ == "__main__":
    main()
