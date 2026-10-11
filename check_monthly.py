"""
check_monthly.py -- is a monthly expansion cycle worth running?

Answers two questions with data instead of argument:

  1. MOVIES: of this month's new films, how many already clear the 400-vote
     serving floor? New releases below it can be rated but never recommended,
     so they only justify an embed cycle if some are already servable.

  2. TV: of series already in the corpus, how many crossed the 300-vote floor
     since the last refresh? That drift is what a refresh actually buys.

Written as a file because the one-liner versions do not survive PowerShell:
nested quotes get eaten, and the TV refresh keys its records on `tmdb_id`
where the movie one uses `id`.

Read-only. Reads data/expansion*/ and the processed parquets, writes nothing.

USAGE
    python check_monthly.py
    python check_monthly.py --month 2026-09
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MOVIE_FLOOR = 400     # recommend_movies min_votes
TV_FLOOR = 300        # recommend_tv min_votes


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def check_movies(month):
    p = Path("data/expansion/details.jsonl")
    print("=" * 64)
    print(f"MOVIES released {month}")
    print("=" * 64)
    if not p.exists():
        print(f"  {p} not found -- run expand_movies --stage details first")
        return
    s = [r for r in load_jsonl(p)
         if str(r.get("release_date", "")).startswith(month)]
    s.sort(key=lambda r: -(r.get("vote_count") or 0))
    over = [r for r in s if (r.get("vote_count") or 0) >= MOVIE_FLOOR]
    print(f"  new films fetched              : {len(s)}")
    print(f"  already >= {MOVIE_FLOOR} votes (servable) : {len(over)}")
    print("  most-voted:")
    for r in s[:10]:
        print(f"    {r.get('vote_count', 0):>6}  {str(r.get('title', ''))[:52]}")


def check_tv():
    p = Path("data/expansion_tv/refresh.jsonl")
    print()
    print("=" * 64)
    print(f"TV series crossing the {TV_FLOOR}-vote floor since last refresh")
    print("=" * 64)
    if not p.exists():
        print(f"  {p} not found")
        return
    rows = load_jsonl(p)
    if not rows:
        print("  refresh file is empty")
        return
    id_k = next((k for k in ("tmdb_id", "id", "tv_id") if k in rows[0]), None)
    if id_k is None:
        print(f"  no id field in refresh records: {sorted(rows[0].keys())}")
        return

    ref = {int(r[id_k]): r["vote_count"] for r in rows
           if r.get(id_k) is not None and r.get("vote_count") is not None}
    d = pd.read_parquet("data/processed/tv_v2_meta.parquet",
                        columns=["tmdb_id", "name", "vote_count"])
    d["new"] = d["tmdb_id"].astype("int64").map(ref)
    h = d[d["new"].notna()]
    cross = h[(h["vote_count"] < TV_FLOOR) & (h["new"] >= TV_FLOOR)]
    print(f"  series refreshed : {len(h):,}")
    print(f"  crossed {TV_FLOOR}      : {len(cross)}")
    for r in cross.sort_values("new", ascending=False).head(15).itertuples():
        print(f"    {int(r.vote_count):>5} -> {int(r.new):>5}  {str(r.name)[:44]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-09")
    a = ap.parse_args()
    check_movies(a.month)
    check_tv()
    print()
    print("For comparison: one month of movie drift promoted 11 films past 400.")
    print("A handful here means quarterly. Dozens means monthly pays for itself.")


if __name__ == "__main__":
    main()
