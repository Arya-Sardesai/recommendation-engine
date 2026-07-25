"""
pilot_tmdb_bulk.py — Pilot for the v2 bulk TMDB enrichment pass.

Fetches details+keywords by TMDB id for a random sample of corpus ids
(~350 movies, ~150 TV). Validates before the full 133K run:
  1. Schema stability across responses
  2. TV keywords response key ("results" vs "keywords")
  3. Keyword coverage rate + density on the REAL corpus (not hand-picked titles)
  4. 404 / error rate
  5. KW_STOP candidates (mood junk, production junk)

Usage:
  $env:TMDB_API_KEY = "..."   # per-session ritual
  python notebooks/pipeline/pilot_tmdb_bulk.py

Resumable: re-running skips ids already in the output JSONL.
"""

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

# --- Config -----------------------------------------------------------------
API_KEY = os.environ.get("TMDB_API_KEY")
if not API_KEY:
    sys.exit("TMDB_API_KEY not set. Run: $env:TMDB_API_KEY = \"...\"")

BASE = "https://api.themoviedb.org/3"
MOVIES_PARQUET = "data/processed/movies.parquet"
TV_PARQUET = "data/processed/tv_corpus.parquet"
OUT_DIR = Path("data/tmdb_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MOVIES = 350
N_TV = 150
SEED = 42
REQ_INTERVAL = 0.05  # 20 req/s, sequential is fine at pilot scale


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def fetch_one(kind: str, tmdb_id: int, session: requests.Session) -> dict:
    """kind: 'movie' or 'tv'. Returns a JSONL-ready record."""
    url = f"{BASE}/{kind}/{tmdb_id}"
    params = {"api_key": API_KEY, "append_to_response": "keywords"}
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return {"id": tmdb_id, "status": "ok",
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "data": r.json()}
            if r.status_code == 404:
                return {"id": tmdb_id, "status": "404",
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "data": None}
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return {"id": tmdb_id, "status": f"http_{r.status_code}",
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "data": None}
        except requests.RequestException as e:
            if attempt == 3:
                return {"id": tmdb_id, "status": f"error:{type(e).__name__}",
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "data": None}
            time.sleep(2 ** attempt)
    return {"id": tmdb_id, "status": "error:retries_exhausted",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": None}


def extract_keywords(kind: str, data: dict) -> list[str] | None:
    """Handles both response shapes; returns None if shape is unrecognized."""
    kw = data.get("keywords")
    if kw is None:
        return None
    inner = kw.get("keywords", kw.get("results"))  # movie vs tv
    if inner is None:
        return None
    return [k["name"] for k in inner]


def run_pilot(kind: str, ids: list[int], out_path: Path):
    done = load_done(out_path)
    todo = [i for i in ids if i not in done]
    print(f"\n[{kind}] {len(done)} already done, {len(todo)} to fetch")
    session = requests.Session()
    with open(out_path, "a", encoding="utf-8") as f:
        for n, tmdb_id in enumerate(todo, 1):
            rec = fetch_one(kind, tmdb_id, session)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
            time.sleep(REQ_INTERVAL)


def analyze(kind: str, out_path: Path):
    print(f"\n{'=' * 60}\n[{kind}] ANALYSIS\n{'=' * 60}")
    records = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    statuses = Counter(r["status"] for r in records)
    print("statuses:", dict(statuses))

    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        print("no successful fetches — stop and investigate")
        return

    # TV response-key verification (finding: "results" vs "keywords")
    shape_keys = Counter(
        tuple(sorted(r["data"]["keywords"].keys()))
        for r in ok if r["data"].get("keywords")
    )
    print("keywords-object shapes:", dict(shape_keys))

    kw_lists = []
    unrecognized = 0
    for r in ok:
        kws = extract_keywords(kind, r["data"])
        if kws is None:
            unrecognized += 1
        else:
            kw_lists.append(kws)
    if unrecognized:
        print(f"WARNING: {unrecognized} responses with unrecognized keyword shape")

    counts = pd.Series([len(k) for k in kw_lists])
    coverage = (counts > 0).mean()
    print(f"keyword coverage (>=1 kw): {coverage:.1%}")
    print(f"density: median={counts.median():.0f}, mean={counts.mean():.1f}, "
          f"p90={counts.quantile(0.9):.0f}, max={counts.max()}")

    all_kw = Counter(kw for kws in kw_lists for kw in kws)
    print(f"\ntop 40 keywords (KW_STOP candidates will be visible here):")
    for kw, c in all_kw.most_common(40):
        print(f"  {c:4d}  {kw}")

    # Schema check: field presence across responses
    field_presence = Counter(k for r in ok for k in r["data"].keys())
    missing_fields = {k: len(ok) - v for k, v in field_presence.items() if v < len(ok)}
    if missing_fields:
        print(f"\nfields NOT present in all responses: {missing_fields}")
    else:
        print("\nschema stable: all fields present in all responses")


if __name__ == "__main__":
    random.seed(SEED)

    movie_ids = pd.read_parquet(MOVIES_PARQUET, columns=["id"])["id"].tolist()
    tv_ids = pd.read_parquet(TV_PARQUET, columns=["tmdb_id"])["tmdb_id"].tolist()

    movie_sample = random.sample(movie_ids, N_MOVIES)
    tv_sample = random.sample(tv_ids, N_TV)

    run_pilot("movie", [int(i) for i in movie_sample], OUT_DIR / "pilot_movies.jsonl")
    run_pilot("tv", [int(i) for i in tv_sample], OUT_DIR / "pilot_tv.jsonl")

    analyze("movie", OUT_DIR / "pilot_movies.jsonl")
    analyze("tv", OUT_DIR / "pilot_tv.jsonl")

    print("\nPilot complete. Review coverage, TV keyword shape, and KW_STOP "
          "candidates before designing the full run.")