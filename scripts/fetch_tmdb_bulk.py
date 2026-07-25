"""
fetch_tmdb_bulk.py — v2 bulk TMDB enrichment pass (step 1: enrich existing corpus).

Fetches details + keywords by TMDB id for the full movie (~110K) and TV (~23K)
corpora. Validated by notebooks/pipeline/pilot_tmdb_bulk.py (schema stable,
movie kw key = "keywords", TV kw key = "results", 404 rate ~0.3%).

Usage:
  $env:TMDB_API_KEY = "..."
  python scripts/fetch_tmdb_bulk.py            # both corpora
  python scripts/fetch_tmdb_bulk.py movies     # movies only
  python scripts/fetch_tmdb_bulk.py tv         # tv only

Output: data/tmdb_raw/movies_raw.jsonl, data/tmdb_raw/tv_raw.jsonl
Resumable: re-running skips ids already present in the output file.
Safe to Ctrl-C at any point; progress is flushed per record.
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
OUT_DIR = Path("data/tmdb_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 10
TARGET_RPS = 20.0          # polite ceiling; TMDB allows ~50
PROGRESS_EVERY = 500


class RateLimiter:
    """Token-bucket-ish: enforces a global minimum interval between request starts."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_ok = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_ok:
                sleep_for = self.next_ok - now
                self.next_ok += self.interval
            else:
                sleep_for = 0.0
                self.next_ok = now + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


LIMITER = RateLimiter(TARGET_RPS)
_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_one(kind: str, tmdb_id: int) -> dict:
    url = f"{BASE}/{kind}/{tmdb_id}"
    params = {"api_key": API_KEY, "append_to_response": "keywords"}
    session = get_session()
    for attempt in range(5):
        LIMITER.wait()
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return {"id": tmdb_id, "status": "ok",
                        "fetched_at": now_iso(), "data": r.json()}
            if r.status_code == 404:
                return {"id": tmdb_id, "status": "404",
                        "fetched_at": now_iso(), "data": None}
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            return {"id": tmdb_id, "status": f"http_{r.status_code}",
                    "fetched_at": now_iso(), "data": None}
        except requests.RequestException as e:
            if attempt == 4:
                return {"id": tmdb_id, "status": f"error:{type(e).__name__}",
                        "fetched_at": now_iso(), "data": None}
            time.sleep(min(2 ** attempt, 30))
    return {"id": tmdb_id, "status": "error:retries_exhausted",
            "fetched_at": now_iso(), "data": None}


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass  # torn line from a Ctrl-C mid-write; will be refetched
    return done


def run(kind: str, ids: list[int], out_path: Path):
    done = load_done(out_path)
    todo = [i for i in ids if i not in done]
    total = len(todo)
    print(f"\n[{kind}] corpus={len(ids)}, already done={len(done)}, to fetch={total}")
    if total == 0:
        return
    est_min = total / TARGET_RPS / 60
    print(f"[{kind}] estimated time at {TARGET_RPS:.0f} req/s: ~{est_min:.0f} min")

    write_lock = threading.Lock()
    completed = 0
    status_counts: dict[str, int] = {}
    t0 = time.monotonic()

    with open(out_path, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, kind, i): i for i in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            completed += 1
            status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1
            if completed % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - t0
                rps = completed / elapsed
                remaining = (total - completed) / rps / 60 if rps > 0 else 0
                print(f"  [{kind}] {completed}/{total} "
                      f"({rps:.1f} req/s, ~{remaining:.0f} min left) "
                      f"statuses={status_counts}")

    print(f"[{kind}] DONE. Final statuses: {status_counts}")
    errors = {k: v for k, v in status_counts.items() if k.startswith("error")}
    if errors:
        print(f"[{kind}] NOTE: {sum(errors.values())} errors — re-run the script; "
              f"resume logic will retry only failed... ")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("movies", "both"):
        movie_ids = [int(i) for i in
                     pd.read_parquet(MOVIES_PARQUET, columns=["id"])["id"]]
        run("movie", movie_ids, OUT_DIR / "movies_raw.jsonl")

    if which in ("tv", "both"):
        tv_ids = [int(i) for i in
                  pd.read_parquet(TV_PARQUET, columns=["tmdb_id"])["tmdb_id"]]
        run("tv", tv_ids, OUT_DIR / "tv_raw.jsonl")

    print("\nBulk fetch complete. Next: parse JSONL → merge into parquets "
          "(keywords column + refreshed metadata), then KW_STOP + taxonomy.")