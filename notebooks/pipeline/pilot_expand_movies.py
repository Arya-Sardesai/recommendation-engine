"""
pilot_expand_movies.py -- v2 pilot: catalogue expansion discovery + schema check.

PILOT ONLY. Writes nothing to data/processed. Answers four questions before a
single row is appended to movies.parquet:

  1. HOW MANY new films are actually out there past the corpus wall, after the
     SAME tiered filter build_movies_corpus.py applied to the Kaggle dump?
  2. Does /discover/movie windowing stay under the 500-page cap per month?
  3. Does the API's JSON normalise to the EXACT schema movies.parquet already
     has -- same columns, same dtypes, same string shapes for genres/studio?
  4. What does the API cost look like for the full catch-up?

WHY DISCOVER, NOT THE DAILY ID EXPORT
-------------------------------------
The daily export is ~900K ids with no release date and no vote count. Diffing
it against the corpus leaves ~790K unknown ids, each needing a details call
just to learn whether it qualifies. /discover/movie filters server-side on
release date AND vote count, so the harvest costs a few hundred calls instead
of ~790K for the same answer.

THE THRESHOLDS ARE IMPORTED, NOT RETYPED
----------------------------------------
build_movies_corpus.py is the single source of truth for what belongs in the
corpus. If those constants ever change, this script changes with them. Retyping
them here is how corpus composition silently drifts and embeddings end up
describing a different population than they used to.

KNOWN, DELIBERATE GAP
---------------------
Windowing by release date finds NEW releases. It does NOT find older films that
were below the vote floor when the Kaggle dump was cut and have since risen
above it. That is a separate backfill, deliberately out of scope here -- but it
means "expansion complete" does not mean "corpus == TMDB".

Usage:
    $env:TMDB_API_KEY = "..."       # per-session ritual
    python notebooks/pipeline/pilot_expand_movies.py

Run from REPO ROOT.
"""

import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from build_movies_corpus import (  # noqa: E402  -- single source of truth
    VOTE_CLEAN_EN, VOTE_CLEAN_OTHER,
    VOTE_FLOOR_EN, VOTE_FLOOR_OTHER,
    MIN_OVERVIEW_CHARS, PROBATION_OVERVIEW_CHARS, PROBATION_MIN_RUNTIME,
    KEEP,
)

API_KEY = os.environ.get("TMDB_API_KEY")
if not API_KEY:
    sys.exit('TMDB_API_KEY not set. Run: $env:TMDB_API_KEY = "..."')

BASE = "https://api.themoviedb.org/3"
ROOT = Path(__file__).parent.parent.parent
CORPUS = ROOT / "data" / "processed" / "movies.parquet"

# --- pilot config ------------------------------------------------------------
PILOT_WINDOWS = ["2024-01", "2024-07", "2025-03"]  # 3 sampled months, not the full range
MAX_DETAILS = 60          # cap the details calls; this is a pilot
REQ_INTERVAL = 0.05       # 20 req/s
# The lowest floor in the tiered filter. Prefilter at this level server-side,
# then apply the real EN/non-EN tiering locally on the details response.
DISCOVER_VOTE_FLOOR = min(VOTE_FLOOR_EN, VOTE_FLOOR_OTHER)


def get(path, **params):
    params["api_key"] = API_KEY
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            print(f"  HTTP {r.status_code} on {path}")
            return None
        except requests.RequestException as e:
            if attempt == 3:
                print(f"  request failed on {path}: {type(e).__name__}")
                return None
            time.sleep(2 ** attempt)
    return None


def month_bounds(ym: str):
    y, m = map(int, ym.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), (m % 12) + 1, 1)
    return start.isoformat(), end.isoformat()


def discover_month(ym: str):
    """-> (ids, total_results, total_pages). Harvest only; no details calls."""
    gte, lt = month_bounds(ym)
    ids, page, total_results, total_pages = [], 1, 0, 1
    while page <= min(total_pages, 500):
        data = get(
            "discover/movie",
            include_adult="false",
            sort_by="primary_release_date.asc",
            **{
                "primary_release_date.gte": gte,
                "primary_release_date.lte": lt,
                "vote_count.gte": DISCOVER_VOTE_FLOOR,
                "page": page,
            },
        )
        if not data:
            break
        total_results = data.get("total_results", 0)
        total_pages = data.get("total_pages", 1)
        ids.extend(m["id"] for m in data.get("results", []))
        page += 1
        time.sleep(REQ_INTERVAL)
    return ids, total_results, total_pages


def normalise(d: dict) -> dict:
    """API details JSON -> the exact shape build_movies_corpus.py produced.

    The two shape mismatches that would silently corrupt the merge:
      genres              API: [{'id','name'}]  corpus: 'Action, Comedy'
      production_companies API: [{'name',...}]  corpus: studio = first name
    """
    genres = ", ".join(g["name"] for g in d.get("genres") or [])
    pcs = d.get("production_companies") or []
    studio = pcs[0]["name"].strip() if pcs else ""
    kw = (d.get("keywords") or {}).get("keywords") or []
    imdb = (d.get("imdb_id") or "").strip()
    rd = d.get("release_date") or ""
    return {
        "id": d["id"],
        "imdb_id": imdb,
        "title": d.get("title") or "",
        "original_title": d.get("original_title") or "",
        "original_language": d.get("original_language") or "",
        "release_date": rd,
        "vote_average": d.get("vote_average"),
        "vote_count": int(d.get("vote_count") or 0),
        "runtime": d.get("runtime"),
        "overview": (d.get("overview") or "").strip(),
        "tagline": d.get("tagline") or "",
        "genres": genres,
        "keywords": "",                                   # legacy col: intentionally empty
        "keywords_tmdb": [k["name"] for k in kw],
        "poster_path": d.get("poster_path") or "",
        "studio": studio,
        "release_year": int(rd[:4]) if rd[:4].isdigit() else None,
        "_status": d.get("status") or "",
        "_adult": bool(d.get("adult")),
    }


def passes(r: dict) -> tuple[bool, str]:
    """Same tiering as build_movies_corpus.passes(), on one normalised record."""
    if r["_status"] != "Released":
        return False, "not released"
    if r["_adult"]:
        return False, "adult"
    if len(r["overview"]) < MIN_OVERVIEW_CHARS:
        return False, "overview too short"

    is_en = r["original_language"] == "en"
    clean_floor = VOTE_CLEAN_EN if is_en else VOTE_CLEAN_OTHER
    prob_floor = VOTE_FLOOR_EN if is_en else VOTE_FLOOR_OTHER
    vc = r["vote_count"]

    if vc >= clean_floor:
        return True, "clean"
    if vc < prob_floor:
        return False, "below vote floor"

    rt = r["runtime"] or 0
    if len(r["overview"]) < PROBATION_OVERVIEW_CHARS:
        return False, "probation: overview"
    if rt < PROBATION_MIN_RUNTIME:
        return False, "probation: runtime"
    if not r["genres"]:
        return False, "probation: genres"
    if not r["poster_path"]:
        return False, "probation: poster"
    return True, "probation"


def main():
    if not CORPUS.exists():
        sys.exit(f"ERROR: missing {CORPUS}")

    corpus = pd.read_parquet(CORPUS)
    known = set(corpus["id"].astype("int64"))
    print(f"corpus: {len(corpus):,} films, {len(corpus.columns)} columns")
    print(f"thresholds (imported): clean EN>={VOTE_CLEAN_EN} other>={VOTE_CLEAN_OTHER} | "
          f"probation EN>={VOTE_FLOOR_EN} other>={VOTE_FLOOR_OTHER}")
    print(f"discover prefilter: vote_count>={DISCOVER_VOTE_FLOOR}\n")

    # ---- 1. harvest ---------------------------------------------------------
    all_new, harvest_calls = [], 0
    for ym in PILOT_WINDOWS:
        ids, total, pages = discover_month(ym)
        harvest_calls += min(pages, 500)
        fresh = [i for i in ids if i not in known]
        page_warn = "  <-- OVER 500-PAGE CAP, needs finer windows" if pages > 500 else ""
        print(f"[{ym}] discover: {total:,} results / {pages} pages | "
              f"{len(ids):,} harvested, {len(fresh):,} not in corpus{page_warn}")
        all_new.extend(fresh)

    all_new = list(dict.fromkeys(all_new))
    print(f"\ncandidate ids across pilot windows: {len(all_new):,}")
    if not all_new:
        print("nothing new in the sampled windows -- widen PILOT_WINDOWS.")
        return

    # ---- 2. details + filter ------------------------------------------------
    sample = all_new[:MAX_DETAILS]
    print(f"fetching details for {len(sample)} of them ...\n")
    kept, reasons, rows = 0, Counter(), []
    for n, mid in enumerate(sample, 1):
        d = get(f"movie/{mid}", append_to_response="keywords")
        time.sleep(REQ_INTERVAL)
        if not d:
            reasons["fetch failed"] += 1
            continue
        r = normalise(d)
        ok, why = passes(r)
        reasons[why] += 1
        if ok:
            kept += 1
            rows.append(r)
        if n % 20 == 0:
            print(f"  {n}/{len(sample)}")

    rate = kept / max(len(sample), 1)
    print(f"\npass rate: {kept}/{len(sample)} = {rate:.1%}")
    print("outcomes:", dict(reasons))

    # ---- 3. schema check (the silent-corruption guard) ----------------------
    print(f"\n{'='*70}\nSCHEMA CHECK vs movies.parquet\n{'='*70}")
    if not rows:
        print("no passing rows to compare -- cannot validate schema.")
        return
    new_df = pd.DataFrame(rows).drop(columns=["_status", "_adult"])

    expected = set(corpus.columns)
    got = set(new_df.columns)
    missing, extra = expected - got, got - expected
    print(f"missing vs corpus: {sorted(missing) or 'none'}")
    print(f"extra vs corpus:   {sorted(extra) or 'none'}")

    print("\nper-column dtype comparison:")
    for c in sorted(expected & got):
        a, b = str(corpus[c].dtype), str(new_df[c].dtype)
        print(f"  {'OK ' if a == b else 'DIFF'} {c:<20} corpus={a:<12} new={b}")

    print("\nstring-shape spot check (these are the ones that corrupt silently):")
    for c in ("genres", "studio", "keywords_tmdb"):
        if c in corpus.columns:
            print(f"  {c}:")
            print(f"    corpus: {corpus[c].dropna().iloc[0]!r}")
            print(f"    new:    {new_df[c].iloc[0]!r}")

    # ---- 4. cost projection -------------------------------------------------
    wall = pd.to_datetime(corpus["release_date"], errors="coerce").max()
    months = max((pd.Timestamp.today() - wall).days // 30, 1)
    per_month = len(all_new) / len(PILOT_WINDOWS)
    est_candidates = per_month * months
    print(f"\n{'='*70}\nFULL-RUN PROJECTION\n{'='*70}")
    print(f"corpus wall: {wall.date() if pd.notna(wall) else '?'}  (~{months} months to cover)")
    print(f"~{per_month:,.0f} new candidates/month -> ~{est_candidates:,.0f} total")
    print(f"~{est_candidates * rate:,.0f} expected to pass the filter")
    print(f"API calls: ~{harvest_calls / len(PILOT_WINDOWS) * months:,.0f} harvest "
          f"+ ~{est_candidates:,.0f} details = ~{harvest_calls / len(PILOT_WINDOWS) * months + est_candidates:,.0f}")
    print("\nPILOT ONLY -- nothing written. Review the schema check above before "
          "writing any expansion script.")


if __name__ == "__main__":
    main()
