"""
expand_movies.py -- catalogue expansion, structured as the reusable pipeline.

The catch-up run and the monthly refresh are the SAME script with different
date bounds:

    catch-up (now):   python notebooks/pipeline/expand_movies.py --since 2022-07 --until 2026-07
    monthly (Sep 1):  python notebooks/pipeline/expand_movies.py --since 2026-08 --until 2026-08

Stages (each resumable, each writes JSONL/state under data/expansion/):

    harvest   /discover/movie per monthly window -> candidate ids
    details   one call per candidate, append_to_response=credits,keywords
              -> normalised rows, filtered through the SAME tiered thresholds
              as build_movies_corpus.py (imported, never retyped)
    refresh   re-fetch vote_count / vote_average for films ALREADY in the
              corpus released in the trailing 12 months (new releases enter
              with low counts; without this pass they sit below the serving
              floor forever)
    merge     dtype-align, append + dedupe on id, update refreshed votes,
              write movies.parquet (backup written first)

Run stages individually (--stage harvest) or all in order (--stage all).
A crash mid-details costs nothing: re-run and it skips ids already fetched.

--limit N caps details calls for a smoke test. ALWAYS smoke-test first:

    python notebooks/pipeline/expand_movies.py --since 2024-01 --until 2024-01 --limit 25 --stage all --dry-run

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
It does not re-embed, rebuild FAISS, or upload. Adding rows to movies.parquet
breaks positional alignment with the live .npy and index. After merge you MUST:
  1. bump EMBED_VARIANT to "v2_kw2" in embed_movies.py AND embed_tv.py
  2. re-embed, rebuild FAISS
  3. verify: parquet rows == npy shape[0] == index.ntotal (+ spot checks)
  4. upload, probe the CDN cache
The merge stage prints this checklist and refuses --dry-run-less runs until
you acknowledge with --i-know-this-breaks-alignment.

Usage:
    $env:TMDB_API_KEY = "..."       # per-session ritual
    python notebooks/pipeline/expand_movies.py --since 2022-07 --until 2026-07 --stage all

Run from REPO ROOT.
"""

import argparse
import json
import os
import shutil
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
)

API_KEY = os.environ.get("TMDB_API_KEY")

BASE = "https://api.themoviedb.org/3"
ROOT = Path(__file__).parent.parent.parent
CORPUS = ROOT / "data" / "processed" / "movies.parquet"
WORK = ROOT / "data" / "expansion"
WORK.mkdir(parents=True, exist_ok=True)

CANDIDATES = WORK / "candidates.jsonl"     # {"window": "2024-01", "ids": [...]}
DETAILS = WORK / "details.jsonl"           # one normalised row per line
REFRESH = WORK / "refresh.jsonl"           # {"id":..., "vote_count":..., "vote_average":...}
STATE = WORK / "state.json"                # {"windows_done": [...]}

REQ_INTERVAL = 0.05                        # 20 req/s
DISCOVER_VOTE_FLOOR = min(VOTE_FLOOR_EN, VOTE_FLOOR_OTHER)
CAST_KEEP = 10                             # top-billed names carried into the corpus
REFRESH_TRAILING_DAYS = 365


# ---------------------------------------------------------------------------
# HTTP (same shape as the pilot)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def month_bounds(ym: str):
    y, m = map(int, ym.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), (m % 12) + 1, 1)
    return start.isoformat(), end.isoformat()


def month_range(since: str, until: str):
    """['2022-07', '2022-08', ..., until] inclusive."""
    y, m = map(int, since.split("-"))
    uy, um = map(int, until.split("-"))
    out = []
    while (y, m) <= (uy, um):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"windows_done": []}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# STAGE 1: harvest
# ---------------------------------------------------------------------------
def discover_month(ym: str):
    gte, lt = month_bounds(ym)
    ids, page, total_pages = [], 1, 1
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
        total_pages = data.get("total_pages", 1)
        ids.extend(m["id"] for m in data.get("results", []))
        page += 1
        time.sleep(REQ_INTERVAL)
    if total_pages > 500:
        print(f"  WARNING [{ym}]: {total_pages} pages exceeds the 500-page cap; "
              f"this month needs finer (weekly) windows. Harvest is INCOMPLETE.")
    return ids


def stage_harvest(windows):
    state = load_state()
    done = set(state["windows_done"])
    todo = [w for w in windows if w not in done]
    if not todo:
        print("harvest: all windows already done (delete data/expansion/state.json to redo)")
        return
    print(f"harvest: {len(todo)} window(s) to do, {len(done)} already done")
    with open(CANDIDATES, "a", encoding="utf-8") as out:
        for ym in todo:
            ids = discover_month(ym)
            out.write(json.dumps({"window": ym, "ids": ids}) + "\n")
            out.flush()
            state["windows_done"].append(ym)
            save_state(state)
            print(f"  [{ym}] {len(ids):,} candidates")
    print("harvest: done")


# ---------------------------------------------------------------------------
# STAGE 2: details (+ filter)
# ---------------------------------------------------------------------------
def normalise(d: dict) -> dict:
    """API details JSON -> corpus row shape. Extends the pilot's normalise with
    credits parsing (director / directors_all / cast come from the same call
    via append_to_response=credits,keywords)."""
    genres = ", ".join(g["name"] for g in d.get("genres") or [])
    pcs = d.get("production_companies") or []
    studio = pcs[0]["name"].strip() if pcs else ""
    kw = (d.get("keywords") or {}).get("keywords") or []
    rd = d.get("release_date") or ""

    crew = (d.get("credits") or {}).get("crew") or []
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    cast_raw = (d.get("credits") or {}).get("cast") or []
    cast = [c["name"] for c in sorted(cast_raw, key=lambda c: c.get("order", 999))][:CAST_KEEP]

    return {
        "id": d["id"],
        "imdb_id": (d.get("imdb_id") or "").strip(),
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
        "director": directors[0] if directors else "",
        "directors_all": directors,
        "cast": cast,
        "_status": d.get("status") or "",
        "_adult": bool(d.get("adult")),
    }


def passes(r: dict) -> tuple[bool, str]:
    """Same tiering as build_movies_corpus.passes(), one record at a time."""
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


def stage_details(limit=None):
    if not CANDIDATES.exists():
        sys.exit("details: no candidates.jsonl -- run --stage harvest first")

    corpus = pd.read_parquet(CORPUS)
    known = set(corpus["id"].astype("int64"))

    # candidate ids, deduped, minus corpus, minus already-fetched
    cand = []
    with open(CANDIDATES, encoding="utf-8") as f:
        for line in f:
            cand.extend(json.loads(line)["ids"])
    cand = [i for i in dict.fromkeys(cand) if i not in known]

    fetched = set()
    if DETAILS.exists():
        with open(DETAILS, encoding="utf-8") as f:
            for line in f:
                fetched.add(json.loads(line)["id"])
    todo = [i for i in cand if i not in fetched]
    if limit:
        todo = todo[:limit]

    print(f"details: {len(cand):,} candidates | {len(fetched):,} already fetched | "
          f"{len(todo):,} to do" + (f" (--limit {limit})" if limit else ""))
    if not todo:
        return

    reasons = Counter()
    kept = 0
    with open(DETAILS, "a", encoding="utf-8") as out:
        for n, mid in enumerate(todo, 1):
            d = get(f"movie/{mid}", append_to_response="credits,keywords")
            time.sleep(REQ_INTERVAL)
            if not d:
                reasons["fetch failed"] += 1
                continue
            r = normalise(d)
            ok, why = passes(r)
            reasons[why] += 1
            if ok:
                r["_pass_tier"] = why
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
                if kept % 200 == 0:
                    out.flush()
            if n % 200 == 0:
                print(f"  {n:,}/{len(todo):,}  (kept {kept:,})")
    print(f"details: done. kept {kept:,} / {len(todo):,}")
    print("outcomes:", dict(reasons))


# ---------------------------------------------------------------------------
# STAGE 3: refresh (vote counts for recent films already in the corpus)
# ---------------------------------------------------------------------------
def stage_refresh(until: str, limit=None):
    corpus = pd.read_parquet(CORPUS)
    _, until_end = month_bounds(until)
    cutoff = (pd.Timestamp(until_end) - pd.Timedelta(days=REFRESH_TRAILING_DAYS))
    rd = pd.to_datetime(corpus["release_date"], errors="coerce")
    recent = corpus.loc[rd >= cutoff, "id"].astype("int64").tolist()

    done = set()
    if REFRESH.exists():
        with open(REFRESH, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    todo = [i for i in recent if i not in done]
    if limit:
        todo = todo[:limit]

    print(f"refresh: {len(recent):,} corpus films released after {cutoff.date()} | "
          f"{len(done):,} done | {len(todo):,} to do")
    if not todo:
        return

    with open(REFRESH, "a", encoding="utf-8") as out:
        for n, mid in enumerate(todo, 1):
            d = get(f"movie/{mid}")
            time.sleep(REQ_INTERVAL)
            if not d:
                continue
            out.write(json.dumps({
                "id": d["id"],
                "vote_count": int(d.get("vote_count") or 0),
                "vote_average": d.get("vote_average"),
            }) + "\n")
            if n % 200 == 0:
                out.flush()
                print(f"  {n:,}/{len(todo):,}")
    print("refresh: done")


# ---------------------------------------------------------------------------
# STAGE 4: merge
# ---------------------------------------------------------------------------
def _match_listcol_shape(new_series, corpus_sample):
    """Corpus list-ish columns (cast, directors_all, keywords_tmdb) may hold
    numpy arrays or comma-joined strings depending on how they were written.
    Match whatever shape the corpus actually has."""
    if isinstance(corpus_sample, str):
        return new_series.apply(lambda v: ", ".join(v) if isinstance(v, list) else (v or ""))
    return new_series  # parquet will coerce lists to arrays on write


def stage_merge(dry_run=True):
    if not DETAILS.exists():
        sys.exit("merge: no details.jsonl -- run --stage details first")

    corpus = pd.read_parquet(CORPUS)
    known = set(corpus["id"].astype("int64"))

    rows = []
    with open(DETAILS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r.pop("_status", None); r.pop("_adult", None); r.pop("_pass_tier", None)
            if r["id"] not in known:
                rows.append(r)
    if not rows:
        print("merge: nothing new to merge")
        return
    new_df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="first")
    print(f"merge: {len(new_df):,} new films to append to {len(corpus):,}")

    # ---- shape list-ish columns to match the corpus ----
    for col in ("cast", "directors_all", "keywords_tmdb"):
        if col in corpus.columns:
            sample = corpus[col].dropna().iloc[0] if corpus[col].notna().any() else None
            new_df[col] = _match_listcol_shape(new_df[col], sample)

    # ---- align columns and dtypes to the corpus (the DIFF fix) ----
    for c in corpus.columns:
        if c not in new_df.columns:
            print(f"  WARNING: corpus column '{c}' absent from new rows -- filling empty")
            new_df[c] = "" if corpus[c].dtype == object else pd.NA
    new_df = new_df[list(corpus.columns)]
    for c in corpus.columns:
        try:
            if str(corpus[c].dtype) != str(new_df[c].dtype) and corpus[c].dtype != object:
                new_df[c] = new_df[c].astype(corpus[c].dtype)
        except (TypeError, ValueError) as e:
            print(f"  WARNING: could not cast '{c}' to {corpus[c].dtype}: {e}")

    merged = pd.concat([corpus, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="id", keep="first")

    # ---- apply vote refresh ----
    if REFRESH.exists():
        ref = {}
        with open(REFRESH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ref[r["id"]] = r
        idx = merged["id"].astype("int64").map(lambda i: i in ref)
        n_ref = int(idx.sum())
        if n_ref:
            merged.loc[idx, "vote_count"] = merged.loc[idx, "id"].map(
                lambda i: ref[int(i)]["vote_count"]).astype(merged["vote_count"].dtype)
            merged.loc[idx, "vote_average"] = merged.loc[idx, "id"].map(
                lambda i: ref[int(i)]["vote_average"])
        print(f"merge: refreshed votes on {n_ref:,} existing films")

    # ---- report ----
    yr = merged["release_year"].value_counts().sort_index()
    print("\nyear histogram (2022+):")
    print(yr[yr.index >= 2022].to_string())
    print(f"\nfinal corpus: {len(merged):,} films (was {len(corpus):,})")

    if dry_run:
        print("\nDRY RUN -- nothing written. Re-run with --write to commit.")
        return

    backup = CORPUS.with_name("movies_pre_expansion.parquet")
    if not backup.exists():
        shutil.copy2(CORPUS, backup)
        print(f"backup written: {backup.name}")
    merged.to_parquet(CORPUS, index=False)
    print(f"WROTE {CORPUS}")
    print("\n" + "=" * 70)
    print("THE CORPUS AND THE LIVE EMBEDDINGS ARE NOW MISALIGNED.")
    print("Before anything touches the Space:")
    print('  1. bump EMBED_VARIANT to "v2_kw2" in embed_movies.py AND embed_tv.py')
    print("  2. re-embed, rebuild FAISS")
    print("  3. verify parquet rows == npy shape[0] == index.ntotal + spot checks")
    print("  4. upload, probe CDN cache, THEN update M_EMB/M_FAISS in app.py")
    print("=" * 70)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="first month, e.g. 2022-07")
    ap.add_argument("--until", required=True, help="last month inclusive, e.g. 2026-07")
    ap.add_argument("--stage", default="all",
                    choices=["harvest", "details", "refresh", "merge", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap details/refresh calls (smoke testing)")
    ap.add_argument("--write", action="store_true",
                    help="merge actually writes movies.parquet (default: dry run)")
    args = ap.parse_args()

    if args.stage in ("harvest", "details", "refresh", "all") and not API_KEY:
        sys.exit('TMDB_API_KEY not set. Run: $env:TMDB_API_KEY = "..."')
    if not CORPUS.exists():
        sys.exit(f"ERROR: missing {CORPUS}")

    windows = month_range(args.since, args.until)
    print(f"windows: {windows[0]} .. {windows[-1]} ({len(windows)} months)\n")

    if args.stage in ("harvest", "all"):
        stage_harvest(windows)
    if args.stage in ("details", "all"):
        stage_details(limit=args.limit)
    if args.stage in ("refresh", "all"):
        stage_refresh(args.until, limit=args.limit)
    if args.stage in ("merge", "all"):
        stage_merge(dry_run=not args.write)


if __name__ == "__main__":
    main()
