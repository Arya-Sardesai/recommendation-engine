"""
expand_tv.py -- TV catalogue expansion, same skeleton as expand_movies.py.

    catch-up (now):   python notebooks/pipeline/expand_tv.py --since 2022-07 --until 2026-07 --stage all
    monthly (Sep 1):  python notebooks/pipeline/expand_tv.py --since 2026-08 --until 2026-08 --stage all

Stages (resumable, JSONL/state under data/expansion_tv/):

    harvest   /discover/tv per monthly window (first_air_date bounds)
    details   /tv/{id}?append_to_response=credits,keywords -> normalised rows,
              filtered through the SAME thresholds as build_tv_corpus.py
              (imported, never retyped). TV quirk: keywords come back under
              keywords.results, not keywords.keywords.
    refresh   series ALREADY in the corpus with start_year >= REFRESH_FROM_YEAR:
              re-fetch vote_count, vote_average, number_of_seasons,
              number_of_episodes, status, end_year. TV needs this more than
              movies -- returning series gain seasons and votes while their
              corpus row stays frozen at harvest time.
    merge     dtype-align to the live corpus, append + dedupe on tmdb_id,
              apply refresh updates in place, write tv_corpus.parquet
              (backup written first)

Differences from the movies builder, mirrored from build_tv_corpus.py:
  - overview >= 80 chars is a HARD gate (not tiered)
  - vote floors: clean EN>=10 / other>=5, probation EN 4-9 / other 2-4
  - probation gated on genres AND poster only (no runtime/status gates)
  - embed_text is a corpus column; new rows get it via the IMPORTED
    make_embed_text() so the recipe cannot drift

After merge, the same alignment law applies: re-embed TV (embed_tv.py is
already set to the v2_kw_notitle variant -> new filenames, inert until
app.py points at them), rebuild FAISS, verify counts, upload, flip constants.

Usage:
    $env:TMDB_API_KEY = "..."
    python notebooks/pipeline/expand_tv.py --since 2022-07 --until 2026-07 --stage all

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
from build_tv_corpus import (  # noqa: E402  -- single source of truth
    MIN_OVERVIEW_CHARS,
    CLEAN_EN_VOTES, CLEAN_OTHER_VOTES,
    PROB_EN_VOTES, PROB_OTHER_VOTES,
    make_embed_text,
)

API_KEY = os.environ.get("TMDB_API_KEY")

BASE = "https://api.themoviedb.org/3"
ROOT = Path(__file__).parent.parent.parent
CORPUS = ROOT / "data" / "processed" / "tv_corpus.parquet"
WORK = ROOT / "data" / "expansion_tv"
WORK.mkdir(parents=True, exist_ok=True)

CANDIDATES = WORK / "candidates.jsonl"
DETAILS = WORK / "details.jsonl"
REFRESH = WORK / "refresh.jsonl"
STATE = WORK / "state.json"

REQ_INTERVAL = 0.05
DISCOVER_VOTE_FLOOR = min(PROB_EN_VOTES, PROB_OTHER_VOTES)
REFRESH_FROM_YEAR = 2021          # returning-series refresh window


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


def month_bounds(ym: str):
    y, m = map(int, ym.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), (m % 12) + 1, 1)
    return start.isoformat(), end.isoformat()


def month_range(since: str, until: str):
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
            "discover/tv",
            include_adult="false",
            sort_by="first_air_date.asc",
            **{
                "first_air_date.gte": gte,
                "first_air_date.lte": lt,
                "vote_count.gte": DISCOVER_VOTE_FLOOR,
                "page": page,
            },
        )
        if not data:
            break
        total_pages = data.get("total_pages", 1)
        ids.extend(s["id"] for s in data.get("results", []))
        page += 1
        time.sleep(REQ_INTERVAL)
    if total_pages > 500:
        print(f"  WARNING [{ym}]: {total_pages} pages exceeds the 500-page cap; "
              f"this month needs finer windows. Harvest is INCOMPLETE.")
    return ids


def stage_harvest(windows):
    state = load_state()
    done = set(state["windows_done"])
    todo = [w for w in windows if w not in done]
    if not todo:
        print("harvest: all windows already done (delete data/expansion_tv/state.json to redo)")
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
    """API /tv/{id} JSON -> tv_corpus row shape.
    TV keywords live under keywords.results (movies: keywords.keywords)."""
    genres = ", ".join(g["name"] for g in d.get("genres") or [])
    networks = ", ".join(n["name"] for n in d.get("networks") or [])
    created_by = ", ".join(p["name"] for p in d.get("created_by") or [])
    kw = (d.get("keywords") or {}).get("results") or []
    origin = d.get("origin_country") or []
    _ert = d.get("episode_run_time") or []
    ert = _ert[0] if _ert else None
    fad = d.get("first_air_date") or ""
    lad = d.get("last_air_date") or ""

    row = {
        "tmdb_id": d["id"],
        "name": d.get("name") or "",
        "original_name": d.get("original_name") or "",
        "start_year": int(fad[:4]) if fad[:4].isdigit() else None,
        "end_year": int(lad[:4]) if lad[:4].isdigit() else None,
        "original_language": d.get("original_language") or "",
        "vote_count": int(d.get("vote_count") or 0),
        "vote_average": d.get("vote_average"),
        "popularity": d.get("popularity"),
        "genres": genres,
        "created_by": created_by,
        "networks": networks,
        "number_of_seasons": d.get("number_of_seasons"),
        "number_of_episodes": d.get("number_of_episodes"),
        "episode_run_time": ert,
        "type": d.get("type") or "",
        "status": d.get("status") or "",
        "origin_country": origin,
        "tagline": d.get("tagline") or "",
        "poster_path": d.get("poster_path") or "",
        "overview": (d.get("overview") or "").strip(),
        "keywords_tmdb": [k["name"] for k in kw],
        "_adult": bool(d.get("adult")),
    }
    # embed_text via the IMPORTED corpus recipe (takes a dict-like row)
    row["embed_text"] = make_embed_text({
        "name": row["name"], "start_year": row["start_year"],
        "genres": row["genres"], "overview": row["overview"],
    })
    return row


def passes(r: dict) -> tuple[bool, str]:
    """Same tiering as build_tv_corpus.py, one record at a time."""
    if r["_adult"]:
        return False, "adult"
    if len(r["overview"]) < MIN_OVERVIEW_CHARS:
        return False, "overview too short"

    is_en = r["original_language"] == "en"
    clean_floor = CLEAN_EN_VOTES if is_en else CLEAN_OTHER_VOTES
    prob_floor = PROB_EN_VOTES if is_en else PROB_OTHER_VOTES
    vc = r["vote_count"]

    if vc >= clean_floor:
        return True, "clean"
    if vc < prob_floor:
        return False, "below vote floor"
    if not r["genres"]:
        return False, "probation: genres"
    if not r["poster_path"]:
        return False, "probation: poster"
    return True, "probation"


def stage_details(limit=None):
    if not CANDIDATES.exists():
        sys.exit("details: no candidates.jsonl -- run --stage harvest first")

    corpus = pd.read_parquet(CORPUS)
    known = set(corpus["tmdb_id"].astype("int64"))

    cand = []
    with open(CANDIDATES, encoding="utf-8") as f:
        for line in f:
            cand.extend(json.loads(line)["ids"])
    cand = [i for i in dict.fromkeys(cand) if i not in known]

    fetched = set()
    if DETAILS.exists():
        with open(DETAILS, encoding="utf-8") as f:
            for line in f:
                fetched.add(json.loads(line)["tmdb_id"])
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
        for n, tid in enumerate(todo, 1):
            d = get(f"tv/{tid}", append_to_response="credits,keywords")
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
# STAGE 3: refresh (returning series already in the corpus)
# ---------------------------------------------------------------------------
def stage_refresh(limit=None):
    corpus = pd.read_parquet(CORPUS)
    recent = corpus.loc[
        pd.to_numeric(corpus["start_year"], errors="coerce") >= REFRESH_FROM_YEAR,
        "tmdb_id"].astype("int64").tolist()

    done = set()
    if REFRESH.exists():
        with open(REFRESH, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["tmdb_id"])
    todo = [i for i in recent if i not in done]
    if limit:
        todo = todo[:limit]

    print(f"refresh: {len(recent):,} corpus series with start_year >= {REFRESH_FROM_YEAR} | "
          f"{len(done):,} done | {len(todo):,} to do")
    if not todo:
        return

    with open(REFRESH, "a", encoding="utf-8") as out:
        for n, tid in enumerate(todo, 1):
            d = get(f"tv/{tid}")
            time.sleep(REQ_INTERVAL)
            if not d:
                continue
            lad = d.get("last_air_date") or ""
            out.write(json.dumps({
                "tmdb_id": d["id"],
                "vote_count": int(d.get("vote_count") or 0),
                "vote_average": d.get("vote_average"),
                "number_of_seasons": d.get("number_of_seasons"),
                "number_of_episodes": d.get("number_of_episodes"),
                "status": d.get("status") or "",
                "end_year": int(lad[:4]) if lad[:4].isdigit() else None,
            }) + "\n")
            if n % 200 == 0:
                out.flush()
                print(f"  {n:,}/{len(todo):,}")
    print("refresh: done")


# ---------------------------------------------------------------------------
# STAGE 4: merge
# ---------------------------------------------------------------------------
def _match_listcol_shape(new_series, corpus_sample):
    if isinstance(corpus_sample, str):
        return new_series.apply(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else (v or ""))
    return new_series


REFRESH_COLS = ["vote_count", "vote_average", "number_of_seasons",
                "number_of_episodes", "status", "end_year"]


def stage_merge(dry_run=True):
    if not DETAILS.exists():
        sys.exit("merge: no details.jsonl -- run --stage details first")

    corpus = pd.read_parquet(CORPUS)
    known = set(corpus["tmdb_id"].astype("int64"))

    rows = []
    with open(DETAILS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r.pop("_adult", None); r.pop("_pass_tier", None)
            if r["tmdb_id"] not in known:
                rows.append(r)
    if not rows:
        print("merge: nothing new to merge")
        return
    new_df = pd.DataFrame(rows).drop_duplicates(subset="tmdb_id", keep="first")
    print(f"merge: {len(new_df):,} new series to append to {len(corpus):,}")

    for col in (
        "origin_country", "keywords_tmdb"):
        if col in corpus.columns:
            sample = corpus[col].dropna().iloc[0] if corpus[col].notna().any() else None
            new_df[col] = _match_listcol_shape(new_df[col], sample)
    if "episode_run_time" in new_df.columns:
        new_df["episode_run_time"] = new_df["episode_run_time"].fillna(0)

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
    merged = merged.drop_duplicates(subset="tmdb_id", keep="first")

    if REFRESH.exists():
        ref = {}
        with open(REFRESH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ref[r["tmdb_id"]] = r
        mask = merged["tmdb_id"].astype("int64").isin(ref)
        n_ref = int(mask.sum())
        if n_ref:
            for col in REFRESH_COLS:
                if col in merged.columns:
                    vals = merged.loc[mask, "tmdb_id"].astype("int64").map(
                        lambda i: ref[i].get(col))
                    try:
                        merged.loc[mask, col] = vals.astype(merged[col].dtype)
                    except (TypeError, ValueError):
                        merged.loc[mask, col] = vals
        print(f"merge: refreshed {n_ref:,} returning series "
              f"(votes, seasons, episodes, status, end_year)")

    yr = pd.to_numeric(merged["start_year"], errors="coerce").value_counts().sort_index()
    print("\nstart_year histogram (2022+):")
    print(yr[yr.index >= 2022].to_string())
    print(f"\nfinal corpus: {len(merged):,} series (was {len(corpus):,})")

    if dry_run:
        print("\nDRY RUN -- nothing written. Re-run with --write to commit.")
        return

    backup = CORPUS.with_name("tv_corpus_pre_expansion.parquet")
    if not backup.exists():
        shutil.copy2(CORPUS, backup)
        print(f"backup written: {backup.name}")
    merged.to_parquet(CORPUS, index=False)
    print(f"WROTE {CORPUS}")
    print("\n" + "=" * 70)
    print("THE TV CORPUS AND THE LIVE EMBEDDINGS ARE NOW MISALIGNED.")
    print("Before anything touches the Space:")
    print("  1. embed_tv.py is set to v2_kw_notitle -> new filenames (inert). Verify.")
    print("  2. re-embed TV, rebuild FAISS")
    print("  3. verify parquet rows == npy shape[0] == index.ntotal + spot checks")
    print("  4. re-run build_keyword_tags.py (TV side picks up new keywords)")
    print("  5. upload corpus + embeddings + index + tv_tags, THEN flip T_EMB/T_FAISS in app.py")
    print("=" * 70)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="first month, e.g. 2022-07")
    ap.add_argument("--until", required=True, help="last month inclusive, e.g. 2026-07")
    ap.add_argument("--stage", default="all",
                    choices=["harvest", "details", "refresh", "merge", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true",
                    help="merge actually writes tv_corpus.parquet (default: dry run)")
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
        stage_refresh(limit=args.limit)
    if args.stage in ("merge", "all"):
        stage_merge(dry_run=not args.write)


if __name__ == "__main__":
    main()
