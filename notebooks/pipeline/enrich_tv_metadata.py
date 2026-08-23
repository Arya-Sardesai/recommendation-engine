"""
enrich_tv_metadata.py — add content ratings (age certification) to the TV corpus.

WHY
Three testers reported children's content reaching adult profiles, and TWO of
the three were TV: Futurama -> children's cartoons, a teen show -> children's
shows. The movie side was fixed with TMDB certifications; the TV tab has had
no age signal at all. Age suitability is not inferable from a plot overview.

WHY THIS IS NOT A RE-HARVEST
Adds COLUMNS to existing rows. Row count and order unchanged, so
tv_embeddings_*.npy and tv_faiss_*.index stay valid: no re-embed. The merge
stage asserts row order and refuses to write if it shifted.

WHY ORDINAL, NOT BINARY
The movie filter gates on `all_ages` certification AND an Animation/Family
genre. That does not transfer to TV, for two reasons:

  1. "Animation" on TV includes all anime. Cowboy Bebop, Code Geass and
     Haikyu!! are Animation and are not children's shows.
  2. The binary "does this profile like kids' content" gate breaks on real
     profiles. A tester rating Gravity Falls (TV-Y7) 5/5 would stand the
     filter down entirely -- and Gravity Falls -> The Flintstones is exactly
     the recommendation being complained about.

So this stores a MATURITY ORDINAL instead of a tier label. The useful question
is not "is this a kids' show" but "how far below this viewer's usual level does
this sit". Futurama (TV-14) -> The Flintstones (TV-Y) is a four-step drop;
Futurama -> The Addams Family (TV-G) is two. Only the first is the complaint.

    TV-Y 0 | TV-Y7 1 | TV-G 2 | TV-PG 3 | TV-14 4 | TV-MA 5

Non-US systems are mapped onto the same scale so a single number works
corpus-wide.

STAGES
  inspect : what's present, how many rows need fetching, cost estimate
  fetch   : resumable -> tv_meta_raw.jsonl
  merge   : add columns, assert row order, write a NEW parquet

USAGE
    $env:TMDB_API_KEY = "..."
    python notebooks\\pipeline\\enrich_tv_metadata.py --stage inspect
    python notebooks\\pipeline\\enrich_tv_metadata.py --stage fetch --limit 200
    python notebooks\\pipeline\\enrich_tv_metadata.py --stage fetch
    python notebooks\\pipeline\\enrich_tv_metadata.py --stage merge
    python notebooks\\pipeline\\enrich_tv_metadata.py --stage merge --write

Run from the REPO ROOT.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

PROCESSED = Path("data/processed")
RAW_DIR = Path("data/raw/tv/metadata_v1")
RAW_PATH = RAW_DIR / "tv_meta_raw.jsonl"
IN_PARQUET = PROCESSED / "tv_corpus.parquet"
OUT_PARQUET = PROCESSED / "tv_v2_meta.parquet"

MIN_VOTES = 300          # matches recommend_tv; below this nothing surfaces
SLEEP = 0.03
CERT_COUNTRIES = ["US", "GB", "IN", "CA", "AU"]

API = "https://api.themoviedb.org/3/tv/{id}"

# Ordinal maturity. Everything is mapped onto the US TV scale so one number
# works across certification systems.
MATURITY = {
    # US television
    "TV-Y": 0, "TV-Y7": 1, "TV-Y7-FV": 1, "TV-G": 2, "TV-PG": 3,
    "TV-14": 4, "TV-MA": 5,
    # US theatrical codes that appear on some TV entries
    "G": 2, "PG": 3, "PG-13": 4, "R": 5, "NC-17": 5,
    # UK
    "U": 2, "Uc": 0, "PG-GB": 3, "12": 4, "12A": 4, "15": 5, "18": 5,
    # India
    "UA": 3, "U/A": 3, "A": 5, "U/A 7+": 1, "U/A 13+": 4, "U/A 16+": 5,
    # Canada / Australia and common numeric systems
    "C": 0, "C8": 1, "G-CA": 2, "PG-CA": 3, "14+": 4, "18+": 5,
    "P": 0, "C7": 1, "M": 4, "MA15+": 5, "R18+": 5,
    "0+": 0, "6+": 1, "7+": 1, "9+": 2, "10+": 2, "12+": 4, "13+": 4,
    "16+": 5, "TV-Y7 FV": 1,
    # --- added after the first merge dry run surfaced them as UNMAPPED ------
    # Bare numeric systems (several European broadcasters + Netflix regional).
    "10": 2, "14": 4, "16": 5,
    # Brazil (ClassInd) L = livre, Spain/Portugal TP = todos los publicos.
    # These are CONTENT-WARNING systems: they mean "nothing objectionable",
    # NOT "made for children". The first dry run put Yo soy Betty, la fea --
    # a telenovela -- at level 0 beside Paw Patrol because of BR:L.
    # Mapped to 2 (TV-G, "suitable for all") rather than 0 (TV-Y, "aimed at
    # young children"), which is what they actually assert. Same category
    # error as a 1968 G rating on the movie side.
    "L": 2,
    "TP": 2,
    # Mexico (RTC): A all ages, B 12+, B-15 15+, C adults, D adults.
    # NOTE "A" already maps to 5 above for India, where it means ADULT. The
    # collision is real and unresolvable from the code alone, so Mexican A is
    # deliberately NOT added -- India's reading is the safer error here
    # (treating an all-ages show as adult only means it is not suppressed).
    # "C" is NOT added: it already means the Canadian children's code (0)
    # above, and Mexico's C means adult. CA is in CERT_COUNTRIES and MX is
    # not, so the Canadian reading wins; adding Mexico's would have silently
    # relabelled Canadian children's TV as adult.
    "B": 4, "B-15": 5, "D": 5,
    # Australia, with the space TMDB actually returns.
    "MA 15+": 5, "MA15+": 5,
}

# NOT a maturity level. "Not rated" is an absence of information, and mapping
# it to any number -- high or low -- invents a signal that does not exist.
# Left out of MATURITY on purpose so maturity_of() returns None.
_UNRATED_CODES = {"NR", "UNRATED", "NOT RATED", "N/A", "--"}


def api_key():
    k = os.environ.get("TMDB_API_KEY")
    if not k:
        raise SystemExit("Set TMDB_API_KEY first: $env:TMDB_API_KEY = 'your_key'")
    return k


def pick_rating(content_ratings):
    """First non-empty rating across preferred countries."""
    if not isinstance(content_ratings, dict):
        return ""
    by_country = {r.get("iso_3166_1"): (r.get("rating") or "").strip()
                  for r in content_ratings.get("results", []) or []}
    for cc in CERT_COUNTRIES:
        v = by_country.get(cc)
        if v:
            return f"{cc}:{v}"
    # fall back to anything at all rather than discarding the row
    for cc, v in by_country.items():
        if v:
            return f"{cc}:{v}"
    return ""


def maturity_of(cert):
    """Ordinal 0-5, or None when unknown. None is NOT 'safe' and must never be
    treated as a low value -- an absent rating carries no information."""
    if not cert:
        return None
    code = cert.split(":")[-1].strip()
    if code in MATURITY:
        return MATURITY[code]
    up = code.upper()
    for k, v in MATURITY.items():
        if k.upper() == up:
            return v
    return None


def load_done():
    done = {}
    if RAW_PATH.exists():
        with RAW_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done[str(rec["id"])] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def stage_inspect():
    df = pd.read_parquet(IN_PARQUET)
    print(f"corpus: {len(df):,} rows")
    print(f"columns: {list(df.columns)}\n")
    for col in ("content_rating", "maturity", "number_of_seasons"):
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"  {col:18} PRESENT  ({nn:,} non-null, {nn / len(df):.0%})")
        else:
            print(f"  {col:18} missing")

    need = df[df["vote_count"] >= MIN_VOTES]
    done = load_done()
    todo = [i for i in need["tmdb_id"].astype(str) if i not in done]
    print(f"\nrecommendable (vote_count >= {MIN_VOTES}): {len(need):,}")
    print(f"already fetched                        : {len(done):,}")
    print(f"still to fetch                         : {len(todo):,}")
    print(f"estimated fetch time                   : ~{len(todo)*SLEEP/60:.0f} min")
    print("\nAdds COLUMNS only — embeddings and FAISS untouched, no re-embed.")


def stage_fetch(limit=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PARQUET, columns=["tmdb_id", "vote_count", "name"])
    need = df[df["vote_count"] >= MIN_VOTES]
    done = load_done()
    todo = [str(i) for i in need["tmdb_id"].astype(str) if str(i) not in done]
    if limit:
        todo = todo[:limit]
    print(f"fetching {len(todo):,} series ({len(done):,} already done)")

    key = api_key()
    ok = fail = 0
    with RAW_PATH.open("a", encoding="utf-8") as fh:
        for n, tid in enumerate(todo, 1):
            try:
                r = requests.get(API.format(id=tid),
                                 params={"api_key": key,
                                         "append_to_response": "content_ratings"},
                                 timeout=30)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 5))
                    print(f"  rate limited; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    fh.write(json.dumps({"id": tid, "content_rating": "",
                                         "gone": True}) + "\n")
                    fail += 1
                    continue
                r.raise_for_status()
                d = r.json()
                fh.write(json.dumps({
                    "id": tid,
                    "content_rating": pick_rating(d.get("content_ratings")),
                    "in_production": bool(d.get("in_production")),
                    "status": d.get("status") or "",
                    "episode_count": d.get("number_of_episodes"),
                }, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                print(f"  {tid} failed ({type(e).__name__}: {e})")
                fail += 1
            if n % 250 == 0:
                fh.flush()
                print(f"  {n:,}/{len(todo):,}  ok={ok:,} fail={fail:,}")
            time.sleep(SLEEP)
    print(f"\nfetch done: ok={ok:,} fail={fail:,} -> {RAW_PATH}")
    print("Re-run to resume; already-fetched ids are skipped.")


def stage_merge(write=False):
    df = pd.read_parquet(IN_PARQUET)
    before_ids = df["tmdb_id"].astype(str).tolist()
    done = load_done()
    if not done:
        raise SystemExit(f"{RAW_PATH} is empty — run --stage fetch first")

    meta = pd.DataFrame([
        {"_id": k,
         "content_rating": v.get("content_rating", ""),
         "in_production_new": bool(v.get("in_production", False)),
         "status_new": v.get("status", ""),
         "episode_count_new": v.get("episode_count")}
        for k, v in done.items()
    ])
    df["_id"] = df["tmdb_id"].astype(str)
    merged = df.merge(meta, on="_id", how="left").drop(columns="_id")

    assert len(merged) == len(df), f"row count changed {len(df)} -> {len(merged)}"
    assert merged["tmdb_id"].astype(str).tolist() == before_ids, \
        "ROW ORDER CHANGED — embeddings would be misaligned; refusing to write"
    print(f"row order preserved: {len(merged):,} rows")

    collided = [c for c in merged.columns if c.endswith(("_x", "_y"))]
    if collided:
        raise SystemExit(f"unexpected column collision: {collided}")

    cert_s = merged["content_rating"].fillna("")
    merged["maturity"] = [maturity_of(c) for c in cert_s]

    rec_mask = merged["vote_count"] >= MIN_VOTES
    n_rec = int(rec_mask.sum())
    print(f"\n  recommendable series (vote_count >= {MIN_VOTES}): {n_rec:,}")
    print(f"  content_rating populated {(cert_s[rec_mask] != '').mean():.1%}"
          f"   <-- decides whether the filter is viable")
    print(f"  maturity resolved        "
          f"{merged.loc[rec_mask, 'maturity'].notna().mean():.1%}"
          f"   (mapped onto the 0-5 scale)")

    print("\n  maturity distribution among recommendable series:")
    LBL = {0: "0 TV-Y", 1: "1 TV-Y7", 2: "2 TV-G", 3: "3 TV-PG",
           4: "4 TV-14", 5: "5 TV-MA"}
    vc = merged.loc[rec_mask, "maturity"].value_counts().sort_index()
    for m, c in vc.items():
        print(f"    {LBL.get(int(m), m):10s} {c:6,}  {c/max(n_rec,1):5.1%}")
    n_missing = int(merged.loc[rec_mask, "maturity"].isna().sum())
    print(f"    {'unknown':10s} {n_missing:6,}  {n_missing/max(n_rec,1):5.1%}"
          f"   <- NOT safe, must pass through")

    unmapped = sorted({c.split(":")[-1] for c in cert_s[rec_mask]
                       if c and maturity_of(c) is None})
    if unmapped:
        print(f"\n  UNMAPPED codes -> add to MATURITY at the top of this file:")
        print(f"    {unmapped[:25]}")

    # The low end is what the filter suppresses, so print MORE of it and show
    # the country. A non-US code at level 0-1 is a mapping bug, not a finding.
    low = merged[rec_mask & merged["maturity"].isin([0, 1])]
    non_us = low[~low["content_rating"].fillna("").str.startswith("US:")]
    if len(non_us):
        print(f"\n  WARNING: {len(non_us)} suppressible series carry a NON-US code.")
        print("  Non-US systems rate content, not target audience — check these:")
        for r in non_us.nlargest(8, "vote_count")[["name", "content_rating"]].itertuples():
            print(f"    {str(r.name)[:44]:46s} [{r.content_rating}]")

    print("\n  sample by level (sanity-check by eye):")
    for m in range(6):
        sub = merged[rec_mask & (merged["maturity"] == m)]
        if len(sub):
            names = sub.nlargest(4, "vote_count")[["name", "content_rating"]]
            shown = ", ".join(f"{r.name[:24]} [{r.content_rating}]"
                              for r in names.itertuples())
            print(f"    {LBL[m]:10s} {shown}")

    if not write:
        print("\nDRY RUN — nothing written. Re-run with --write.")
        return
    merged.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET} ({len(merged):,} rows)")
    print("Upload the PARQUET ONLY; embeddings and index are unchanged.")
    print("Then point T_CORPUS at the new filename.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["inspect", "fetch", "merge"], default="inspect")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.stage == "inspect":
        stage_inspect()
    elif args.stage == "fetch":
        stage_fetch(limit=args.limit)
    else:
        stage_merge(write=args.write)


if __name__ == "__main__":
    main()
