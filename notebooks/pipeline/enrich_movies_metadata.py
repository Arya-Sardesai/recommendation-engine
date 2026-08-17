"""
enrich_movies_metadata.py — add runtime + certification to the movie corpus.

WHY THIS IS NOT A RE-HARVEST
This ADDS COLUMNS to existing rows. Row count and row order are unchanged, so
movie_embeddings_*.npy and movie_faiss_*.index stay valid: no re-embed, no
index rebuild, no alignment risk. Only the parquet is re-uploaded. The merge
stage asserts row order explicitly and refuses to write if it shifted.

Contrast with relaxing a filter (e.g. the TV overview floor), which ADMITS NEW
ROWS and therefore forces a full re-embed. Keep the two jobs separate.

WHAT IT FIXES
  runtime       -> "Team Thor" is a 4-minute Marvel promo short sitting in a
                   film corpus and was recommended as a film. Nothing in the
                   data distinguished it.
  certification -> three separate testers reported children's content reaching
                   adult profiles (Futurama -> kids' cartoons; a teen show ->
                   children's shows). Age suitability is not inferable from a
                   plot overview, and no amount of embedding work will fix it.

SCOPE
Only films that can actually be recommended are fetched. recommend_movies
applies min_votes=400, so anything below that never surfaces and is not worth
a request. That is the difference between ~15K calls and ~121K.

TMDB's append_to_response returns details and release_dates together, so
runtime and certification cost ONE request per film, not two.

STAGES
  inspect : what's already present, how many rows need fetching, cost estimate
  fetch   : resumable -> meta_raw.jsonl
  merge   : add columns, assert row order, write a NEW parquet

READING THE MERGE OUTPUT
Coverage is reported over RECOMMENDABLE films (vote_count >= MIN_VOTES), not
over the whole corpus. Only recommendable films were ever fetched, so a whole-
corpus rate is bounded at ~8% however well the fetch went, and would make a
healthy result look like a failure. The decision thresholds are:
  >= 80% certification coverage -> a hard kids-content filter is viable
  <  60%                        -> missing cert is indistinguishable from an
                                   adult one; treat as a soft signal or ship
                                   as a stated limitation

USAGE
    $env:TMDB_API_KEY = "..."
    python notebooks\\pipeline\\enrich_movies_metadata.py --stage inspect
    python notebooks\\pipeline\\enrich_movies_metadata.py --stage fetch --limit 200
    python notebooks\\pipeline\\enrich_movies_metadata.py --stage fetch
    python notebooks\\pipeline\\enrich_movies_metadata.py --stage merge
    python notebooks\\pipeline\\enrich_movies_metadata.py --stage merge --write

Run from the REPO ROOT — the paths below are relative to it.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

PROCESSED = Path("data/processed")
RAW_DIR = Path("data/raw/movies/metadata_v1")
RAW_PATH = RAW_DIR / "meta_raw.jsonl"
IN_PARQUET = PROCESSED / "movies.parquet"
OUT_PARQUET = PROCESSED / "movies_v2_meta.parquet"

MIN_VOTES = 400          # matches recommend_movies; below this nothing surfaces
SLEEP = 0.03             # ~33 req/s, under TMDB's limit
CERT_COUNTRIES = ["US", "GB", "IN"]   # first match wins

API = "https://api.themoviedb.org/3/movie/{id}"

# Certification codes, matched EXACTLY against the part AFTER the country
# prefix. The previous version ran the unanchored regex "G|PG|U|TV-Y" against
# the whole "US:PG-13"-style string, so the U in every US: value and the G in
# every GB: value matched: 10 of 11 test certifications came back as
# children's content, GB:18 among them.
CERT_ALL_AGES = {"G", "U", "TV-Y", "TV-Y7", "TV-G", "0+", "6+", "AL"}
CERT_PARENTAL = {"PG", "PG-13", "PG13", "TV-PG", "TV-14", "UA", "U/A",
                 "12", "12A", "13", "9+", "12+"}
CERT_ADULT = {"R", "NC-17", "TV-MA", "A", "X", "15", "16", "18", "16+", "18+"}
# "not rated" is NOT the same as "we never fetched one", and neither is a safe
# class. Kept as its own tier so it cannot be mistaken for a finding.
CERT_UNRATED = {"NR", "UNRATED", "NOT RATED", "N/A"}


def api_key():
    k = os.environ.get("TMDB_API_KEY")
    if not k:
        raise SystemExit("Set TMDB_API_KEY first: $env:TMDB_API_KEY = 'your_key'")
    return k


def pick_certification(release_dates):
    """First non-empty certification across preferred countries."""
    if not isinstance(release_dates, dict):
        return ""
    by_country = {r.get("iso_3166_1"): r.get("release_dates", [])
                  for r in release_dates.get("results", []) or []}
    for cc in CERT_COUNTRIES:
        for entry in by_country.get(cc, []) or []:
            cert = (entry.get("certification") or "").strip()
            if cert:
                return f"{cc}:{cert}"
    return ""


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

    for col in ("runtime", "certification", "adult"):
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"  {col:14} PRESENT  ({nn:,} non-null, {nn / len(df):.0%})")
        else:
            print(f"  {col:14} missing")

    need = df[df["vote_count"] >= MIN_VOTES]
    done = load_done()
    todo = [i for i in need["id"].astype(str) if i not in done]
    print(f"\nrecommendable (vote_count >= {MIN_VOTES}): {len(need):,}")
    print(f"already fetched                        : {len(done):,}")
    print(f"still to fetch                         : {len(todo):,}")
    mins = len(todo) * SLEEP / 60
    print(f"estimated fetch time                   : ~{mins:.0f} min")
    print("\nNOTE: this adds COLUMNS only. Embeddings and FAISS index are")
    print("untouched, so no re-embed is required.")


def stage_fetch(limit=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PARQUET, columns=["id", "vote_count", "title"])
    need = df[df["vote_count"] >= MIN_VOTES]
    done = load_done()
    todo = [str(i) for i in need["id"].astype(str) if str(i) not in done]
    if limit:
        todo = todo[:limit]
    print(f"fetching {len(todo):,} films ({len(done):,} already done)")

    key = api_key()
    ok = fail = 0
    with RAW_PATH.open("a", encoding="utf-8") as fh:
        for n, mid in enumerate(todo, 1):
            try:
                r = requests.get(API.format(id=mid),
                                 params={"api_key": key,
                                         "append_to_response": "release_dates"},
                                 timeout=30)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 5))
                    print(f"  rate limited; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    fh.write(json.dumps({"id": mid, "runtime": None,
                                         "certification": "", "gone": True}) + "\n")
                    fail += 1
                    continue
                r.raise_for_status()
                d = r.json()
                fh.write(json.dumps({
                    "id": mid,
                    "runtime": d.get("runtime"),
                    "adult": bool(d.get("adult")),
                    "certification": pick_certification(d.get("release_dates")),
                }, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                print(f"  {mid} failed ({type(e).__name__}: {e})")
                fail += 1
            if n % 250 == 0:
                fh.flush()
                print(f"  {n:,}/{len(todo):,}  ok={ok:,} fail={fail:,}")
            time.sleep(SLEEP)
    print(f"\nfetch done: ok={ok:,} fail={fail:,} -> {RAW_PATH}")
    print("Re-run this stage to resume; already-fetched ids are skipped.")


def stage_merge(write=False):
    df = pd.read_parquet(IN_PARQUET)
    before_ids = df["id"].astype(str).tolist()
    done = load_done()
    if not done:
        raise SystemExit(f"{RAW_PATH} is empty — run --stage fetch first")

    # runtime_new, not runtime: movies.parquet ALREADY carries a runtime column
    # (--stage inspect reported it 100% populated). Merging two columns of the
    # same name makes pandas emit runtime_x / runtime_y and no runtime at all,
    # and every read below raises KeyError — after the row-order asserts have
    # already printed "row order preserved", which reads like the guard fired.
    meta = pd.DataFrame([
        {"_id": k, "runtime_new": v.get("runtime"),
         "certification": v.get("certification", ""),
         "adult_flag": bool(v.get("adult", False))}
        for k, v in done.items()
    ])
    df["_id"] = df["id"].astype(str)
    merged = df.merge(meta, on="_id", how="left").drop(columns="_id")

    # THE SAFETY PROPERTY: same rows, same order -> .npy and FAISS still valid.
    assert len(merged) == len(df), f"row count changed {len(df)} -> {len(merged)}"
    assert merged["id"].astype(str).tolist() == before_ids, \
        "ROW ORDER CHANGED — embeddings would be misaligned; refusing to write"
    print(f"row order preserved: {len(merged):,} rows")

    collided = [c for c in merged.columns if c.endswith(("_x", "_y"))]
    if collided:
        raise SystemExit(
            f"unexpected column collision: {collided}\n"
            f"a fetched field shares a name with an existing corpus column.\n"
            f"Rename it in the `meta` DataFrame above and coalesce explicitly."
        )

    # Coalesce runtime: trust the fresh TMDB value, fall back to the dump.
    if "runtime_new" in merged.columns:
        new_rt = pd.to_numeric(merged["runtime_new"], errors="coerce")
        if "runtime" in df.columns:
            base_rt = pd.to_numeric(merged["runtime"], errors="coerce")
            disagree = int((base_rt.notna() & new_rt.notna() & (base_rt != new_rt)).sum())
            print(f"  runtime: {disagree:,} rows where TMDB disagrees with the dump"
                  f" (TMDB wins)")
            merged["runtime"] = new_rt.fillna(base_rt)
        else:
            merged["runtime"] = new_rt
        merged = merged.drop(columns="runtime_new")

    # ---- coverage, over films that can actually SURFACE --------------------
    # Only vote_count >= MIN_VOTES was ever fetched, so a whole-corpus rate is
    # bounded at ~8% no matter how well the fetch went. The design thresholds
    # (>=80% viable / <60% ship-as-limitation) are about recommendable films.
    rec_mask = merged["vote_count"] >= MIN_VOTES
    n_rec = int(rec_mask.sum())
    cert_s = merged["certification"].fillna("")

    print(f"\n  recommendable films (vote_count >= {MIN_VOTES}): {n_rec:,}")
    print(f"  runtime populated       "
          f"{merged.loc[rec_mask, 'runtime'].notna().mean():.1%}"
          f"   (whole corpus {merged['runtime'].notna().mean():.1%})")
    print(f"  certification populated "
          f"{(cert_s[rec_mask] != '').mean():.1%}"
          f"   <-- THIS decides the kids-content filter")
    print(f"      (whole corpus {(cert_s != '').mean():.1%} — not the number to read;")
    print(f"       unfetched low-vote films can never have one)")

    # ---- certification tiers ------------------------------------------------
    code = cert_s.str.split(":").str[-1].str.strip().str.upper()
    tier = pd.Series("none", index=merged.index)
    tier[code.isin(CERT_ADULT)] = "adult"
    tier[code.isin(CERT_PARENTAL)] = "parental"
    tier[code.isin(CERT_ALL_AGES)] = "all_ages"
    tier[code.isin(CERT_UNRATED)] = "unrated"
    tier[cert_s == ""] = "missing"
    merged["cert_tier"] = tier

    print("\n  certification tiers among recommendable films:")
    print("    (unrated/none/missing are NOT suppressible — no cert is not a")
    print("     safe cert, and must not be treated as one by the filter)")
    vc = tier[rec_mask].value_counts()
    for t in ("all_ages", "parental", "adult", "unrated", "none", "missing"):
        if t in vc:
            print(f"    {t:10s} {vc[t]:7,}  {vc[t] / max(n_rec, 1):6.1%}")

    unmapped = sorted(set(code[rec_mask & (cert_s != "") & (tier == "none")]))
    if unmapped:
        print(f"\n  UNMAPPED codes -> add to the CERT_* sets at the top of this file:")
        print(f"    {unmapped[:25]}")
        print(f"    ({len(unmapped)} distinct; they are currently in neither tier)")

    print("\n  sample of what each tier caught (sanity-check by eye):")
    for t in ("all_ages", "parental", "adult"):
        sub = merged[rec_mask & (tier == t)]
        if len(sub):
            names = sub.nlargest(4, "vote_count")[["title", "certification"]]
            shown = ", ".join(f"{r.title[:26]} [{r.certification}]"
                              for r in names.itertuples())
            print(f"    {t:10s} {shown}")

    # ---- runtime floor: decide MOVIE_MIN_RUNTIME from this output -----------
    shorts = merged[merged["runtime"].notna() & (merged["runtime"] < 40) & rec_mask]
    print(f"\n  recommendable films under 40 min: {len(shorts):,}")
    if len(shorts):
        print(shorts.nlargest(6, "vote_count")[["title", "runtime", "vote_count"]]
              .to_string(index=False))
    print("\n  MOVIE_MIN_RUNTIME candidates (null/zero runtime is never excluded):")
    for floor in (10, 15, 20, 30, 40):
        n = int(((merged["runtime"] > 0) & (merged["runtime"] < floor) & rec_mask).sum())
        print(f"    a {floor}-min floor removes {n:,} recommendable films")

    if not write:
        print("\nDRY RUN — nothing written. Re-run with --write.")
        return
    merged.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET} ({len(merged):,} rows)")
    print("Upload the PARQUET ONLY; embeddings and index are unchanged.")
    print("Then point M_CORPUS at the new filename.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["inspect", "fetch", "merge"], default="inspect")
    ap.add_argument("--limit", type=int, default=None, help="cap fetch (smoke test)")
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
