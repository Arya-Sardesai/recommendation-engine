"""
merge_tmdb_bulk.py — v2 step 1b: merge bulk TMDB fetch into the corpus parquets.

Reads data/tmdb_raw/{movies_raw,tv_raw}.jsonl and:
  1. Builds a diff report (old vs new metadata) -> data/tmdb_raw/enrichment_diff_report.json
  2. Backs up current parquets to *_pre_v2.parquet (one-time insurance,
     delete after validation)
  3. Overwrites refreshed metadata columns IN PLACE (one schema, one truth)
  4. Adds a `keywords` column (list[str], raw/unfiltered — KW_STOP is applied
     at embed-text assembly, not at storage)

404/error rows: old data kept as-is, keywords = [].

Usage:
  python scripts/merge_tmdb_bulk.py            # both
  python scripts/merge_tmdb_bulk.py movies
  python scripts/merge_tmdb_bulk.py tv
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/tmdb_raw")
MOVIES_PARQUET = Path("data/processed/movies.parquet")
TV_PARQUET = Path("data/processed/tv_corpus.parquet")
REPORT_PATH = RAW_DIR / "enrichment_diff_report.json"

# Columns refreshed from TMDB, mapped old_col -> TMDB response field.
# Deliberately NOT refreshed: genres (old parquet has parsed format; TMDB
# returns list-of-dicts — remap later if wanted), director/cast (IMDb-sourced),
# release_date/release_year (dump values fine; placeholders get healed via
# overview/votes anyway), embed_text (rebuilt at re-embed).
MOVIE_REFRESH = {
    "vote_average": "vote_average",
    "vote_count": "vote_count",
    "runtime": "runtime",
    "overview": "overview",
    "tagline": "tagline",
    "poster_path": "poster_path",
}
TV_REFRESH = {
    "vote_average": "vote_average",
    "vote_count": "vote_count",
    "popularity": "popularity",
    "number_of_seasons": "number_of_seasons",
    "number_of_episodes": "number_of_episodes",
    "status": "status",
    "overview": "overview",
    "tagline": "tagline",
    "poster_path": "poster_path",
}


def load_raw(path: Path, kw_key: str) -> dict[int, dict]:
    """id -> {'fields': {...}, 'keywords': [...], 'status': str}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line
            entry = {"status": rec["status"], "fields": None, "keywords": []}
            if rec["status"] == "ok" and rec["data"]:
                entry["fields"] = rec["data"]
                kw_obj = rec["data"].get("keywords") or {}
                entry["keywords"] = [k["name"] for k in kw_obj.get(kw_key, [])]
            out[rec["id"]] = entry  # later lines win (resume-rerun duplicates)
    return out


def merge_corpus(name: str, parquet_path: Path, raw_path: Path,
                 id_col: str, refresh: dict[str, str], kw_key: str) -> dict:
    print(f"\n{'=' * 60}\n[{name}] merging\n{'=' * 60}")
    df = pd.read_parquet(parquet_path)
    n_rows = len(df)
    raw = load_raw(raw_path, kw_key)
    print(f"parquet rows: {n_rows}, raw records: {len(raw)}")

    missing = set(df[id_col].astype(int)) - set(raw)
    if missing:
        print(f"WARNING: {len(missing)} corpus ids absent from raw JSONL "
              f"(fetch incomplete?). Their rows will be left untouched.")

    statuses = pd.Series(
        [raw.get(int(i), {}).get("status", "missing") for i in df[id_col]]
    )
    print("status breakdown:", statuses.value_counts().to_dict())

    ok_mask = (statuses == "ok").to_numpy()

    # ---- Diff report (BEFORE overwrite) ------------------------------------
    report: dict = {"corpus": name, "rows": n_rows,
                    "status_breakdown": statuses.value_counts().to_dict(),
                    "column_drift": {}}

    new_vals = {}
    for old_col, field in refresh.items():
        new_vals[old_col] = [
            raw[int(i)]["fields"].get(field) if ok else None
            for i, ok in zip(df[id_col], ok_mask)
        ]

    for col in refresh:
        old_s = df[col]
        new_s = pd.Series(new_vals[col], index=df.index)
        if pd.api.types.is_numeric_dtype(old_s):
            new_num = pd.to_numeric(new_s, errors="coerce")
            delta = (new_num - old_s)[ok_mask]
            delta = delta.dropna()
            entry = {
                "median_delta": float(delta.median()) if len(delta) else None,
                "mean_delta": float(delta.mean()) if len(delta) else None,
                "pct_changed": float((delta != 0).mean()) if len(delta) else None,
            }
            if col == "vote_count" and len(delta):
                top = delta.nlargest(10)
                title_col = "title" if "title" in df.columns else "name"
                entry["top10_gainers"] = [
                    {"title": str(df.loc[i, title_col]),
                     "old": int(old_s.loc[i]) if pd.notna(old_s.loc[i]) else None,
                     "new": int(new_num.loc[i]),
                     "gain": int(v)}
                    for i, v in top.items()
                ]
            report["column_drift"][col] = entry
        else:
            old_f = old_s.fillna("").astype(str)
            new_f = new_s.fillna("").astype(str)
            changed = (old_f != new_f) & ok_mask
            report["column_drift"][col] = {"pct_changed": float(changed.mean())}

    kw_lists = [raw.get(int(i), {}).get("keywords", []) for i in df[id_col]]
    kw_counts = pd.Series([len(k) for k in kw_lists])
    report["keywords"] = {
        "coverage_pct": float((kw_counts > 0).mean()),
        "median_density_nonzero": float(kw_counts[kw_counts > 0].median())
        if (kw_counts > 0).any() else 0,
        "mean_density": float(kw_counts.mean()),
    }
    print(f"keyword coverage: {report['keywords']['coverage_pct']:.1%}")

    # ---- Backup, then overwrite --------------------------------------------
    backup = parquet_path.with_name(parquet_path.stem + "_pre_v2.parquet")
    if not backup.exists():
        shutil.copy2(parquet_path, backup)
        print(f"backup written: {backup}")
    else:
        print(f"backup already exists (not overwritten): {backup}")

    for col in refresh:
        fresh = pd.Series(new_vals[col], index=df.index)
        if pd.api.types.is_numeric_dtype(df[col]):
            fresh = pd.to_numeric(fresh, errors="coerce")
        # only overwrite where fetch succeeded AND value is not null
        take_new = ok_mask & fresh.notna().to_numpy()
        df[col] = df[col].where(~take_new, fresh)

    df["keywords_tmdb"] = kw_lists

    # ---- Validation before writing -----------------------------------------
    assert len(df) == n_rows, "row count changed during merge!"
    for col in refresh:
        null_pct = df[col].isna().mean()
        if null_pct > 0.5:
            raise RuntimeError(f"{col} is {null_pct:.0%} null post-merge — aborting write")

    df.to_parquet(parquet_path, index=False)
    print(f"written: {parquet_path} ({len(df)} rows, "
          f"+keywords_tmdb column)")
    return report


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    reports = {}

    if which in ("movies", "both"):
        reports["movies"] = merge_corpus(
            "movies", MOVIES_PARQUET, RAW_DIR / "movies_raw.jsonl",
            "id", MOVIE_REFRESH, kw_key="keywords")

    if which in ("tv", "both"):
        reports["tv"] = merge_corpus(
            "tv", TV_PARQUET, RAW_DIR / "tv_raw.jsonl",
            "tmdb_id", TV_REFRESH, kw_key="results")

    if REPORT_PATH.exists():
        existing = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        existing.update(reports)
        reports = existing
    REPORT_PATH.write_text(json.dumps(reports, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\ndiff report: {REPORT_PATH}")
    print("Next: eyeball the report, spot-check a few rows, run the app "
          "smoke test, then KW_STOP + taxonomy design.")