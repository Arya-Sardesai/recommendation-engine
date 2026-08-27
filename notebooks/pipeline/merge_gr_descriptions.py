"""
merge_gr_descriptions.py — rescue the books Hardcover catalogues but does not
describe, using descriptions from the 2017 UCSD Goodreads scrape.

THE CHAIN THAT LED HERE
  fetch_hardcover_targeted_v2 --stage fetch   8 of 27 works had a described
                                              record; 19 did not
  Open Library (works + editions)             2 of 19
  Wikipedia                                   19 of 19 "hits", all of them the
                                              AUTHOR's article -- thirteen Zaidi
                                              titles sharing one 500-char bio
  Google Books                                rate-limited without a key
  UCSD Goodreads dump (2.36M books, 2017)     12 of 19, author-verified,
                                              real reader-facing synopses

The descriptions were never missing. They were behind an API that Goodreads
retired in 2020, and the academic scrape still has them.

AUTHOR VERIFICATION IS NOT OPTIONAL. Matching on title alone returned David
Baldacci's Zero Day and Catherine Coulter's Eleventh Hour for S. Hussain Zaidi.
Across 2.36M books, generic titles collide constantly. gr_desc.json must be
built with the author check in place.

WHAT THIS PRODUCES
Rows for books that are NOT in the corpus at all (they were rejected at harvest
for having no description). Hardcover supplies the metadata -- cover, genres,
moods, tags, ratings -- and the Goodreads dump supplies only the description
field. Mixing at that granularity keeps every other column on one source.

PROVENANCE: rows carry source="hardcover+goodreads_dump_2017" so the mixed
origin is visible in the parquet rather than implied. The dump is academic-use
only (Wan & McAuley, RecSys'18), not for redistribution -- embedding derived
vectors is fine, republishing the text is not.

STALENESS: these descriptions are from 2017. The corpus moved OFF a frozen 2017
Goodreads dump on purpose. This re-introduces that staleness for ~12 rows, in
exchange for books that are otherwise unrecommendable. State it, do not hide it.

USAGE
    $env:HARDCOVER_TOKEN = "..."
    python notebooks\\pipeline\\merge_gr_descriptions.py --stage fetch
    python notebooks\\pipeline\\merge_gr_descriptions.py --stage merge
    python notebooks\\pipeline\\merge_gr_descriptions.py --stage merge --write
"""
import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

ENDPOINT = "https://api.hardcover.app/v1/graphql"
RAW_DIR = Path("data/raw/books/hardcover_v2")
PROCESSED = Path("data/processed")
REJECTS = RAW_DIR / "targeted_rejects.jsonl"
GR_DESC = Path("gr_desc.json")               # written by the dump probe
TARGET_RAW = RAW_DIR / "targeted_raw.jsonl"  # the 8 Hardcover already described
RESCUED_RAW = RAW_DIR / "rescued_raw.jsonl"
BASE_PARQUET = PROCESSED / "books_v2_hc.parquet"
OUT_PARQUET = PROCESSED / "books_v2_hc2.parquet"

MIN_DESC_CHARS = 100
SLEEP = 0.35

BOOK_BY_ID = """
query ById($ids: [Int!]) {
  books(where: {id: {_in: $ids}}) {
    id
    slug
    title
    subtitle
    description
    release_date
    pages
    rating
    ratings_count
    users_count
    cached_tags
    cached_image
    contributions { author { name } }
  }
}
"""


def headers():
    tok = os.environ.get("HARDCOVER_TOKEN")
    if not tok:
        raise SystemExit("Set HARDCOVER_TOKEN first")
    return {"authorization": tok, "Content-Type": "application/json"}


def run_query(query, variables=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, headers=headers(),
                              json={"query": query, "variables": variables or {}},
                              timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            if "errors" in d:
                print("   GraphQL errors:", json.dumps(d["errors"])[:300])
                return None
            return d.get("data")
        except Exception as e:
            print(f"   request failed ({type(e).__name__}); retrying")
            time.sleep(3 * (attempt + 1))
    return None


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(title):
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def _tag_list(cached_tags, category):
    if not isinstance(cached_tags, dict):
        return []
    items = cached_tags.get(category)
    if not isinstance(items, list):
        return []
    out = []
    for t in items:
        if isinstance(t, dict):
            name = t.get("tag") or t.get("name") or ""
            if name:
                out.append(str(name))
        elif isinstance(t, str) and t:
            out.append(t)
    return out


def cover_url(ci):
    if isinstance(ci, dict):
        return ci.get("url") or ""
    return ci if isinstance(ci, str) else ""


def primary_author(contribs):
    if not isinstance(contribs, list):
        return ""
    for c in contribs:
        if isinstance(c, dict):
            a = c.get("author")
            if isinstance(a, dict) and a.get("name"):
                return str(a["name"])
    return ""


def load_gr():
    if not GR_DESC.exists():
        raise SystemExit(f"{GR_DESC} not found — run the dump probe first")
    raw = json.loads(GR_DESC.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        desc = v.get("desc") if isinstance(v, dict) else v
        if isinstance(desc, str) and len(desc.strip()) >= MIN_DESC_CHARS:
            out[k] = desc.strip()
    return out


def stage_fetch():
    gr = load_gr()
    print(f"goodreads descriptions available: {len(gr)}")

    rejects = []
    with REJECTS.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("reason") == "no_described_record":
                rejects.append(r)

    wanted = [(r, gr[_core(r.get("title", ""))])
              for r in rejects if _core(r.get("title", "")) in gr]
    print(f"rejected works with a usable description: {len(wanted)}\n")
    if not wanted:
        raise SystemExit("nothing to rescue")

    ids = [int(r["id"]) for r, _ in wanted if r.get("id") is not None]
    data = run_query(BOOK_BY_ID, {"ids": ids})
    time.sleep(SLEEP)
    if not data or not data.get("books"):
        raise SystemExit("Hardcover returned nothing for those ids")

    by_id = {int(b["id"]): b for b in data["books"]}
    print(f"Hardcover returned metadata for {len(by_id)} of {len(ids)} ids\n")

    rescued = 0
    with RESCUED_RAW.open("w", encoding="utf-8") as fh:
        for r, desc in wanted:
            bid = int(r["id"]) if r.get("id") is not None else None
            b = by_id.get(bid)
            if not b:
                print(f"   MISS  {str(r.get('title'))[:50]} (id {bid} not returned)")
                continue
            # ONLY the description comes from the dump. Everything else stays
            # Hardcover, so one field is mixed-provenance rather than the row.
            b["description"] = desc
            b["_desc_source"] = "goodreads_dump_2017"
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
            rescued += 1
            print(f"   OK    {str(b.get('title'))[:50]:52s} {len(desc):>5} chars")

    print(f"\nrescued {rescued} -> {RESCUED_RAW}")


def stage_merge(write=False):
    import pandas as pd

    if not BASE_PARQUET.exists():
        raise SystemExit(f"{BASE_PARQUET} not found")
    base = pd.read_parquet(BASE_PARQUET)
    existing = set(base["book_id"].astype(str))
    print(f"base corpus: {len(base):,} rows")

    def rows_from(path, desc_source):
        out = []
        if not path.exists():
            print(f"  ({path.name} not present, skipping)")
            return out
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                b = json.loads(line)
                bid = str(b["id"])
                if bid in existing:
                    continue
                desc = (b.get("description") or "").strip()
                if len(desc) < MIN_DESC_CHARS:
                    continue
                rd = b.get("release_date") or ""
                year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
                ct = b.get("cached_tags")
                src = ("hardcover+goodreads_dump_2017"
                       if b.get("_desc_source") == "goodreads_dump_2017"
                       else "hardcover")
                out.append({
                    "book_id": bid, "slug": b.get("slug") or "",
                    "title": b.get("title") or "", "subtitle": b.get("subtitle") or "",
                    "description": desc,
                    "primary_author": primary_author(b.get("contributions")),
                    "genres": _tag_list(ct, "Genre"), "moods": _tag_list(ct, "Mood"),
                    "tags": _tag_list(ct, "Tag"),
                    "publication_year": year, "release_date": rd,
                    "num_pages": b.get("pages"), "average_rating": b.get("rating"),
                    "ratings_count": b.get("ratings_count") or 0,
                    "users_count": b.get("users_count") or 0,
                    "image_url": cover_url(b.get("cached_image")),
                    "source": src, "work_id": bid,
                })
                existing.add(bid)
        return out

    rows = rows_from(TARGET_RAW, "hardcover") + rows_from(RESCUED_RAW, "gr")
    if not rows:
        print("nothing new to append.")
        return

    new = pd.DataFrame(rows).drop_duplicates(subset="book_id").reset_index(drop=True)

    built = set(rows[0])
    base_cols = set(base.columns)
    dropped = sorted(built - base_cols)
    filled = sorted(base_cols - built)
    if dropped:
        raise SystemExit(f"fields would be DISCARDED by the reindex: {dropped}")
    if filled:
        print(f"  NOTE: appended rows get NaN in {filled}")

    new = new.reindex(columns=list(base.columns))
    out = pd.concat([base, new], ignore_index=True)

    assert len(out) == len(base) + len(new), "row count mismatch"
    assert out["book_id"].iloc[:len(base)].tolist() == base["book_id"].tolist(), \
        "existing rows moved — refusing to write"
    assert out["book_id"].is_unique, "duplicate book_id"
    assert list(out.columns) == list(base.columns), "column set changed"

    print(f"\nappending {len(new)} rows:")
    for _, r in new.iterrows():
        mark = "GR" if "goodreads" in str(r["source"]) else "HC"
        print(f"   [{mark}] {str(r['title'])[:52]:54s} {str(r['primary_author'])[:24]}")
    print(f"\n  by description source:")
    print(new["source"].value_counts().to_string())
    print(f"\n  append-only verified: rows 0..{len(base)-1} unchanged")
    print(f"  new rows occupy {len(base):,}..{len(out)-1:,}")
    print(f"  EMBED ONLY FROM INDEX {len(base):,} ONWARD")

    if not write:
        print("\nDRY RUN — nothing written. Re-run with --write.")
        return
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET} ({len(out):,} rows)")
    print("\nNEXT, IN THIS ORDER:")
    print(f"  1. embed rows [{len(base)}:] and concatenate to the existing .npy")
    print("  2. rebuild FAISS from the concatenated array")
    print("  3. re-run the tag build over the new parquet")
    print("  4. verify rows == npy.shape[0] == index.ntotal, self-match probe")
    print("  5. upload embeddings + index FIRST, parquet SECOND, flip B_* LAST")
    print("  6. re-run eval_goodbooks to confirm no regression")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fetch", "merge"], default="fetch")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.stage == "fetch":
        stage_fetch()
    else:
        stage_merge(write=a.write)


if __name__ == "__main__":
    main()
