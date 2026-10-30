"""
merge_new_books.py -- the November books merge: Google Books blurbs + award
books, appended to the catalogue the same way the 882-book author sweep was
(sweep_authors.py --stage merge).

    python notebooks\\pipeline\\merge_new_books.py            (dry run: report only)
    python notebooks\\pipeline\\merge_new_books.py --write

INPUTS
  data/processed/books_v2_hc3.parquet                     the live catalogue (base)
  data/raw/books/hardcover_v2/raw.jsonl                   Hardcover records
  data/raw/books/hardcover_v2/gbooks_audit_usable.jsonl   audit_gbooks.py: id -> cleaned Google blurb
  data/raw/books/awards/award_lookup.jsonl                lookup_award_books.py --stage report
  data/raw/books/awards/award_hc.jsonl                    the full Hardcover record per award book

OUTPUTS (--write)
  data/processed/books_v2_hc4.parquet       base rows UNCHANGED, new rows appended
  data/raw/books/hardcover_v2/merge_v4_raw.jsonl
        the Hardcover record of every appended row, so build_book_tags_hc.py
        and build_content_warnings.py can read their tags (both list it)
  data/processed/book_awards_v4.parquet     book_id -> awards (for a future
        book award badge; nothing reads it yet)

SOURCES
  gbooks_blurb  a Hardcover book Hardcover had no blurb for. Normal 10-rating
                floor applies: these were queued at 25+ Hardcover readers.
  award_books   won or shortlisted for a prize (build_award_list.py). Median 1
                Hardcover rating, so they need the floor EXEMPTION that
                targeted_authors has -- app.py CURATED_SOURCES and
                src/space_ranker.py must both include "award_books" before the
                new catalogue goes live, or they are ratable but never shown.

FILTERS (imported from sweep_authors.py, never retyped)
  English description, English title (translated editions are duplicates
  under a name nobody searches), no anthologies/magazines; then duplicates
  against the catalogue by id, by title+author key, and by the looser
  surname + title check (Babel = Babel, or The Necessity of Violence).

APPEND-ONLY, asserted: rows 0..N-1 keep their exact positions, so every stored
corpus_index in every user's browser stays valid.

NEXT (printed after --write):
  embed_books_append.py, build_book_tags_hc.py, build_content_warnings.py,
  upload embeddings + index FIRST, then parquet + tags, flip B_* LAST.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_authors as SW            # noqa: E402  filters + row parsing, as the sweep did
import audit_gbooks as AG             # noqa: E402  title/author rules
import lookup_award_books as LA       # noqa: E402  looser duplicate check

PROCESSED = Path("data/processed")
RAW_DIR = Path("data/raw/books/hardcover_v2")
AW_DIR = Path("data/raw/books/awards")
BASE = PROCESSED / "books_v2_hc3.parquet"
OUT = PROCESSED / "books_v2_hc4.parquet"
OUT_RAW = RAW_DIR / "merge_v4_raw.jsonl"
OUT_AWARDS = PROCESSED / "book_awards_v4.parquet"
HC_RAW = RAW_DIR / "raw.jsonl"
GB_USABLE = RAW_DIR / "gbooks_audit_usable.jsonl"
AW_LOOKUP = AW_DIR / "award_lookup.jsonl"
AW_HC = AW_DIR / "award_hc.jsonl"

MIN_DESC_CHARS = SW.MIN_DESC_CHARS


def load_jsonl(p):
    return AG.load_jsonl(p)


def book_key(title, author):
    ct = SW._core(title)
    ca = " ".join(sorted(SW._norm(author).split()))
    return (ct, ca) if ct and ca else None


def build_row(b, description, author, source):
    """A Hardcover record -> a catalogue row, the same fields
    sweep_authors.stage_merge builds."""
    rd = b.get("release_date") or ""
    year = int(rd[:4]) if len(rd) >= 4 and rd[:4].isdigit() else None
    ct = b.get("cached_tags")
    if isinstance(ct, str):
        try:
            ct = json.loads(ct)
        except json.JSONDecodeError:
            ct = {}
    bid = str(b["id"])
    return {
        "book_id": bid,
        "slug": b.get("slug") or "",
        "title": b.get("title") or "",
        "subtitle": b.get("subtitle") or "",
        "description": description.strip(),
        "primary_author": author or SW.primary_author(b.get("contributions")),
        "genres": [t["tag"] for t in SW._tag_list(ct, "Genre")],
        "moods": [t["tag"] for t in SW._tag_list(ct, "Mood")],
        "tags": [t["tag"] for t in SW._tag_list(ct, "Tag")],
        "publication_year": year,
        "release_date": rd,
        "num_pages": b.get("pages"),
        "average_rating": b.get("rating"),
        "ratings_count": b.get("ratings_count") or 0,
        "users_count": b.get("users_count") or 0,
        "image_url": SW.cover_url(b.get("cached_image")),
        "source": source,
        "work_id": bid,
    }


def google_only_row(r):
    """An award book Hardcover does not have: Google metadata only."""
    return {
        "book_id": r["book_id"], "slug": "", "title": r["title"],
        "subtitle": r.get("subtitle") or "", "description": r["description"].strip(),
        "primary_author": r["primary_author"], "genres": [], "moods": [], "tags": [],
        "publication_year": r.get("publication_year"), "release_date": "",
        "num_pages": r.get("num_pages"), "average_rating": None, "ratings_count": 0,
        "users_count": 0, "image_url": "", "source": "award_books", "work_id": r["book_id"],
    }


def candidates():
    """(row, raw_record_or_None, awards_info_or_None) for every proposed book;
    award books first, so an award book also in the Google set keeps its awards
    and its exemption."""
    out = []
    # --- award books
    hc = {r["key"]: r for r in load_jsonl(AW_HC)}
    by_bid = {}
    for k, r in hc.items():
        if r.get("status") == "found" and r.get("book"):
            by_bid[str(r["book"]["id"])] = r["book"]
    for r in load_jsonl(AW_LOOKUP):
        info = {"awards": r.get("awards", []), "groups": r.get("award_groups", []),
                "won": bool(r.get("award_won"))}
        raw = by_bid.get(str(r["book_id"]))
        if raw is not None:
            out.append((build_row(raw, r["description"], r["primary_author"], "award_books"),
                        raw, info))
        else:
            out.append((google_only_row(r), None, info))
    n_award = len(out)
    # --- Google-blurb books: their Hardcover record is in raw.jsonl
    usable = {str(r["id"]): r for r in load_jsonl(GB_USABLE)}
    found = 0
    with HC_RAW.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = usable.get(str(b.get("id")))
            if u is None:
                continue
            found += 1
            out.append((build_row(b, u["description"], None, "gbooks_blurb"), b, None))
    print(f"proposed: {n_award:,} award books, {found:,} Google-blurb books "
          f"({len(usable) - found:,} Google ids not found in raw.jsonl)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the new parquet (default: dry run)")
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()

    for p in (BASE, HC_RAW):
        if not p.exists():
            sys.exit(f"{p} not found -- run from the repo root")
    if not GB_USABLE.exists() and not AW_LOOKUP.exists():
        sys.exit("nothing to merge: run audit_gbooks.py and/or "
                 "lookup_award_books.py --stage report first")

    base = pd.read_parquet(BASE)
    base_ids = set(base["book_id"].astype(str))
    base_keys = {k for k in (book_key(t, a) for t, a in
                             zip(base["title"], base["primary_author"])) if k}
    by_sn = defaultdict(list)
    for t, a in zip(base["title"].astype(str), base["primary_author"].astype(str)):
        sn = AG.surname(a)
        if sn:
            by_sn[sn].append(t)
    print(f"base catalogue: {len(base):,} rows ({BASE.name})")

    keep, drops, seen_ids, seen_keys = [], Counter(), set(), set()
    drop_eg = defaultdict(list)
    for row, raw, info in candidates():
        bid, key = row["book_id"], book_key(row["title"], row["primary_author"])
        why = None
        if bid in base_ids:
            why = "already in catalogue (id)"
        elif key and key in base_keys:
            why = "already in catalogue (title + author)"
        elif bid in seen_ids or (key and key in seen_keys):
            why = "duplicate within this merge"
        elif len(row["description"]) < MIN_DESC_CHARS:
            why = "description too short"
        elif not SW.looks_english(row["description"]):
            why = "description not English"
        elif not SW._title_looks_english(row["title"]):
            why = "translated edition (title)"
        elif SW._COLLECTION_PAT.search(row["title"]) or SW._READER_PAT.search(row["title"]):
            why = "anthology / magazine"
        elif LA.in_catalogue(by_sn, [row["title"]], row["primary_author"]):
            why = "already in catalogue (looser title check)"
        if why:
            drops[why] += 1
            if len(drop_eg[why]) < 3:
                drop_eg[why].append(f"{row['title'][:40]} / {row['primary_author'][:20]}")
            continue
        seen_ids.add(bid)
        if key:
            seen_keys.add(key)
        keep.append((row, raw, info))

    print(f"\nkept {len(keep):,}, dropped {sum(drops.values()):,}")
    for why, n in drops.most_common():
        print(f"  {why:<44}{n:>7,}   e.g. {'; '.join(drop_eg[why])}")
    if not keep:
        print("nothing new to append.")
        return

    new = pd.DataFrame([r for r, _, _ in keep])
    src = Counter(new["source"])
    print(f"\nappending {len(new):,}: " + ", ".join(f"{s} {n:,}" for s, n in src.most_common()))
    for s in src:
        sub = new[new["source"] == s]
        print(f"  {s:<14} genres {(sub['genres'].str.len() > 0).mean():.0%} | "
              f"moods {(sub['moods'].str.len() > 0).mean():.0%} | "
              f"cover {(sub['image_url'] != '').mean():.0%} | "
              f"median ratings {sub['ratings_count'].median():.0f} | "
              f"10+ ratings {(sub['ratings_count'] >= 10).mean():.0%}")
    print("  award_books rows sit under the 10-rating floor: they need the")
    print('  CURATED_SOURCES exemption ("award_books") in app.py and space_ranker.py.')

    # --- schema reconciliation, as sweep_authors does it ---------------------
    built, base_cols = set(new.columns), set(base.columns)
    dropped = sorted(built - base_cols)
    if dropped:
        sys.exit(f"these fields would be DISCARDED by the column reindex: {dropped}\n"
                 f"base columns are: {sorted(base_cols)}")
    filled = sorted(base_cols - built)
    if filled:
        print(f"\n  WARNING: appended rows will be NaN in {filled}")
    new = new.reindex(columns=list(base.columns))

    print("\n  sample:")
    for _, r in new.sample(min(args.sample, len(new)), random_state=1).iterrows():
        print(f"    {str(r['publication_year'] or '----')[:4]}  {r['title'][:44]:46s}"
              f"{str(r['primary_author'])[:24]:26s}{r['source']}")

    out = pd.concat([base, new], ignore_index=True)
    assert len(out) == len(base) + len(new), "row count mismatch"
    assert out["book_id"].iloc[:len(base)].tolist() == base["book_id"].tolist(), \
        "existing rows moved -- refusing to write"
    assert out["book_id"].astype(str).is_unique, "duplicate book_id after concat"
    assert list(out.columns) == list(base.columns), "column set changed"
    print(f"\n  append-only verified: rows 0..{len(base) - 1:,} unchanged")
    print(f"  new rows occupy {len(base):,}..{len(out) - 1:,} -> EMBED FROM {len(base):,}")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write.")
        return

    out.to_parquet(OUT, index=False)
    with OUT_RAW.open("w", encoding="utf-8") as fh:
        for _, raw, _ in keep:
            if raw is not None:
                fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
    aw = [{"book_id": r["book_id"], "awards": info["awards"], "groups": info["groups"],
           "won": info["won"]} for r, _, info in keep if info]
    if aw:
        pd.DataFrame(aw).to_parquet(OUT_AWARDS, index=False)
    print(f"\nwrote {OUT}  ({len(out):,} rows)")
    print(f"wrote {OUT_RAW}  (Hardcover records of the appended rows)")
    if aw:
        print(f"wrote {OUT_AWARDS}  ({len(aw):,} award books)")
    print("\nNEXT, IN THIS ORDER (live catalogue untouched until step 6):")
    print("  1. python notebooks\\pipeline\\embed_books_append.py --dry-run, then without")
    print("  2. python notebooks\\pipeline\\build_book_tags_hc.py")
    print("  3. python notebooks\\pipeline\\build_content_warnings.py --stage build")
    print("  4. eval_goodbooks + probe_why_books on your profile, before vs after")
    print("  5. upload embeddings + index FIRST, then parquet + tags + content_warnings")
    print("  6. app.py: B_CORPUS/B_EMB/B_FAISS/B_TAGS -> hc4, push")


if __name__ == "__main__":
    main()
