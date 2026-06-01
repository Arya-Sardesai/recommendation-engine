"""
Merge Hardcover recent books into the Goodreads corpus.

Handles:
- schema normalization (Hardcover -> Goodreads-style columns)
- language filtering (drop obvious non-English)
- popularity-scale mismatch (Hardcover users_count << Goodreads ratings_count)
- dedup against existing corpus (fuzzy title+author)
- description length filter (>= 100 chars, same as Goodreads pipeline)

Output: data/processed/books_v1.parquet  (corpus ready for embedding the new rows)

Run inside the notebook environment, or as a script. Assumes the existing
deduped Goodreads parquet is present.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

GOODREADS_CORPUS = PROCESSED_DIR / "books_v0_deduped.parquet"
HARDCOVER_JSONL = RAW_DIR / "hardcover_recent.jsonl"
OUTPUT = PROCESSED_DIR / "books_v1.parquet"

# Hardcover's popularity scale is much smaller than Goodreads.
# Goodreads ratings_count for popular books = 100K-5M.
# Hardcover users_count for popular books = ~1K-15K.
# To make the recommender's min_ratings filter + popularity weighting behave
# sanely across both, we scale Hardcover users_count up by a factor so the
# distributions roughly overlap. Empirically Goodreads is ~30-50x larger.
HARDCOVER_POPULARITY_SCALE = 40


def looks_english(text):
    """Cheap heuristic: mostly ASCII letters. Filters obvious non-English."""
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return (ascii_letters / total_letters) > 0.85


def normalize_hardcover(rec):
    """Map a Hardcover book record to the Goodreads corpus schema."""
    desc = (rec.get("description") or "").strip()
    title = (rec.get("title") or "").strip()

    # author = first contribution
    contribs = rec.get("contributions") or []
    author = ""
    for c in contribs:
        if c.get("author") and c["author"].get("name"):
            author = c["author"]["name"]
            break

    # year from release_date "YYYY-MM-DD"
    year = None
    rd = rec.get("release_date")
    if rd and len(rd) >= 4 and rd[:4].isdigit():
        year = int(rd[:4])

    users = rec.get("users_count") or 0
    scaled_pop = int(users * HARDCOVER_POPULARITY_SCALE)

    return {
        "book_id": f"hc_{rec['id']}",          # prefix to avoid GR id collisions
        "work_id": "",                          # Hardcover has no work_id
        "title": title,
        "description": desc,
        "author_ids": [],                       # not used downstream; author name below
        "genres": [],                           # could enrich later
        "language_code": "eng",
        "num_pages": rec.get("pages"),
        "publication_year": year,
        "average_rating": rec.get("rating"),
        "ratings_count": scaled_pop,            # scaled to ~Goodreads range
        "text_reviews_count": 0,
        "is_ebook": False,
        "image_url": "",
        "url": "",
        "primary_author": author,
        "source": "hardcover",                  # tag the source
    }


def main():
    # load existing corpus
    print("Loading Goodreads corpus...")
    gr = pd.read_parquet(GOODREADS_CORPUS)
    if "source" not in gr.columns:
        gr["source"] = "goodreads"
    print(f"  {len(gr):,} Goodreads books")

    # load + normalize Hardcover
    print("Loading + normalizing Hardcover books...")
    hc_rows = []
    with open(HARDCOVER_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            norm = normalize_hardcover(rec)
            # filters
            if len(norm["description"]) < 100:
                continue
            if not norm["title"]:
                continue
            if not looks_english(norm["description"]):
                continue
            hc_rows.append(norm)
    hc = pd.DataFrame(hc_rows)
    print(f"  {len(hc):,} Hardcover books after filters")

    # dedup against Goodreads: build a normalized key for fast pre-filtering,
    # then fuzzy-check collisions
    def norm_key(title, author):
        t = re.sub(r"[^a-z0-9]", "", str(title).lower())[:20]
        a = re.sub(r"[^a-z0-9]", "", str(author).lower())[:10]
        return f"{t}|{a}"

    gr["_key"] = [norm_key(t, a) for t, a in zip(gr["title"], gr["primary_author"])]
    gr_keys = set(gr["_key"])

    keep = []
    dropped_dupe = 0
    for _, row in hc.iterrows():
        k = norm_key(row["title"], row["primary_author"])
        if k in gr_keys:
            dropped_dupe += 1
            continue
        keep.append(row)
    hc_dedup = pd.DataFrame(keep)
    print(f"  Dropped {dropped_dupe:,} exact-key duplicates already in Goodreads")
    print(f"  {len(hc_dedup):,} genuinely new Hardcover books")

    # also dedup Hardcover against itself (same book, multiple editions)
    hc_dedup["_key"] = [norm_key(t, a) for t, a in zip(hc_dedup["title"], hc_dedup["primary_author"])]
    before = len(hc_dedup)
    hc_dedup = hc_dedup.drop_duplicates(subset="_key", keep="first")
    print(f"  Dropped {before - len(hc_dedup):,} internal Hardcover duplicates")

    # combine
    gr = gr.drop(columns=["_key"])
    hc_dedup = hc_dedup.drop(columns=["_key"])
    combined = pd.concat([gr, hc_dedup], ignore_index=True)
    print(f"\nCombined corpus: {len(combined):,} books ({len(gr):,} GR + {len(hc_dedup):,} HC)")

    # normalize mixed-type columns before writing (Goodreads stored is_ebook as
    # strings, Hardcover as bools - coerce everything to string for consistency)
    combined["is_ebook"] = combined["is_ebook"].astype(str)

    combined.to_parquet(OUTPUT, index=False)
    print(f"Saved to {OUTPUT}")

    # report: how many new books per year
    print("\nNew Hardcover books by year:")
    print(hc_dedup["publication_year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()