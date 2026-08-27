"""
build_book_tags_hc.py — tag layer for the v2 Hardcover book corpus.

Emits a long-format (book_id, tag, score) parquet, the same shape app.py's
load_books_tags() pivots into a re-rank matrix. This is the RANKING layer;
embed_books.py handles the same fields as RETRIEVAL text. Two different jobs:
which candidates come back vs how they are ordered.

WHY THIS REPLACES THE OLD BOOK TAGS
-----------------------------------
book_tags_knn.parquet was 99.3% kNN-PROPAGATED -- inferred from the same
BGE-M3 embeddings it was meant to correct. That circularity is why it could not
fix `Love in the Time of Cholera` -> generic romance: the tags agreed with the
embedding because they came from it. Hardcover tags are INDEPENDENT EVIDENCE,
applied by readers, with a `count` of how many applied each one.

SOURCES (from cached_tags, harvested by fetch_hardcover_v2.py)
  Genre  90.5% coverage -- category signal
  Mood   35.7% coverage -- TONE, the thing plot descriptions never carry.
                           Sparse, but it is signal the old corpus lacked
                           entirely. Sparse coverage in the tail is expected;
                           TMDB keywords behave the same way (~33% of movies
                           have none).

SCORING
  raw   = tag count for that book
  score = raw / max(raw for that book)      -> in (0, 1], per-book relative
Counts scale with popularity: a book with 10,000 readers has larger counts than
a good match with 50, and raw counts would make the tag matrix a popularity
proxy. Per-book normalisation makes each score mean "share of attention this
book's taggers gave this tag", which is comparable across books.

DROPPED TAGS
Genres appearing on more than MAX_DOC_FRAC of the corpus carry no discriminative
signal -- `Fiction` is on 55% of books. They stay in the embed text (harmless
there) but are dropped here, where they would add a constant to every vector.

Run from REPO ROOT:
    python notebooks\\pipeline\\build_book_tags_hc.py
    python notebooks\\pipeline\\build_book_tags_hc.py --dry-run
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/books/hardcover_v2")
FILTERED_PATH = RAW_DIR / "filtered.jsonl"

# Tags come from the HARVEST jsonl, not the parquet -- the parquet has no
# cached_tags column. Books appended by the targeted and rescue paths are
# absent from filtered.jsonl, so reading only that file silently left them
# untagged: the build reported "read 74,054 filtered rows" against a
# 74,074-row corpus and printed coverage as if nothing were missing.
#
# Those books sit below app.py's serving floor so they are never RECOMMENDED,
# but they ARE ratable as anchors, and an anchor's tags feed the user's tag
# profile. An untagged anchor contributes nothing there.
TAG_SOURCES = [
    FILTERED_PATH,
    RAW_DIR / "targeted_raw.jsonl",
    RAW_DIR / "rescued_raw.jsonl",
]

PROCESSED = Path("data/processed")
CORPUS = PROCESSED / "books_v2_hc2.parquet"
OUT = PROCESSED / "book_tags_v2_hc2.parquet"

USE_CATEGORIES = ("Genre", "Mood")   # not "Tag" (noisy) or "Content Warning"
MIN_COUNT = 1            # a tag one person applied is not evidence
MAX_TAGS_PER_BOOK = 15
MAX_DOC_FRAC = 0.35      # drop tags on >35% of books (e.g. Fiction at 55%)
MIN_DOC_COUNT = 50       # drop tags appearing on fewer than this many books

# Always dropped regardless of frequency: shelf-management and format labels
# that describe the reader's relationship to the book, not the book.
STOP_TAGS = {
    "fiction", "nonfiction", "non-fiction", "general", "books", "ebook",
    "audiobook", "to-read", "currently-reading", "owned", "favorites",
    "book club", "adult", "adult fiction", "literature", "novels",
}


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
            count = t.get("count") or t.get("tagged_count") or 0
            if name:
                out.append((str(name), int(count) if count else 0))
        elif isinstance(t, str) and t:
            out.append((t, 0))
    return out


def norm_tag(t: str) -> str:
    """Fold case and whitespace. Hardcover mixes 'dark' and 'Adventurous'."""
    return " ".join(str(t).strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    if not any(p.exists() for p in TAG_SOURCES):
        raise SystemExit(f"no tag source found -- run fetch_hardcover_v2.py --stage filter")
    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} not found -- run fetch_hardcover_v2.py --stage merge --write")

    corpus_ids = set(pd.read_parquet(CORPUS, columns=["book_id"])["book_id"].astype(str))
    print(f"corpus: {len(corpus_ids):,} books")

    # pass 1: collect raw (book, tag, count), and per-tag document frequency
    per_book = defaultdict(dict)
    doc_freq = Counter()
    n_rows = 0
    for src in TAG_SOURCES:
        if not src.exists():
            print(f"  ({src.name} absent, skipping)")
            continue
        before = n_rows
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                b = json.loads(line)
                bid = str(b.get("id"))
                if bid not in corpus_ids:
                    continue
                n_rows += 1
                ct = b.get("cached_tags")
                seen_here = set()
                for cat in USE_CATEGORIES:
                    for name, count in _tag_list(ct, cat):
                        t = norm_tag(name)
                        if not t or t in STOP_TAGS or count < MIN_COUNT:
                            continue
                        # keep the larger count if a tag appears in two categories
                        if count > per_book[bid].get(t, 0):
                            per_book[bid][t] = count
                        seen_here.add(t)
                for t in seen_here:
                    doc_freq[t] += 1
        print(f"  {src.name:24s} {n_rows - before:>7,} rows in corpus")

    print(f"read {n_rows:,} harvest rows; {len(per_book):,} have >=1 usable tag "
          f"({len(per_book) / max(len(corpus_ids), 1):.1%} coverage)")

    # This mismatch is how the 20 appended books went untagged in silence.
    # A tag build that reads fewer rows than the corpus has is missing a source.
    missing = len(corpus_ids) - n_rows
    if missing > 0:
        print(f"  WARNING: {missing:,} corpus books have NO tag source at all.")
        print(f"  Every book in the parquet should appear in one of TAG_SOURCES;")
        print(f"  if a new harvest path was added, add its jsonl to that list.")
    print(f"vocabulary before frequency pruning: {len(doc_freq):,} tags")

    # pass 2: prune the vocabulary
    n_books = max(len(per_book), 1)
    too_common = {t for t, c in doc_freq.items() if c / n_books > MAX_DOC_FRAC}
    too_rare = {t for t, c in doc_freq.items() if c < MIN_DOC_COUNT}
    keep_vocab = {t for t in doc_freq if t not in too_common and t not in too_rare}

    print(f"  dropped {len(too_common):,} as too common (>{MAX_DOC_FRAC:.0%} of books):")
    for t in sorted(too_common, key=lambda x: -doc_freq[x])[:10]:
        print(f"      {t:<28} {doc_freq[t]:,}  ({doc_freq[t] / n_books:.1%})")
    print(f"  dropped {len(too_rare):,} as too rare (<{MIN_DOC_COUNT} books)")
    print(f"  vocabulary kept: {len(keep_vocab):,} tags")

    # pass 3: per-book normalise and emit
    rows = []
    for bid, tagmap in per_book.items():
        kept = {t: c for t, c in tagmap.items() if t in keep_vocab}
        if not kept:
            continue
        top = sorted(kept.items(), key=lambda kv: -kv[1])[:MAX_TAGS_PER_BOOK]
        mx = max(c for _, c in top)
        if mx <= 0:
            continue
        for t, c in top:
            rows.append({"book_id": bid, "tag": t, "score": round(c / mx, 4)})

    tdf = pd.DataFrame(rows)
    tagged_books = tdf["book_id"].nunique() if len(tdf) else 0
    print()
    print(f"tag rows        {len(tdf):,}")
    print(f"tagged books    {tagged_books:,} ({tagged_books / max(len(corpus_ids), 1):.1%} of corpus)")
    if len(tdf):
        print(f"tags per book   median {tdf.groupby('book_id').size().median():.0f}")
        print()
        print("top tags by document frequency:")
        for t, c in Counter(tdf["tag"]).most_common(25):
            print(f"    {t:<28} {c:,}")

        # a spot check that the scores mean something
        probe = tdf[tdf["book_id"] == str(pd.read_parquet(CORPUS, columns=["book_id"]).iloc[0]["book_id"])]
        if len(probe):
            print("\nspot check, first corpus row:")
            for _, r in probe.sort_values("score", ascending=False).head(8).iterrows():
                print(f"    {r['tag']:<28} {r['score']:.3f}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    tdf.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("Upload with the other v2 artifacts; flip B_TAGS in app.py last.")


if __name__ == "__main__":
    main()
