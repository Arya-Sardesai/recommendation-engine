"""
tag_taxonomy.py

Single source of truth for the tag vocabulary across ALL modalities
(books, movies, TV). Every per-modality tag builder imports from here so
that a tag name means the same thing everywhere -- this is what lets the
future cross-modal supervisor bridge taste ("time-travel books" ->
"time-travel movies") with a plain set intersection, no fuzzy matching.

THREE GROUPS:
  UNIVERSAL_TAGS  -- concepts that exist in every modality. The supervisor
                     ONLY bridges on these.
  BOOK_ONLY_TAGS  -- prose/narration concepts with no screen equivalent.
  MOVIE_ONLY_TAGS -- visual/cinematic concepts with no book equivalent.
  (TV_ONLY_TAGS   -- added when the TV layer is built.)

IMPORTANT: UNIVERSAL_TAGS names MUST match the tag strings already used in
data/processed/book_tags.parquet exactly. If they don't, the bridge silently
misses. Verify with the one-liner at the bottom of this file before relying
on cross-modal routing.
"""

# ---------------------------------------------------------------------------
# UNIVERSAL -- shared across books / movies / TV. Bridge tags.
# ---------------------------------------------------------------------------
UNIVERSAL_TAGS = {
    # --- speculative / sci-fi themes (confirmed present in book taxonomy) ---
    "science-fiction",
    "dystopian",
    "space-opera",
    "cyberpunk",
    "climate-fiction",
    "time-travel",
    "time-loop",
    # --- broader content/genre themes ---
    "post-apocalyptic",
    "fantasy",
    "historical",
    "mystery",
    "thriller",
    "horror",
    "romance",
    "war",
    "crime",
    "coming-of-age",
    "satire",
    # --- mood ---
    "dark",
    "hopeful",
    "melancholic",
    "humorous",
    "bleak",
    "feel-good",
    "tense",
    "whimsical",
    # --- experience / pace ---
    "slow-burn",
    "fast-paced",
    "mind-bending",
    "tearjerker",
    "atmospheric",
    "twist-ending",
    "nonlinear",
    # --- relationships / conflict ---
    "found-family",
    "enemies-to-lovers",
    "forbidden-love",
    "survival",
    "revenge",
    "redemption",
    "man-vs-nature",
    "political",
}

# ---------------------------------------------------------------------------
# BOOK-ONLY -- prose/narration; no clean screen equivalent.
# (Reconcile against book_tags.parquet; extend as needed.)
# ---------------------------------------------------------------------------
BOOK_ONLY_TAGS = {
    "unreliable-narrator",
    "lyrical-prose",
    "first-person",
    "second-person",
    "multiple-povs",
    "epistolary",
    "stream-of-consciousness",
    "omniscient",          # candidate for removal per handoff (only 2 books)
    "sparse-prose",
    "dense-prose",
}

# ---------------------------------------------------------------------------
# MOVIE-ONLY -- visual/cinematic; no book equivalent.
# ---------------------------------------------------------------------------
MOVIE_ONLY_TAGS = {
    "visually-stunning",
    "long-take",
    "practical-effects",
    "stylized-violence",
    "ensemble-cast",
    "black-and-white",
    "found-footage",
    "musical",
    "animation",
    "documentary-style",
}

# TV layer fills this in later.
TV_ONLY_TAGS: set[str] = set()


# ---------------------------------------------------------------------------
# Convenience views
# ---------------------------------------------------------------------------
ALL_MOVIE_TAGS = UNIVERSAL_TAGS | MOVIE_ONLY_TAGS
ALL_BOOK_TAGS = UNIVERSAL_TAGS | BOOK_ONLY_TAGS


def bridgeable(tag: str) -> bool:
    """True if this tag can be used for cross-modal recommendation."""
    return tag in UNIVERSAL_TAGS


if __name__ == "__main__":
    # Reconciliation check: which UNIVERSAL tags are actually present in the
    # shipped book tag file? Run from repo root once book_tags.parquet exists.
    from pathlib import Path
    import pandas as pd

    ROOT = Path(__file__).parent.parent.parent
    bt = ROOT / "data" / "processed" / "book_tags.parquet"
    if not bt.exists():
        print(f"(book_tags.parquet not found at {bt} -- skipping reconciliation)")
    else:
        book_tags = set(pd.read_parquet(bt)["tag"].unique())
        missing = UNIVERSAL_TAGS - book_tags
        present = UNIVERSAL_TAGS & book_tags
        print(f"UNIVERSAL tags present in book_tags.parquet: {len(present)}/{len(UNIVERSAL_TAGS)}")
        if missing:
            print("\nIn UNIVERSAL but NOT in book_tags (bridge will miss these):")
            for t in sorted(missing):
                print(f"  - {t}")
        extra = book_tags - ALL_BOOK_TAGS
        if extra:
            print("\nIn book_tags but NOT declared in taxonomy (add to UNIVERSAL or BOOK_ONLY):")
            for t in sorted(extra):
                print(f"  - {t}")