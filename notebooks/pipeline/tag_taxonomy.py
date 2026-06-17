"""
tag_taxonomy.py  (RECONCILED DRAFT)

Single source of truth for tags across books / movies / TV. Merged from the
prior declared taxonomy + the shipped book_tags.parquet vocabulary, so the new
20K book tagging and all future modalities speak one language.

RECONCILIATION RULES APPLIED:
  - Routing: a tag is UNIVERSAL if the concept survives both prose and screen
    (theme / mood / pace / relationship / conflict); BOOK_ONLY if it is a
    prose/narration/structure concept with no screen equivalent.
  - Near-duplicates folded to ONE canonical form, keeping the spelling already
    well-used in book_tags.parquet (so the existing 8,800 tags stay valid):
        satire        <- satirical (kept 'satirical', 502 books)  [SEE NOTE]
        non-linear    <- nonlinear
        lyrical       <- lyrical-prose
        dense         <- dense-prose / sparse-prose
        political-intrigue kept distinct from 'political' (plot vs theme)
  - >>> JUDGMENT CALLS are marked  # ?? -- these are yours to confirm/move. <<<

After you settle the # ?? lines, run the reconciliation block at the bottom; it
must report 0 undeclared tags before the full tagging run.
"""

# ---------------------------------------------------------------------------
# UNIVERSAL -- shared across books / movies / TV. The bridge set.
# ---------------------------------------------------------------------------
UNIVERSAL_TAGS = {
    # --- speculative / genre themes ---
    "science-fiction", "dystopian", "space-opera", "cyberpunk",
    "climate-fiction", "time-travel", "time-loop", "post-apocalyptic",
    "fantasy", "high-fantasy", "urban-fantasy",          # ?? fantasy subtypes: universal or book-only?
    "historical", "mystery", "thriller", "horror", "romance",
    "war", "crime", "coming-of-age", "satire",           # ?? 'satire' vs shipped 'satirical' -- pick one canonical
    "heist", "dark-academia",                            # ?? present in book_tags; bridge to screen? (likely yes)
    # --- mood ---
    "dark", "hopeful", "melancholic", "humorous", "bleak", "feel-good",
    "tense", "whimsical", "eerie", "atmospheric", "witty", "comfort-read",
    "heart-wrenching", "meditative", "thought-provoking",
    # --- experience / pace ---
    "slow-burn", "fast-paced", "mind-bending", "tearjerker",
    "twist-ending", "page-turner", "immersive", "accessible",  # ?? 'accessible' reads book-ish to me
    # --- relationships / conflict / theme ---
    "found-family", "enemies-to-lovers", "forbidden-love", "forced-proximity",
    "survival", "revenge", "redemption", "man-vs-nature", "political",
    "political-intrigue", "class-struggle", "power", "legacy", "hubris",
    "identity", "grief", "family", "existential", "psychological",
    "queer-rep",
}

# ---------------------------------------------------------------------------
# BOOK-ONLY -- prose / narration / structure; no clean screen equivalent.
# ---------------------------------------------------------------------------
BOOK_ONLY_TAGS = {
    "unreliable-narrator", "first-person", "second-person", "multiple-povs",
    "omniscient", "epistolary", "stream-of-consciousness", "framed-narrative",
    "literary-prose", "lyrical", "dense", "dialogue-heavy",
    "dual-timeline", "non-linear",                       # ?? 'non-linear' structure vs UNIVERSAL 'twist-ending' -- distinct, keep both
    "slow-burn-romance",                                 # ?? romance pacing; could collapse into 'romance'+'slow-burn'
    "single-sitting",
    "contemporary-realism", "soft-magic", "hard-magic",  # ?? magic-system tags: book-only or move to UNIVERSAL fantasy?
}

# ---------------------------------------------------------------------------
# MOVIE-ONLY -- visual / cinematic; no book equivalent. (unchanged)
# ---------------------------------------------------------------------------
MOVIE_ONLY_TAGS = {
    "visually-stunning", "long-take", "practical-effects", "stylized-violence",
    "ensemble-cast", "black-and-white", "found-footage", "musical",
    "animation", "documentary-style",
}

TV_ONLY_TAGS: set[str] = set()

ALL_MOVIE_TAGS = UNIVERSAL_TAGS | MOVIE_ONLY_TAGS
ALL_BOOK_TAGS = UNIVERSAL_TAGS | BOOK_ONLY_TAGS


def bridgeable(tag: str) -> bool:
    return tag in UNIVERSAL_TAGS


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    ROOT = Path(__file__).parent.parent.parent
    bt = ROOT / "data" / "processed" / "book_tags.parquet"
    if not bt.exists():
        print(f"(book_tags.parquet not found at {bt} -- skipping)")
    else:
        book_tags = set(pd.read_parquet(bt)["tag"].unique())
        present = UNIVERSAL_TAGS & book_tags
        print(f"UNIVERSAL present in shipped book_tags: {len(present)}/{len(UNIVERSAL_TAGS)}")
        extra = book_tags - ALL_BOOK_TAGS
        print(f"shipped book_tags NOT yet declared: {len(extra)}")
        for t in sorted(extra):
            print(f"  - {t}")
        print(f"\nMerged totals -> UNIVERSAL {len(UNIVERSAL_TAGS)} | "
              f"BOOK_ONLY {len(BOOK_ONLY_TAGS)} | ALL_BOOK {len(ALL_BOOK_TAGS)}")