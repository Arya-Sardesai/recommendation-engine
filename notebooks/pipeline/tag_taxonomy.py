"""
tag_taxonomy.py

Single source of truth for tags across books / movies / TV.

TIERS
-----
    UNIVERSAL   concept survives prose AND screen -- the cross-modal bridge set
    SCREEN      movies + TV, but not books (source-medium, anime demographics)
    BOOK_ONLY   prose / narration / structure, or reader-experience
    MOVIE_ONLY  cinematic craft
    TV_ONLY     serial format

WHAT CHANGED IN THIS REVISION
-----------------------------
1. NEW TIER: SCREEN. Movies and TV share vocabulary that books cannot have
   (a novel is not a "book-adaptation"). Previously there was nowhere to put
   these, so they didn't exist and the largest block of TMDB keyword mass had
   no slot to map into.

2. TV_ONLY IS NO LONGER EMPTY. miniseries (1,209 series), sitcom (829),
   telenovela (282). For television, format IS content -- a 6-episode
   miniseries and a 9-season sitcom are different products and taste splits
   on exactly that axis.

3. +17 TAGS, chosen by measured keyword mass in the corpus, not by intuition:
     source medium ~6,150 movie / ~3,324 TV instances -- the single largest
     unmapped block, and the literal books<->screen bridge
     anime vocabulary -- TV is 11.8% anime with no way to express it; genre
     "Animation" cannot separate Mushi-Shi from Naruto
     sub-genres TMDB's genre column cannot express: superhero, noir, sports,
     biographical, true-crime, supernatural

4. FIVE DEMOTIONS, UNIVERSAL -> BOOK_ONLY: accessible, comfort-read,
   forced-proximity, immersive, page-turner. Measured unreachable by BOTH
   genome and keywords -- they are reader-experience concepts a screen
   folksonomy has no reason to carry. "Universal" was overstated by five, and
   bridgeable() was returning True for tags no film can ever receive.

NOTE: the seven new UNIVERSAL tags enter ALL_BOOK_TAGS, so the next book
tagging run will offer them to the model. That is intended -- biographical,
true-crime, sports, superhero, noir, supernatural and slice-of-life are all
real book categories -- but it does mean book vocabulary changed.

Run this file directly for the reconciliation report; it must show 0 undeclared
shipped tags before any tagging run.
"""

# ---------------------------------------------------------------------------
# UNIVERSAL -- books / movies / TV. The bridge set.
# ---------------------------------------------------------------------------
UNIVERSAL_TAGS = {
    # --- speculative / genre themes ---
    "science-fiction", "dystopian", "space-opera", "cyberpunk",
    "climate-fiction", "time-travel", "time-loop", "post-apocalyptic",
    "fantasy", "high-fantasy", "urban-fantasy",
    "historical", "mystery", "thriller", "horror", "romance",
    "war", "crime", "coming-of-age", "satire",
    "heist", "dark-academia",
    # --- sub-genres the TMDB genre column cannot express (NEW) ---
   "superhero", "noir", "sports", "martial-arts", "biographical",
    "true-crime", "supernatural",
    # --- mood ---
    "dark", "hopeful", "melancholic", "humorous", "bleak", "feel-good",
    "tense", "whimsical", "eerie", "atmospheric", "witty",
    "heart-wrenching", "meditative", "thought-provoking", "emotionally-heavy",
    # --- experience / pace / structure ---
    "slow-burn", "fast-paced", "mind-bending", "tearjerker",
    "twist-ending", "non-linear", "slice-of-life",          # slice-of-life NEW
    # --- relationships / conflict / theme ---
    "found-family", "enemies-to-lovers", "forbidden-love",
    "survival", "revenge", "redemption", "man-vs-nature", "political",
    "political-intrigue", "class-struggle", "power", "legacy", "hubris",
    "identity", "grief", "family", "existential", "psychological",
    "queer-rep",
}

# ---------------------------------------------------------------------------
# SCREEN -- movies + TV, never books.  (NEW TIER)
# ---------------------------------------------------------------------------
SCREEN_TAGS = {
    # source medium: the cross-modal bridge. A film tagged book-adaptation
    # should preferentially surface books.
    "book-adaptation", "comic-adaptation", "manga-adaptation", "game-adaptation",
    # anime demographic categories
    "shounen", "seinen", "mecha",
}

# ---------------------------------------------------------------------------
# BOOK-ONLY -- prose / narration / structure, and reader experience.
# ---------------------------------------------------------------------------
BOOK_ONLY_TAGS = {
    "unreliable-narrator", "first-person", "second-person", "multiple-povs",
    "omniscient", "epistolary", "stream-of-consciousness", "framed-narrative",
    "literary-prose", "lyrical", "dense", "dialogue-heavy",
    "dual-timeline",
    "slow-burn-romance",
    "single-sitting",
    "contemporary-realism", "soft-magic", "hard-magic",
    # demoted from UNIVERSAL: unreachable by genome AND keywords on screen
    "accessible", "comfort-read", "forced-proximity", "immersive", "page-turner",
}

# ---------------------------------------------------------------------------
# MOVIE-ONLY -- cinematic craft; no book equivalent.
# ---------------------------------------------------------------------------
MOVIE_ONLY_TAGS = {
    "visually-stunning", "long-take", "practical-effects", "stylized-violence",
    "ensemble-cast", "black-and-white", "found-footage", "musical",
    "animation", "documentary-style", "short-film",
}

# ---------------------------------------------------------------------------
# TV-ONLY -- serial format. Format IS content for television.
# ---------------------------------------------------------------------------
TV_ONLY_TAGS = {
    "miniseries", "sitcom", "telenovela",
}

ALL_MOVIE_TAGS = UNIVERSAL_TAGS | SCREEN_TAGS | MOVIE_ONLY_TAGS
ALL_TV_TAGS = UNIVERSAL_TAGS | SCREEN_TAGS | TV_ONLY_TAGS
ALL_BOOK_TAGS = UNIVERSAL_TAGS | BOOK_ONLY_TAGS

# sanity: tiers must not overlap
for _a, _b in (("UNIVERSAL", "BOOK_ONLY"), ("UNIVERSAL", "SCREEN"),
               ("UNIVERSAL", "MOVIE_ONLY"), ("UNIVERSAL", "TV_ONLY"),
               ("SCREEN", "BOOK_ONLY"), ("MOVIE_ONLY", "TV_ONLY")):
    _x = {"UNIVERSAL": UNIVERSAL_TAGS, "SCREEN": SCREEN_TAGS,
          "BOOK_ONLY": BOOK_ONLY_TAGS, "MOVIE_ONLY": MOVIE_ONLY_TAGS,
          "TV_ONLY": TV_ONLY_TAGS}
    _o = _x[_a] & _x[_b]
    assert not _o, f"tier overlap {_a} & {_b}: {sorted(_o)}"


def bridgeable(tag: str) -> bool:
    """True if this tag can appear on BOTH a book and a screen title."""
    return tag in UNIVERSAL_TAGS


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    ROOT = Path(__file__).parent.parent.parent
    print(f"UNIVERSAL {len(UNIVERSAL_TAGS)} | SCREEN {len(SCREEN_TAGS)} | "
          f"BOOK_ONLY {len(BOOK_ONLY_TAGS)} | MOVIE_ONLY {len(MOVIE_ONLY_TAGS)} | "
          f"TV_ONLY {len(TV_ONLY_TAGS)}")
    print(f"ALL_BOOK {len(ALL_BOOK_TAGS)} | ALL_MOVIE {len(ALL_MOVIE_TAGS)} | "
          f"ALL_TV {len(ALL_TV_TAGS)}\n")
    bt = ROOT / "data" / "processed" / "book_tags.parquet"
    if not bt.exists():
        print(f"(book_tags.parquet not found at {bt} -- skipping)")
    else:
        book_tags = set(pd.read_parquet(bt)["tag"].unique())
        print(f"UNIVERSAL present in shipped book_tags: "
              f"{len(UNIVERSAL_TAGS & book_tags)}/{len(UNIVERSAL_TAGS)}")
        extra = book_tags - ALL_BOOK_TAGS
        print(f"shipped book_tags NOT yet declared: {len(extra)}")
        for t in sorted(extra):
            print(f"  - {t}")
