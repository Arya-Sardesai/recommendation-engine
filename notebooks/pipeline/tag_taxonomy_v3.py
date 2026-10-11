"""
tag_taxonomy_v3.py — the tag vocabulary for the LLM tagging run.

WHY THIS EXISTS AND WHAT CHANGED
The v1 taxonomy (85 tags, in book_tags.parquet) described how a book READS:
literary-prose, dense, unreliable-narrator, non-linear, meditative. That is the
axis a description embedding flattens, which is the whole reason an LLM pass is
worth paying for.

The Hardcover folksonomy that replaced it describes what SHELF a book sits on:
adventure, young adult, science fiction, classics, lgbtq. The embedding already
knows a book is science fiction from its description. It does not know the prose
is dense. That is the likeliest reason the books tag layer measures as noise
(0.014 vs 0.016 full, inside noise) while the same mechanism is a 31% effect on
films, where MovieLens genome tags carry exactly this register signal.

TWO ADDITIONS TO THE v1 SET

1. FORM (new). v1 had no fiction/non-fiction axis at all, which is why
   Dongri to Dubai -- non-fiction reporting on the Mumbai underworld -- returns
   the Daevabad fantasy trilogy. 60% of non-fiction anchors' recommendations are
   novels. A description embedding cannot tell a book ABOUT an underworld from a
   novel SET in one; they share every content word. A form label can.

2. MOOD gaps. v1 covered dark/bleak/melancholic/hopeful/tense well but missed
   registers Hardcover users reach for constantly: reflective, unsettling,
   earnest, wry, propulsive.

WHAT IS DELIBERATELY NOT HERE
No genre expansion. Genre is the one thing the embedding already gets right from
the description, and the measured failure of the Hardcover layer is evidence that
piling on more category labels adds nothing. Every tag below is something you
could only know by READING the book, not by skimming its subject.

SCORING
The model returns tag:score, 0.2-1.0, 5-8 tags per book. 1.0 is reserved for a
defining quality. Scores feed the pool-centred tag adjustment in recommend_books.
"""

# ---------------------------------------------------------------------------
# FORM -- what KIND of book this is. New in v3, and the point of the exercise
# for the non-fiction bleed.
# ---------------------------------------------------------------------------
FORM_TAGS = {
    # non-fiction forms
    "memoir",              # first-person life writing
    "biography",           # a life, written by someone else
    "reportage",           # journalism; the writer went and looked
    "history",             # scholarly or popular account of the past
    "essays",              # a collection of discrete arguments
    "criticism",           # writing about art, books, film
    "science-writing",     # popular science, nature, medicine
    "travel-writing",
    "self-help",           # instructional, advice-giving
    "reference",           # consulted rather than read through
    "true-crime",          # non-fiction crime
    "philosophy",
    "polemic",             # argues a position, takes a side
    # fiction forms
    "novel",
    "short-stories",
    "novella",
    "poetry",
    "graphic-novel",
    "play",
    # "epic" was here and was removed after the pilot: the model reads it as a
    # SCALE word ("epic fantasy"), so it landed beside `novel` on 59 of 300
    # books and gave them two forms. A Mahabharata retelling is a `novel`.
}

# ---------------------------------------------------------------------------
# MOOD -- the tonal register. v1 set plus the gaps.
# ---------------------------------------------------------------------------
MOOD_TAGS = {
    # v1
    "dark", "bleak", "melancholic", "hopeful", "feel-good", "tense",
    "whimsical", "witty", "eerie", "atmospheric", "humorous",
    "heart-wrenching", "meditative", "thought-provoking", "emotionally-heavy",
    # added
    "reflective",          # turns inward, considers rather than asserts
    "unsettling",          # wrong-footed rather than frightened
    "earnest",             # sincere, unironic
    "wry",                 # dry, amused, understated
    "propulsive",          # pulls you forward by force of plot
    "tender",
    "angry",               # the BOOK is angry, not the reader
    "cosy",
}

# ---------------------------------------------------------------------------
# PROSE & STRUCTURE -- v1's strongest block, kept whole. This is the axis
# cosine similarity over a plot summary cannot see at all.
# ---------------------------------------------------------------------------
PROSE_TAGS = {
    "literary-prose", "lyrical", "dense", "dialogue-heavy", "accessible",
    "first-person", "second-person", "multiple-povs", "omniscient",
    "unreliable-narrator", "epistolary", "stream-of-consciousness",
    "framed-narrative", "non-linear", "dual-timeline",
    "vignette",            # short discrete sections rather than chapters
    "digressive",          # wanders on purpose
}

# ---------------------------------------------------------------------------
# PACE & EXPERIENCE
# ---------------------------------------------------------------------------
PACE_TAGS = {
    "slow-burn", "fast-paced", "page-turner", "single-sitting", "immersive",
    "comfort-read", "mind-bending", "tearjerker", "twist-ending",
    "demanding",           # requires attention; rewards it
}

# ---------------------------------------------------------------------------
# THEME -- kept from v1. Narrower than genre: what the book is ABOUT at the
# level of human situation, not subject matter.
# ---------------------------------------------------------------------------
THEME_TAGS = {
    "coming-of-age", "identity", "grief", "family", "existential",
    "class-struggle", "power", "legacy", "redemption", "revenge",
    "survival", "man-vs-nature", "political", "political-intrigue",
    "found-family", "forbidden-love", "enemies-to-lovers", "forced-proximity",
    "queer-rep", "war", "psychological", "hubris",
    "displacement",        # migration, exile, not-belonging
    "faith",               # religion as lived experience
    "work",                # labour, craft, vocation
}

# ---------------------------------------------------------------------------
# GENRE -- the minimum needed for the model to place a book. Deliberately
# small: this is the axis the embedding already handles.
# ---------------------------------------------------------------------------
GENRE_TAGS = {
    "fantasy", "high-fantasy", "urban-fantasy", "science-fiction",
    "space-opera", "cyberpunk", "dystopian", "post-apocalyptic",
    "climate-fiction", "time-travel", "time-loop",
    "horror", "mystery", "thriller", "crime", "romance", "historical",
    "satire", "heist", "dark-academia", "contemporary-realism",
    "soft-magic", "hard-magic",
}

ALL_BOOK_TAGS_V3 = (
    FORM_TAGS | MOOD_TAGS | PROSE_TAGS | PACE_TAGS | THEME_TAGS | GENRE_TAGS
)

CATEGORIES = {
    "form": FORM_TAGS,
    "mood": MOOD_TAGS,
    "prose": PROSE_TAGS,
    "pace": PACE_TAGS,
    "theme": THEME_TAGS,
    "genre": GENRE_TAGS,
}

# Non-fiction forms, exported so the ranker can gate on shelf without
# re-deriving the list. A non-fiction anchor should preferentially retrieve
# non-fiction; this is the set that says which is which.
NONFICTION_FORMS = {
    "memoir", "biography", "reportage", "history", "essays", "criticism",
    "science-writing", "travel-writing", "self-help", "reference",
    "true-crime", "philosophy", "polemic",
}

assert NONFICTION_FORMS <= FORM_TAGS, "a non-fiction form is not in FORM_TAGS"

for _a, _b in [("form", "genre"), ("mood", "pace"), ("theme", "genre"),
               ("prose", "pace"), ("mood", "theme")]:
    _o = CATEGORIES[_a] & CATEGORIES[_b]
    assert not _o, f"tier overlap {_a} & {_b}: {sorted(_o)}"


if __name__ == "__main__":
    print(f"{len(ALL_BOOK_TAGS_V3)} tags total\n")
    for name, s in CATEGORIES.items():
        print(f"  {name:<8} {len(s):>3}")
    print(f"\n  non-fiction forms: {len(NONFICTION_FORMS)}")
    try:
        from tag_taxonomy import ALL_BOOK_TAGS as V1
        print(f"\n  vs v1 ({len(V1)} tags):")
        print(f"    added   {len(ALL_BOOK_TAGS_V3 - V1):>3}")
        print(f"    dropped {len(V1 - ALL_BOOK_TAGS_V3):>3}: "
              f"{', '.join(sorted(V1 - ALL_BOOK_TAGS_V3))}")
    except ImportError:
        pass
