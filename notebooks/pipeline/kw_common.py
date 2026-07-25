"""
kw_common.py  -- shared keyword handling for the v2_kw embed recipes.

WHY THIS FILE EXISTS
--------------------
KW_STOP and the term-coercion helper were about to be duplicated across
embed_movies.py and embed_tv.py. Duplicated stoplists drift, and a drifted
stoplist means movies and TV get silently different embed recipes -- exactly
the class of bug that makes cross-modal comparisons meaningless. One copy.

STOPLIST PHILOSOPHY
-------------------
Strip a keyword only if it describes the artifact's PRODUCTION or its
RELATIONSHIP TO ANOTHER TITLE, never if it describes content, form, or
reception. Measured against the enriched corpus (see
data/tmdb_raw/kw_freq_*.csv):

  KEEP  "based on novel or book"  rank 2 movies (3.88%) / rank 4 TV (5.31%)
  KEEP  "based on manga"          rank 3 TV (5.53%)
        -> these are the books<->screen cross-modal bridge, free. The old
           stoplist deleted them, which was the single most expensive line
           in the file for a multi-modal recommender.

  STRIP "woman director"          rank 1 movies (5,694 films, 5.18%)
        -> the most frequent keyword in the movie corpus is a crew
           demographic. Left in, cosine clusters films by who directed them
           rather than what they are.

  STRIP "sequel"/"prequel"/"remake"/"reboot"/"spin off"
        -> describes a title's relationship to another title. Left in, every
           sequel is cosine-adjacent to every other sequel regardless of
           subject, which also fights the sequel suppression in ranking.

  KEEP  "silent film", "black and white", "short film", "3d animation",
        "miniseries", "sitcom", "telenovela"
        -> form/format IS content-predictive. Not metadata.
"""

import numpy as np

# Production/exhibition metadata + inter-title relationships. Lowercase.
KW_STOP = {
    # --- production & exhibition junk ---
    "aftercreditsstinger",
    "duringcreditsstinger",
    "woman director",          # rank 1 movies, 5.18% -- crew demographic

    # --- relationship-to-another-title (not content) ---
    "sequel",
    "prequel",
    "remake",
    "reboot",
    "spin off",
    "spinoff",

    # --- TMDB mood vocabulary (affect labels, not subject matter) ---
    # NOTE: this set is from the pilot frequency tables and is probably
    # INCOMPLETE. Sweep the full tables before the re-embed:
    #   Select-String -Path data\tmdb_raw\kw_freq_movies.csv `
    #     -Pattern "^(amused|gloomy|joyous|angry|sad|happy|tense|calm)," 
    "amused",
    "gloomy",
    "joyous",
}

# Deliberately NOT stopped (documented so nobody "fixes" this later):
#   based on novel or book / based on comic / based on manga /
#   based on light novel / based on young adult novel /
#   based on play or musical / based on true story / based on tv series
KW_KEEP_EXPLICIT = {
    "based on novel or book", "based on comic", "based on manga",
    "based on light novel", "based on young adult novel",
    "based on play or musical", "based on true story", "based on tv series",
}
assert not (KW_STOP & KW_KEEP_EXPLICIT), "stoplist collides with the keep-list"


def to_terms(val, max_n=None, stop=None):
    """Coerce a comma-string OR an array/list column value into clean terms.

    Handles both str columns (genres) and list/ndarray columns (keywords_tmdb,
    cast, directors_all) uniformly. Drops empties / 'nan' / stoplisted tokens.
    Stoplist comparison is case-insensitive; original casing is preserved.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        terms = [str(x).strip() for x in val]
    else:
        s = str(val)
        if s.strip().lower() in ("", "nan", "none", "[]"):
            return []
        terms = [t.strip() for t in s.split(",")]
    out = []
    for t in terms:
        if not t or t.lower() in ("nan", "none"):
            continue
        if stop and t.lower() in stop:
            continue
        out.append(t)
    return out[:max_n] if max_n else out
