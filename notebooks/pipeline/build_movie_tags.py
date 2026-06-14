"""
build_movie_tags.py

Builds movie_tags.parquet by mapping the MovieLens genome (~1,100 tags,
relevance-scored for ~13K films) onto the curated cross-modal taxonomy in
tag_taxonomy.py, joined to the TMDB corpus via TMDB id.

Join path:
    genome-scores.csv (movieId, tagId, relevance)
        + genome-tags.csv (tagId -> tag string)
        + links.csv (movieId -> tmdbId)
        -> match tmdbId against movies.parquet 'id'

Output mirrors book_tags.parquet: long/sparse format, one row per
(movie_id, tag, score), score >= THRESHOLD only.

Run from REPO ROOT:
    python notebooks/pipeline/build_movie_tags.py
"""

import sys
from pathlib import Path
import pandas as pd

# import the shared taxonomy (same dir)
sys.path.insert(0, str(Path(__file__).parent))
from tag_taxonomy import ALL_MOVIE_TAGS  # noqa: E402

ROOT = Path(__file__).parent.parent.parent
PROC = ROOT / "data" / "processed"
CORPUS = PROC / "movies.parquet"
OUT = PROC / "movie_tags.parquet"

# MovieLens files may sit under data/raw/movies/ml-25m/ (zip extracts a
# top-level ml-25m/ folder) or directly in data/raw/movies/. Auto-detect.
_MOVIES = ROOT / "data" / "raw" / "movies"
if (_MOVIES / "ml-25m" / "genome-scores.csv").exists():
    ML = _MOVIES / "ml-25m"
else:
    ML = _MOVIES

THRESHOLD = 0.30   # same sparse cutoff as books
MAX_TAGS_PER_FILM = 7  # match book per-item cap

# ---------------------------------------------------------------------------
# CURATED TAG -> source genome tag(s).
# Keys are taxonomy names (must be in ALL_MOVIE_TAGS). Values are lists of
# raw genome tag strings (lowercased) whose relevance feeds this curated tag.
# A film's curated score = MAX relevance across its matching genome tags.
#
# Only genome tags that actually exist in genome-tags.csv will match; unknowns
# are reported and skipped, so over-listing candidates is safe.
# ---------------------------------------------------------------------------
GENOME_MAP = {
    # --- speculative themes ---
    "science-fiction": ["sci-fi", "science fiction", "scifi", "sci fi", "futuristic"],
    "dystopian": ["dystopia", "dystopic future", "distopia", "post-apocalyptic"],
    "space-opera": ["space opera", "space", "space travel", "aliens", "astronauts"],
    "cyberpunk": ["cyberpunk", "artificial intelligence", "virtual reality", "robots", "androids"],
    "climate-fiction": ["global warming", "environmental", "environment", "ecology", "nature"],
    "time-travel": ["time travel"],
    "time-loop": ["time loop"],
    "post-apocalyptic": ["post-apocalyptic", "post apocalyptic", "apocalypse", "end of the world"],
    "fantasy": ["fantasy", "magic", "fairy tale", "mythology", "high fantasy", "fantasy world"],
    "historical": ["history", "historical", "period piece", "based on a true story", "true story"],
    "mystery": ["mystery", "murder mystery", "detective", "investigation"],
    "thriller": ["suspense", "thriller", "suspenseful", "psychological"],
    "horror": ["horror", "supernatural", "zombies", "vampires", "slasher", "scary"],
    "romance": ["romance", "love", "romantic", "love story"],
    "war": ["war", "world war ii", "military", "wartime", "war movie"],
    "crime": ["crime", "heist", "mafia", "gangster", "organized crime"],
    "coming-of-age": ["coming of age", "coming-of-age", "teenager", "adolescence"],
    "satire": ["satire", "satirical", "social commentary"],

    # --- mood ---
    "dark": ["dark", "dark comedy", "bleak", "disturbing"],
    "hopeful": ["inspirational", "inspiring", "heartwarming"],
    "melancholic": ["melancholy", "melancholic", "bittersweet", "sad", "depressing"],
    "humorous": ["funny", "comedy", "hilarious", "humor", "humorous"],
    "bleak": ["bleak", "depressing", "grim"],
    "feel-good": ["feel good movie", "feel-good", "heartwarming"],
    "tense": ["tense", "suspenseful", "intense"],
    "whimsical": ["quirky", "whimsical", "eccentricity", "absurd"],

    # --- experience / pace ---
    "slow-burn": ["slow", "slow paced"],
    "fast-paced": ["fast paced", "action packed", "action"],
    "mind-bending": ["mindfuck", "surreal", "psychedelic", "confusing"],
    "tearjerker": ["tear jerker", "heartbreaking", "emotional"],
    "atmospheric": ["atmospheric", "moody", "stylish"],
    "twist-ending": ["twist ending", "plot twist", "surprise ending", "twist"],
    "nonlinear": ["nonlinear", "non-linear", "multiple storylines"],

    # --- relationships / conflict ---
    "found-family": ["friendship", "family", "family bonds"],
    "enemies-to-lovers": [],  # no genome equivalent
    "forbidden-love": ["infidelity", "adultery", "interracial romance"],
    "survival": ["survival", "stranded", "wilderness"],
    "revenge": ["revenge", "vengeance"],
    "redemption": ["redemption"],
    "man-vs-nature": ["nature", "disaster", "natural disaster"],
    "political": ["politics", "political", "conspiracy"],

    # --- movie-only ---
    "visually-stunning": ["visually stunning", "visually appealing", "beautiful", "cinematography",
                          "great cinematography", "amazing cinematography", "beautifully filmed"],
    "long-take": [],  # no genome equivalent
    "practical-effects": ["special effects", "effects"],
    "stylized-violence": ["stylized", "violent", "violence", "bloody", "gore"],
    "ensemble-cast": ["ensemble cast"],
    "black-and-white": ["black and white"],
    "found-footage": ["fake documentary", "mockumentary"],
    "musical": ["musical", "music"],
    "animation": ["animation", "anime", "animated", "computer animation"],
    "documentary-style": ["documentary", "mockumentary"],
}


def main():
    for p in (CORPUS, ML / "genome-scores.csv", ML / "genome-tags.csv", ML / "links.csv"):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")

    # sanity: every curated key must be a declared taxonomy tag
    unknown_keys = set(GENOME_MAP) - ALL_MOVIE_TAGS
    if unknown_keys:
        sys.exit(f"ERROR: GENOME_MAP keys not in taxonomy: {sorted(unknown_keys)}")

    print("Loading corpus + MovieLens bridge files ...")
    corpus_ids = set(pd.read_parquet(CORPUS, columns=["id"])["id"].astype("int64"))
    links = pd.read_csv(ML / "links.csv")  # movieId, imdbId, tmdbId
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype("int64")
    # keep only MovieLens films that exist in our TMDB corpus
    links = links[links["tmdbId"].isin(corpus_ids)]
    print(f"  MovieLens films matching corpus: {len(links):,}")

    gtags = pd.read_csv(ML / "genome-tags.csv")  # tagId, tag
    gtags["tag"] = gtags["tag"].str.strip().str.lower()
    tag_str_to_id = dict(zip(gtags["tag"], gtags["tagId"]))

    # resolve curated -> set of genome tagIds, report unmatched source strings
    curated_to_tagids = {}
    missing_sources = []
    for curated, sources in GENOME_MAP.items():
        ids = []
        for s in sources:
            tid = tag_str_to_id.get(s.lower())
            if tid is None:
                missing_sources.append((curated, s))
            else:
                ids.append(tid)
        if ids:
            curated_to_tagids[curated] = set(ids)
    if missing_sources:
        print(f"\n  Note: {len(missing_sources)} source genome tags not found "
              f"(safe to ignore, listed for tuning):")
        for c, s in missing_sources[:25]:
            print(f"    {c:<20} <- '{s}'")
        if len(missing_sources) > 25:
            print(f"    ... and {len(missing_sources) - 25} more")

    needed_tagids = set().union(*curated_to_tagids.values())
    print(f"\n  Curated tags with >=1 genome source: {len(curated_to_tagids)}/{len(GENOME_MAP)}")

    # stream genome-scores (435MB) in chunks, keep only needed films+tags
    valid_movieids = set(links["movieId"])
    print("Streaming genome-scores.csv ...")
    keep = []
    reader = pd.read_csv(ML / "genome-scores.csv", chunksize=2_000_000)
    for i, chunk in enumerate(reader):
        c = chunk[chunk["movieId"].isin(valid_movieids) & chunk["tagId"].isin(needed_tagids)]
        keep.append(c)
        print(f"  chunk {i}: kept {len(c):,}")
    scores = pd.concat(keep, ignore_index=True)
    print(f"  total relevant score rows: {len(scores):,}")

    # tagId -> curated (a genome tag can feed multiple curated tags)
    tagid_to_curated = {}
    for curated, ids in curated_to_tagids.items():
        for tid in ids:
            tagid_to_curated.setdefault(tid, []).append(curated)

    # explode score rows to (movieId, curated, relevance)
    rows = []
    for movieid, tagid, rel in zip(scores["movieId"], scores["tagId"], scores["relevance"]):
        for curated in tagid_to_curated.get(tagid, ()):
            rows.append((movieid, curated, rel))
    long = pd.DataFrame(rows, columns=["movieId", "tag", "relevance"])

    # curated score = MAX relevance across contributing genome tags
    long = long.groupby(["movieId", "tag"], as_index=False)["relevance"].max()
    long = long.rename(columns={"relevance": "score"})

    # threshold
    long = long[long["score"] >= THRESHOLD]

    # cap tags per film: keep top-N by score
    long = (long.sort_values(["movieId", "score"], ascending=[True, False])
                .groupby("movieId").head(MAX_TAGS_PER_FILM))

    # map movieId -> TMDB id (our canonical movie_id)
    ml_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))
    long["movie_id"] = long["movieId"].map(ml_to_tmdb).astype("int64")
    out = long[["movie_id", "tag", "score"]].sort_values(["movie_id", "score"], ascending=[True, False])
    out["score"] = out["score"].round(4)

    out.to_parquet(OUT, index=False)
    n_films = out["movie_id"].nunique()
    print(f"\nWrote {len(out):,} tag rows across {n_films:,} films -> {OUT}")

    # sanity
    print("\n--- sanity ---")
    print(f"Avg tags per tagged film: {len(out) / n_films:.1f}")
    print(f"Median score: {out['score'].median():.3f}")
    print("\nMost common tags:")
    print(out["tag"].value_counts().head(15).to_string())
    print("\nUniversal-tag coverage (cross-modal bridgeable films):")
    from tag_taxonomy import UNIVERSAL_TAGS
    uni = out[out["tag"].isin(UNIVERSAL_TAGS)]
    print(f"  films with >=1 universal tag: {uni['movie_id'].nunique():,}")


if __name__ == "__main__":
    main()