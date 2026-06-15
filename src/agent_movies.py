"""
Movie recommendation agent — multi-step reasoning over the movie recommender.

Mirrors the books agent (src/agent.py) structure exactly: Claude (Haiku) is
given tools wrapping the movie recommender and runs an agentic loop. UI-agnostic:
takes the loaded corpus/embeddings/index + user ratings, exposes run_agent(...).

Mirrors books, with movie-schema adaptations:
  book_id -> id, primary_author -> director, ratings_count -> vote_count,
  publication_year -> release_year, work_id -> (none), book_tags -> movie_tags.

ADDS movie-only tools enabled by the richer IMDb data:
  - search_by_people : films by a director OR actor (auteur / star following)
  - search_by_studio : films from a studio (A24, Blumhouse, Ghibli, ...)
  and a `language` filter on semantic search (e.g. Korean / Hindi cinema).

Set key:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

MIN_VOTES = 400  # popularity floor, matches the app

# language display <-> code helpers (small, common subset)
LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "ja": "Japanese",
    "it": "Italian", "de": "German", "ru": "Russian", "zh": "Chinese",
    "pt": "Portuguese", "ko": "Korean", "hi": "Hindi", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "th": "Thai", "fa": "Persian",
    "sv": "Swedish", "da": "Danish", "nl": "Dutch", "pl": "Polish",
}
NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}


def _lang_code(name_or_code):
    """Accept 'Korean' or 'ko' -> 'ko'. Returns None if unrecognized."""
    if not name_or_code:
        return None
    s = str(name_or_code).strip().lower()
    if s in LANG_NAMES:
        return s
    return NAME_TO_CODE.get(s)


# ---------------------------------------------------------------------------
# Tool implementations — wrap the real movie recommender logic.
# ---------------------------------------------------------------------------

def _get_query_embedder(state):
    """Lazy-load and cache the embedding model on state for query encoding."""
    if "query_embedder" not in state:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        state["query_embedder"] = model
    return state["query_embedder"]


def _tool_search_user_ratings(state, query=None, top_k=8):
    """Return the user's rated films. With a query, rank them by SEMANTIC
    similarity (so 'my heist films' finds them by meaning)."""
    ratings = state["ratings"]
    if not ratings:
        return {"ratings": [], "note": "User has not rated any films yet."}

    if not query:
        return {"ratings": [{"title": r["matched_title"],
                             "director": r.get("matched_director", ""),
                             "rating": r["rating"]} for r in ratings],
                "count": len(ratings)}

    embeddings = state["embeddings"]
    model = _get_query_embedder(state)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    scored = []
    for r in ratings:
        fvec = embeddings[r["corpus_index"]]
        sim = float(np.dot(qvec, fvec))
        scored.append((sim, r))
    scored.sort(key=lambda x: -x[0])

    top = scored[:top_k]
    return {
        "query": query,
        "matched_films": [
            {"title": r["matched_title"], "director": r.get("matched_director", ""),
             "rating": r["rating"], "relevance": round(sim, 3)}
            for sim, r in top
        ],
        "note": "Ranked by semantic similarity to the query.",
    }


def _resolve_anchor(state, title):
    """Find a film by title: first in user ratings, then in catalog."""
    df = state["df"]
    ratings = state["ratings"]
    for r in ratings:
        if title.lower() in r["matched_title"].lower() or r["matched_title"].lower() in title.lower():
            return r
    mask = df["title"].str.contains(title, case=False, na=False, regex=False)
    if mask.any():
        row = df[mask].nlargest(1, "vote_count").iloc[0]
        pos = int(df.index[df["id"] == row["id"]][0])
        return {"matched_title": row["title"],
                "matched_director": row["director"] if "director" in df.columns and isinstance(row["director"], str) else "",
                "corpus_index": pos, "movie_id": row["id"], "rating": 5.0}
    return None


def _tool_get_recommendations(state, anchor_titles, n=10, min_year=None, max_year=None,
                              language=None):
    """Core recommender. Finds anchor films (rated or catalog), returns nearest
    unseen films, with optional year + language filters and director-exclusion."""
    df = state["df"]; embeddings = state["embeddings"]; index = state["index"]
    ratings = state["ratings"]

    anchors = [a for a in (_resolve_anchor(state, t) for t in anchor_titles) if a]
    if not anchors:
        return {"recommendations": [], "note": f"Could not find any of {anchor_titles} to anchor on."}

    lang = _lang_code(language)
    rated_ids = {str(r["movie_id"]) for r in ratings}
    rated_directors = {r.get("matched_director", "").lower() for r in ratings if r.get("matched_director")}
    has_director = "director" in df.columns

    anchor_vectors = np.array([embeddings[a["corpus_index"]] for a in anchors]).astype("float32")
    sims, idxs = index.search(anchor_vectors, 200)

    queues = []
    for i in range(len(anchors)):
        q = []
        for s, ix in zip(sims[i], idxs[i]):
            if ix < 0:        # FAISS pads with -1 when fewer neighbors than k exist
                continue
            f = df.iloc[ix]
            if str(f["id"]) in rated_ids:
                continue
            if int(f["vote_count"]) < MIN_VOTES:
                continue
            if (has_director and isinstance(f["director"], str) and f["director"]
                    and f["director"].lower() in rated_directors):
                continue
            yr = f["release_year"]
            if min_year is not None and (pd.isna(yr) or yr < min_year):
                continue
            if max_year is not None and (pd.isna(yr) or yr > max_year):
                continue
            if lang is not None and f["original_language"] != lang:
                continue
            q.append((int(ix), float(s)))
        queues.append(q)

    results, seen = [], set()
    while len(results) < n:
        added = 0
        for qi, q in enumerate(queues):
            taken = 0
            while q and taken < 2:
                ix, s = q.pop(0)
                if ix in seen:
                    continue
                seen.add(ix)
                f = df.iloc[ix]
                yr = int(f["release_year"]) if pd.notna(f["release_year"]) else None
                results.append({
                    "title": f["title"],
                    "director": f["director"] if has_director and isinstance(f["director"], str) else "",
                    "cast": list(f["cast"])[:4] if "cast" in df.columns and f["cast"] is not None else [],
                    "year": yr,
                    "language": LANG_NAMES.get(f["original_language"], f["original_language"]),
                    "because_of": anchors[qi]["matched_title"],
                })
                taken += 1; added += 1
                if len(results) >= n: break
            if len(results) >= n: break
        if added == 0: break
    return {"recommendations": results}


def _tool_search_catalog(state, query, limit=5):
    df = state["df"]
    mask = df["title"].str.contains(query, case=False, na=False, regex=False)
    hits = df[mask].nlargest(limit, "vote_count")
    has_director = "director" in df.columns
    return {"results": [{"title": r["title"],
                         "director": r["director"] if has_director and isinstance(r["director"], str) else "",
                         "year": int(r["release_year"]) if pd.notna(r["release_year"]) else None}
                        for _, r in hits.iterrows()]}


def _tool_search_catalog_semantic(state, query, limit=15, min_year=None, max_year=None,
                                  language=None):
    """Semantic search across the ENTIRE catalog. Use for genres/vibes outside
    the user's rated taste — combine requested genre with qualities you inferred."""
    df = state["df"]; embeddings = state["embeddings"]; index = state["index"]
    ratings = state["ratings"]

    model = _get_query_embedder(state)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(qvec, 300)

    lang = _lang_code(language)
    rated_ids = {str(r["movie_id"]) for r in ratings}
    rated_directors = {r.get("matched_director", "").lower() for r in ratings if r.get("matched_director")}
    has_director = "director" in df.columns

    results = []
    for s, ix in zip(sims[0], idxs[0]):
        if ix < 0:        # FAISS pads with -1 when fewer neighbors than k exist
            continue
        f = df.iloc[ix]
        if str(f["id"]) in rated_ids:
            continue
        if int(f["vote_count"]) < MIN_VOTES:
            continue
        if (has_director and isinstance(f["director"], str) and f["director"]
                and f["director"].lower() in rated_directors):
            continue
        yr = f["release_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue
        if lang is not None and f["original_language"] != lang:
            continue
        results.append({
            "title": f["title"],
            "director": f["director"] if has_director and isinstance(f["director"], str) else "",
            "year": int(yr) if pd.notna(yr) else None,
            "language": LANG_NAMES.get(f["original_language"], f["original_language"]),
            "genres": [g.strip() for g in f["genres"].split(",")] if isinstance(f["genres"], str) else [],
            "relevance": round(float(s), 3),
        })
        if len(results) >= limit:
            break

    return {"query": query, "results": results,
            "note": "Catalog-wide semantic results for genres/vibes outside the user's rated taste."}


def _tool_search_by_tags(state, positive_tags=None, negative_tags=None, limit=15,
                         min_year=None, max_year=None):
    """Search the tagged subset (~13.5K films) by quality-tags."""
    df = state["df"]; ratings = state["ratings"]

    if "tags_df" not in state:
        tags_path = Path("data/processed/movie_tags.parquet")
        if not tags_path.exists():
            return {"results": [], "note": "No tag data available."}
        tdf = pd.read_parquet(tags_path)
        tdf["movie_id"] = tdf["movie_id"].astype(str)
        state["tags_df"] = tdf.pivot_table(
            index="movie_id", columns="tag", values="score", fill_value=0.0)
    tags_df = state["tags_df"]

    positive_tags = positive_tags or []
    negative_tags = negative_tags or []
    available = set(tags_df.columns)
    pos = [t for t in positive_tags if t in available]
    neg = [t for t in negative_tags if t in available]
    invalid = [t for t in positive_tags + negative_tags if t not in available]

    if not pos and not neg:
        return {"results": [], "note": f"None of those tags exist. Available: {sorted(available)}"}

    score = tags_df[pos].sum(axis=1) if pos else pd.Series(0.0, index=tags_df.index)
    if neg:
        score = score - tags_df[neg].sum(axis=1)

    rated_ids = {str(r["movie_id"]) for r in ratings}
    rated_directors = {r.get("matched_director", "").lower() for r in ratings if r.get("matched_director")}
    has_director = "director" in df.columns
    ranked = score.sort_values(ascending=False)

    results = []
    for movie_id, s in ranked.items():
        if str(movie_id) in rated_ids:
            continue
        row = df[df["id"].astype(str) == str(movie_id)]
        if row.empty:
            continue
        f = row.iloc[0]
        if int(f["vote_count"]) < MIN_VOTES:
            continue
        if (has_director and isinstance(f["director"], str) and f["director"]
                and f["director"].lower() in rated_directors):
            continue
        yr = f["release_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue

        top_tags = tags_df.loc[movie_id].nlargest(5)
        top_tags = [(t, round(float(v), 2)) for t, v in top_tags.items() if v > 0]
        results.append({
            "title": f["title"],
            "director": f["director"] if has_director and isinstance(f["director"], str) else "",
            "year": int(yr) if pd.notna(yr) else None,
            "tag_score": round(float(s), 2),
            "top_tags": top_tags,
        })
        if len(results) >= limit:
            break

    return {"positive_tags_used": pos, "negative_tags_used": neg,
            "invalid_tags": invalid, "results": results,
            "note": f"Searched {len(tags_df):,} tagged films (~13.5K subset, not the full catalog)."}


def _tool_search_by_people(state, person, role="any", limit=15):
    """Find films by a DIRECTOR or top-billed ACTOR. role: 'director', 'actor', or 'any'.
    Excludes films the user already rated."""
    df = state["df"]; ratings = state["ratings"]
    rated_ids = {str(r["movie_id"]) for r in ratings}
    p = person.strip().lower()

    has_director = "director" in df.columns
    has_cast = "cast" in df.columns

    def director_match(row):
        return has_director and isinstance(row["director"], str) and p in row["director"].lower()

    def cast_match(row):
        if not has_cast or row["cast"] is None:
            return False
        return any(p in str(a).lower() for a in row["cast"])

    if role == "director":
        mask = df.apply(director_match, axis=1)
    elif role == "actor":
        mask = df.apply(cast_match, axis=1)
    else:
        mask = df.apply(lambda r: director_match(r) or cast_match(r), axis=1)

    hits = df[mask].nlargest(limit + len(rated_ids), "vote_count")
    results = []
    for _, f in hits.iterrows():
        if str(f["id"]) in rated_ids:
            continue
        as_director = director_match(f)
        results.append({
            "title": f["title"],
            "year": int(f["release_year"]) if pd.notna(f["release_year"]) else None,
            "director": f["director"] if has_director and isinstance(f["director"], str) else "",
            "cast": list(f["cast"])[:4] if has_cast and f["cast"] is not None else [],
            "role_of_person": "director" if as_director else "actor",
            "vote_count": int(f["vote_count"]),
        })
        if len(results) >= limit:
            break

    if not results:
        return {"person": person, "results": [],
                "note": f"No films found for '{person}'. Check spelling, or they may be outside the catalog."}
    return {"person": person, "role_searched": role, "results": results,
            "note": "Films involving this person, by popularity. Excludes the user's already-rated films."}


def _tool_search_by_studio(state, studio, limit=15, min_year=None, max_year=None):
    """Find films from a production studio (e.g. A24, Blumhouse, Studio Ghibli)."""
    df = state["df"]; ratings = state["ratings"]
    if "studio" not in df.columns:
        return {"results": [], "note": "Studio data not available."}
    rated_ids = {str(r["movie_id"]) for r in ratings}
    s = studio.strip().lower()
    mask = df["studio"].astype(str).str.lower().str.contains(s, na=False, regex=False)
    hits = df[mask].nlargest(limit + len(rated_ids), "vote_count")
    has_director = "director" in df.columns

    results = []
    for _, f in hits.iterrows():
        if str(f["id"]) in rated_ids:
            continue
        yr = f["release_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue
        results.append({
            "title": f["title"],
            "director": f["director"] if has_director and isinstance(f["director"], str) else "",
            "year": int(yr) if pd.notna(yr) else None,
            "studio": f["studio"],
        })
        if len(results) >= limit:
            break

    if not results:
        return {"studio": studio, "results": [],
                "note": f"No films found for studio '{studio}'."}
    return {"studio": studio, "results": results,
            "note": "Films from this studio by popularity. Excludes already-rated films."}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "search_user_ratings",
        "description": "Look at the films the user has rated. Provide a query describing a vibe/genre/mood (e.g. 'heist', 'slow character drama', 'crime') to rank their rated films by SEMANTIC similarity to it. Omit the query to get all ratings. Use to find which of the user's films match a description like 'my crime films'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional vibe/genre/mood, e.g. 'heist' or 'dark thriller'. Omit for all ratings."}
            },
        },
    },
    {
        "name": "get_recommendations",
        "description": "Get film recommendations from a set of anchor films (titles the user likes). Optionally restrict by year and/or original language. Returns unseen films similar to the anchors, each tagged with which anchor it came from. Excludes the user's rated films and films by directors they've already rated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "anchor_titles": {"type": "array", "items": {"type": "string"},
                                  "description": "Titles to base recommendations on (rated or any catalog film)."},
                "n": {"type": "integer", "description": "How many to return (default 10)."},
                "min_year": {"type": "integer", "description": "Only films released in or after this year."},
                "max_year": {"type": "integer", "description": "Only films released in or before this year."},
                "language": {"type": "string", "description": "Optional original-language filter, e.g. 'Korean', 'Hindi', 'French' (or code 'ko')."},
            },
            "required": ["anchor_titles"],
        },
    },
    {
        "name": "search_catalog",
        "description": "Check whether a film exists in the catalog and get its details. Use to verify a title before anchoring on it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title to search for."},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_catalog_semantic",
        "description": "Semantic search across the ENTIRE catalog by free-text (genre + qualities), with optional language filter. Use when the user asks for a genre/vibe OUTSIDE their rated taste. Pattern: infer qualities they enjoy from their ratings, then query combining the requested genre AND those qualities — e.g. 'cyberpunk noir with stylish action'. For world cinema, set language (e.g. 'Korean'). Excludes rated films and rated directors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text genre + qualities, e.g. 'heist thriller with twists and ensemble cast'."},
                "limit": {"type": "integer", "description": "How many results (default 15)."},
                "min_year": {"type": "integer"},
                "max_year": {"type": "integer"},
                "language": {"type": "string", "description": "Optional language, e.g. 'Korean', 'Hindi', 'French'."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_by_tags",
        "description": "Search films by quality-tags (NOT genres). Returns films from the tagged subset (~13.5K popular films) scoring high on positive tags and optionally low on negative tags. Use for qualities like 'slow-burn crime', 'mind-bending nonlinear', 'feel-good with found-family', 'stylish revenge without romance'. Available tags span pace (fast-paced, slow-burn), mood (dark, atmospheric, melancholic, whimsical, tense, bleak, hopeful, feel-good), experience (mind-bending, twist-ending, nonlinear, tearjerker), themes (coming-of-age, revenge, redemption, survival, found-family, forbidden-love, political), and genre-ish (science-fiction, dystopian, cyberpunk, time-travel, space-opera, climate-fiction, horror, thriller, crime, war, satire) plus movie-specific (visually-stunning, stylized-violence, ensemble-cast, animation, musical). Returns each film's top contributing tags. Only ~13.5K films are tagged — if results are weak, fall back to search_catalog_semantic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "positive_tags": {"type": "array", "items": {"type": "string"},
                                  "description": "Tags the film SHOULD score high on."},
                "negative_tags": {"type": "array", "items": {"type": "string"},
                                  "description": "Tags the film should score LOW on."},
                "limit": {"type": "integer"},
                "min_year": {"type": "integer"},
                "max_year": {"type": "integer"},
            },
        },
    },
    {
        "name": "search_by_people",
        "description": "Find films by a specific DIRECTOR or top-billed ACTOR. Use for auteur or star following: 'films by Bong Joon-ho', 'movies starring Morgan Freeman', 'more Christopher Nolan'. Set role to 'director', 'actor', or 'any' (default). Excludes the user's already-rated films, sorted by popularity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "Director or actor name, e.g. 'Christopher Nolan' or 'Morgan Freeman'."},
                "role": {"type": "string", "enum": ["director", "actor", "any"],
                         "description": "Restrict to their director credits, acting credits, or either (default 'any')."},
                "limit": {"type": "integer"},
            },
            "required": ["person"],
        },
    },
    {
        "name": "search_by_studio",
        "description": "Find films from a production studio or company, e.g. 'A24', 'Blumhouse', 'Studio Ghibli', 'Marvel Studios'. Use when the user follows a studio's house style. Excludes already-rated films, sorted by popularity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "studio": {"type": "string", "description": "Studio name, e.g. 'A24'."},
                "limit": {"type": "integer"},
                "min_year": {"type": "integer"},
                "max_year": {"type": "integer"},
            },
            "required": ["studio"],
        },
    },
]

TOOL_IMPLS = {
    "search_user_ratings": _tool_search_user_ratings,
    "get_recommendations": _tool_get_recommendations,
    "search_catalog": _tool_search_catalog,
    "search_catalog_semantic": _tool_search_catalog_semantic,
    "search_by_tags": _tool_search_by_tags,
    "search_by_people": _tool_search_by_people,
    "search_by_studio": _tool_search_by_studio,
}

SYSTEM_PROMPT = """You are a thoughtful film recommendation assistant. You help the user find movies to watch based on their taste and their requests.

You are strictly a film recommendation assistant. If the user asks for anything unrelated to movies (general questions, code, other topics), politely redirect: "I'm focused on film recommendations — what kind of movie are you in the mood for?"

You have seven tools:
- search_user_ratings: inspect what the user has rated, with optional semantic query
- get_recommendations: find films similar to anchor films they've rated (best when their taste matches the request); supports year + language filters
- search_catalog: look up a specific title
- search_catalog_semantic: search the WHOLE catalog by free-text (genre + qualities), with optional language filter — use for requests outside their rated taste
- search_by_tags: search the ~13.5K tagged films by quality-tags (pace, mood, themes, structure)
- search_by_people: find films by a director or actor (auteur / star following)
- search_by_studio: find films from a studio (A24, Blumhouse, Ghibli, ...)

Decide which path the request needs:

**Path A — request fits their taste** (e.g. "more like my heist films", "something recent like the crime films I enjoy"):
1. search_user_ratings WITH a query to find which rated films fit the description.
2. get_recommendations with those matched films as anchors.
3. Explain why each pick fits.

**Path B — request is OUTSIDE their rated taste** (e.g. they ask for horror/romance/sci-fi but their ratings are mostly crime):
1. search_user_ratings WITHOUT a query to see the full taste picture.
2. Identify 2-3 qualities the user reliably enjoys (twists, stylish violence, fast pace, ensemble casts, dark humor, character focus). Reference specific films they rated highly.
3. Call search_catalog_semantic with a query combining the requested genre + those qualities. e.g. "horror with stylish tension and a twist".
4. Present picks by EXPLICITLY BRIDGING: "You rated [film] highly for [quality] — here's a horror film with that same [quality]." Don't just dump genre results; show why these fit them.

**Path C — request is QUALITY-based** (e.g. "mind-bending nonlinear films", "slow-burn revenge", "feel-good found-family"):
1. Use search_by_tags with positive_tags for the qualities, optionally negative_tags to avoid.
2. Cross-reference with their taste — note which returned films share tags with what they rated highly.
3. Explain picks by referencing the specific tags.
4. Only ~13.5K films are tagged — if results are weak, fall back to search_catalog_semantic.

**Path D — request is PEOPLE or STUDIO based** (e.g. "movies by Bong Joon-ho", "films with Denzel Washington", "more A24"):
1. Use search_by_people (set role to director/actor/any) or search_by_studio.
2. Optionally cross-reference taste: note which picks align with qualities they enjoy.

**Language / world cinema** ("Korean thrillers", "Bollywood", "French crime"): add the language filter to get_recommendations or search_catalog_semantic.

You can COMBINE paths: for "Korean revenge thrillers like the crime films I like," search_by_tags(positive=[revenge, thriller]) or search_catalog_semantic('revenge thriller', language='Korean'), then bridge to their taste.

CRITICAL — GROUNDING RULE (read carefully):
Every single film you name in a recommendation MUST appear, by title, in a tool result you received in THIS conversation. This is an absolute rule:
- Do NOT add films from your own knowledge, even if they fit the request perfectly and you are confident they exist. If a great film comes to mind but it is not in your tool results, you may NOT name it — instead, run another tool search to try to surface it.
- Never recommend films the user has already rated (visible via search_user_ratings).
- If your tool results don't contain enough good matches, say so honestly and offer to search differently, rather than padding the list with films from memory.
- Prefer the EXACT titles as they appear in the tool results. Do not rename, retranslate, or "correct" them.
- When in doubt about whether a film is in the catalog, call search_catalog to verify before naming it.
The recommender's entire value is that it is grounded in this specific catalog and the user's actual ratings. A recommendation the tools didn't return is worse than no recommendation — it breaks trust and may point to a film that isn't in the catalog at all.

When a user asks for "something to watch" without specifying, treat it as Path B: infer qualities from their ratings, then search_catalog_semantic for fresh unseen films matching those qualities + any constraint they gave.

Other guidance:
- For mood shifts ("lighter", "darker", "faster") the tools can't filter directly, use judgment to pick from results and explain.
- Be concise and warm. Reference the user's taste in your explanations.
- If recommendations would be weak, say so honestly rather than forcing them.
- The catalog is films only (no books/TV), spanning world cinema up to 2025."""


def _titles_from_result(result):
    """Pull every film title out of a tool result dict (any tool's shape)."""
    titles = set()
    if not isinstance(result, dict):
        return titles
    for key in ("recommendations", "results", "matched_films", "ratings"):
        items = result.get(key)
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("title"):
                    titles.add(it["title"])
    return titles


def _check_faithfulness(reply, seen_titles):
    """Find film titles in the reply that were never returned by any tool.

    Precision matters: we only treat a phrase as a *film title* if it is
    shaped like one — i.e. immediately followed by a parenthetical containing
    a 4-digit year and/or a director, e.g.:
        "Smokin' Aces (Joe Carnahan, 2006)"
        "Ocean's Eleven (2001)"
        "**Snatch** (Guy Ritchie, 2000)"
    This avoids flagging prose fragments ("Why these fit:", "exactly") that a
    looser bold/numbered-list scan would wrongly catch.
    Returns a list of suspicious (likely-hallucinated) titles.
    """
    import re

    # Title candidate = text right before a parenthetical that contains a year.
    # Capture the run of words (optionally bolded) preceding "(... 19xx/20xx ...)".
    pattern = re.compile(
        r"(?:\*\*)?"                       # optional opening bold
        r"([A-Z0-9][^\n*()]{1,70}?)"       # the title text (starts capitalized)
        r"(?:\*\*)?"                       # optional closing bold
        r"\s*\((?:[^)]*?\b(?:19|20)\d{2}\b[^)]*)\)"  # ( ... yyyy ... )
    )
    cands = [m.group(1).strip().rstrip(":–-—").strip() for m in pattern.finditer(reply)]
    # strip any leading list-number prefix ("1. Title" -> "Title")
    cands = [re.sub(r"^\d+\.\s*", "", c).strip() for c in cands]

    def norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())

    seen_norm = {norm(t) for t in seen_titles}
    suspicious, suspicious_norms = [], set()
    for c in cands:
        cn = norm(c)
        if len(cn) < 3 or cn in suspicious_norms:
            continue
        if cn in seen_norm:
            continue
        if any(cn in s or s in cn for s in seen_norm if len(s) >= 4):
            continue
        suspicious.append(c)
        suspicious_norms.add(cn)
    return suspicious


def run_agent(user_message, state, conversation_history=None, max_turns=6, verbose=True):
    """Run the agentic loop. Returns (final_text, updated_history).

    Tracks every title returned by tools and, after the model's final answer,
    checks whether the answer names any film the tools never returned (a sign
    of the model freelancing from memory). When verbose, prints the returned
    titles and any suspicious (likely-hallucinated) ones.
    """
    client = Anthropic()
    history = conversation_history or []
    history = history + [{"role": "user", "content": user_message}]

    seen_titles = set()  # every film title any tool returned this turn

    for turn in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=history,
        )
        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b.text for b in resp.content if b.type == "text"]
        history.append({"role": "assistant", "content": resp.content})

        if not tool_calls:
            reply = "\n".join(text_blocks)
            # faithfulness check against everything the tools returned
            suspicious = _check_faithfulness(reply, seen_titles)
            if verbose and suspicious:
                print(f"  [!] WARNING: {len(suspicious)} title(s) in the reply were "
                      f"NOT in any tool result (possible hallucination):")
                for s in suspicious:
                    print(f"        - {s}")
            elif verbose:
                print(f"  [ok] all recommended titles were grounded in tool results "
                      f"({len(seen_titles)} films seen across tools)")
            return reply, history

        tool_results = []
        for tc in tool_calls:
            impl = TOOL_IMPLS.get(tc.name)
            try:
                result = impl(state, **tc.input)
            except Exception as e:
                result = {"error": str(e)}
            returned = _titles_from_result(result)
            seen_titles |= returned
            if verbose:
                print(f"  [agent calls {tc.name}({json.dumps(tc.input)})]")
                if returned:
                    preview = list(returned)[:8]
                    more = f" (+{len(returned) - 8} more)" if len(returned) > 8 else ""
                    print(f"        -> returned {len(returned)} films: {', '.join(preview)}{more}")
                else:
                    print(f"        -> (no film titles in result)")
            tool_results.append({
                "type": "tool_result", "tool_use_id": tc.id,
                "content": json.dumps(result),
            })
        history.append({"role": "user", "content": tool_results})

    return "(agent hit max turns without finishing)", history