"""
Film recommendation agent -- Claude (Haiku) with tools over the film catalogue.

Ranking is NOT implemented here. get_recommendations calls the Space's own
recommend_movies through space_ranker, with the arguments the Movies tab
passes, so with no anchors the agent's list IS the website's "For you" list
(kids filter, runtime floor, quality floor, tag layer and all). The other
tools are searches that apply the same floors.

Movie-only tools: search_by_people (director or actor) and search_by_studio,
plus a language filter on recommendations and semantic search.

Set key:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import numpy as np
import pandas as pd

import space_ranker as sr
from agent_common import embed_query, run_agent_loop

KIND = "movie"


def _b(state):
    return state["bundle"]


def _lang_code(b, name_or_code):
    """'Korean' or 'ko' -> 'ko', using the Space's own language table."""
    if not name_or_code:
        return None
    names = b["ns"].get("LANG_NAMES", {})
    s = str(name_or_code).strip().lower()
    if s in names:
        return s
    return {v.lower(): k for k, v in names.items()}.get(s)


def _row(b, pos):
    r = b["df"].iloc[int(pos)]
    out = {"title": str(r["title"]), "director": sr.text(r.get("director")),
           "year": sr.year_of(r.get("release_year")),
           "language": b["ns"]["lang_label"](r.get("original_language")),
           "genres": sr.split_genres(r.get("genres"))[:3]}
    blurb = sr.short(r.get("overview"), 160)
    if blurb:
        out["blurb"] = blurb
    return out


def _rec_row(rec):
    out = {"title": str(rec["title"]), "director": sr.text(rec.get("director")),
           "cast": [str(c) for c in sr.as_list(rec.get("cast"))[:3]],
           "year": sr.year_of(rec.get("year")), "language": sr.text(rec.get("language")),
           "genres": sr.split_genres(rec.get("genres"))[:3],
           "vote_average": rec.get("vote_average"),
           "because_of": rec.get("because_of")}
    also = rec.get("also_because")
    if isinstance(also, str) and also:
        out["also_because"] = also
    tags = [t for t in sr.as_list(rec.get("match_tags")) if t]
    if tags:
        out["matched_tags"] = tags
    blurb = sr.short(rec.get("overview"))
    if blurb:
        out["blurb"] = blurb
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def _tool_search_user_ratings(state, query=None, top_k=8):
    ratings = state["ratings"]
    if not ratings:
        return {"ratings": [], "note": "The user has not rated any films yet."}
    if not query:
        return {"ratings": [{"title": r["matched_title"], "director": r.get("matched_director", ""),
                             "rating": r["rating"]}
                            for r in sorted(ratings, key=lambda r: -r["rating"])],
                "count": len(ratings)}
    b = _b(state)
    q = embed_query(state, query)[0]
    scored = sorted(((float(np.dot(q, b["emb"][r["corpus_index"]])), r) for r in ratings),
                    key=lambda x: -x[0])
    return {"query": query,
            "ratings": [{"title": r["matched_title"], "director": r.get("matched_director", ""),
                         "rating": r["rating"], "relevance": round(s, 3)}
                        for s, r in scored[:int(top_k or 8)]],
            "note": "The user's rated films, ranked by similarity in meaning to the query."}


def _tool_get_recommendations(state, anchor_titles=None, n=10, min_year=None, max_year=None,
                              language=None, avoid_tags=None):
    b, ratings = _b(state), state["ratings"]
    avoid_cols = sr.resolve_tags(b, avoid_tags)[0] if b["tags"] is not None else []
    n = max(1, min(int(n or 10), 20))
    lang = _lang_code(b, language)
    if language and lang is None:
        return {"recommendations": [], "note": f"Unknown language '{language}'."}
    anchors, unresolved = [], []
    for t in anchor_titles or []:
        p = sr.resolve_title(b, t, ratings)
        if p is None:
            unresolved.append(t)
        elif p not in anchors:
            anchors.append(p)
    if anchor_titles and not anchors:
        return {"recommendations": [],
                "note": f"None of {anchor_titles} are in the catalogue, so there is nothing to anchor on."}

    filtered = (min_year is not None or max_year is not None or lang is not None
                or bool(avoid_tags))
    recs, top = sr.rank(b, ratings, state.get("skipped"),
                        n=80 if filtered else None, anchor_positions=anchors)
    want_lang = b["ns"]["lang_label"](lang) if lang else None
    rows, avoided = [], []
    for _, rec in recs.iterrows():
        if not sr.year_ok(rec.get("year"), min_year, max_year):
            continue
        if want_lang and rec.get("language") != want_lang:
            continue
        why = sr.avoid_reason(b, rec["corpus_index"], avoid_tags, avoid_cols) if avoid_tags else None
        if why:
            avoided.append(f"{rec['title']} ({why})")
            continue
        rows.append(_rec_row(rec))
        if len(rows) >= n:
            break

    mode = ("anchored on: " + ", ".join(str(b["df"].iloc[p]["title"]) for p in anchors)
            if anchors else "whole profile (the same list the website's Movies tab shows)")
    out = {"mode": mode, "recommendations": rows}
    if top is not None and len(top):
        out["taste_signature"] = [str(t) for t in top.index[:5]]
    if unresolved:
        out["unresolved_anchors"] = unresolved
    if avoided:
        out["removed_by_avoid_tags"] = len(avoided)
    if not rows:
        note = "Nothing came back."
        if filtered:
            note += (" The filters removed everything from the ranked list; for a language "
                     "request try search_catalog_semantic with the language set.")
        out["note"] = note
    return out


def _tool_search_catalog(state, query, limit=5):
    b = _b(state)
    hits = b["ns"]["search_movies"](query, b["df"], b["search_idx"], limit=int(limit or 5))
    rated = sr.excluded_positions(b, state["ratings"])
    return {"results": [dict(_row(b, p), recommendable=bool(b["servable"][p]),
                             already_rated=int(p) in rated)
                        for p in hits.index]}


def _tool_search_catalog_semantic(state, query, limit=15, min_year=None, max_year=None,
                                  language=None):
    b, ratings = _b(state), state["ratings"]
    df = b["df"]
    lang = _lang_code(b, language)
    if language and lang is None:
        return {"results": [], "note": f"Unknown language '{language}'."}
    q = embed_query(state, query)
    sims, idxs = b["index"].search(q, 600 if lang else 400)
    skip = sr.excluded_positions(b, ratings, state.get("skipped"))
    people = sr.rated_people(b, ratings)
    results = []
    for s, p in zip(sims[0], idxs[0]):
        p = int(p)
        if p < 0 or p in skip or not b["servable"][p] or sr.shares_people(b, p, people):
            continue
        row = df.iloc[p]
        if lang and row.get("original_language") != lang:
            continue
        if not sr.year_ok(row.get("release_year"), min_year, max_year):
            continue
        results.append(dict(_row(b, p), relevance=round(float(s), 3)))
        if len(results) >= int(limit or 15):
            break
    return {"query": query, "results": results,
            "note": "Catalogue-wide results by meaning. Excludes rated films and rated directors."}


def _tool_search_by_tags(state, positive_tags=None, negative_tags=None, limit=15,
                         min_year=None, max_year=None):
    b = _b(state)
    if b["tags"] is None:
        return {"results": [], "note": "No tag file is loaded."}
    pos, bad1, s1 = sr.resolve_tags(b, positive_tags)
    neg, bad2, s2 = sr.resolve_tags(b, negative_tags)
    if not pos and not neg:
        return {"results": [], "invalid_tags": bad1 + bad2, "suggestions": {**s1, **s2},
                "common_tags": sr.common_tags(b),
                "note": "None of those tags exist. Retry with tags from suggestions or common_tags."}
    df, people = b["df"], sr.rated_people(b, state["ratings"])

    def keep(p):
        return (not sr.shares_people(b, p, people)
                and sr.year_ok(df.iloc[p].get("release_year"), min_year, max_year))

    hits = sr.tag_search(b, pos, neg, sr.excluded_positions(b, state["ratings"], state.get("skipped")),
                         keep=keep, limit=int(limit or 15))
    return {"positive_tags_used": pos, "negative_tags_used": neg,
            "invalid_tags": bad1 + bad2, "suggestions": {**s1, **s2},
            "results": [dict(_row(b, p), tag_score=round(s, 2), top_tags=t) for p, s, t in hits],
            "note": f"Searched {len(b['tags']):,} tagged films (about 40% of the catalogue)."}


def _tool_search_by_people(state, person, role="any", limit=15):
    b = _b(state)
    df, si = b["df"], b["search_idx"]
    q = b["ns"]["_normalize"](person or "")
    if not q:
        return {"results": [], "note": "Give a name to search for."}
    d = pd.Series(si["directors"]).str.contains(q, regex=False, na=False).to_numpy()
    c = pd.Series(si["cast"]).str.contains(q, regex=False, na=False).to_numpy()
    m = d if role == "director" else c if role == "actor" else (d | c)
    m = m & b["servable"]
    skip = sr.excluded_positions(b, state["ratings"], state.get("skipped"))
    vc = sr._num(df, "vote_count").fillna(0).to_numpy()
    hits = sorted((int(p) for p in np.flatnonzero(m) if int(p) not in skip), key=lambda p: -vc[p])
    results = []
    for p in hits[:int(limit or 15)]:
        r = df.iloc[p]
        results.append(dict(_row(b, p), cast=[str(x) for x in sr.as_list(r.get("cast"))[:4]],
                            role_of_person="director" if d[p] else "actor"))
    if not results:
        return {"person": person, "results": [],
                "note": f"No recommendable films found for '{person}'. Check the spelling."}
    return {"person": person, "role_searched": role, "results": results,
            "note": "Their films that clear the website's quality floor, most-voted first. "
                    "Excludes films the user rated."}


def _tool_search_by_studio(state, studio, limit=15, min_year=None, max_year=None):
    b = _b(state)
    df = b["df"]
    if "studio" not in df.columns:
        return {"results": [], "note": "Studio data is not available."}
    s = (studio or "").strip().lower()
    m = df["studio"].astype(str).str.lower().str.contains(s, regex=False, na=False).to_numpy()
    m = m & b["servable"]
    skip = sr.excluded_positions(b, state["ratings"], state.get("skipped"))
    vc = sr._num(df, "vote_count").fillna(0).to_numpy()
    hits = sorted((int(p) for p in np.flatnonzero(m) if int(p) not in skip), key=lambda p: -vc[p])
    results = []
    for p in hits:
        r = df.iloc[p]
        if not sr.year_ok(r.get("release_year"), min_year, max_year):
            continue
        results.append(dict(_row(b, p), studio=sr.text(r.get("studio"))))
        if len(results) >= int(limit or 15):
            break
    if not results:
        return {"studio": studio, "results": [], "note": f"No recommendable films for studio '{studio}'."}
    return {"studio": studio, "results": results,
            "note": "Most-voted first. Excludes films the user rated."}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_recommendations",
        "description": (
            "The website's own film recommender. With NO anchor_titles it returns exactly the "
            "website's 'For you' list for this user's whole profile: use it for open requests "
            "like 'what should I watch tonight'. With anchor_titles it steers the same ranker "
            "toward those films; the user's other likes still count as seen and dislikes still "
            "apply. Each pick carries because_of / also_because (the rated films it is closest "
            "to), matched_tags (tags that lifted it), cast and a blurb. Optional year and "
            "original-language filters, and avoid_tags to drop anything strongly tagged or "
            "genred as e.g. 'horror' or 'romance'. Start here for any 'for me' request, "
            "putting the user's exclusions in avoid_tags."),
        "input_schema": {"type": "object", "properties": {
            "anchor_titles": {"type": "array", "items": {"type": "string"},
                              "description": "Optional. Films to steer toward: rated films or any catalogue title."},
            "n": {"type": "integer", "description": "How many to return, up to 20 (default 10)."},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"},
            "language": {"type": "string", "description": "e.g. 'Korean', 'Hindi', 'French', or a code like 'ko'."},
            "avoid_tags": {"type": "array", "items": {"type": "string"},
                           "description": "Things the user doesn't want, e.g. ['horror']."}}},
    },
    {
        "name": "search_user_ratings",
        "description": (
            "The films the user has rated. With a query ('heist', 'slow character drama'), "
            "returns their rated films ranked by similarity in meaning. Omit the query for all "
            "ratings, highest first."),
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "search_catalog",
        "description": (
            "Look a film up by title, director or cast member. Says whether it is recommendable "
            "and whether the user already rated it. Use it to verify a title before naming it."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_catalog_semantic",
        "description": (
            "Search the whole catalogue by meaning, with an optional language filter. Use for "
            "requests outside the user's rated taste, combining the genre with qualities they "
            "enjoy ('horror with stylish tension and a twist'), and for world cinema ('revenge "
            "thriller', language='Korean'). Excludes rated films and rated directors."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"},
            "language": {"type": "string"}},
            "required": ["query"]},
    },
    {
        "name": "search_by_tags",
        "description": (
            "Search films by quality tags (about 40% of the catalogue is tagged). Tags cover pace "
            "(fast-paced, slow-burn), mood (dark, atmospheric, tense, bleak, feel-good), experience "
            "(mind-bending, twist-ending, non-linear, tearjerker), themes (revenge, redemption, "
            "survival, found-family, coming-of-age), genre-ish tags (noir, heist, superhero, "
            "martial-arts, true-crime, satire, cyberpunk...), and source (book-adaptation, "
            "comic-adaptation, manga-adaptation). If a tag doesn't exist, the result lists real "
            "tags to retry with. Fall back to search_catalog_semantic if results are weak."),
        "input_schema": {"type": "object", "properties": {
            "positive_tags": {"type": "array", "items": {"type": "string"}},
            "negative_tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"}}},
    },
    {
        "name": "search_by_people",
        "description": (
            "Films by a director or top-billed actor ('films by Bong Joon-ho', 'movies with "
            "Denzel Washington'). role: 'director', 'actor' or 'any'. Most-voted first, "
            "excluding films the user rated."),
        "input_schema": {"type": "object", "properties": {
            "person": {"type": "string"},
            "role": {"type": "string", "enum": ["director", "actor", "any"]},
            "limit": {"type": "integer"}},
            "required": ["person"]},
    },
    {
        "name": "search_by_studio",
        "description": "Films from a studio ('A24', 'Blumhouse', 'Studio Ghibli'). Most-voted first.",
        "input_schema": {"type": "object", "properties": {
            "studio": {"type": "string"}, "limit": {"type": "integer"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"}},
            "required": ["studio"]},
    },
]

TOOL_IMPLS = {
    "get_recommendations": _tool_get_recommendations,
    "search_user_ratings": _tool_search_user_ratings,
    "search_catalog": _tool_search_catalog,
    "search_catalog_semantic": _tool_search_catalog_semantic,
    "search_by_tags": _tool_search_by_tags,
    "search_by_people": _tool_search_by_people,
    "search_by_studio": _tool_search_by_studio,
}

SYSTEM_PROMPT = """You are a thoughtful film recommendation assistant. You help the user find films to watch, based on their ratings and their request.

You only do film recommendations. If asked about anything else, say: "I'm focused on film recommendations — what kind of movie are you in the mood for?"

Choose a path:

**Open request** ("what should I watch tonight?", "a must-see list for me"): get_recommendations with no anchors. That is the website's own list for their profile. Put anything they rule out ("no horror") in avoid_tags. Only add catalogue or tag searches if that list is too thin, and say which picks came from the wider search.

**Path A — fits their taste** ("more like my heist films"): search_user_ratings with a query, then get_recommendations with the matching films as anchor_titles.

**Path B — outside their taste** ("I want horror" from someone who rates crime): search_user_ratings without a query, pick 2-3 qualities they reliably enjoy (name the films), then search_catalog_semantic combining the genre with those qualities. Bridge explicitly: "You rated X highly for its Y; this has the same Y."

**Path C — quality request** ("mind-bending non-linear", "slow-burn revenge"): search_by_tags, falling back to search_catalog_semantic if thin.

**Path D — people or studio** ("films by Bong Joon-ho", "more A24"): search_by_people or search_by_studio.

**World cinema** ("Korean thrillers", "Bollywood"): set the language on get_recommendations or search_catalog_semantic.

EXPLAINING PICKS: for get_recommendations results, the reasons are because_of, also_because and matched_tags; those are why the ranker chose the film. For search results, explain by the query, tags or person that found it. Describe a film only from its blurb, genres and tags. If a result has no blurb, don't summarise its plot from your own knowledge, even if you know the film well.

GROUNDING RULE — absolute:
- Every film you name must appear, by title, in a tool result from THIS conversation. Never add a film from your own knowledge, however sure you are; run another search instead.
- Never recommend a film the user has already rated.
- Use titles exactly as the tools returned them.
- Write each recommended film as **Title** (Director, Year), using the tool's values.
- If the results are weak, say so and offer to search differently rather than padding the list.

Be concise and warm, and refer to the user's own films when explaining."""


def run_agent(user_message, state, conversation_history=None, max_turns=8, verbose=True):
    """Returns (reply, history). The grounding result is left in state["_last_check"]."""
    return run_agent_loop(user_message, state, system=SYSTEM_PROMPT, tools=TOOLS,
                          impls=TOOL_IMPLS, noun="films",
                          conversation_history=conversation_history,
                          max_turns=max_turns, verbose=verbose)
