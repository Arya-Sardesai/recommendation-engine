"""
TV recommendation agent -- Claude (Haiku) with tools over the TV catalogue.

Ranking is NOT implemented here. get_recommendations calls the Space's own
recommend_tv through space_ranker, with the arguments the TV tab passes, so
with no anchors the agent's list IS the website's "For you" list (maturity
filter, same-title reboot exclusion, creator exclusion, tag layer).

Ratings use the Space's keys: tv_id and matched_creator. Older sessions stored
tmdb_id; space_ranker.normalize_ratings renames it.

TV tags come from TMDB keywords: themes and FORMAT (miniseries, sitcom,
telenovela) are well covered, pace and mood are sparse. The tool description
says so, so the model routes mood requests to semantic search.

Set key:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import numpy as np

import space_ranker as sr
from agent_common import embed_query, run_agent_loop

KIND = "tv"


def _b(state):
    return state["bundle"]


def _row(b, pos):
    r = b["df"].iloc[int(pos)]
    seasons = r.get("number_of_seasons")
    out = {"name": str(r["name"]), "creator": sr.text(r.get("created_by")),
           "network": b["ns"]["network_label"](r.get("networks")),
           "year": sr.year_of(r.get("start_year")),
           "language": b["ns"]["lang_label"](r.get("original_language")),
           "genres": sr.split_genres(r.get("genres"))[:3]}
    if sr.has_value(seasons):
        out["seasons"] = int(seasons)
    blurb = sr.short(r.get("overview"), 160)
    if blurb:
        out["blurb"] = blurb
    return out


def _rec_row(b, rec):
    out = {"name": str(rec["name"]), "creator": sr.text(rec.get("creator")),
           "network": b["ns"]["network_label"](rec.get("network")),
           "year": sr.year_of(rec.get("year")), "language": sr.text(rec.get("language")),
           "genres": sr.split_genres(rec.get("genres"))[:3],
           "vote_average": rec.get("vote_average"),
           "because_of": rec.get("because_of")}
    if sr.has_value(rec.get("seasons")):
        out["seasons"] = int(rec["seasons"])
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
        return {"ratings": [], "note": "The user has not rated any series yet."}
    if not query:
        return {"ratings": [{"name": r["matched_title"], "rating": r["rating"]}
                            for r in sorted(ratings, key=lambda r: -r["rating"])],
                "count": len(ratings)}
    b = _b(state)
    q = embed_query(state, query)[0]
    scored = sorted(((float(np.dot(q, b["emb"][r["corpus_index"]])), r) for r in ratings),
                    key=lambda x: -x[0])
    return {"query": query,
            "ratings": [{"name": r["matched_title"], "rating": r["rating"],
                         "relevance": round(s, 3)} for s, r in scored[:int(top_k or 8)]],
            "note": "The user's rated series, ranked by similarity in meaning to the query."}


def _tool_get_recommendations(state, anchor_names=None, n=10, min_year=None, max_year=None,
                              avoid_tags=None):
    b, ratings = _b(state), state["ratings"]
    avoid_cols = sr.resolve_tags(b, avoid_tags)[0] if b["tags"] is not None else []
    n = max(1, min(int(n or 10), 20))
    anchors, unresolved = [], []
    for t in anchor_names or []:
        p = sr.resolve_title(b, t, ratings)
        if p is None:
            unresolved.append(t)
        elif p not in anchors:
            anchors.append(p)
    if anchor_names and not anchors:
        return {"recommendations": [],
                "note": f"None of {anchor_names} are in the catalogue, so there is nothing to anchor on."}

    filtered = min_year is not None or max_year is not None or bool(avoid_tags)
    recs, top = sr.rank(b, ratings, state.get("skipped"),
                        n=80 if filtered else None, anchor_positions=anchors)
    rows, avoided = [], []
    for _, rec in recs.iterrows():
        if not sr.year_ok(rec.get("year"), min_year, max_year):
            continue
        why = sr.avoid_reason(b, rec["corpus_index"], avoid_tags, avoid_cols) if avoid_tags else None
        if why:
            avoided.append(f"{rec['name']} ({why})")
            continue
        rows.append(_rec_row(b, rec))
        if len(rows) >= n:
            break

    mode = ("anchored on: " + ", ".join(str(b["df"].iloc[p]["name"]) for p in anchors)
            if anchors else "whole profile (the same list the website's TV tab shows)")
    out = {"mode": mode, "recommendations": rows}
    if top is not None and len(top):
        out["taste_signature"] = [str(t) for t in top.index[:5]]
    if unresolved:
        out["unresolved_anchors"] = unresolved
    if avoided:
        out["removed_by_avoid_tags"] = len(avoided)
    if not rows:
        out["note"] = ("Nothing came back. The ranker needs at least one rating of 4+, "
                       "or the filters are too narrow.")
    return out


def _tool_search_catalog(state, query, limit=5):
    b = _b(state)
    hits = b["ns"]["search_tv"](query, b["df"], b["search_idx"], limit=int(limit or 5))
    rated = sr.excluded_positions(b, state["ratings"])
    return {"results": [dict(_row(b, p), recommendable=bool(b["servable"][p]),
                             already_rated=int(p) in rated)
                        for p in hits.index]}


def _tool_search_catalog_semantic(state, query, limit=15, min_year=None, max_year=None):
    b, ratings = _b(state), state["ratings"]
    df = b["df"]
    q = embed_query(state, query)
    sims, idxs = b["index"].search(q, 400)
    skip = sr.excluded_positions(b, ratings, state.get("skipped"))
    people = sr.rated_people(b, ratings)
    results = []
    for s, p in zip(sims[0], idxs[0]):
        p = int(p)
        if p < 0 or p in skip or not b["servable"][p] or sr.shares_people(b, p, people):
            continue
        if not sr.year_ok(df.iloc[p].get("start_year"), min_year, max_year):
            continue
        results.append(dict(_row(b, p), relevance=round(float(s), 3)))
        if len(results) >= int(limit or 15):
            break
    return {"query": query, "results": results,
            "note": "Catalogue-wide results by meaning. Excludes rated series and their creators."}


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
                and sr.year_ok(df.iloc[p].get("start_year"), min_year, max_year))

    hits = sr.tag_search(b, pos, neg, sr.excluded_positions(b, state["ratings"], state.get("skipped")),
                         keep=keep, limit=int(limit or 15))
    return {"positive_tags_used": pos, "negative_tags_used": neg,
            "invalid_tags": bad1 + bad2, "suggestions": {**s1, **s2},
            "results": [dict(_row(b, p), tag_score=round(s, 2), top_tags=t) for p, s, t in hits],
            "note": (f"Searched {len(b['tags']):,} tagged series (about half the catalogue, "
                     "from TMDB keywords). Tag scores are coarse: read them as 'has this quality'.")}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_recommendations",
        "description": (
            "The website's own TV recommender. With NO anchor_names it returns exactly the "
            "website's 'For you' list for this user's whole profile: use it for open requests "
            "like 'what should I start next'. With anchor_names it steers the same ranker toward "
            "those series; the user's other likes still count as watched and dislikes still "
            "apply. Each pick carries because_of / also_because (the rated series it is closest "
            "to), matched_tags, seasons, network and a blurb. Optional start-year filters, and "
            "avoid_tags to drop anything strongly tagged or genred as e.g. 'animation' or "
            "'sitcom'. Start here for any 'for me' request, putting the user's exclusions in "
            "avoid_tags."),
        "input_schema": {"type": "object", "properties": {
            "anchor_names": {"type": "array", "items": {"type": "string"},
                             "description": "Optional. Series to steer toward: rated or any catalogue series."},
            "n": {"type": "integer", "description": "How many to return, up to 20 (default 10)."},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"},
            "avoid_tags": {"type": "array", "items": {"type": "string"},
                           "description": "Things the user doesn't want, e.g. ['animation']."}}},
    },
    {
        "name": "search_user_ratings",
        "description": (
            "The series the user has rated. With a query ('crime dramas', 'cozy comfort shows'), "
            "returns their rated series ranked by similarity in meaning. Omit the query for all "
            "ratings, highest first."),
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "search_catalog",
        "description": (
            "Look a series up by name or creator. Says whether it is recommendable and whether "
            "the user already rated it. Use it to verify a name before naming it."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_catalog_semantic",
        "description": (
            "Search the whole TV catalogue by meaning. Use for requests outside the user's rated "
            "taste, and for pace or mood requests (slow-burn, atmospheric) that tags cover "
            "poorly. Combine the genre with qualities they enjoy, e.g. 'nordic crime drama with "
            "slow pacing and moral ambiguity'. Excludes rated series and their creators."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_by_tags",
        "description": (
            "Search series by tags from TMDB keywords (about half the catalogue). STRONG on "
            "themes (crime, heist, revenge, grief, political-intrigue, survival, supernatural, "
            "true-crime, dystopian, time-travel), source (book-adaptation, manga-adaptation, "
            "comic-adaptation), anime categories (shounen, seinen, slice-of-life, mecha) and "
            "FORMAT: miniseries, sitcom, telenovela ('something I can finish in a weekend' means "
            "positive_tags=['miniseries']). WEAK on pace and mood: use search_catalog_semantic "
            "for those. If a tag doesn't exist, the result lists real tags to retry with."),
        "input_schema": {"type": "object", "properties": {
            "positive_tags": {"type": "array", "items": {"type": "string"}},
            "negative_tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"}}},
    },
]

TOOL_IMPLS = {
    "get_recommendations": _tool_get_recommendations,
    "search_user_ratings": _tool_search_user_ratings,
    "search_catalog": _tool_search_catalog,
    "search_catalog_semantic": _tool_search_catalog_semantic,
    "search_by_tags": _tool_search_by_tags,
}

SYSTEM_PROMPT = """You are a thoughtful TV recommendation assistant. You help the user find series to watch, based on their ratings and their request.

You only do TV recommendations. If asked about anything else, say: "I'm focused on TV recommendations — what are you in the mood to watch?"

Choose a path:

**Open request** ("what should I start next?", "a must-watch list for me"): get_recommendations with no anchors. That is the website's own list for their profile. Put anything they rule out ("no anime") in avoid_tags. Only add catalogue or tag searches if that list is too thin, and say which picks came from the wider search.

**Path A — fits their taste** ("more like my crime shows"): search_user_ratings with a query, then get_recommendations with the matching series as anchor_names.

**Path B — outside their taste** (K-drama for someone who rates US sitcoms): search_user_ratings without a query, pick 2-3 qualities they reliably enjoy (name the series), then search_catalog_semantic combining the genre with those qualities. Bridge explicitly to their taste.

**Path C — theme or format** ("a limited series about a con", "a long sitcom to live in"): search_by_tags. Format tags (miniseries, sitcom, telenovela) are the right tool for length requests. For pace or mood ("slow-burn", "atmospheric"), use search_catalog_semantic instead.

EXPLAINING PICKS: for get_recommendations results, the reasons are because_of, also_because and matched_tags; those are why the ranker chose the series. Use seasons and network too. For search results, explain by the query or tags that found it. Describe a series only from its blurb, genres and tags. If a result has no blurb, don't summarise its plot from your own knowledge, even if you know the show well.

GROUNDING RULE — absolute:
- Every series you name must appear, by name, in a tool result from THIS conversation. Never add one from your own knowledge, however sure you are; run another search instead. That includes spin-offs and sequels: verify with search_catalog or describe without naming.
- Never recommend a series the user has already rated.
- Use names exactly as the tools returned them.
- Write each recommended series as **Name** (Year), using the tool's values.
- If the results are weak, say so and offer to search differently rather than padding the list.

Be concise and warm, and refer to the user's own series when explaining."""


def run_agent(user_message, state, conversation_history=None, max_turns=8, verbose=True):
    """Returns (reply, history). The grounding result is left in state["_last_check"]."""
    return run_agent_loop(user_message, state, system=SYSTEM_PROMPT, tools=TOOLS,
                          impls=TOOL_IMPLS, noun="series",
                          conversation_history=conversation_history,
                          max_turns=max_turns, verbose=verbose)
