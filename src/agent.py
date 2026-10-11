"""
Book recommendation agent -- Claude (Haiku) with tools over the book catalogue.

Ranking is NOT implemented here. get_recommendations calls the Space's own
recommend_books through space_ranker, with the arguments the Books tab passes,
so with no anchors the agent's list IS the website's list. The other tools are
searches (catalogue, semantic, tags) that apply the same floors the ranker
does, so they never surface a book the website would refuse to recommend.

The app supplies state["bundle"] (space_ranker.load("book")), state["ratings"]
and state["skipped"]. The agent adds caches to the same dict.

Set key:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import numpy as np

import space_ranker as sr
from agent_common import embed_query, run_agent_loop

KIND = "book"


def _b(state):
    return state["bundle"]


def _form(b, book_id):
    return b["forms"].get(str(book_id)) if b["forms"] else None


def _row(b, pos):
    """A catalogue row as the model sees it."""
    r = b["df"].iloc[int(pos)]
    out = {"title": str(r["title"]), "author": sr.text(r.get("primary_author")),
           "year": sr.year_of(r.get("publication_year")),
           "genres": sr.book_genres(b, r.get("genres"))}
    f = _form(b, r["book_id"])
    if f:
        out["form"] = f
    blurb = sr.short(r.get("description"), 160)
    if blurb:
        out["blurb"] = blurb
    return out


def _rec_row(b, rec):
    """A recommend_books result row: the pick plus the ranker's own reasons."""
    out = {"title": str(rec["title"]), "author": sr.text(rec.get("author")),
           "year": sr.year_of(rec.get("year")),
           "genres": sr.book_genres(b, rec.get("genres")),
           "because_of": rec.get("because_of")}
    also = rec.get("also_because")
    if isinstance(also, str) and also:
        out["also_because"] = also
    tags = [t for t in sr.as_list(rec.get("match_tags")) if t]
    if tags:
        out["matched_tags"] = tags
    f = _form(b, rec.get("book_id"))
    if f:
        out["form"] = f
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
        return {"ratings": [], "note": "The user has not rated any books yet."}
    if not query:
        return {"ratings": [{"title": r["matched_title"], "author": r.get("matched_author", ""),
                             "rating": r["rating"]}
                            for r in sorted(ratings, key=lambda r: -r["rating"])],
                "count": len(ratings)}
    b = _b(state)
    q = embed_query(state, query)[0]
    scored = sorted(((float(np.dot(q, b["emb"][r["corpus_index"]])), r) for r in ratings),
                    key=lambda x: -x[0])
    return {"query": query,
            "ratings": [{"title": r["matched_title"], "author": r.get("matched_author", ""),
                         "rating": r["rating"], "relevance": round(s, 3)}
                        for s, r in scored[:int(top_k or 8)]],
            "note": "The user's rated books, ranked by similarity in meaning to the query."}


def _tool_get_recommendations(state, anchor_titles=None, n=10, min_year=None, max_year=None,
                              avoid_tags=None):
    b, ratings = _b(state), state["ratings"]
    avoid_cols = sr.resolve_tags(b, avoid_tags)[0] if b["tags"] is not None else []
    n = max(1, min(int(n or 10), 20))
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

    filtered = min_year is not None or max_year is not None or bool(avoid_tags)
    recs, top = sr.rank(b, ratings, state.get("skipped"),
                        n=80 if filtered else None, anchor_positions=anchors)
    rows, avoided = [], []
    for _, rec in recs.iterrows():
        if not sr.year_ok(rec.get("year"), min_year, max_year):
            continue
        why = sr.avoid_reason(b, rec["corpus_index"], avoid_tags, avoid_cols) if avoid_tags else None
        if why:
            avoided.append(f"{rec['title']} ({why})")
            continue
        rows.append(_rec_row(b, rec))
        if len(rows) >= n:
            break

    mode = ("anchored on: " + ", ".join(str(b["df"].iloc[p]["title"]) for p in anchors)
            if anchors else "whole profile (the same list the website's Books tab shows)")
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
    hits = b["ns"]["search_books"](query, b["df"], b["search_idx"], limit=int(limit or 5))
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
    results, works = [], set()
    for s, p in zip(sims[0], idxs[0]):
        p = int(p)
        if p < 0 or p in skip or not b["servable"][p] or sr.shares_people(b, p, people):
            continue
        row = df.iloc[p]
        w = row["work_id"] if "work_id" in df.columns else None
        if sr.has_value(w):
            if w in works:
                continue
            works.add(w)
        if not sr.year_ok(row.get("publication_year"), min_year, max_year):
            continue
        results.append(dict(_row(b, p), relevance=round(float(s), 3)))
        if len(results) >= int(limit or 15):
            break
    return {"query": query, "results": results,
            "note": "Catalogue-wide results by meaning. Excludes rated books and their authors."}


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
                and sr.year_ok(df.iloc[p].get("publication_year"), min_year, max_year))

    hits = sr.tag_search(b, pos, neg, sr.excluded_positions(b, state["ratings"], state.get("skipped")),
                         keep=keep, limit=int(limit or 15))
    return {"positive_tags_used": pos, "negative_tags_used": neg,
            "invalid_tags": bad1 + bad2, "suggestions": {**s1, **s2},
            "results": [dict(_row(b, p), tag_score=round(s, 2), top_tags=t) for p, s, t in hits],
            "note": f"Searched {len(b['tags']):,} tagged books (Hardcover reader tags)."}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_recommendations",
        "description": (
            "The website's own book recommender. With NO anchor_titles it returns exactly the "
            "list the website's Books tab shows for this user's whole profile: use it for open "
            "requests like 'what should I read next'. With anchor_titles it steers the same "
            "ranker toward those books; the user's other likes still count as read and their "
            "dislikes still apply. Each pick carries because_of / also_because (the rated books "
            "it is closest to), matched_tags (reader tags that lifted it), form (novel, memoir, "
            "reportage...) and a blurb. Optional publication-year filters, and avoid_tags to "
            "drop anything strongly tagged or shelved as e.g. 'romance' or 'horror'. Start "
            "here for any 'for me' request, putting the user's exclusions in avoid_tags."),
        "input_schema": {"type": "object", "properties": {
            "anchor_titles": {"type": "array", "items": {"type": "string"},
                              "description": "Optional. Books to steer toward: rated books or any catalogue title."},
            "n": {"type": "integer", "description": "How many to return, up to 20 (default 10)."},
            "min_year": {"type": "integer"},
            "max_year": {"type": "integer"},
            "avoid_tags": {"type": "array", "items": {"type": "string"},
                           "description": "Things the user doesn't want, e.g. ['romance']."}}},
    },
    {
        "name": "search_user_ratings",
        "description": (
            "The books the user has rated. With a query describing a vibe or genre ('noir crime', "
            "'funny', 'literary fiction'), returns their rated books ranked by similarity in "
            "meaning. Omit the query for all ratings, highest first."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Optional vibe/genre/mood text."}}},
    },
    {
        "name": "search_catalog",
        "description": (
            "Look a book up by title or author. Says whether it is recommendable (clears the "
            "website's quality floor) and whether the user already rated it. Use it to verify a "
            "title before naming or anchoring on it."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_catalog_semantic",
        "description": (
            "Search the whole catalogue by meaning with a free-text query. Use for requests "
            "outside the user's rated taste: combine the requested genre with qualities they "
            "reliably enjoy, e.g. 'cyberpunk with literary depth and character focus'. Excludes "
            "rated books and their authors."),
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
            "min_year": {"type": "integer"}, "max_year": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_by_tags",
        "description": (
            "Search books by Hardcover reader tags: genres and moods applied by readers, on "
            "about 86% of the catalogue. Good for genre-plus-mood requests ('dark mystery', "
            "'funny fantasy, not romance'). Moods like funny, emotional and adventurous sit on "
            "most books and say little alone; pair them with something specific. If a tag "
            "doesn't exist, the result lists real tags to retry with."),
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

SYSTEM_PROMPT = """You are a thoughtful book recommendation assistant. You help the user find books to read, based on their ratings and their request.

You only do book recommendations. If asked about anything else, say: "I'm focused on book recommendations — what kind of book are you in the mood for?"

Choose a path:

**Open request** ("what should I read next?", "a must-read list for me", "surprise me"): get_recommendations with no anchors. That is the website's own list for their profile. Put anything they rule out ("no romance") in avoid_tags. Only add catalogue or tag searches if that list is too thin, and say which picks came from the wider search.

**Path A — fits their taste** ("more like my noir books"): search_user_ratings with a query to find the matching rated books, then get_recommendations with those as anchor_titles.

**Path B — outside their taste** ("I want cyberpunk" from someone who reads literary fiction): search_user_ratings without a query, pick 2-3 qualities they reliably enjoy (name the books), then search_catalog_semantic combining the genre with those qualities. Bridge explicitly: "You rated X highly for its Y; this has the same Y."

**Path C — tag request** ("dark mystery", "funny fantasy without romance"): search_by_tags. If results are thin, fall back to search_catalog_semantic.

Fiction or non-fiction: results carry a form field (novel, memoir, reportage, history...). If the user asks for non-fiction, don't offer novels, and the reverse.

EXPLAINING PICKS: for get_recommendations results, the reasons are because_of, also_because and matched_tags; those are why the ranker chose the book. For search results, explain by the query or tags that found it. Describe a book only from its blurb, genres and tags. If a result has no blurb, don't summarise its plot from your own knowledge, even if you know the book well.

GROUNDING RULE — absolute:
- Every book you name must appear, by title, in a tool result from THIS conversation. Never add a book from your own knowledge, however sure you are; run another search instead.
- Never recommend a book the user has already rated.
- Use titles exactly as the tools returned them.
- Write each recommended book as **Title** (Author, Year), using the tool's values.
- If the results are weak, say so and offer to search differently rather than padding the list.

Be concise and warm, and refer to the user's own books when explaining."""


def run_agent(user_message, state, conversation_history=None, max_turns=8, verbose=True):
    """Returns (reply, history). The grounding result is left in state["_last_check"]."""
    return run_agent_loop(user_message, state, system=SYSTEM_PROMPT, tools=TOOLS,
                          impls=TOOL_IMPLS, noun="books",
                          conversation_history=conversation_history,
                          max_turns=max_turns, verbose=verbose)
