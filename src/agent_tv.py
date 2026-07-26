"""
TV recommendation agent — multi-step reasoning over the TV recommender.

Clone of the movies agent pattern with TV-specific differences:

  1. TAGS ARE KEYWORD-ONLY. tv_tags.parquet comes entirely from the TMDB
     keyword->taxonomy map (49.9% of the 23,464-series corpus). There is no
     genome layer (that is MovieLens, movies-only) and no LLM layer. Scores are
     two-valued (0.7 primary / 0.5 supporting), so ranking within a tag is
     coarse -- ties are broken by vote count as a popularity prior.
     Consequence, stated in the tool description so the model routes correctly:
     THEME tags (crime, queer-rep, grief, heist) are well covered; EXPERIENCE
     tags (slow-burn, atmospheric, twist-ending) are sparse on TV, and requests
     for those should fall back to semantic search early.

  2. FORMAT TAGS ARE A NEW SEARCH AXIS. miniseries / sitcom / telenovela --
     for TV, format IS content. "A limited series I can finish in a weekend"
     is a tag query, not a semantic one.

  3. CREATOR EXCLUSION mirrors the ranking layer's rule (the TV analogue of
     the books agent's author exclusion).

Set key:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import json
import numpy as np
import pandas as pd
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

MIN_VOTES = 150          # same floor as the TV ranking layer


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _get_query_embedder(state):
    if "query_embedder" not in state:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state["query_embedder"] = SentenceTransformer("BAAI/bge-m3", device=device)
    return state["query_embedder"]


def _creators_of(row):
    """Normalized creator set for a series row; tolerant of missing column."""
    val = row.get("created_by") if hasattr(row, "get") else None
    if val is None:
        return set()
    if isinstance(val, str):
        parts = val.split(",")
    else:
        try:
            parts = list(val)
        except TypeError:
            return set()
    return {str(p).strip().lower() for p in parts if str(p).strip()}


def _tool_search_user_ratings(state, query=None, top_k=8):
    """The user's rated series; with a query, ranked by semantic similarity."""
    ratings = state["ratings"]
    if not ratings:
        return {"ratings": [], "note": "User has not rated any series yet."}
    if not query:
        return {"ratings": [{"name": r["matched_title"],
                             "rating": r["rating"]} for r in ratings],
                "count": len(ratings)}
    embeddings = state["embeddings"]
    model = _get_query_embedder(state)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scored = sorted(((float(np.dot(qvec, embeddings[r["corpus_index"]])), r)
                     for r in ratings), key=lambda x: -x[0])
    return {"query": query,
            "matched_series": [{"name": r["matched_title"], "rating": r["rating"],
                                "relevance": round(s, 3)} for s, r in scored[:top_k]],
            "note": "Ranked by semantic similarity to the query."}


def _tool_get_recommendations(state, anchor_names, n=10, min_year=None, max_year=None):
    """Nearest unwatched series to the anchors, round-robin, with the ranking
    layer's floors: min_votes and creator exclusion."""
    df = state["df"]; embeddings = state["embeddings"]; index = state["index"]
    ratings = state["ratings"]

    anchors = []
    for an in anchor_names:
        for r in ratings:
            if an.lower() in r["matched_title"].lower() or r["matched_title"].lower() in an.lower():
                anchors.append(r); break
    if not anchors:
        for an in anchor_names:
            mask = df["name"].str.contains(an, case=False, na=False, regex=False)
            if mask.any():
                row = df[mask].nlargest(1, "vote_count").iloc[0]
                anchors.append({"matched_title": row["name"],
                                "corpus_index": int(df.index[df["tmdb_id"] == row["tmdb_id"]][0]),
                                "tmdb_id": row["tmdb_id"], "rating": 5.0})
    if not anchors:
        return {"recommendations": [], "note": f"Could not find any of {anchor_names} to anchor on."}

    rated_ids = {r["tmdb_id"] for r in ratings if "tmdb_id" in r}
    rated_rows = [df.iloc[r["corpus_index"]] for r in ratings if "corpus_index" in r]
    rated_creators = set().union(*(_creators_of(rr) for rr in rated_rows)) if rated_rows else set()

    vecs = np.array([embeddings[a["corpus_index"]] for a in anchors]).astype("float32")
    sims, idxs = index.search(vecs, 150)

    queues = []
    for i in range(len(anchors)):
        q = []
        for s, ix in zip(sims[i], idxs[i]):
            if ix < 0:      # FAISS pads with -1; iloc[-1] is the silent-last-row trap
                continue
            b = df.iloc[ix]
            if b["tmdb_id"] in rated_ids:
                continue
            if b["vote_count"] < MIN_VOTES:
                continue
            if rated_creators and (_creators_of(b) & rated_creators):
                continue
            yr = b["start_year"]
            if min_year is not None and (pd.isna(yr) or yr < min_year):
                continue
            if max_year is not None and (pd.isna(yr) or yr > max_year):
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
                b = df.iloc[ix]
                yr = int(b["start_year"]) if pd.notna(b["start_year"]) else None
                results.append({"name": b["name"], "year": yr,
                                "because_of": anchors[qi]["matched_title"]})
                taken += 1; added += 1
                if len(results) >= n: break
            if len(results) >= n: break
        if added == 0: break
    return {"recommendations": results}


def _tool_search_catalog(state, query, limit=5):
    df = state["df"]
    mask = df["name"].str.contains(query, case=False, na=False, regex=False)
    hits = df[mask].nlargest(limit, "vote_count")
    return {"results": [{"name": r["name"],
                         "year": int(r["start_year"]) if pd.notna(r["start_year"]) else None,
                         "genres": r.get("genres", "")}
                        for _, r in hits.iterrows()]}


def _tool_search_catalog_semantic(state, query, limit=15, min_year=None, max_year=None):
    """Semantic search across the whole TV corpus."""
    df = state["df"]; index = state["index"]
    ratings = state["ratings"]
    model = _get_query_embedder(state)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(qvec, 200)

    rated_ids = {r["tmdb_id"] for r in ratings if "tmdb_id" in r}
    results = []
    for s, ix in zip(sims[0], idxs[0]):
        if ix < 0:
            continue
        b = df.iloc[ix]
        if b["tmdb_id"] in rated_ids:
            continue
        if b["vote_count"] < MIN_VOTES:
            continue
        yr = b["start_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue
        results.append({"name": b["name"],
                        "year": int(yr) if pd.notna(yr) else None,
                        "genres": b.get("genres", ""),
                        "relevance": round(float(s), 3)})
        if len(results) >= limit:
            break
    return {"query": query, "results": results}


def _tool_search_by_tags(state, positive_tags=None, negative_tags=None, limit=15, min_year=None, max_year=None):
    """Tag search over tv_tags.parquet (keyword-derived, 49.9% of corpus).
    Scores are coarse (0.7/0.5), so within-tag ties are broken by vote count."""
    df = state["df"]
    ratings = state["ratings"]

    if "tags_df" not in state:
        from pathlib import Path
        tags_path = Path("data/processed/tv_tags.parquet")
        if not tags_path.exists():
            return {"results": [], "note": "No TV tag data available. Run build_keyword_tags.py."}
        tags = pd.read_parquet(tags_path)
        state["tags_df"] = tags.pivot_table(
            index="tv_id", columns="tag", values="score", fill_value=0.0
        ).astype("float32")
        state["tvid_pos"] = {int(t): i for i, t in enumerate(df["tmdb_id"])}
        # popularity prior for tie-breaking coarse scores
        state["votes_by_id"] = dict(zip(df["tmdb_id"].astype(int), df["vote_count"]))

    tags_df = state["tags_df"]
    tvid_pos = state["tvid_pos"]
    votes_by_id = state["votes_by_id"]

    positive_tags = positive_tags or []
    negative_tags = negative_tags or []
    available = set(tags_df.columns)
    pos = [t for t in positive_tags if t in available]
    neg = [t for t in negative_tags if t in available]
    invalid = [t for t in positive_tags + negative_tags if t not in available]

    if not pos and not neg:
        return {"results": [], "note": f"None of those tags exist. Available tags include: {sorted(available)[:30]}..."}

    score = tags_df[pos].sum(axis=1) if pos else pd.Series(0.0, index=tags_df.index)
    if neg:
        score = score - tags_df[neg].sum(axis=1)

    # coarse scores -> many exact ties; log-vote tiebreak keeps it stable and
    # popularity-aware without letting votes outrank an actual extra tag match
    tie = pd.Series({i: np.log1p(votes_by_id.get(int(i), 0)) * 1e-3 for i in score.index})
    ranked = (score + tie).sort_values(ascending=False)

    rated_ids = {r["tmdb_id"] for r in ratings if "tmdb_id" in r}
    results = []
    for tv_id, s in ranked.items():
        if s <= 0:
            break
        if tv_id in rated_ids:
            continue
        pos_ix = tvid_pos.get(int(tv_id))
        if pos_ix is None:
            continue
        b = df.iloc[pos_ix]
        if b["vote_count"] < MIN_VOTES:
            continue
        yr = b["start_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue
        top = tags_df.loc[tv_id].nlargest(5)
        results.append({
            "name": b["name"],
            "year": int(yr) if pd.notna(yr) else None,
            "tag_score": round(float(score.loc[tv_id]), 2),
            "top_tags": [(t, round(float(v), 2)) for t, v in top.items() if v > 0],
        })
        if len(results) >= limit:
            break

    return {
        "positive_tags_used": pos,
        "negative_tags_used": neg,
        "invalid_tags": invalid,
        "results": results,
        "note": (f"Searched {len(tags_df):,} tagged series (~50% of catalog, "
                 "TMDB-keyword-derived). Theme tags are well covered; "
                 "experience/pace tags are sparse on TV."),
    }


TOOLS = [
    {
        "name": "search_user_ratings",
        "description": "Look at the series the user has rated. With a query describing a vibe/genre/mood (e.g. 'crime dramas', 'cozy comfort shows'), returns their rated series ranked by SEMANTIC similarity to it. Omit the query to get all ratings.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Optional vibe/genre/mood text. Omit for all ratings."}}},
    },
    {
        "name": "get_recommendations",
        "description": "Get TV recommendations based on anchor series (names the user likes). Optionally restrict by start year. Returns unwatched series similar to the anchors, each tagged with which anchor it came from. Excludes series by creators the user has already watched.",
        "input_schema": {"type": "object", "properties": {
            "anchor_names": {"type": "array", "items": {"type": "string"},
                             "description": "Series names to base recommendations on."},
            "n": {"type": "integer", "description": "How many to return (default 10)."},
            "min_year": {"type": "integer", "description": "Only series starting in or after this year."},
            "max_year": {"type": "integer", "description": "Only series starting in or before this year."}},
            "required": ["anchor_names"]},
    },
    {
        "name": "search_catalog",
        "description": "Check whether a series exists in the catalog and get its details. Use to verify a name before anchoring on it.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_catalog_semantic",
        "description": "Semantic search across the ENTIRE TV catalog by free-text query (genre + qualities). Use when the user asks for something outside their rated taste, or when tag search comes back weak. Combine the requested genre with qualities inferred from their taste, e.g. 'nordic crime drama with slow pacing and moral ambiguity'.",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "min_year": {"type": "integer"},
            "max_year": {"type": "integer"}},
            "required": ["query"]},
    },
    {
        "name": "search_by_tags",
        "description": "Search series by quality-tags (NOT genres). Covers ~11.7K series (~50% of catalog) with TMDB-keyword-derived tags. STRONG coverage: themes (crime, queer-rep, grief, heist, political-intrigue, revenge, survival, family, identity, class-struggle, psychological, supernatural, true-crime, sports, martial-arts, war, dystopian, post-apocalyptic, time-travel), source medium (book-adaptation, manga-adaptation, comic-adaptation, game-adaptation), FORMAT (miniseries, sitcom, telenovela — use these for 'limited series I can finish fast' or 'a long sitcom to live in'), and anime categories (slice-of-life, shounen, seinen, mecha). WEAK/SPARSE on TV: experience and pace tags (slow-burn, atmospheric, twist-ending, tearjerker) — for those qualities, prefer search_catalog_semantic instead of forcing tag search. Returns each series' top contributing tags for explanation.",
        "input_schema": {"type": "object", "properties": {
            "positive_tags": {"type": "array", "items": {"type": "string"},
                              "description": "Tags the series SHOULD score high on."},
            "negative_tags": {"type": "array", "items": {"type": "string"},
                              "description": "Tags the series should score LOW on."},
            "limit": {"type": "integer"},
            "min_year": {"type": "integer"},
            "max_year": {"type": "integer"}}},
    },
]

TOOL_IMPLS = {
    "search_user_ratings": _tool_search_user_ratings,
    "get_recommendations": _tool_get_recommendations,
    "search_catalog": _tool_search_catalog,
    "search_catalog_semantic": _tool_search_catalog_semantic,
    "search_by_tags": _tool_search_by_tags,
}

SYSTEM_PROMPT = """You are a thoughtful TV recommendation assistant. You help the user find series to watch based on their taste and their requests.

You are strictly a TV recommendation assistant. If the user asks for anything unrelated to TV or watching (general questions, code, other topics), politely redirect: "I'm focused on TV recommendations — what are you in the mood to watch?"

You have five tools:
- search_user_ratings: inspect what the user has rated, with optional semantic query
- get_recommendations: find series similar to anchors they've rated
- search_catalog: look up a specific series
- search_catalog_semantic: search the WHOLE catalog by free-text (genre + qualities)
- search_by_tags: search by quality/theme/format tags across ~50% of the catalog

Decide which path the request needs:

**Path A — request fits their taste** ("more like my crime shows"):
1. search_user_ratings WITH a query to find which rated series fit.
2. get_recommendations with those as anchors.
3. Explain why each pick fits.

**Path B — request is OUTSIDE their rated taste** (they ask for K-drama but rate mostly US sitcoms):
1. search_user_ratings WITHOUT a query for the full picture.
2. Identify 2-3 qualities they reliably enjoy, referencing specific series.
3. search_catalog_semantic combining the requested genre + those qualities.
4. Present picks by EXPLICITLY BRIDGING to their taste.

**Path C — request is QUALITY, THEME, or FORMAT based** ("a limited series about a con", "long sitcom to live in", "shounen but emotionally heavy"):
1. search_by_tags with positive_tags (and negative_tags to exclude qualities).
   Format tags (miniseries, sitcom, telenovela) are first-class here — "something
   I can finish in a weekend" means positive_tags=["miniseries"].
2. Cross-reference with their taste; explain picks via the specific tags.
3. TV tag coverage is thematic, not experiential: if the request hinges on pace
   or mood (slow-burn, atmospheric), go to search_catalog_semantic instead of
   forcing tags.
4. Tag scores on TV are coarse (keyword-derived) — treat them as "has this
   quality", not as a strength ranking. If results are weak, fall back to
   search_catalog_semantic.

CRITICAL — GROUNDING RULE:
Every series you name MUST appear, by name, in a tool result from THIS conversation.
- Do NOT add series from your own knowledge, even if perfect and certain to exist. Run another search instead.
- This includes spin-offs, sequels, and related series. Verify via search_catalog before naming, or refer without naming ("there's also a spin-off").
- Never recommend series the user has already rated.
- Prefer EXACT names as returned by tools. Do not rename or "correct" them.
- If tool results are thin, say so honestly rather than padding from memory.
The recommender's value is grounding in this catalog and the user's actual ratings. An ungrounded recommendation is worse than none.

Other guidance:
- For mood shifts ("lighter", "shorter", "slower") the tools can't filter directly, use judgment on the results and explain.
- Be concise and warm. Reference the user's taste in explanations.
- The catalog is TV series only (no films), up to 2025."""


def _titles_from_result(result):
    titles = set()
    if not isinstance(result, dict):
        return titles
    for key in ("recommendations", "results", "matched_series", "ratings"):
        items = result.get(key)
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("name"):
                    titles.add(it["name"])
    return titles


def _check_faithfulness(reply, seen_titles):
    """Title-shaped-pattern checker (name followed by a year parenthetical),
    same discipline as the books agent: precise enough not to flag prose."""
    import re
    pattern = re.compile(
        r"(?:\*\*)?([A-Z0-9][^\n*()]{1,70}?)(?:\*\*)?"
        r"\s*\((?:[^)]*?\b(?:19|20)\d{2}\b[^)]*)\)"
    )
    cands = [m.group(1).strip().rstrip(":–-—").strip() for m in pattern.finditer(reply)]
    cands = [re.sub(r"^\d+\.\s*", "", c).strip() for c in cands]

    def norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())

    seen_norm = {norm(t) for t in seen_titles}
    suspicious, sus_norms = [], set()
    for c in cands:
        cn = norm(c)
        if len(cn) < 3 or cn in sus_norms or cn in seen_norm:
            continue
        if any(cn in s or s in cn for s in seen_norm if len(s) >= 4):
            continue
        suspicious.append(c); sus_norms.add(cn)
    return suspicious


def run_agent(user_message, state, conversation_history=None, max_turns=6, verbose=True):
    """Agentic loop. Returns (final_text, updated_history)."""
    client = Anthropic()
    history = (conversation_history or []) + [{"role": "user", "content": user_message}]
    # Conversation-scoped, not turn-scoped. The grounding rule says "in THIS
    # conversation", and a follow-up like "tell me about the third one" makes
    # no tool calls -- with a per-turn reset it flagged the answer as a
    # hallucination. Lives on `state` so it survives across messages.
    seen_titles = state.setdefault("_seen_titles", set())

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
            suspicious = _check_faithfulness(reply, seen_titles)
            if verbose and suspicious:
                print(f"  [!] WARNING: {len(suspicious)} name(s) in the reply were "
                      f"NOT in any tool result (possible hallucination):")
                for s in suspicious:
                    print(f"        - {s}")
            elif verbose:
                print(f"  [ok] all recommended series were grounded in tool results "
                      f"({len(seen_titles)} series seen across tools)")
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
                    more = f" (+{len(returned)-8} more)" if len(returned) > 8 else ""
                    print(f"        -> returned {len(returned)} series: {', '.join(preview)}{more}")
                else:
                    print(f"        -> (no series names in result)")
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id,
                                 "content": json.dumps(result)})
        history.append({"role": "user", "content": tool_results})

    return "(agent hit max turns without finishing)", history
