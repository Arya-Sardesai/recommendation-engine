"""
Recommendation agent — multi-step reasoning over the book recommender.

The agent (Claude) is given tools that wrap the existing recommender. It can:
  - inspect the user's ratings (semantic search over what they've rated)
  - get recommendations from anchor books, with optional year filters
  - look up books in the catalog

It runs an agentic loop: Claude decides which tools to call, we execute them,
feed results back, and Claude continues until it produces a final answer.

This module is UI-agnostic — it takes the loaded corpus/embeddings/index and a
list of user ratings, and exposes `run_agent(user_message, ...)`.

Set key:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import json
import numpy as np
import pandas as pd
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Tool implementations — these wrap the real recommender logic.
# Each takes the shared state (df, embeddings, index, ratings) plus tool args.
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
    """Return the user's rated books. If a query is given, rank the user's books
    by SEMANTIC similarity to the query (so 'my noir books' finds crime/noir by
    meaning, not string match). Otherwise return all ratings."""
    ratings = state["ratings"]
    if not ratings:
        return {"ratings": [], "note": "User has not rated any books yet."}

    if not query:
        return {"ratings": [{"title": r["matched_title"],
                             "author": r.get("matched_author", ""),
                             "rating": r["rating"]} for r in ratings],
                "count": len(ratings)}

    # semantic: embed the query, compare to each rated book's corpus embedding
    embeddings = state["embeddings"]
    model = _get_query_embedder(state)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    scored = []
    for r in ratings:
        bvec = embeddings[r["corpus_index"]]
        sim = float(np.dot(qvec, bvec))  # both normalized -> cosine
        scored.append((sim, r))
    scored.sort(key=lambda x: -x[0])

    top = scored[:top_k]
    return {
        "query": query,
        "matched_books": [
            {"title": r["matched_title"], "author": r.get("matched_author", ""),
             "rating": r["rating"], "relevance": round(sim, 3)}
            for sim, r in top
        ],
        "note": "Ranked by semantic similarity to the query. Higher relevance = closer match to what the user described."
    }


def _tool_get_recommendations(state, anchor_titles, n=10, min_year=None, max_year=None):
    """Core recommender. Finds the anchor books among the user's ratings (by title),
    uses their embeddings, returns nearest unread books with optional year filters."""
    df = state["df"]; embeddings = state["embeddings"]; index = state["index"]
    ratings = state["ratings"]

    # match anchor_titles to the user's rated books (so we use their corpus_index)
    anchors = []
    for at in anchor_titles:
        for r in ratings:
            if at.lower() in r["matched_title"].lower() or r["matched_title"].lower() in at.lower():
                anchors.append(r)
                break
    if not anchors:
        # fall back: search the catalog directly for these titles
        for at in anchor_titles:
            mask = df["title"].str.contains(at, case=False, na=False, regex=False)
            if mask.any():
                row = df[mask].nlargest(1, "ratings_count").iloc[0]
                anchors.append({"matched_title": row["title"],
                                "matched_author": row["primary_author"],
                                "corpus_index": int(df.index[df["book_id"] == row["book_id"]][0]),
                                "book_id": row["book_id"], "rating": 5.0})
    if not anchors:
        return {"recommendations": [], "note": f"Could not find any of {anchor_titles} to anchor on."}

    rated_ids = {r["book_id"] for r in ratings}
    rated_authors = {r.get("matched_author", "").lower() for r in ratings}

    anchor_vectors = np.array([embeddings[a["corpus_index"]] for a in anchors]).astype("float32")
    sims, idxs = index.search(anchor_vectors, 150)

    # round-robin across anchors
    queues = []
    for i in range(len(anchors)):
        q = []
        for s, ix in zip(sims[i], idxs[i]):
            b = df.iloc[ix]
            if b["book_id"] in rated_ids:
                continue
            if b["primary_author"].lower() in rated_authors:
                continue
            if b["ratings_count"] < 2000:
                continue
            yr = b["publication_year"]
            if min_year is not None and (pd.isna(yr) or yr < min_year):
                continue
            if max_year is not None and (pd.isna(yr) or yr > max_year):
                continue
            q.append((int(ix), float(s)))
        queues.append(q)

    results, seen, seen_w = [], set(), set()
    while len(results) < n:
        added = 0
        for qi, q in enumerate(queues):
            taken = 0
            while q and taken < 2:
                ix, s = q.pop(0)
                b = df.iloc[ix]; w = b["work_id"]
                if ix in seen or (w and w in seen_w):
                    continue
                seen.add(ix)
                if w: seen_w.add(w)
                yr = int(b["publication_year"]) if pd.notna(b["publication_year"]) else None
                results.append({"title": b["title"], "author": b["primary_author"],
                                "year": yr, "because_of": anchors[qi]["matched_title"]})
                taken += 1; added += 1
                if len(results) >= n: break
            if len(results) >= n: break
        if added == 0: break
    return {"recommendations": results}


def _tool_search_catalog(state, query, limit=5):
    df = state["df"]
    mask = df["title"].str.contains(query, case=False, na=False, regex=False)
    hits = df[mask].nlargest(limit, "ratings_count")
    return {"results": [{"title": r["title"], "author": r["primary_author"],
                         "year": int(r["publication_year"]) if pd.notna(r["publication_year"]) else None}
                        for _, r in hits.iterrows()]}

def _tool_search_catalog_semantic(state, query, limit=15, min_year=None, max_year=None):
    """Semantic search across the ENTIRE catalog (not anchored to user ratings).
    Use when the user asks for a genre/vibe outside their rated taste — combine
    the requested genre with qualities you've inferred from their taste."""
    df = state["df"]
    embeddings = state["embeddings"]
    index = state["index"]
    ratings = state["ratings"]

    model = _get_query_embedder(state)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # search a wider net so we can filter by year/popularity and still have enough
    sims, idxs = index.search(qvec, 200)

    rated_ids = {r["book_id"] for r in ratings}
    rated_authors = {r.get("matched_author", "").lower() for r in ratings if r.get("matched_author")}

    results = []
    seen_works = set()
    for s, ix in zip(sims[0], idxs[0]):
        b = df.iloc[ix]
        if b["book_id"] in rated_ids:
            continue
        if b["primary_author"].lower() in rated_authors:
            continue
        if b["ratings_count"] < 2000:
            continue
        w = b["work_id"]
        if w and w in seen_works:
            continue
        if w:
            seen_works.add(w)
        yr = b["publication_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue
        results.append({
            "title": b["title"],
            "author": b["primary_author"],
            "year": int(yr) if pd.notna(yr) else None,
            "genres": list(b["genres"]) if b["genres"] is not None else [],
            "relevance": round(float(s), 3),
        })
        if len(results) >= limit:
            break

    return {
        "query": query,
        "results": results,
        "note": "Catalog-wide semantic results. Use these when the user asks for a genre outside their rated taste."
    }

def _tool_search_by_tags(state, positive_tags=None, negative_tags=None, limit=15, min_year=None, max_year=None):
    """Search the tagged subset by quality-tags. Returns books scoring high on positive
    tags and low on negative tags. Only works for the ~5K books that have been tagged."""
    df = state["df"]
    ratings = state["ratings"]
    
    # Lazy-load the tags dataframe on first call
    if "tags_df" not in state:
        from pathlib import Path
        tags_path = Path("data/processed/book_tags.parquet")
        if not tags_path.exists():
            return {"results": [], "note": "No tag data available."}
        tdf = pd.read_parquet(tags_path)
        # Pivot to wide: rows=book_id, cols=tag, values=score
        state["tags_df"] = tdf.pivot_table(
            index="book_id", columns="tag", values="score", fill_value=0.0
        )
    tags_df = state["tags_df"]
    
    positive_tags = positive_tags or []
    negative_tags = negative_tags or []
    
    # Validate tags exist
    available = set(tags_df.columns)
    pos = [t for t in positive_tags if t in available]
    neg = [t for t in negative_tags if t in available]
    invalid = [t for t in positive_tags + negative_tags if t not in available]
    
    if not pos and not neg:
        return {"results": [], "note": f"None of those tags exist. Available tags include: {sorted(available)[:30]}..."}
    
    # Score: sum of positive tag scores - sum of negative tag scores
    score = tags_df[pos].sum(axis=1) if pos else pd.Series(0.0, index=tags_df.index)
    if neg:
        score = score - tags_df[neg].sum(axis=1)
    
    # Sort, filter out already-rated, apply year filters
    rated_ids = {r["book_id"] for r in ratings}
    rated_authors = {r.get("matched_author", "").lower() for r in ratings if r.get("matched_author")}
    
    ranked = score.sort_values(ascending=False)
    
    results = []
    for book_id, s in ranked.items():
        if str(book_id) in {str(rid) for rid in rated_ids}:
            continue
        row = df[df["book_id"].astype(str) == str(book_id)]
        if row.empty:
            continue
        b = row.iloc[0]
        if b["primary_author"].lower() in rated_authors:
            continue
        if b["ratings_count"] < 2000:
            continue
        yr = b["publication_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue
        
        # Show which tags drove the score
        book_tags = tags_df.loc[book_id]
        top_tags = book_tags.nlargest(5)
        top_tags = [(t, round(float(v), 2)) for t, v in top_tags.items() if v > 0]
        
        results.append({
            "title": b["title"],
            "author": b["primary_author"],
            "year": int(yr) if pd.notna(yr) else None,
            "tag_score": round(float(s), 2),
            "top_tags": top_tags,
        })
        if len(results) >= limit:
            break
    
    return {
        "positive_tags_used": pos,
        "negative_tags_used": neg,
        "invalid_tags": invalid,
        "results": results,
        "note": f"Searched {len(tags_df):,} tagged books. Tagged subset is ~5K popular books — not the full catalog.",
    }


# tool schemas exposed to Claude
TOOLS = [
    {
        "name": "search_user_ratings",
        "description": "Look at the books the user has rated. If you provide a query describing a vibe/genre/mood (e.g. 'noir crime', 'funny', 'literary fiction'), it returns the user's rated books ranked by SEMANTIC similarity to that query — so it finds books by meaning, not keyword. Omit the query to get all ratings. Use this to find which of the user's books match a description like 'my noir books'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional vibe/genre/mood text, e.g. 'crime noir' or 'funny'. Omit to get all ratings."}
            },
        },
    },
    {
        "name": "get_recommendations",
        "description": "Get book recommendations based on a set of anchor books (titles the user likes). Optionally restrict by publication year. Returns unread books similar to the anchors, each tagged with which anchor it came from.",
        "input_schema": {
            "type": "object",
            "properties": {
                "anchor_titles": {"type": "array", "items": {"type": "string"},
                                  "description": "Titles to base recommendations on. Can be the user's rated books or any book."},
                "n": {"type": "integer", "description": "How many recommendations to return (default 10)."},
                "min_year": {"type": "integer", "description": "Optional: only books published in or after this year."},
                "max_year": {"type": "integer", "description": "Optional: only books published in or before this year."},
            },
            "required": ["anchor_titles"],
        },
    },
    {
        "name": "search_catalog",
        "description": "Check whether a book exists in the catalog and get its details. Use to verify a title before anchoring on it.",
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
    "description": "Semantic search across the ENTIRE catalog by free-text query (genre + qualities). Use this when the user asks for a genre they haven't rated in (e.g. they ask for cyberpunk but their ratings are all literary fiction). The right pattern: first look at their ratings to infer qualities they enjoy (twists, character-driven, witty, fast-paced, etc.), then call this with a query combining the requested genre AND those qualities — e.g. 'cyberpunk with literary depth and character focus'. Excludes books they've rated and authors they've read.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text query combining the requested genre with qualities from the user's taste. e.g. 'cyberpunk with twists and witty dialogue'."},
            "limit": {"type": "integer", "description": "How many results to return (default 15)."},
            "min_year": {"type": "integer"},
            "max_year": {"type": "integer"},
        },
        "required": ["query"],},
    },
    {
    "name": "search_by_tags",
    "description": "Search books by quality-tags (NOT genres). Returns books from the tagged subset (~5,000 popular books) scoring high on positive tags and optionally low on negative tags. Use when the user asks for specific qualities like 'books with unreliable narrators,' 'slow-burn dark academia,' 'fast-paced sci-fi without romance,' etc. Available tags include pace (fast-paced, slow-burn), mood (dark, atmospheric, melancholic, whimsical), narrator (unreliable-narrator, first-person, multiple-povs), structure (non-linear, epistolary, dual-timeline), style (literary-prose, witty, lyrical, satirical), themes (coming-of-age, grief, identity, found-family, legacy, hubris), experience (page-turner, thought-provoking, mind-bending, comfort-read), and genres (high-fantasy, urban-fantasy, dark-academia, science-fiction, cyberpunk, dystopian, space-opera, climate-fiction, time-travel). Returns books with each book's top contributing tags for explanation. Note: only ~5K popular books have tags, so this won't cover the full catalog.",
    "input_schema": {
        "type": "object",
        "properties": {
            "positive_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags the book SHOULD score high on."
            },
            "negative_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags the book should score LOW on (for excluding qualities)."
            },
            "limit": {"type": "integer"},
            "min_year": {"type": "integer"},
            "max_year": {"type": "integer"},
        },
    },
},

]

TOOL_IMPLS = {
    "search_user_ratings": _tool_search_user_ratings,
    "get_recommendations": _tool_get_recommendations,
    "search_catalog": _tool_search_catalog,
    "search_catalog_semantic": _tool_search_catalog_semantic,
    "search_by_tags": _tool_search_by_tags
}

SYSTEM_PROMPT = """You are a thoughtful book recommendation assistant. You help the user find books to read based on their taste and their requests.

You are strictly a book recommendation assistant. If the user asks for anything unrelated to books or reading (general questions, code, advice on other topics), politely redirect them: "I'm focused on book recommendations — what kind of book are you in the mood for?"
You have four tools:
- search_user_ratings: inspect what the user has rated, with optional semantic query
- get_recommendations: find books similar to anchor books they've rated (best when their taste matches the request)
- search_catalog: look up a specific title
- search_catalog_semantic: search the WHOLE catalog by free-text (genre + qualities), use when the user asks for something outside their rated taste

Decide which path the request needs:

**Path A — request fits their taste** (e.g. "more like my noir books", "something recent like the literary fiction I enjoy"):
1. search_user_ratings WITH a query to find which rated books fit the description.
2. get_recommendations with those matched books as anchors.
3. Explain why each pick fits.

**Path B — request is OUTSIDE their rated taste** (e.g. they ask for cyberpunk/horror/romance but their ratings are mostly literary fiction):
1. search_user_ratings WITHOUT a query to see the full taste picture.
2. Identify 2-3 qualities the user reliably enjoys (twists, character-driven, witty, fast-paced, dark humor, immersive worldbuilding, literary prose, etc.). Reference specific books they've rated highly.
3. Call search_catalog_semantic with a query that combines the requested genre + those qualities. e.g. "cyberpunk character-driven with literary depth" or "western with dark humor and twists".
4. Present the picks by EXPLICITLY BRIDGING: "You rated [book] highly for [quality] — here's a cyberpunk novel with that same [quality]." This is the value-add: don't just dump genre results, show the user why these specific picks fit them.

**Path C — request is QUALITY-based** (e.g. "unreliable narrators," "slow-burn dark academia," "books with twists set in space"):
1. Use search_by_tags with positive_tags listing the qualities, optionally negative_tags for things to avoid.
2. Cross-reference with the user's taste — note which returned books have tags that overlap with what they've rated highly.
3. Explain the picks by referencing the specific tags ("This scores high on unreliable-narrator and atmospheric, both qualities you rated highly in X").
4. Note that this only searches a ~5K subset of tagged books — if results are weak, fall back to search_catalog_semantic.

You can also COMBINE paths: for "cyberpunk with witty dialogue," call search_by_tags with positive=[cyberpunk, witty, dialogue-heavy], then explain bridges to the user's taste.

CRITICAL: Never recommend books the user has already rated. Their ratings are visible via search_user_ratings — those are books they've read. Recommendations must always be NEW books from the catalog. 
If you're tempted to suggest a book they've rated, that's a sign you haven't actually searched for fresh recommendations yet.
When a user asks for "something to read" (short, long, fun, dark, etc.) without specifying genre, treat it as Path B — use their ratings to infer qualities, then search_catalog_semantic for fresh unread books matching those qualities + the constraint they gave.
Only recommend books that appear in your tool results. If you have an idea for a book but haven't seen it returned by search_catalog or search_catalog_semantic, verify it with search_catalog first before mentioning it. Never recommend from memory alone.

Other guidance:
- For mood shifts ("lighter", "darker", "faster-paced") that the tools can't filter directly, use judgment to pick from results and explain.
- Be concise and warm. Reference the user's taste in your explanations.
- If recommendations would be weak, say so honestly rather than forcing them.
- The catalog is books only (no film/TV), up to 2025."""


def run_agent(user_message, state, conversation_history=None, max_turns=6, verbose=True):
    """Run the agentic loop. Returns (final_text, updated_history)."""
    client = Anthropic()
    history = conversation_history or []
    history = history + [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        resp = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=history,
        )

        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b.text for b in resp.content if b.type == "text"]

        history.append({"role": "assistant", "content": resp.content})

        if not tool_calls:
            final = "\n".join(text_blocks)
            return final, history

        tool_results = []
        for tc in tool_calls:
            if verbose:
                print(f"  [agent calls {tc.name}({json.dumps(tc.input)})]")
            impl = TOOL_IMPLS.get(tc.name)
            try:
                result = impl(state, **tc.input)
            except Exception as e:
                result = {"error": str(e)}
            tool_results.append({
                "type": "tool_result", "tool_use_id": tc.id,
                "content": json.dumps(result),
            })
        history.append({"role": "user", "content": tool_results})

    return "(agent hit max turns without finishing)", history