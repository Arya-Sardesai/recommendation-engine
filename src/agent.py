"""
Recommendation agent — multi-step reasoning over the book recommender.

The agent (Claude) is given tools that wrap the existing recommender. It can:
  - inspect the user's ratings (semantic search over what they've rated)
  - get recommendations from anchor books, with optional year filters
  - look up books in the catalog
  - search by quality-tags across the near-full catalog (LLM + kNN-propagated)

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
        model = SentenceTransformer("BAAI/bge-m3", device=device)
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
            if ix < 0:        # FAISS pads with -1 when fewer neighbors than k exist
                continue
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
        if ix < 0:        # FAISS pads with -1 when fewer neighbors than k exist
            continue
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
    """Search by quality-tags across the FULL catalog: LLM tags (~33K books,
    weight 1.0) + kNN-propagated tags (~397K books, weight 0.5). Propagated
    tags are inference (holdout precision 0.541), so they rank lower and are
    marked as inferred in the output."""
    df = state["df"]
    ratings = state["ratings"]

    KNN_WEIGHT = 0.5

    if "tags_df" not in state:
        from pathlib import Path
        llm_path = Path("data/processed/book_tags.parquet")
        knn_path = Path("data/processed/book_tags_knn.parquet")
        if not llm_path.exists():
            return {"results": [], "note": "No tag data available."}
        llm = pd.read_parquet(llm_path)
        llm["source"] = "llm"
        llm["book_id"] = llm["book_id"].astype(str)
        if knn_path.exists():
            knn = pd.read_parquet(knn_path)[["book_id", "tag", "score"]]
            knn["source"] = "knn"
            knn["book_id"] = knn["book_id"].astype(str)
            knn["score"] = knn["score"] * KNN_WEIGHT
            both = pd.concat([llm, knn], ignore_index=True)
            # LLM wins on any (book_id, tag) collision. Make the priority
            # EXPLICIT rather than relying on alphabetical accident:
            both["_prio"] = (both["source"] == "llm").astype(int)
            both = (both.sort_values("_prio")
                        .drop_duplicates(subset=["book_id", "tag"], keep="last")
                        .drop(columns="_prio"))
        else:
            both = llm
        state["tags_df"] = both.pivot_table(
            index="book_id", columns="tag", values="score", fill_value=0.0
        ).astype("float32")
        # (book_id, tag) -> source, for marking inferred tags in output
        state["tag_source"] = {(str(b), t): s for b, t, s in
                               zip(both.book_id, both.tag, both.source)}
        # book_id -> positional row, so ranked iteration isn't O(n) per lookup
        state["bookid_pos"] = {str(b): i for i, b in enumerate(df["book_id"])}

    tags_df = state["tags_df"]
    tag_source = state["tag_source"]
    bookid_pos = state["bookid_pos"]

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

    rated_ids = {str(r["book_id"]) for r in ratings}
    rated_authors = {r.get("matched_author", "").lower() for r in ratings if r.get("matched_author")}
    ranked = score.sort_values(ascending=False)

    results = []
    for book_id, s in ranked.items():
        if s <= 0:
            break                      # scores only fall from here; nothing left
        if str(book_id) in rated_ids:
            continue
        pos_ix = bookid_pos.get(str(book_id))
        if pos_ix is None:
            continue
        b = df.iloc[pos_ix]
        if b["primary_author"].lower() in rated_authors:
            continue
        if b["ratings_count"] < 2000:
            continue
        yr = b["publication_year"]
        if min_year is not None and (pd.isna(yr) or yr < min_year):
            continue
        if max_year is not None and (pd.isna(yr) or yr > max_year):
            continue

        book_tags = tags_df.loc[book_id].nlargest(5)
        top_tags = [(t + (" (inferred)" if tag_source.get((str(book_id), t)) == "knn" else ""),
                     round(float(v), 2))
                    for t, v in book_tags.items() if v > 0]

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
        "note": (f"Searched {len(tags_df):,} tagged books (near-full catalog). "
                 "Tags marked (inferred) come from embedding-neighbour propagation "
                 "and are less certain than unmarked LLM-assigned tags."),
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
            "required": ["query"],
        },
    },
    {
        "name": "search_by_tags",
        "description": "Search books by quality-tags (NOT genres). Covers nearly the FULL catalog: ~33K popular books carry high-confidence LLM-assigned tags, and ~397K more carry inferred tags propagated from embedding neighbours (marked '(inferred)' in results, ranked lower). Use when the user asks for specific qualities like 'books with unreliable narrators,' 'slow-burn dark academia,' 'fast-paced sci-fi without romance,' etc. Available tags include pace (fast-paced, slow-burn), mood (dark, atmospheric, melancholic, whimsical), narrator (unreliable-narrator, first-person, multiple-povs), structure (non-linear, epistolary, dual-timeline), style (literary-prose, witty, lyrical, satire), themes (coming-of-age, grief, identity, found-family, legacy, hubris), experience (page-turner, thought-provoking, mind-bending, comfort-read), and genres (high-fantasy, urban-fantasy, dark-academia, science-fiction, cyberpunk, dystopian, space-opera, climate-fiction, time-travel). Returns books with each book's top contributing tags for explanation. Inferred tags are reliable for broad qualities (dark, romance, thriller, psychological) and weak for rare structural ones (epistolary, second-person, time-loop) — for those, results will skew to the LLM-tagged subset, which is correct.",
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
You have five tools:
- search_user_ratings: inspect what the user has rated, with optional semantic query
- get_recommendations: find books similar to anchor books they've rated (best when their taste matches the request)
- search_catalog: look up a specific title
- search_catalog_semantic: search the WHOLE catalog by free-text (genre + qualities), use when the user asks for something outside their rated taste
- search_by_tags: search by quality-tags across nearly the full catalog

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
4. Tag search now covers nearly the full catalog. Unmarked tags are high-confidence; tags marked (inferred) are propagated and less certain — when a pick rests mainly on inferred tags, soften the claim ("this looks like a slow-burn dark academia pick") rather than asserting it. If results are weak, fall back to search_catalog_semantic.

You can also COMBINE paths: for "cyberpunk with witty dialogue," call search_by_tags with positive=[cyberpunk, witty, dialogue-heavy], then explain bridges to the user's taste.

CRITICAL — GROUNDING RULE (read carefully):
Every single book you name in a recommendation MUST appear, by title, in a tool result you received in THIS conversation. This is an absolute rule:
- Do NOT add books from your own knowledge, even if they fit the request perfectly and you are confident they exist. If a great book comes to mind but it is not in your tool results, you may NOT name it — instead, run another tool search to try to surface it.
- Never recommend books the user has already rated (visible via search_user_ratings — those are books they've read).
- If your tool results don't contain enough good matches, say so honestly and offer to search differently, rather than padding the list with books from memory.
- Prefer the EXACT titles as they appear in the tool results. Do not rename, retranslate, or "correct" them.
- When in doubt about whether a book is in the catalog, call search_catalog to verify before naming it.
The recommender's entire value is that it is grounded in this specific catalog and the user's actual ratings. A recommendation the tools didn't return is worse than no recommendation — it breaks trust and may point to a book that isn't in the catalog at all.

When a user asks for "something to read" (short, long, fun, dark, etc.) without specifying genre, treat it as Path B — use their ratings to infer qualities, then search_catalog_semantic for fresh unread books matching those qualities + the constraint they gave.

Other guidance:
- For mood shifts ("lighter", "darker", "faster-paced") that the tools can't filter directly, use judgment to pick from results and explain.
- Be concise and warm. Reference the user's taste in your explanations.
- If recommendations would be weak, say so honestly rather than forcing them.
- The catalog is books only (no film/TV), up to 2025."""


def _titles_from_result(result):
    """Pull every book title out of a tool result dict (any tool's shape)."""
    titles = set()
    if not isinstance(result, dict):
        return titles
    for key in ("recommendations", "results", "matched_books", "ratings"):
        items = result.get(key)
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("title"):
                    titles.add(it["title"])
    return titles


def _check_faithfulness(reply, seen_titles):
    """Find book titles in the reply that were never returned by any tool.

    Precision matters: we only treat a phrase as a *title* if it is shaped like
    one — immediately followed by a parenthetical containing a 4-digit year
    and/or an author, e.g. "Piranesi (Susanna Clarke, 2020)" or "Dune (1965)".
    This avoids flagging prose fragments that a looser bold/numbered-list scan
    would wrongly catch. Returns a list of suspicious (likely-hallucinated) titles.
    """
    import re

    pattern = re.compile(
        r"(?:\*\*)?"                       # optional opening bold
        r"([A-Z0-9][^\n*()]{1,70}?)"       # the title text (starts capitalized)
        r"(?:\*\*)?"                       # optional closing bold
        r"\s*\((?:[^)]*?\b(?:19|20)\d{2}\b[^)]*)\)"  # ( ... yyyy ... )
    )
    cands = [m.group(1).strip().rstrip(":–-—").strip() for m in pattern.finditer(reply)]
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
    checks whether the answer names any book the tools never returned (a sign of
    the model freelancing from memory). When verbose, prints returned titles and
    any suspicious (likely-hallucinated) ones.
    """
    client = Anthropic()
    history = conversation_history or []
    history = history + [{"role": "user", "content": user_message}]

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
                print(f"  [!] WARNING: {len(suspicious)} title(s) in the reply were "
                      f"NOT in any tool result (possible hallucination):")
                for s in suspicious:
                    print(f"        - {s}")
            elif verbose:
                print(f"  [ok] all recommended titles were grounded in tool results "
                      f"({len(seen_titles)} books seen across tools)")
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
                    print(f"        -> returned {len(returned)} books: {', '.join(preview)}{more}")
                else:
                    print(f"        -> (no book titles in result)")
            tool_results.append({
                "type": "tool_result", "tool_use_id": tc.id,
                "content": json.dumps(result),
            })
        history.append({"role": "user", "content": tool_results})

    return "(agent hit max turns without finishing)", history
