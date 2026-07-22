"""
TV Recommendation Engine — v1 (standalone, local build)

Content-based TV-series recommender that mirrors app_movies.py architecture:
rate shows -> taste profile -> round-robin recommendations across liked shows.

This is the TV equivalent of the movies build. Kept standalone for now so it can
be tested in isolation; tabbing it together with books+movies is the next step.

Schema differences from MOVIES (handled throughout):
  id            -> tmdb_id
  title         -> name
  release_year  -> start_year
  director      -> created_by   (showrunner; multi-valued "A, B"; ~52% coverage)
  studio        -> networks     (HBO/Netflix/...; ~96% coverage)
  cast          -> (none; no IMDb credit join in v1) -> dropped
  movie_tags    -> (none; no TV genome in v1)         -> tag layer dropped
  min_votes 400 -> 20           (TV vote counts run far lower; p95=55, p99=480)
  + seasons shown in the rec card (TV-specific field)

Run from REPO ROOT (use python -m to dodge the stale venv streamlit shim):
    python -m streamlit run src/app/app_tv.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import faiss

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file (local build; loads from disk)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CORPUS_PATH = PROCESSED_DIR / "tv_corpus.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "tv_embeddings_minilm_v1.npy"
FAISS_PATH = PROCESSED_DIR / "tv_faiss_minilm_v1.index"

# ---------------------------------------------------------------------------
# Page config + styling (same identity as the movies app, TV vernacular)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TV Recommendation Engine", page_icon="📺", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #14110f; }
    h1, h2, h3 { color: #f4f1ea; font-family: Georgia, 'Times New Roman', serif; }
    .show-card {
        background: #1f1b18; border: 1px solid #34302b; border-radius: 8px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    .show-title { font-size: 1.05rem; font-weight: 600; color: #f4f1ea; }
    .show-lang { color: #c9a86a; font-size: 0.9rem; }
    .show-meta { color: #8a8178; font-size: 0.8rem; margin-top: 4px; }
    .because { color: #9bbf9b; font-size: 0.82rem; font-style: italic; margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ISO 639-1 -> display name for the languages most common in the corpus
LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "ja": "Japanese",
    "it": "Italian", "de": "German", "ru": "Russian", "zh": "Chinese",
    "pt": "Portuguese", "ko": "Korean", "hi": "Hindi", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "cn": "Chinese", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "no": "Norwegian", "fa": "Persian", "ar": "Arabic",
}


def lang_label(code) -> str:
    if not isinstance(code, str):
        return ""
    return LANG_NAMES.get(code, code.upper())


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_corpus():
    df = pd.read_parquet(CORPUS_PATH)
    df = df.reset_index(drop=True)
    return df


@st.cache_resource
def load_embeddings():
    return np.load(EMBEDDINGS_PATH).astype("float32")


@st.cache_resource
def load_index():
    return faiss.read_index(str(FAISS_PATH))


# words too common to count as a meaningful title collision
_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with",
    "is", "it", "his", "her", "my", "our", "your", "part", "ii", "iii", "iv",
    "2", "3", "4", "show", "series", "season",
}


def _title_tokens(title: str) -> set:
    """Significant lowercased word tokens in a title (stopwords/short removed)."""
    if not isinstance(title, str):
        return set()
    toks = set()
    for w in title.lower().replace(":", " ").replace("-", " ").split():
        w = "".join(ch for ch in w if ch.isalnum())
        if len(w) >= 3 and w not in _TITLE_STOPWORDS:
            toks.add(w)
    return toks


def _creator_names(s) -> set:
    """Split a multi-valued created_by string ('A, B') into a name set."""
    if not isinstance(s, str) or not s.strip():
        return set()
    return {p.strip().lower() for p in s.split(",") if p.strip()}


# ---------------------------------------------------------------------------
# Recommender (round-robin across liked shows; creator-exclusion + title penalty)
# ---------------------------------------------------------------------------
def recommend(ratings, df, embeddings, index, n=20,
              min_votes=20, like_threshold=3.5, per_show=2,
              exclude_rated_creators=True, title_token_penalty=0.15):
    liked = [r for r in ratings if r["rating"] >= like_threshold]
    liked = sorted(liked, key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame()

    liked_vectors = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    rated_tv_ids = {str(t) for t in {r["tv_id"] for r in ratings}}

    # union of individual creator names across all rated shows
    rated_creators = set()
    for r in ratings:
        rated_creators |= _creator_names(r.get("matched_creator"))

    k_per_show = 100
    all_sims, all_idxs = index.search(liked_vectors, k_per_show)

    has_creator = "created_by" in df.columns

    per_show_queues = []
    for like_i, (sims, idxs) in enumerate(zip(all_sims, all_idxs)):
        seed_tokens = _title_tokens(liked[like_i].get("matched_title", ""))
        queue = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:                      # FAISS padding guard
                continue
            show = df.iloc[idx]
            if str(show["tmdb_id"]) in rated_tv_ids:
                continue
            if int(show["vote_count"]) < min_votes:
                continue
            if exclude_rated_creators and has_creator and rated_creators:
                if _creator_names(show["created_by"]) & rated_creators:
                    continue

            final_sim = float(sim)

            # Down-rank candidates that share a significant title word with the
            # seed show — catches coincidental title collisions ("Game of Thrones"
            # -> "The Real War of Thrones") that the title-in-embedding inflates.
            if title_token_penalty and seed_tokens:
                if seed_tokens & _title_tokens(show["name"]):
                    final_sim -= title_token_penalty

            queue.append((int(idx), final_sim))
        queue.sort(key=lambda x: -x[1])
        per_show_queues.append({"like": liked[like_i], "queue": queue})

    results = []
    seen_idxs = set()
    while len(results) < n:
        added = 0
        for psq in per_show_queues:
            taken = 0
            while psq["queue"] and taken < per_show:
                idx, sim = psq["queue"].pop(0)
                if idx in seen_idxs:
                    continue
                seen_idxs.add(idx)
                show = df.iloc[idx]
                genres_str = show["genres"] if isinstance(show["genres"], str) else ""
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
                seasons = (int(show["number_of_seasons"])
                           if "number_of_seasons" in df.columns
                           and pd.notna(show["number_of_seasons"]) else 0)
                results.append({
                    "name": show["name"],
                    "creator": show["created_by"] if has_creator and isinstance(show["created_by"], str) else "",
                    "network": show["networks"] if "networks" in df.columns and isinstance(show["networks"], str) else "",
                    "language": lang_label(show["original_language"]),
                    "genres": genres,
                    "year": show["start_year"],
                    "seasons": seasons,
                    "vote_count": int(show["vote_count"]),
                    "vote_average": float(show["vote_average"]) if pd.notna(show["vote_average"]) else None,
                    "similarity": sim,
                    "because_of": psq["like"]["matched_title"],
                })
                taken += 1
                added += 1
                if len(results) >= n:
                    break
            if len(results) >= n:
                break
        if added == 0:
            break
    return pd.DataFrame(results)


import unicodedata
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Lowercase, strip accents, drop punctuation -> bare alphanumerics + spaces.
    'Marvel's Daredevil' -> 'marvels daredevil'; 'Pokémon' -> 'pokemon'."""
    if not isinstance(text, str):
        return ""
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    text = text.lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace() or ch in "-:/&":
            out.append(" ")
    return " ".join("".join(out).split())


@st.cache_resource
def build_search_index(_df):
    """Precompute normalized titles once (keyed off df identity via cache)."""
    return _df["name"].apply(_normalize).tolist()


def search_corpus(query, df, norm_titles, limit=12):
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    q = _normalize(query)
    if not q:
        return df.iloc[0:0]
    q_words = q.split()

    norm = pd.Series(norm_titles, index=df.index)

    # Tier 1: exact normalized substring (fast, most precise)
    exact = norm.str.contains(q, na=False, regex=False)
    if exact.any():
        return df[exact].nlargest(limit, "vote_count")

    # Tier 2: all query words present somewhere in the title (order-independent)
    mask = pd.Series(True, index=df.index)
    for w in q_words:
        mask &= norm.str.contains(w, na=False, regex=False)
    if mask.any():
        return df[mask].nlargest(limit, "vote_count")

    # Tier 3: fuzzy fallback for typos, on a popularity-pruned shortlist.
    shortlist = df.nlargest(8000, "vote_count")
    short_norm = norm.loc[shortlist.index]
    scores = short_norm.apply(lambda t: SequenceMatcher(None, q, t).ratio())
    good = scores[scores >= 0.7].sort_values(ascending=False)
    if len(good):
        idx = good.index[:limit]
        return shortlist.loc[idx]
    return df.iloc[0:0]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def add_rating(tv_id, name, creator, corpus_index):
    rating = st.session_state.get(f"rate_{tv_id}")
    if rating is None:
        return
    if any(r["tv_id"] == tv_id for r in st.session_state.tv_ratings):
        return
    st.session_state.tv_ratings.append({
        "tv_id": tv_id, "matched_title": name,
        "matched_creator": creator,
        "corpus_index": int(corpus_index), "rating": float(rating),
    })


def remove_rating(tv_id):
    st.session_state.tv_ratings = [
        r for r in st.session_state.tv_ratings if r["tv_id"] != tv_id
    ]


def clear_all():
    st.session_state.tv_ratings = []


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
with st.spinner("Loading the TV recommendation engine..."):
    df = load_corpus()
    embeddings = load_embeddings()
    index = load_index()
    norm_titles = build_search_index(df)

if "tv_ratings" not in st.session_state:
    st.session_state.tv_ratings = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("TV Recommendation Engine")
st.caption(f"Content-based TV recommender - {len(df):,} series - learns your taste from the shows you rate")

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("Your taste profile")
    st.button("Clear all", use_container_width=True, on_click=clear_all)
    min_votes = st.slider(
        "Min. votes (popularity floor)", 0, 200, 20, 5,
        help="TV vote counts run low; 20 is a sensible floor for this corpus.",
    )

    st.markdown("**Add a show you've seen:**")
    query = st.text_input("Search by title", key="search_box",
                          label_visibility="collapsed", placeholder="e.g. Breaking Bad")

    if query:
        matches = search_corpus(query, df, norm_titles)
        if len(matches) == 0:
            st.info("No matches found. Try a different title.")
        else:
            already_rated = {r["tv_id"] for r in st.session_state.tv_ratings}
            for pos, show in matches.iterrows():
                if show["tmdb_id"] in already_rated:
                    continue
                sc1, sc2 = st.columns([3, 1.2])
                with sc1:
                    yr = f" ({int(show['start_year'])})" if pd.notna(show['start_year']) else ""
                    subtitle = (show["created_by"]
                                if "created_by" in df.columns and isinstance(show["created_by"], str) and show["created_by"]
                                else lang_label(show["original_language"]))
                    st.markdown(f"**{show['name']}**{yr}  \n*{subtitle}*")
                with sc2:
                    st.selectbox(
                        "Rate", [None, 1, 2, 3, 4, 5],
                        key=f"rate_{show['tmdb_id']}",
                        label_visibility="collapsed",
                        on_change=add_rating,
                        args=(show["tmdb_id"], show["name"],
                              show["created_by"] if "created_by" in df.columns and isinstance(show["created_by"], str) else "",
                              pos),
                    )

    st.markdown("---")
    if st.session_state.tv_ratings:
        st.markdown(f"**{len(st.session_state.tv_ratings)} shows rated:**")
        for r in sorted(st.session_state.tv_ratings, key=lambda x: -x["rating"]):
            rc1, rc2 = st.columns([5, 1])
            with rc1:
                creator_line = f"<span class='show-lang'>{r['matched_creator']}</span><br>" if r.get("matched_creator") else ""
                st.markdown(
                    f"<div class='show-card'><span class='show-title'>{r['matched_title']}</span><br>"
                    f"{creator_line}"
                    f"<span style='color:#c9a86a'>{r['rating']}/5</span></div>",
                    unsafe_allow_html=True,
                )
            with rc2:
                st.button("X", key=f"rm_{r['tv_id']}", on_click=remove_rating, args=(r["tv_id"],))
    else:
        st.info("No shows rated yet. Search above to add some.")

with right:
    st.subheader("Recommended for you")
    if not st.session_state.tv_ratings:
        st.markdown("_Rate a few shows to see recommendations._")
    else:
        liked_count = len([r for r in st.session_state.tv_ratings if r["rating"] >= 3.5])
        if liked_count == 0:
            st.warning("Rate at least one show 4 or higher to get recommendations.")
        else:
            recs = recommend(
                st.session_state.tv_ratings, df, embeddings, index,
                n=20, min_votes=min_votes,
            )
            if len(recs) == 0:
                st.info("No recommendations found. Try rating more shows you enjoyed, or lowering the votes floor.")
            else:
                for _, rec in recs.iterrows():
                    yr = f" - {int(rec['year'])}" if pd.notna(rec['year']) else ""
                    genres = ", ".join(rec["genres"][:2]) if len(rec["genres"]) else ""
                    rating_str = f"{rec['vote_average']:.1f}/10" if rec["vote_average"] else ""
                    seasons_str = (f"{rec['seasons']} season" + ("s" if rec["seasons"] != 1 else "")) if rec["seasons"] else ""
                    headline = rec["creator"] if rec.get("creator") else rec["language"]
                    meta_bits = " &middot; ".join(
                        b for b in [rec.get("network", ""), genres, seasons_str,
                                    f"{rec['vote_count']:,} votes", rating_str] if b
                    )
                    st.markdown(
                        f"<div class='show-card'>"
                        f"<span class='show-title'>{rec['name']}</span>{yr}<br>"
                        f"<span class='show-lang'>{headline}</span>"
                        f"<div class='show-meta'>{meta_bits}</div>"
                        f"<div class='because'>because you liked <b>{rec['because_of']}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )