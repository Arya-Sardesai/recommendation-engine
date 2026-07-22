"""
Movie Recommendation Engine — v1 (standalone, local build)

Content-based film recommender that mirrors the books app architecture:
rate films -> taste profile -> round-robin recommendations across liked films,
with optional tag-based re-ranking from movie_tags.parquet.

This is the MOVIES equivalent of the books deployment build. Kept standalone
for now so the movie recommender can be tested in isolation; tabbing it together
with books is the next step.

Schema differences from books (handled throughout):
  book_id          -> id
  publication_year -> release_year
  ratings_count    -> vote_count
  primary_author   -> (none; films have no author field) -> show language instead
  work_id          -> (none; dedup on id)
  genres (list)    -> genres (comma-joined string)
  source/25x scale -> (none; not applicable to movies)
  book_tags.book_id-> movie_tags.movie_id

Run from REPO ROOT:
    streamlit run src/app/app_movies.py
"""
from pathlib import Path
import json

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

CORPUS_PATH = PROCESSED_DIR / "movies.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "movie_embeddings_minilm_v1.npy"
FAISS_PATH = PROCESSED_DIR / "movie_faiss_minilm_v1.index"
TAGS_PATH = PROCESSED_DIR / "movie_tags.parquet"

# ---------------------------------------------------------------------------
# Page config + styling (same identity as the books app, film vernacular)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommendation Engine", page_icon="clapper", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #14110f; }
    h1, h2, h3 { color: #f4f1ea; font-family: Georgia, 'Times New Roman', serif; }
    .film-card {
        background: #1f1b18; border: 1px solid #34302b; border-radius: 8px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    .film-title { font-size: 1.05rem; font-weight: 600; color: #f4f1ea; }
    .film-lang { color: #c9a86a; font-size: 0.9rem; }
    .film-meta { color: #8a8178; font-size: 0.8rem; margin-top: 4px; }
    .because { color: #9bbf9b; font-size: 0.82rem; font-style: italic; margin-top: 6px; }
    .tag-hint { color: #7a9ec9; font-size: 0.78rem; margin-top: 4px; }
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


@st.cache_resource
def load_tags():
    """Load tag scores, pivot to wide (movie_id index, tag columns)."""
    try:
        tdf = pd.read_parquet(TAGS_PATH)
        tdf["movie_id"] = tdf["movie_id"].astype(str)
        wide = tdf.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)
        return wide
    except Exception as e:
        print(f"Tag data unavailable: {e}")
        return None


# words too common to count as a meaningful title collision
_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with",
    "is", "it", "his", "her", "my", "our", "your", "part", "ii", "iii", "iv",
    "2", "3", "4", "movie", "film",
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


# ---------------------------------------------------------------------------
# Recommender (round-robin across liked films, optional tag re-ranking)
# ---------------------------------------------------------------------------
def recommend(ratings, df, embeddings, index, tags_df=None, n=20,
              min_votes=400, like_threshold=3.5, per_film=2, tag_weight=0.3,
              exclude_rated_directors=True, title_token_penalty=0.15):
    liked = [r for r in ratings if r["rating"] >= like_threshold]
    liked = sorted(liked, key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame(), None

    liked_vectors = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    rated_movie_ids = {str(m) for m in {r["movie_id"] for r in ratings}}
    rated_directors = {
        r["matched_director"].lower()
        for r in ratings if r.get("matched_director")
    }

    # Build user's tag profile from liked films that are tagged
    user_tag_profile = None
    top_user_tags = None
    if tags_df is not None:
        liked_ids = [str(r["movie_id"]) for r in liked]
        present = [mid for mid in liked_ids if mid in tags_df.index]
        if present:
            ratings_by_id = {str(r["movie_id"]): r["rating"] for r in liked}
            weights = np.array([ratings_by_id[mid] for mid in present])
            liked_tagged = tags_df.loc[present]
            user_tag_profile = (liked_tagged.T.values @ weights) / weights.sum()
            top_user_tags = pd.Series(user_tag_profile, index=tags_df.columns).nlargest(5)

    k_per_film = 100
    all_sims, all_idxs = index.search(liked_vectors, k_per_film)

    has_director = "director" in df.columns

    per_film_queues = []
    for like_i, (sims, idxs) in enumerate(zip(all_sims, all_idxs)):
        seed_tokens = _title_tokens(liked[like_i].get("matched_title", ""))
        queue = []
        for sim, idx in zip(sims, idxs):
            film = df.iloc[idx]
            if str(film["id"]) in rated_movie_ids:
                continue
            if int(film["vote_count"]) < min_votes:
                continue
            if (exclude_rated_directors and has_director
                    and isinstance(film["director"], str) and film["director"]
                    and film["director"].lower() in rated_directors):
                continue

            final_sim = float(sim)
            tag_boost_applied = False
            if user_tag_profile is not None and str(film["id"]) in tags_df.index:
                film_tags = tags_df.loc[str(film["id"])].values
                norm_u = np.linalg.norm(user_tag_profile)
                norm_f = np.linalg.norm(film_tags)
                if norm_u > 0 and norm_f > 0:
                    tag_sim = float(np.dot(user_tag_profile, film_tags) / (norm_u * norm_f))
                    final_sim = (1 - tag_weight) * float(sim) + tag_weight * tag_sim
                    tag_boost_applied = True

            # Down-rank candidates that share a significant title word with the
            # seed film — catches coincidental title collisions ("Heat" ->
            # "City Heat") that the title-in-embedding inflates.
            if title_token_penalty and seed_tokens:
                if seed_tokens & _title_tokens(film["title"]):
                    final_sim -= title_token_penalty

            queue.append((int(idx), final_sim, tag_boost_applied))
        queue.sort(key=lambda x: -x[1])
        per_film_queues.append({"like": liked[like_i], "queue": queue})

    results = []
    seen_idxs = set()
    while len(results) < n:
        added = 0
        for pfq in per_film_queues:
            taken = 0
            while pfq["queue"] and taken < per_film:
                idx, sim, tag_boost = pfq["queue"].pop(0)
                if idx in seen_idxs:
                    continue
                seen_idxs.add(idx)
                film = df.iloc[idx]
                genres_str = film["genres"] if isinstance(film["genres"], str) else ""
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
                results.append({
                    "title": film["title"],
                    "director": film["director"] if has_director and isinstance(film["director"], str) else "",
                    "cast": list(film["cast"]) if "cast" in df.columns and film["cast"] is not None else [],
                    "studio": film["studio"] if "studio" in df.columns and isinstance(film["studio"], str) else "",
                    "language": lang_label(film["original_language"]),
                    "genres": genres,
                    "year": film["release_year"],
                    "vote_count": int(film["vote_count"]),
                    "vote_average": float(film["vote_average"]) if pd.notna(film["vote_average"]) else None,
                    "similarity": sim,
                    "because_of": pfq["like"]["matched_title"],
                    "tag_boost": tag_boost,
                })
                taken += 1
                added += 1
                if len(results) >= n:
                    break
            if len(results) >= n:
                break
        if added == 0:
            break
    return pd.DataFrame(results), top_user_tags


import unicodedata
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Lowercase, strip accents, drop punctuation -> bare alphanumerics + spaces.
    'Ocean's Eleven' -> 'oceans eleven'; 'Amélie' -> 'amelie'."""
    if not isinstance(text, str):
        return ""
    # strip accents
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    text = text.lower()
    # punctuation -> nothing (so apostrophes/hyphens vanish), keep spaces
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace() or ch in "-:/&":
            out.append(" ")
        # other punctuation (apostrophes, periods, commas) dropped entirely
    return " ".join("".join(out).split())


@st.cache_resource
def build_search_index(_df):
    """Precompute normalized titles once (keyed off df identity via cache)."""
    return _df["title"].apply(_normalize).tolist()


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

    # Tier 3: fuzzy fallback for typos. Only on a popularity-pruned shortlist
    # (top 8000 by votes) to keep it fast over 109K titles.
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
def add_rating(movie_id, title, director, corpus_index):
    rating = st.session_state.get(f"rate_{movie_id}")
    if rating is None:
        return
    if any(r["movie_id"] == movie_id for r in st.session_state.movie_ratings):
        return
    st.session_state.movie_ratings.append({
        "movie_id": movie_id, "matched_title": title,
        "matched_director": director,
        "corpus_index": int(corpus_index), "rating": float(rating),
    })


def remove_rating(movie_id):
    st.session_state.movie_ratings = [
        r for r in st.session_state.movie_ratings if r["movie_id"] != movie_id
    ]


def clear_all():
    st.session_state.movie_ratings = []


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
with st.spinner("Loading the movie recommendation engine..."):
    df = load_corpus()
    embeddings = load_embeddings()
    index = load_index()
    tags_df = load_tags()
    norm_titles = build_search_index(df)

if "movie_ratings" not in st.session_state:
    st.session_state.movie_ratings = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Movie Recommendation Engine")
tag_status = f" - {len(tags_df):,} tagged for quality re-ranking" if tags_df is not None else ""
st.caption(f"Content-based film recommender - {len(df):,} films{tag_status} - learns your taste from the films you rate")

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("Your taste profile")
    st.button("Clear all", use_container_width=True, on_click=clear_all)

    st.markdown("**Add a film you've seen:**")
    query = st.text_input("Search by title", key="search_box",
                          label_visibility="collapsed", placeholder="e.g. Inception")

    if query:
        matches = search_corpus(query, df, norm_titles)
        if len(matches) == 0:
            st.info("No matches found. Try a different title.")
        else:
            already_rated = {r["movie_id"] for r in st.session_state.movie_ratings}
            for pos, film in matches.iterrows():
                if film["id"] in already_rated:
                    continue
                fc1, fc2 = st.columns([3, 1.2])
                with fc1:
                    yr = f" ({int(film['release_year'])})" if pd.notna(film['release_year']) else ""
                    director = film["director"] if "director" in df.columns and isinstance(film["director"], str) and film["director"] else lang_label(film["original_language"])
                    st.markdown(f"**{film['title']}**{yr}  \n*{director}*")
                with fc2:
                    st.selectbox(
                        "Rate", [None, 1, 2, 3, 4, 5],
                        key=f"rate_{film['id']}",
                        label_visibility="collapsed",
                        on_change=add_rating,
                        args=(film["id"], film["title"],
                              film["director"] if "director" in df.columns and isinstance(film["director"], str) else "",
                              pos),
                    )

    st.markdown("---")
    if st.session_state.movie_ratings:
        st.markdown(f"**{len(st.session_state.movie_ratings)} films rated:**")
        for r in sorted(st.session_state.movie_ratings, key=lambda x: -x["rating"]):
            rc1, rc2 = st.columns([5, 1])
            with rc1:
                director_line = f"<span class='film-lang'>{r['matched_director']}</span><br>" if r.get("matched_director") else ""
                st.markdown(
                    f"<div class='film-card'><span class='film-title'>{r['matched_title']}</span><br>"
                    f"{director_line}"
                    f"<span style='color:#c9a86a'>{r['rating']}/5</span></div>",
                    unsafe_allow_html=True,
                )
            with rc2:
                st.button("X", key=f"rm_{r['movie_id']}", on_click=remove_rating, args=(r["movie_id"],))
    else:
        st.info("No films rated yet. Search above to add some.")

with right:
    st.subheader("Recommended for you")
    if not st.session_state.movie_ratings:
        st.markdown("_Rate a few films to see recommendations._")
    else:
        liked_count = len([r for r in st.session_state.movie_ratings if r["rating"] >= 3.5])
        if liked_count == 0:
            st.warning("Rate at least one film 4 or higher to get recommendations.")
        else:
            recs, top_user_tags = recommend(
                st.session_state.movie_ratings, df, embeddings, index,
                tags_df=tags_df, n=20,
            )
            if top_user_tags is not None and len(top_user_tags) > 0:
                tag_list = ", ".join(top_user_tags.index[:5])
                st.markdown(
                    f"<div class='tag-hint'>Inferred taste signature: {tag_list}</div>",
                    unsafe_allow_html=True,
                )

            if len(recs) == 0:
                st.info("No recommendations found. Try rating more films you enjoyed.")
            else:
                for _, rec in recs.iterrows():
                    yr = f" - {int(rec['year'])}" if pd.notna(rec['year']) else ""
                    genres = ", ".join(rec["genres"][:2]) if len(rec["genres"]) else ""
                    rating_str = f"{rec['vote_average']:.1f}/10" if rec["vote_average"] else ""
                    # headline line: director if known, else language
                    headline = rec["director"] if rec.get("director") else rec["language"]
                    meta_bits = " &middot; ".join(
                        b for b in [rec.get("studio", ""), genres,
                                    f"{rec['vote_count']:,} votes", rating_str] if b
                    )
                    tag_badge = " &middot; <span style='color:#7a9ec9'>tag-matched</span>" if rec.get("tag_boost") else ""
                    cast_line = ""
                    if rec.get("cast"):
                        cast_line = f"<div class='film-meta' style='color:#9a9088'>with {', '.join(rec['cast'][:4])}</div>"
                    st.markdown(
                        f"<div class='film-card'>"
                        f"<span class='film-title'>{rec['title']}</span>{yr}<br>"
                        f"<span class='film-lang'>{headline}</span>"
                        f"<div class='film-meta'>{meta_bits}{tag_badge}</div>"
                        f"{cast_line}"
                        f"<div class='because'>because you liked <b>{rec['because_of']}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )