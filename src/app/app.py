"""
Recommendation Engine — v0
A content-based book recommender that learns your taste from explicit ratings
and recommends books across your distinct interests.
"""
from pathlib import Path
import json
import gzip

import numpy as np
import pandas as pd
import streamlit as st
import faiss

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so it works regardless of CWD
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"

CORPUS_PATH = PROCESSED_DIR / "books_v0_deduped.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings_minilm_v0_deduped.npy"
FAISS_PATH = PROCESSED_DIR / "faiss_minilm_v0_deduped.index"
DEFAULT_RATINGS_PATH = PROCESSED_DIR / "my_matched_ratings.json"
AUTHORS_PATH = RAW_DIR / "goodreads_book_authors.json.gz"

# ---------------------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Recommendation Engine", page_icon="book", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #14110f; }
    h1, h2, h3 { color: #f4f1ea; font-family: Georgia, 'Times New Roman', serif; }
    .book-card {
        background: #1f1b18; border: 1px solid #34302b; border-radius: 8px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    .book-title { font-size: 1.05rem; font-weight: 600; color: #f4f1ea; }
    .book-author { color: #c9a86a; font-size: 0.9rem; }
    .book-meta { color: #8a8178; font-size: 0.8rem; margin-top: 4px; }
    .because { color: #9bbf9b; font-size: 0.82rem; font-style: italic; margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_corpus():
    df = pd.read_parquet(CORPUS_PATH)
    author_id_to_name = {}
    with gzip.open(AUTHORS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            author_id_to_name[a["author_id"]] = a["name"]

    def primary_author(author_ids):
        if author_ids is None or len(author_ids) == 0:
            return ""
        return author_id_to_name.get(author_ids[0], "")

    df["primary_author"] = df["author_ids"].apply(primary_author)
    df = df.reset_index(drop=True)
    return df


@st.cache_resource
def load_embeddings():
    return np.load(EMBEDDINGS_PATH).astype("float32")


@st.cache_resource
def load_index():
    return faiss.read_index(str(FAISS_PATH))


@st.cache_data
def load_default_ratings():
    if DEFAULT_RATINGS_PATH.exists():
        with open(DEFAULT_RATINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Recommender (round-robin across liked books)
# ---------------------------------------------------------------------------
def recommend(ratings, df, embeddings, index, n=20,
              min_ratings=2000, like_threshold=3.5, per_book=2,
              exclude_rated_authors=True):
    liked = [r for r in ratings if r["rating"] >= like_threshold]
    liked = sorted(liked, key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame()

    liked_vectors = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")

    rated_book_ids = {r["book_id"] for r in ratings}
    rated_authors = {r["matched_author"].lower() for r in ratings if r.get("matched_author")}

    k_per_book = 100
    all_sims, all_idxs = index.search(liked_vectors, k_per_book)

    per_book_queues = []
    for like_i, (sims, idxs) in enumerate(zip(all_sims, all_idxs)):
        queue = []
        for sim, idx in zip(sims, idxs):
            book = df.iloc[idx]
            if book["book_id"] in rated_book_ids:
                continue
            if exclude_rated_authors and book["primary_author"].lower() in rated_authors:
                continue
            if book["ratings_count"] < min_ratings:
                continue
            queue.append((int(idx), float(sim)))
        per_book_queues.append({"like": liked[like_i], "queue": queue})

    results = []
    seen_idxs, seen_works = set(), set()
    while len(results) < n:
        added = 0
        for pbq in per_book_queues:
            taken = 0
            while pbq["queue"] and taken < per_book:
                idx, sim = pbq["queue"].pop(0)
                book = df.iloc[idx]
                work_id = book["work_id"]
                if idx in seen_idxs or (work_id and work_id in seen_works):
                    continue
                seen_idxs.add(idx)
                if work_id:
                    seen_works.add(work_id)
                results.append({
                    "title": book["title"], "author": book["primary_author"],
                    "genres": list(book["genres"]) if book["genres"] is not None else [],
                    "year": book["publication_year"], "ratings_count": int(book["ratings_count"]),
                    "similarity": sim, "because_of": pbq["like"]["matched_title"],
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


def search_corpus(query, df, limit=12):
    if not query or len(query) < 2:
        return df.iloc[0:0]
    mask = df["title"].str.contains(query, case=False, na=False, regex=False)
    return df[mask].nlargest(limit, "ratings_count")


# ---------------------------------------------------------------------------
# Callbacks — run BEFORE the rerun, so state is clean
# ---------------------------------------------------------------------------
def add_rating(book_id, title, author, corpus_index):
    rating = st.session_state.get(f"rate_{book_id}")
    if rating is None:
        return
    if any(r["book_id"] == book_id for r in st.session_state.ratings):
        return
    st.session_state.ratings.append({
        "book_id": book_id, "matched_title": title, "matched_author": author,
        "corpus_index": int(corpus_index), "rating": float(rating),
    })


def remove_rating(book_id):
    st.session_state.ratings = [r for r in st.session_state.ratings if r["book_id"] != book_id]


def load_sample():
    st.session_state.ratings = [dict(r) for r in load_default_ratings()]


def clear_all():
    st.session_state.ratings = []


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
with st.spinner("Loading the recommendation engine..."):
    df = load_corpus()
    embeddings = load_embeddings()
    index = load_index()

if "ratings" not in st.session_state:
    st.session_state.ratings = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Recommendation Engine")
st.caption(f"Content-based book recommender - {len(df):,} books - learns your taste from the books you rate")

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("Your taste profile")
    c1, c2 = st.columns(2)
    c1.button("Load sample profile", use_container_width=True, on_click=load_sample)
    c2.button("Clear all", use_container_width=True, on_click=clear_all)

    st.markdown("**Add a book you've read:**")
    query = st.text_input("Search by title", key="search_box",
                          label_visibility="collapsed", placeholder="e.g. The Hobbit")

    if query:
        matches = search_corpus(query, df)
        if len(matches) == 0:
            st.info("No matches found. Try a different title.")
        else:
            already_rated = {r["book_id"] for r in st.session_state.ratings}
            for pos, book in matches.iterrows():
                if book["book_id"] in already_rated:
                    continue
                bc1, bc2 = st.columns([3, 1.2])
                with bc1:
                    yr = f" ({int(book['publication_year'])})" if pd.notna(book['publication_year']) else ""
                    st.markdown(f"**{book['title']}**{yr}  \n*{book['primary_author']}*")
                with bc2:
                    st.selectbox(
                        "Rate", [None, 1, 2, 3, 4, 5],
                        key=f"rate_{book['book_id']}",
                        label_visibility="collapsed",
                        on_change=add_rating,
                        args=(book["book_id"], book["title"], book["primary_author"], pos),
                    )

    st.markdown("---")
    if st.session_state.ratings:
        st.markdown(f"**{len(st.session_state.ratings)} books rated:**")
        for r in sorted(st.session_state.ratings, key=lambda x: -x["rating"]):
            full = int(r["rating"])
            half = (r["rating"] - full) >= 0.5
            stars = "*" * full + ("." if half else "") + "-" * (5 - full - (1 if half else 0))
            rc1, rc2 = st.columns([5, 1])
            with rc1:
                st.markdown(
                    f"<div class='book-card'><span class='book-title'>{r['matched_title']}</span><br>"
                    f"<span class='book-author'>{r['matched_author']}</span> &middot; "
                    f"<span style='color:#c9a86a'>{r['rating']}/5</span></div>",
                    unsafe_allow_html=True,
                )
            with rc2:
                st.button("X", key=f"rm_{r['book_id']}", on_click=remove_rating, args=(r["book_id"],))
    else:
        st.info("No books rated yet. Search above to add some, or load the sample profile.")

with right:
    st.subheader("Recommended for you")
    if not st.session_state.ratings:
        st.markdown("_Rate a few books to see recommendations._")
    else:
        liked_count = len([r for r in st.session_state.ratings if r["rating"] >= 3.5])
        if liked_count == 0:
            st.warning("Rate at least one book 4 or higher to get recommendations.")
        else:
            recs = recommend(st.session_state.ratings, df, embeddings, index, n=20)
            if len(recs) == 0:
                st.info("No recommendations found. Try rating more books you enjoyed.")
            else:
                for _, rec in recs.iterrows():
                    yr = f" - {int(rec['year'])}" if pd.notna(rec['year']) else ""
                    genres = ", ".join(rec["genres"][:2]) if len(rec["genres"]) else ""
                    st.markdown(
                        f"<div class='book-card'>"
                        f"<span class='book-title'>{rec['title']}</span>{yr}<br>"
                        f"<span class='book-author'>{rec['author']}</span>"
                        f"<div class='book-meta'>{genres} &middot; {rec['ratings_count']:,} ratings</div>"
                        f"<div class='because'>because you liked <b>{rec['because_of']}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )