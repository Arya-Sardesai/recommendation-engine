"""
Movie Recommendation Engine — local agent build.
Left: build your taste profile (search + rate films).
Right: chat with the recommendation agent, which uses your ratings + the
recommender tools (incl. director/cast/studio/tag search) to answer
natural-language requests.

Run locally:  streamlit run src/app_agent_movies.py
Requires:  $env:ANTHROPIC_API_KEY = "sk-ant-..."  (set before launching)
Reads local data from data/processed/ (the movie v1 artifacts).
"""
from pathlib import Path
import sys
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
import faiss

# make the agent module importable (it's in src/)
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from agent_movies import run_agent  # noqa: E402

REPO_ROOT = APP_DIR.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CORPUS_PATH = PROCESSED_DIR / "movies.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "movie_embeddings_minilm_v1.npy"
FAISS_PATH = PROCESSED_DIR / "movie_faiss_minilm_v1.index"

st.set_page_config(page_title="Movie Recommendation Engine — Agent", page_icon="clapper", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #14110f; }
    h1, h2, h3 { color: #f4f1ea; font-family: Georgia, 'Times New Roman', serif; }
    .film-card { background: #1f1b18; border: 1px solid #34302b; border-radius: 8px;
                 padding: 12px 14px; margin-bottom: 8px; }
    .film-title { font-size: 1rem; font-weight: 600; color: #f4f1ea; }
    .film-sub { color: #c9a86a; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "ja": "Japanese",
    "it": "Italian", "de": "German", "ru": "Russian", "zh": "Chinese",
    "pt": "Portuguese", "ko": "Korean", "hi": "Hindi", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "th": "Thai", "fa": "Persian",
    "sv": "Swedish", "da": "Danish", "nl": "Dutch", "pl": "Polish",
}


def lang_label(code):
    return LANG_NAMES.get(code, code.upper()) if isinstance(code, str) else ""


@st.cache_resource
def load_all():
    df = pd.read_parquet(CORPUS_PATH).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    index = faiss.read_index(str(FAISS_PATH))
    return df, embeddings, index


# ---- forgiving search (mirrors app_movies) ----
def _normalize(text):
    if not isinstance(text, str):
        return ""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
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
    return _df["title"].apply(_normalize).tolist()


def search_corpus(query, df, norm_titles, limit=12):
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    q = _normalize(query)
    if not q:
        return df.iloc[0:0]
    q_words = q.split()
    norm = pd.Series(norm_titles, index=df.index)
    exact = norm.str.contains(q, na=False, regex=False)
    if exact.any():
        return df[exact].nlargest(limit, "vote_count")
    mask = pd.Series(True, index=df.index)
    for w in q_words:
        mask &= norm.str.contains(w, na=False, regex=False)
    if mask.any():
        return df[mask].nlargest(limit, "vote_count")
    shortlist = df.nlargest(8000, "vote_count")
    short_norm = norm.loc[shortlist.index]
    scores = short_norm.apply(lambda t: SequenceMatcher(None, q, t).ratio())
    good = scores[scores >= 0.7].sort_values(ascending=False)
    if len(good):
        return shortlist.loc[good.index[:limit]]
    return df.iloc[0:0]


# --- callbacks ---
def add_rating(movie_id, title, director, corpus_index):
    rating = st.session_state.get(f"rate_{movie_id}")
    if rating is None:
        return
    if any(r["movie_id"] == movie_id for r in st.session_state.movie_ratings):
        return
    st.session_state.movie_ratings.append({
        "movie_id": movie_id, "matched_title": title, "matched_director": director,
        "corpus_index": int(corpus_index), "rating": float(rating),
    })

def remove_rating(movie_id):
    st.session_state.movie_ratings = [r for r in st.session_state.movie_ratings if r["movie_id"] != movie_id]

def clear_all():
    st.session_state.movie_ratings = []
    st.session_state.chat = []
    st.session_state.agent_history = []


df, embeddings, index = load_all()
norm_titles = build_search_index(df)

if "movie_ratings" not in st.session_state:
    st.session_state.movie_ratings = []
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role","content"} for display
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # full message history (multi-turn memory)

st.title("Movie Recommendation Engine")
st.caption(f"{len(df):,} films · rate films on the left, ask the agent on the right")

left, right = st.columns([1, 1.4], gap="large")

# ----- LEFT: taste profile -----
with left:
    st.subheader("Your taste profile")
    st.button("Clear all", use_container_width=True, on_click=clear_all)

    query = st.text_input("Search a film to rate", key="search_box",
                          placeholder="e.g. Inception")
    if query:
        matches = search_corpus(query, df, norm_titles)
        already = {r["movie_id"] for r in st.session_state.movie_ratings}
        for pos, film in matches.iterrows():
            if film["id"] in already:
                continue
            fc1, fc2 = st.columns([3, 1.2])
            with fc1:
                yr = f" ({int(film['release_year'])})" if pd.notna(film['release_year']) else ""
                director = film["director"] if "director" in df.columns and isinstance(film["director"], str) and film["director"] else lang_label(film["original_language"])
                st.markdown(f"**{film['title']}**{yr}  \n*{director}*")
            with fc2:
                st.selectbox("Rate", [None, 1, 2, 3, 4, 5],
                             key=f"rate_{film['id']}", label_visibility="collapsed",
                             on_change=add_rating,
                             args=(film["id"], film["title"],
                                   film["director"] if "director" in df.columns and isinstance(film["director"], str) else "", pos))

    st.markdown("---")
    if st.session_state.movie_ratings:
        st.markdown(f"**{len(st.session_state.movie_ratings)} films rated:**")
        for r in sorted(st.session_state.movie_ratings, key=lambda x: -x["rating"]):
            rc1, rc2 = st.columns([5, 1])
            dline = f"<span class='film-sub'>{r['matched_director']}</span> · " if r.get("matched_director") else ""
            rc1.markdown(
                f"<div class='film-card'><span class='film-title'>{r['matched_title']}</span><br>"
                f"{dline}<span style='color:#c9a86a'>{r['rating']}/5</span></div>",
                unsafe_allow_html=True)
            rc2.button("X", key=f"rm_{r['movie_id']}", on_click=remove_rating, args=(r["movie_id"],))
    else:
        st.info("Rate some films to get started.")

# ----- RIGHT: agent chat -----
with right:
    st.subheader("Ask the agent")
    st.caption("Try: \"Korean revenge thrillers like the crime films I like\", \"more A24\", or \"films by Bong Joon-ho\"")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("What are you in the mood to watch?")
    if prompt:
        if not st.session_state.movie_ratings:
            st.warning("Rate a few films on the left first so the agent knows your taste.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    state = {"df": df, "embeddings": embeddings, "index": index,
                             "ratings": st.session_state.movie_ratings}
                    try:
                        reply, st.session_state.agent_history = run_agent(
                            prompt, state,
                            conversation_history=st.session_state.agent_history,
                            verbose=False,
                        )
                    except Exception as e:
                        reply = f"Something went wrong: {e}"
                st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})