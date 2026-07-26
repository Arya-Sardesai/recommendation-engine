"""
TV Recommendation Engine — local agent build.
Left: build your taste profile (search + rate series).
Right: chat with the recommendation agent, which uses your ratings + the
recommender tools (incl. tag/format search) to answer natural-language requests.

Run locally:  python -m streamlit run src/app_agent_tv.py
Requires:  $env:ANTHROPIC_API_KEY = "sk-ant-..."  (set before launching)
Reads local data from data/processed/ (the TV v2 artifacts).

Layout/theme mirrors app_agent_movies.py so the three agent apps feel like one
product family.
"""
import os
import warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # silence the __path__ deprecation flood
warnings.filterwarnings("ignore", message=".*__path__.*")

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
from agent_tv import run_agent  # noqa: E402

REPO_ROOT = APP_DIR.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CORPUS_PATH = PROCESSED_DIR / "tv_corpus.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "tv_embeddings_bgem3_v2_notitle.npy"
FAISS_PATH = PROCESSED_DIR / "tv_faiss_bgem3_v2_notitle.index"

st.set_page_config(page_title="TV Recommendation Engine — Agent", page_icon="tv", layout="wide")

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


@st.cache_resource
def load_all():
    for p in (CORPUS_PATH, EMBEDDINGS_PATH, FAISS_PATH):
        if not p.exists():
            listing = "\n".join(sorted(x.name for x in PROCESSED_DIR.glob("tv_*")))
            st.error(f"Missing {p.name}. tv_* files present:\n{listing}")
            st.stop()
    df = pd.read_parquet(CORPUS_PATH).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    index = faiss.read_index(str(FAISS_PATH))
    assert embeddings.shape[0] == len(df) == index.ntotal, (
        f"row mismatch: corpus {len(df)}, embeddings {embeddings.shape[0]}, index {index.ntotal}"
    )
    return df, embeddings, index


# ---- forgiving search (mirrors app_agent_movies) ----
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
    return _df["name"].apply(_normalize).tolist()


def search_corpus(query, df, norm_names, limit=12):
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    q = _normalize(query)
    if not q:
        return df.iloc[0:0]
    q_words = q.split()
    norm = pd.Series(norm_names, index=df.index)
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


def _sub_line(row):
    """Secondary line under a series name: genres, else first creator."""
    g = row.get("genres", "")
    if isinstance(g, str) and g:
        return g
    c = row.get("created_by", "")
    if isinstance(c, str) and c:
        return c.split(",")[0].strip()
    return ""


# --- callbacks ---
def add_rating(tv_id, name, corpus_index):
    rating = st.session_state.get(f"rate_{tv_id}")
    if rating is None:
        return
    if any(r["tmdb_id"] == tv_id for r in st.session_state.tv_ratings):
        return
    st.session_state.tv_ratings.append({
        "tmdb_id": tv_id, "matched_title": name,
        "corpus_index": int(corpus_index), "rating": float(rating),
    })

def remove_rating(tv_id):
    st.session_state.tv_ratings = [r for r in st.session_state.tv_ratings if r["tmdb_id"] != tv_id]

def clear_all():
    st.session_state.tv_ratings = []
    st.session_state.chat = []
    st.session_state.agent_history = []
    st.session_state.agent_extras = {}


df, embeddings, index = load_all()
norm_names = build_search_index(df)

if "tv_ratings" not in st.session_state:
    st.session_state.tv_ratings = []
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role","content"} for display
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # full message history (multi-turn memory)
if "agent_extras" not in st.session_state:
    st.session_state.agent_extras = {}   # lazy caches: tag pivot, query embedder

st.title("TV Recommendation Engine")
st.caption(f"{len(df):,} series · rate series on the left, ask the agent on the right")

left, right = st.columns([1, 1.4], gap="large")

# ----- LEFT: taste profile -----
with left:
    st.subheader("Your taste profile")
    st.button("Clear all", use_container_width=True, on_click=clear_all)

    query = st.text_input("Search a series to rate", key="search_box",
                          placeholder="e.g. Breaking Bad")
    if query:
        matches = search_corpus(query, df, norm_names)
        already = {r["tmdb_id"] for r in st.session_state.tv_ratings}
        for pos, show in matches.iterrows():
            if show["tmdb_id"] in already:
                continue
            fc1, fc2 = st.columns([3, 1.2])
            with fc1:
                yr = f" ({int(show['start_year'])})" if pd.notna(show["start_year"]) else ""
                st.markdown(f"**{show['name']}**{yr}  \n*{_sub_line(show)}*")
            with fc2:
                st.selectbox("Rate", [None, 1, 2, 3, 4, 5],
                             key=f"rate_{show['tmdb_id']}", label_visibility="collapsed",
                             on_change=add_rating,
                             args=(show["tmdb_id"], show["name"], pos))

    st.markdown("---")
    if st.session_state.tv_ratings:
        st.markdown(f"**{len(st.session_state.tv_ratings)} series rated:**")
        for r in sorted(st.session_state.tv_ratings, key=lambda x: -x["rating"]):
            rc1, rc2 = st.columns([5, 1])
            rc1.markdown(
                f"<div class='film-card'><span class='film-title'>{r['matched_title']}</span><br>"
                f"<span style='color:#c9a86a'>{r['rating']}/5</span></div>",
                unsafe_allow_html=True)
            rc2.button("X", key=f"rm_{r['tmdb_id']}", on_click=remove_rating, args=(r["tmdb_id"],))
    else:
        st.info("Rate some series to get started.")

# ----- RIGHT: agent chat -----
with right:
    st.subheader("Ask the agent")
    st.caption("Try: \"a limited series about a con\", \"something like Breaking Bad but anime\", "
               "or \"a long sitcom to live in\"")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("What are you in the mood to watch?")
    if prompt:
        if not st.session_state.tv_ratings:
            st.warning("Rate a few series on the left first so the agent knows your taste.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking (tool trace in terminal)..."):
                    # persistent extras dict so the tag pivot and query embedder
                    # are built once per session, not once per message
                    state = st.session_state.agent_extras
                    state.update({"df": df, "embeddings": embeddings, "index": index,
                                  "ratings": st.session_state.tv_ratings})
                    try:
                        reply, st.session_state.agent_history = run_agent(
                            prompt, state,
                            conversation_history=st.session_state.agent_history,
                            verbose=True,
                        )
                    except Exception as e:
                        reply = f"Something went wrong: {e}"
                st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
