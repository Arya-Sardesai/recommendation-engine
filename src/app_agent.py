"""
Recommendation Engine — local agent build.
Left: build your taste profile (search + rate books).
Right: chat with the recommendation agent, which uses your ratings + the
recommender tools to answer natural-language requests.

Run locally:  streamlit run src/app_agent.py
Requires:  $env:ANTHROPIC_API_KEY = "sk-ant-..."  (set before launching)
Reads local data from data/processed/ (the v1 artifacts).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st
import faiss

# make the agent module importable (it's in src/)
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from agent import run_agent  # noqa: E402

REPO_ROOT = APP_DIR.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CORPUS_PATH = PROCESSED_DIR / "books_v1.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings_bgem3.npy"
FAISS_PATH = PROCESSED_DIR / "faiss_bgem3.index"
DEFAULT_RATINGS_PATH = PROCESSED_DIR / "my_matched_ratings.json"

st.set_page_config(page_title="Recommendation Engine — Agent", page_icon="book", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #14110f; }
    h1, h2, h3 { color: #f4f1ea; font-family: Georgia, 'Times New Roman', serif; }
    .book-card { background: #1f1b18; border: 1px solid #34302b; border-radius: 8px;
                 padding: 12px 14px; margin-bottom: 8px; }
    .book-title { font-size: 1rem; font-weight: 600; color: #f4f1ea; }
    .book-author { color: #c9a86a; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all():
    df = pd.read_parquet(CORPUS_PATH).reset_index(drop=True)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    index = faiss.read_index(str(FAISS_PATH))
    return df, embeddings, index


@st.cache_data
def load_default_ratings():
    if DEFAULT_RATINGS_PATH.exists():
        with open(DEFAULT_RATINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def search_corpus(query, df, limit=12):
    if not query or len(query) < 2:
        return df.iloc[0:0]
    mask = df["title"].str.contains(query, case=False, na=False, regex=False)
    return df[mask].nlargest(limit, "ratings_count")


# --- callbacks ---
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
    st.session_state.chat = []
    st.session_state.agent_history = []


df, embeddings, index = load_all()

if "ratings" not in st.session_state:
    st.session_state.ratings = []
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role","content"} for display
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # full message history for the agent (multi-turn memory)

st.title("Recommendation Engine")
st.caption(f"{len(df):,} books · rate books on the left, ask the agent on the right")

left, right = st.columns([1, 1.4], gap="large")

# ----- LEFT: taste profile -----
with left:
    st.subheader("Your taste profile")
    c1, c2 = st.columns(2)
    c1.button("Load sample", use_container_width=True, on_click=load_sample)
    c2.button("Clear all", use_container_width=True, on_click=clear_all)

    query = st.text_input("Search a book to rate", key="search_box",
                          placeholder="e.g. The Hobbit")
    if query:
        matches = search_corpus(query, df)
        already = {r["book_id"] for r in st.session_state.ratings}
        for pos, book in matches.iterrows():
            if book["book_id"] in already:
                continue
            bc1, bc2 = st.columns([3, 1.2])
            with bc1:
                yr = f" ({int(book['publication_year'])})" if pd.notna(book['publication_year']) else ""
                st.markdown(f"**{book['title']}**{yr}  \n*{book['primary_author']}*")
            with bc2:
                st.selectbox("Rate", [None, 1, 2, 3, 4, 5],
                             key=f"rate_{book['book_id']}", label_visibility="collapsed",
                             on_change=add_rating,
                             args=(book["book_id"], book["title"], book["primary_author"], pos))

    st.markdown("---")
    if st.session_state.ratings:
        st.markdown(f"**{len(st.session_state.ratings)} books rated:**")
        for r in sorted(st.session_state.ratings, key=lambda x: -x["rating"]):
            rc1, rc2 = st.columns([5, 1])
            rc1.markdown(
                f"<div class='book-card'><span class='book-title'>{r['matched_title']}</span><br>"
                f"<span class='book-author'>{r['matched_author']}</span> · "
                f"<span style='color:#c9a86a'>{r['rating']}/5</span></div>",
                unsafe_allow_html=True)
            rc2.button("X", key=f"rm_{r['book_id']}", on_click=remove_rating, args=(r["book_id"],))
    else:
        st.info("Rate some books, or click 'Load sample'.")

# ----- RIGHT: agent chat -----
with right:
    st.subheader("Ask the agent")
    st.caption("Try: \"something like my noir books but lighter\" or \"recent literary fiction\"")

    # render chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("What are you in the mood to read?")
    if prompt:
        if not st.session_state.ratings:
            st.warning("Rate a few books on the left first so the agent knows your taste.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    state = {"df": df, "embeddings": embeddings, "index": index,
                             "ratings": st.session_state.ratings}
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