"""
Recommendation Engine — unified three-tab deployment build (Books | Movies | TV)

Downloads all artifacts from the HF dataset repo at startup, then runs three
content-based recommenders in separate tabs. No GPU / embedding model needed
at runtime — new ratings reuse precomputed corpus embeddings.

Books tab: BGE-M3 + FAISS + tag re-ranking (kNN-propagated book tags),
           source-aware display.
Movies tab: BGE-M3 + FAISS + tag re-ranking (keyword+genome tags),
            director/cast/studio, director-exclusion, language display,
            forgiving search.
TV tab: BGE-M3 + FAISS + tag re-ranking (TMDB keyword tags),
        creator(showrunner)/network, creator-exclusion, title-collision
        penalty, forgiving search.

Session state is namespaced (ratings vs movie_ratings vs tv_ratings) so the
tabs don't collide. Agent chat is intentionally NOT included here (local-only).
"""
from pathlib import Path
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
import faiss
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
DATASET_REPO = "Arya2305/recommendation-engine-data"

# --- books v2: Hardcover corpus (replaces the frozen 2017 Goodreads dump) ---
# Goodreads gave 432K rows but only ~58K cleared the serving floor, with tags
# that were 99.3% kNN-propagated from the same embeddings they were meant to
# correct. Hardcover gives 74K rows with reader-applied genre + MOOD tags
# (independent evidence), 99.9% cover coverage, and current release dates.
# Rollback lines below are the v1 Goodreads artifacts, kept live in the repo.
# B_CORPUS = "books_v1.parquet"                    # v1 Goodreads — rollback
# B_EMB = "embeddings_bgem3.npy"                   # v1 Goodreads — rollback
# B_FAISS = "faiss_bgem3.index"                    # v1 Goodreads — rollback
# B_TAGS = "book_tags_knn.parquet"                 # v1 kNN tags — rollback
B_CORPUS = "books_v2_hc2.parquet"
B_EMB    = "book_embeddings_bgem3_v2_hc2_notitle.npy"
B_FAISS  = "book_faiss_bgem3_v2_hc2_notitle.index"
B_TAGS   = "book_tags_v2_hc2.parquet"
# movies
# M_CORPUS = "movies.parquet"                      # pre-enrichment — rollback
# v2 meta: adds runtime (refreshed from TMDB), certification and cert_tier.
# COLUMNS ONLY — enrich_movies_metadata.py asserts row order and refuses to
# write if it shifted, so M_EMB and M_FAISS below are unchanged and stay valid.
M_CORPUS = "movies_v2_meta.parquet"
# M_EMB = "movie_embeddings_minilm_v1.npy"  # v1 MiniLM — rollback
# M_FAISS = "movie_faiss_minilm_v1.index"   # v1 MiniLM — rollback
M_EMB = "movie_embeddings_bgem3_v2_kw2.npy"     # v2 BGE-M3
M_FAISS = "movie_faiss_bgem3_v2_kw2.index"      # v2 BGE-M3
M_TAGS = "movie_tags_v2.parquet"
# tv (v1: no tag file — ships tagless)
# T_CORPUS = "tv_corpus.parquet"                   # pre-enrichment — rollback
# v2 meta: adds content_rating + maturity (0-5 ordinal, 95% of servable rows).
# COLUMNS ONLY — enrich_tv_metadata.py asserts row order and refuses to write
# if it shifted, so T_EMB and T_FAISS below are unchanged and stay valid.
# The maturity filter in recommend_tv is inert until this constant is flipped.
T_CORPUS = "tv_v2_meta.parquet"
T_TAGS = "tv_tags.parquet"
# T_EMB = "tv_embeddings_minilm_v1.npy"           # v1 MiniLM — rollback
# T_FAISS = "tv_faiss_minilm_v1.index"            # v1 MiniLM — rollback
# T_EMB = "tv_embeddings_bgem3_v2_notitle.npy"    # v2 BGE-M3, 23K corpus, no keywords — rollback
# T_FAISS = "tv_faiss_bgem3_v2_notitle.index"     # v2 BGE-M3, 23K corpus, no keywords — rollback
T_EMB = "tv_embeddings_bgem3_v2_kw_notitle.npy"   # v2.1 BGE-M3, 30K corpus + keywords
T_FAISS = "tv_faiss_bgem3_v2_kw_notitle.index"    # v2.1 BGE-M3, 30K corpus + keywords

# ---------------------------------------------------------------------------
# Page config + styling (shared identity)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Recommendation Engine", page_icon="sparkles", layout="wide")

st.markdown("""
<style>
@media (max-width: 640px) {
  /* 1. Force st.columns([3,1.2]) to stack instead of crushing the dropdown */
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.75rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
  }

  /* 2. Tap targets: 44px min, 16px font (16px also stops iOS auto-zoom on focus) */
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 44px !important;
    font-size: 16px !important;
  }
  .stButton button {
    min-height: 44px !important;
    font-size: 16px !important;
  }
  [data-testid="stTextInput"] input {
    font-size: 16px !important;
  }
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    /* ---------------------------------------------------------------------
       Design tokens. The palette is a film-archive register: warm near-black
       paper, cream type, brass accent, sage marginalia. Kept from v1 -- it is
       the one part of the UI that already had a point of view. What changed is
       everything AROUND it: borders were too loud (#4a443c on #1f1b18 read as
       a table grid), there was no depth, and only one element had a
       transition, so the app felt static rather than responsive.
       --------------------------------------------------------------------- */
    :root {
        --bg:          #14110f;
        --surface:     #1f1b18;
        --surface-2:   #1a1613;
        --border:      #2e2822;   /* quiet by default */
        --border-lift: #4a4136;   /* only on hover/focus */
        --text:        #f4f1ea;
        --text-2:      #a89f93;
        --text-3:      #7d7468;
        --brass:       #c9a86a;
        --sage:        #9bbf9b;
        --slate:       #7a9ec9;
        --radius:      10px;
        --radius-sm:   6px;
        --ease:        cubic-bezier(0.2, 0.6, 0.3, 1);
        --t:           160ms var(--ease);
        --shadow:      0 1px 2px rgba(0,0,0,0.35);
        --shadow-lift: 0 6px 20px rgba(0,0,0,0.45);
    }

    .stApp { background-color: var(--bg); }

    /* Type scale: headings get tighter tracking, body stays comfortable. A
       flat scale was part of why everything read as one undifferentiated mass. */
    h1, h2, h3 {
        color: var(--text);
        font-family: Georgia, 'Times New Roman', serif;
        letter-spacing: -0.015em;
    }
    h1 { font-weight: 600; }

    /* ---------- cards ---------- */
    .item-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 14px 16px; margin-bottom: 10px;
        box-shadow: var(--shadow);
        transition: border-color var(--t), background var(--t);
    }
    .item-card:hover { border-color: var(--border-lift); background: #221e1a; }

    .item-title { font-size: 1.05rem; font-weight: 600; color: var(--text);
                  letter-spacing: -0.01em; }
    .item-sub   { color: var(--brass); font-size: 0.9rem; }
    .item-meta  { color: var(--text-2); font-size: 0.8rem; margin-top: 4px; }
    .item-cast  { color: #b0a69a; font-size: 0.8rem; margin-top: 2px; }
    .because    { color: var(--sage); font-size: 0.82rem; font-style: italic;
                  margin-top: 6px; }
    .tag-hint   { color: var(--slate); font-size: 0.78rem; margin-top: 4px; }
    .blurb      { color: #b8b0a6; font-size: 0.86rem; line-height: 1.5;
                  margin-bottom: 8px; }
    .rate-prompt{ color: var(--sage); font-size: 0.82rem; }
    .list-count { color: var(--text-3); font-size: 0.8rem; padding-top: 9px; }
    .rated-flag { color: var(--sage); font-size: 0.85rem; padding: 4px 0; }
    .rated-flag b { color: var(--brass); }

    /* ---------- the signature element: the recommendation card ----------
       Poster sits on a ring rather than flush, so artwork of any brightness
       still reads as a distinct object against the card. */
    .rec-row { display: flex; gap: 14px; align-items: flex-start; }
    /* Streamlit themes bare <a> inside stMarkdown (blue + underline) and that
       rule beats a bare .watch-link selector. Match on the anchor element and
       force the two properties it sets, or the link renders as a default
       browser-blue hyperlink regardless of what is written here. */
    .rec-body a.watch-link,
    a.watch-link {
        display: inline-block; margin-top: 6px; font-size: 0.78rem;
        letter-spacing: 0.02em;
        color: var(--muted) !important;
        text-decoration: none !important;
        border-bottom: 1px solid transparent;
        transition: color var(--ease), border-color var(--ease);
    }
    .rec-body a.watch-link:hover,
    a.watch-link:hover {
        color: var(--brass) !important;
        border-bottom-color: var(--brass);
        text-decoration: none !important;
    }
    .rec-body a.watch-link:visited,
    a.watch-link:visited { color: var(--muted) !important; }
    .rec-poster {
        width: 92px; height: 138px; object-fit: cover;
        border-radius: var(--radius-sm); flex-shrink: 0; background: #2a2520;
        box-shadow: 0 0 0 1px rgba(244,241,234,0.07), 0 2px 8px rgba(0,0,0,0.4);
        transition: transform var(--t), box-shadow var(--t);
    }
    .rec-body { flex: 1; min-width: 0; }
    [data-testid="stVerticalBlockBorderWrapper"]:hover .rec-poster {
        transform: scale(1.03);
        box-shadow: 0 0 0 1px rgba(201,168,106,0.35), 0 4px 14px rgba(0,0,0,0.5);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        transition: transform var(--t), border-color var(--t), box-shadow var(--t);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        border-color: var(--border-lift) !important;
        box-shadow: var(--shadow-lift);
    }

    /* ---------- buttons ----------
       Streamlit ships a filled accent button; every action on the page read as
       a primary CTA. These are mostly secondary actions (watchlist, dismiss,
       page) so they get a quiet surface and earn emphasis on hover. */
    .stButton button {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-2) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.85rem !important; font-weight: 500 !important;
        transition: background var(--t), border-color var(--t),
                    color var(--t), transform var(--t) !important;
    }
    .stButton button:hover:not(:disabled) {
        background: #262119 !important; border-color: var(--border-lift) !important;
        color: var(--text) !important; transform: translateY(-1px);
    }
    .stButton button:active:not(:disabled) { transform: translateY(0); }
    .stButton button:disabled {
        opacity: 1 !important; background: #262119 !important;
        color: var(--text-3) !important; border-color: var(--border) !important;
    }
    /* The one genuinely destructive action keeps a warning tone. */
    .stButton button[kind="primary"] {
        background: #4a2420 !important; border-color: #6e352e !important;
        color: #f0d9d4 !important;
    }
    .stButton button[kind="primary"]:hover { background: #5c2c27 !important; }

    /* ---------- inputs ---------- */
    [data-testid="stTextInput"] input {
        background: var(--surface-2) !important;
        border-radius: var(--radius-sm) !important;
        border-color: var(--border) !important;
        transition: border-color var(--t), box-shadow var(--t) !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: var(--border-lift) !important;
        box-shadow: 0 0 0 3px rgba(201,168,106,0.12) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: var(--surface-2); border-color: var(--border);
        border-radius: var(--radius-sm); font-size: 0.85rem; min-height: 34px;
        transition: border-color var(--t);
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--border-lift); box-shadow: none;
    }

    /* ---------- tabs: shadcn segmented control ----------
       Canonical shadcn markup is:
         TabsList    bg-muted rounded-lg p-[3px]
         TabsTrigger rounded-md ... data-[state=active]:bg-background shadow-sm
       i.e. a recessed track with the active tab raised OUT of it. Streamlit
       ships an underline instead, so the two indicator elements it draws
       (tab-highlight, tab-border) have to be suppressed or they fight the pill.
       Reproduced here in the archive palette rather than shadcn's neutral grey. */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 3px;
        gap: 2px;
        display: inline-flex;
        width: fit-content;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-3);
        font-size: 0.875rem; font-weight: 500;
        padding: 6px 14px;
        border-radius: var(--radius-sm);
        background: transparent;
        transition: background var(--t), color var(--t), box-shadow var(--t);
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text-2); }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.4), 0 0 0 1px rgba(244,241,234,0.05);
    }
    /* Streamlit's own indicators, neutralised. */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { background: transparent !important; height: 0 !important; }

    /* ---------- expander ---------- */
    [data-testid="stExpander"] details {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        background: var(--surface-2) !important;
        transition: border-color var(--t);
    }
    [data-testid="stExpander"] details:hover { border-color: var(--border-lift) !important; }
    [data-testid="stExpander"] summary { font-size: 0.85rem !important; color: var(--text-2) !important; }

    /* ---------- stars ---------- */
    [data-testid="stFeedback"] [data-baseweb="button"] {
        color: #6b6458 !important; transition: color var(--t), transform var(--t) !important;
    }
    [data-testid="stFeedback"] [aria-checked="true"],
    [data-testid="stFeedback"] [data-baseweb="button"]:hover { color: var(--brass) !important; }
    [data-testid="stFeedback"] [data-baseweb="button"]:hover { transform: scale(1.15); }

    /* ---------- alerts ----------
       Streamlit's defaults are saturated blue/green/amber blocks that read as
       system chrome dropped onto the page. Recoloured to the archive palette
       and flattened to a left rule, so a message informs without shouting. */
    [data-testid="stAlert"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--text-3) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-2) !important;
        box-shadow: none !important;
    }
    [data-testid="stAlert"] p { font-size: 0.86rem !important; }
    [data-testid="stAlertContentSuccess"] { border-left-color: var(--sage) !important; }
    [data-testid="stAlertContentInfo"]    { border-left-color: var(--slate) !important; }
    [data-testid="stAlertContentWarning"] { border-left-color: var(--brass) !important; }

    /* ---------- dropdown popover ----------
       The select trigger was themed but the menu it opens was not, so choosing
       a sort order flashed a light panel over a dark page. */
    div[data-baseweb="popover"] div[role="listbox"],
    div[data-baseweb="popover"] ul {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: var(--shadow-lift) !important;
    }
    div[data-baseweb="popover"] li {
        font-size: 0.85rem !important; color: var(--text-2) !important;
        transition: background var(--t), color var(--t);
    }
    div[data-baseweb="popover"] li:hover {
        background: #262119 !important; color: var(--text) !important;
    }

    /* ---------- file uploader (profile restore) ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-2) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius-sm) !important;
        transition: border-color var(--t);
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--border-lift) !important; }

    /* ---------- vertical rhythm ----------
       Streamlit stacks every element on the same margin, so headings did not
       read as starting anything. A little space above a heading does more for
       hierarchy than any amount of type styling. */
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] h3 { margin-top: 0.4rem; }
    [data-testid="stCaptionContainer"] p { color: var(--text-3) !important; font-size: 0.8rem !important; }
    [data-testid="stExpander"] { margin-bottom: 0.5rem; }

    /* ---------- misc ---------- */
    hr.row-div { margin: 6px 0 12px; border: none; border-top: 1px solid var(--border); }
    .wl-row { display: flex; gap: 10px; align-items: flex-start; }
    .wl-poster { width: 40px; height: 60px; object-fit: cover; border-radius: 4px;
                 flex-shrink: 0; background: #2a2520;
                 box-shadow: 0 0 0 1px rgba(244,241,234,0.06); }
    .cover-blank { border: 1px dashed #3a342c; background: var(--surface-2); box-shadow: none; }

    /* The localStorage component mounts an iframe; collapse it so it leaves no
       gap. Height-zero (not display:none) so its JS still runs. */
    iframe[title="st_local_storage.st_local_storage"],
    iframe[title="st_local_storage"] { height: 0 !important; min-height: 0 !important; border: none !important; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: #2e2822; border-radius: 5px;
                                border: 2px solid var(--bg); }
    ::-webkit-scrollbar-thumb:hover { background: #4a4136; }

    /* Keyboard focus stays visible even though hover states carry most of the
       feedback -- the app is fully operable by tab. */
    .stButton button:focus-visible,
    [data-testid="stTextInput"] input:focus-visible {
        outline: 2px solid var(--brass) !important; outline-offset: 2px !important;
    }

    /* NOTE: deliberately no entrance/keyframe animations. Streamlit rebuilds
       the DOM on every rerun, and every button click is a rerun -- a fade-in
       would replay on each interaction and strobe. Motion here is confined to
       hover and focus, which fire only on real user intent. */
    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover { transform: none; }
        .stButton button:hover:not(:disabled) { transform: none; }
    }
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
    "tr": "Turkish", "ar": "Arabic", "no": "Norwegian",
}


def lang_label(code):
    return LANG_NAMES.get(code, code.upper()) if isinstance(code, str) else ""


# TMDB poster CDN. poster_path is a PATH ("/abc.jpg"); prepend the sized base.
# NOTE: this CDN may be blocked on some ISPs (e.g. India) so posters can fail to
# load locally while loading fine for viewers elsewhere.
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"


# --- "where can I watch this" ---------------------------------------------
# Three testers asked for streaming availability. TMDB carries it (via
# JustWatch) but deliberately does NOT expose deep links: the documented
# integration is to link to TMDB's own /watch page, which resolves the
# viewer's region itself and is always current.
#
# We link rather than STORE availability, on purpose:
#   * availability churns weekly; a value baked into the parquet in August is
#     wrong by October, and a wrong "on Netflix" is worse than no badge --
#     same reasoning that deferred TV season tracking.
#   * displaying the provider data obliges per-item JustWatch attribution.
#     Linking out carries no such obligation.
#   * zero fetch, zero new column, zero re-embed.
# Provider badges are a v2 job, and need a refresh schedule before they are
# honest. No locale param: TMDB geolocates, which is right for testers in
# India and Canada alike without adding a region selector.
def network_label(value, limit=2):
    """TMDB returns EVERY broadcaster, which on anime means every regional
    affiliate: BLUELOCK carries 24 networks and Dragon Ball DAIMA 27, and the
    card printed all of them, swamping the genre and vote metadata beside it.

    First one or two, then a count. The first entry is the primary broadcaster
    in practice, so truncating loses nothing a viewer wants.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return ""
    if len(parts) <= limit:
        return ", ".join(parts)
    return f"{', '.join(parts[:limit])} +{len(parts) - limit} more"


def watch_link(kind, tmdb_id):
    try:
        n = int(tmdb_id)
    except (TypeError, ValueError):
        return ""
    return f"https://www.themoviedb.org/{kind}/{n}/watch"


def watch_link_html(kind, tmdb_id, label="Where to watch"):
    url = watch_link(kind, tmdb_id)
    if not url:
        return ""
    return (f"<a class='watch-link' href='{url}' target='_blank' "
            f"rel='noopener noreferrer'>{label} &rarr;</a>")


def poster_url(path):
    if not isinstance(path, str) or not path.strip():
        return None
    return TMDB_IMG_BASE + "/" + path.lstrip("/")


def year_label(y):
    """' (1998)' or '' -- never ' (18)'.

    342 of 74,054 Hardcover rows carry an unusable publication_year: genuine
    antiquity encoded as a small number (The Art of War -> 501, Meditations ->
    180) and plain junk (Charlotte's Web -> 2, Voyage of the Basilisk -> 18).
    Both render as a broken-looking parenthetical, so anything outside a
    plausible printed-book range is omitted rather than shown.
    """
    try:
        if y is None or (isinstance(y, float) and y != y):
            return ""
        y = int(y)
    except (TypeError, ValueError):
        return ""
    return f" ({y})" if 1400 <= y <= 2032 else ""


# Hardcover's image_url is already a full CDN URL (unlike TMDB's poster_path,
# which is a bare path needing TMDB_IMG_BASE prepended), so this just validates
# it rather than building it. Hardcover's CDN rate-limits bulk loads, so every
# <img> built from this must carry loading="lazy" -- see book_img_tag().
def book_cover_url(url):
    if not isinstance(url, str) or not url.strip():
        return None
    return url


# 110 of 74,054 Hardcover rows have no cover (50 of them clear min_ratings=10
# and so can actually be recommended). Emitting nothing would leave those cards
# with their text flush left while every neighbour is indented by the poster
# width, which reads as a rendering bug. Emit a same-size empty box instead so
# the flex row stays aligned.
def book_img_tag(url, css_class):
    u = book_cover_url(url)
    if not u:
        return f"<div class='{css_class} cover-blank'></div>"
    return f"<img class='{css_class}' loading='lazy' src='{u}'>"


# ---------------------------------------------------------------------------
# BOOK GENRE DISPLAY CLEANUP
# ---------------------------------------------------------------------------
# Hardcover genres carry two kinds of noise, both measured on the live corpus
# (7,535 distinct labels over 74,054 books):
#
#   1. COMMA-SPLIT FRAGMENTS. Something upstream splits LOC subject headings on
#      commas, so "Comic books, strips, etc." arrives as three labels and
#      "World War, 1939-1945" as two. 452 distinct fragments, 2,337 instances,
#      every one identifiable by its leading space. FIX THIS IN
#      fetch_hardcover_v2.py -- the guard here is a display-side patch over a
#      data bug, and the monthly refresh will keep regenerating it.
#   2. LOC SUBJECT HEADINGS. "Apartment dwellers" (6 books), "Sex
#      discrimination against women" (1). Real genres are common; these sit in
#      the extreme tail, so a frequency floor separates them cleanly.
#
# The allowlist is derived from the corpus rather than hand-written, so it
# re-derives itself after a corpus swap instead of going stale.
# --- movie genre tabs ------------------------------------------------------
# Measured on a 70-film profile: a 400-candidate pool splits into five viable
# first-genre buckets. MOVIE_TAB_N * 2 is the depth bar for showing a tab.
# The movie corpus mixes feature films with promo shorts, featurettes and
# concert films. TMDB runtime is 100% populated, so a floor is the cheapest
# possible fix -- no fetch, no re-embed, one comparison per candidate.
# Measured against the real corpus (9,274 recommendable films), the floor
# removes: 10min->54, 15min->66, 20min->81, 30min->103, 40min->115.
# 40 was too aggressive: it took A Trip to the Moon (15), Kung Fury (32) and
# The Wonderful Story of Henry Sugar (39) -- and two Wes Anderson films sit at
# 4/5 in the reference profile. 20 keeps those and still kills Team Thor (4)
# and Frozen Fever (8).
#
# NO FLOOR SEPARATES Paperman (7, Oscar winner) FROM Team Thor (4, a promo).
# Any cut that removes one removes the other. This is a collateral-damage
# setting, not a fix -- 15 would additionally save the 15-20min band
# (A Trip to the Moon) at a cost of 15 more junk shorts.
MOVIE_MIN_RUNTIME = 20

MOVIE_POOL_N = 400
MOVIE_TAB_N = 10
MOVIE_MAX_TABS = 4
# Genre tabs drop films you have almost certainly already seen -- Joker
# (27,944 votes), Reservoir Dogs (15,600), The Godfather Part II (14,077).
# Correct as recommendations, wasted as 3 of 10 slots.
#
# This was first tried as a per-genre MEDIAN cut, on the theory that genre
# popularity scales differ ~10x. That was wrong in practice: a relative cut
# always removes half a bucket, including buckets that were already obscure,
# where the top half was the good half. It rescued Crime and Drama and
# gutted Action (best pick fell from Hard Boiled at 7.5 to Stolen at 5.5).
# "Already famous" turns out to be closer to an absolute property -- 15k TMDB
# votes is famous in any genre -- so an absolute ceiling leaves already-deep
# buckets untouched while still trimming the top-heavy ones.
MOVIE_TAB_VOTE_CEILING = 8000
MOVIE_DEEP_CUTS = True

# --- TV genre tabs ---------------------------------------------------------
# Measured with probe_genre_tabs --kind tv on an 86-show profile before
# building (300-candidate pool):
#
#   MULTI-LABEL bucketing scored 4 "distinct" tabs at Jaccard <= 0.35 and was
#   WRONG. The contents were Drama=[Avatar, Afro Samurai, To Your Eternity],
#   Sci-Fi=[Naruto, Chainsaw Man], Comedy=[One-Punch Man, MASHLE] -- four anime
#   lists with four labels. Jaccard compares ids, and four disjoint sets of
#   anime are still four anime lists. Animation vs Action & Adventure came back
#   at 0.82 overlap, which is the same fact from the other side.
#
#   PRIMARY-GENRE bucketing separates them properly, because Animation-as-first
#   genre absorbs the anime and leaves the rest live-action: Animation 102
#   (Naruto, Bleach), Drama 46 (Alice in Borderland, Elementary), Action &
#   Adventure 39, Comedy 32 (Seinfeld, Friends), Sci-Fi & Fantasy 29, Crime 23.
#
# Falsifier passed: only 35% of multi-genre shows are ordered by ascending
# TMDB id, so first-genre carries real information rather than id order.
TV_POOL_N = 300
TV_TAB_N = 10
TV_MAX_TABS = 4

# NOT 8000. TV vote counts run an order of magnitude below film: the servable
# TV median is 657 and p90 is 2,550, so the movie ceiling would hide 30 of
# 1,750 series (1.7%) and the "deep cuts" tab would be the same list again.
# 2,000 removes 13.3%, which is roughly what 8,000 removes on the movie side.
# The lesson from the movie median cut was that an absolute property needs an
# absolute threshold -- but the threshold still has to be calibrated to ITS
# OWN distribution, not borrowed from another one.
TV_TAB_VOTE_CEILING = 2000
TV_DEEP_CUTS = True

# --- Kids-content suppression ---------------------------------------------
# Three testers reported children's content reaching adult profiles. Age
# suitability is not inferable from a plot overview at all, so this needs an
# external signal: TMDB certifications, fetched for every recommendable film
# (92.9% coverage) and bucketed into cert_tier by enrich_movies_metadata.py.
#
# Only the all_ages tier is suppressible, and it is small (401 films, 4.3%)
# and unambiguous -- US:G, IN:U, TV-Y. That is the point: this filter does not
# need high coverage, it needs precision on the class it removes. The 13.8%
# with no usable certification (unrated / missing) pass through untouched,
# because NO CERT IS NOT A SAFE CERT and must never be treated as one.
#
# CONDITIONAL, not absolute: suppression is skipped entirely for anyone who
# has rated an all_ages film positively. A blanket block would make Pixar
# unreachable for every user, which is a worse defect than the one being
# fixed.
MOVIE_KIDS_FILTER = True
MOVIE_KIDS_TIER = "all_ages"
# Certification ALONE is not enough. The G rating predates PG-13 (1984), so it
# was applied to films that are in no sense children's content: 2001: A Space
# Odyssey is US:G, as are Planet of the Apes and much of the 60s-70s catalogue.
# Measured consequence: one 5/5 rating of 2001 in a 122-film adult profile
# tripped the "this user likes all-ages films" gate and stood the whole filter
# down, letting Atlantis: Milo's Return through.
# Requiring an Animation/Family genre alongside the certification separates
# "rated G" from "made for children", and fixes both directions at once: 2001
# no longer disables the gate, and 2001 is no longer suppressible either.
MOVIE_KIDS_GENRES = {"Animation", "Family"}

# A SINGLE positively-rated children's film is not evidence of taste. Measured:
# one 5/5 of Pokémon: The First Movie in a 122-film profile (1 of 53 liked
# films, 1.9%) stood the whole filter down and let Atlantis: Milo's Return
# through. Nostalgia ratings are common and cheap; a real preference shows up
# as either repetition or share.
# Either condition is enough, which keeps sparse profiles safe: a tester who
# rates 6 films including one Pixar reads as 17% and is left alone, while the
# same single rating inside 53 reads as noise.
# --- Popularity prior (For you tab only) ----------------------------------
# MovieLens says a plain popularity ranking beats this ranker at every k
# (recall@20 0.080 vs 0.043). Popularity is a crude stand-in for the
# collaborative signal a content-based system cannot see, so blending a little
# of it back recovers some of that gap.
#
# Applied to the MAIN list only. The genre tabs already cut at
# MOVIE_TAB_VOTE_CEILING and exist to surface deep cuts -- pushing famous films
# into them would defeat their purpose.
#
# SCALE DISCIPLINE: this is the tag-blend bug waiting to happen again.
# emb_score sits ~0.7-0.9; vote_count runs 400-30,000. Adding them raw lets
# popularity swamp similarity entirely. So the term is a PERCENTILE within the
# candidate pool, centred on the pool mean -- signed, bounded, and on the same
# scale as the tag adjustment.
#
# 0.0 = off. Do not raise it without an eval run: this trades taste for
# familiarity, and the whole point of the app is the former.
MOVIE_POP_WEIGHT = 0.0

# --- TV maturity filter ----------------------------------------------------
# Two of the three kids-content complaints were TV: Futurama -> children's
# cartoons, a teen show -> children's shows. The movie approach does not
# transfer. Gating on certification + Animation/Family would catch every anime
# in the corpus, and a binary "does this profile like kids' shows" gate stands
# down for anyone who rated Gravity Falls (TV-Y7) highly -- which is exactly
# the profile that produced the complaint.
#
# So this is ORDINAL. enrich_tv_metadata.py maps every content rating onto a
# 0-5 scale (TV-Y .. TV-MA). A show is suppressed when it sits far BELOW the
# viewer's usual level, not when it is "for kids" in the abstract:
#
#   Futurama (4) -> The Flintstones (0)   drop of 4   suppressed
#   Futurama (4) -> The Addams Family (2) drop of 2   kept
#
# Unknown maturity (5% of recommendable series) always passes through: an
# absent rating is not a low rating.
TV_MATURITY_DROP = 3      # suppress at this many levels below the profile
TV_MATURITY_MIN_RATED = 5  # need this many rated shows before trusting a median

MOVIE_KIDS_LIKE_MIN = 2       # this many children's films = a preference
MOVIE_KIDS_LIKE_RATIO = 0.15  # or this share of the liked set


def _is_kids_film(row):
    """all_ages certification AND a children's genre. Both, never either."""
    try:
        if str(row["cert_tier"]) != MOVIE_KIDS_TIER:
            return False
    except (KeyError, TypeError):
        return False
    g = row.get("genres") if hasattr(row, "get") else row["genres"]
    if isinstance(g, str):
        gs = {x.strip() for x in g.split(",") if x.strip()}
    elif isinstance(g, (list, tuple, set)):
        gs = {str(x).strip() for x in g}
    else:
        gs = set()
    return bool(gs & MOVIE_KIDS_GENRES)

MIN_GENRE_COUNT = 50
GENRE_STOP = {"general", "etc", "strips", "texts"}
GENRE_GENERIC = {"fiction", "science", "literature", "stories", "novel", "general"}


@st.cache_data(show_spinner=False)
def _book_genre_allow(_df, min_count=MIN_GENRE_COUNT):
    """lowercase label -> canonical casing, for labels clearing the floor."""
    if "genres" not in _df.columns:
        return {}, {}
    counts, casing = Counter(), defaultdict(Counter)
    for g in _df["genres"]:
        if g is None:
            continue
        for raw in g:
            if not isinstance(raw, str) or raw != raw.strip() or not raw.strip():
                continue                      # comma-split fragment
            s = raw.strip()
            counts[s.lower()] += 1
            casing[s.lower()][s] += 1
    allow = {k: casing[k].most_common(1)[0][0] for k, v in counts.items()
             if v >= min_count and k not in GENRE_STOP}
    return allow, dict(counts)


def clean_genres(genres, allow, counts, limit=3):
    if genres is None:
        return []
    kept, seen = [], set()
    for raw in genres:
        if not isinstance(raw, str) or raw != raw.strip():
            continue
        k = raw.strip().lower()
        if k in allow and k not in seen:
            seen.add(k)
            kept.append(allow[k])
    # Collapse a label whose words are wholly contained in another kept label.
    # Two cases, because "more verbose" is not the same as "better":
    #   * a bare generic ("Fiction" beside "Science Fiction") always yields,
    #     even though it is the more common label corpus-wide
    #   * otherwise the MORE common label wins, so "Adventure" (21.5K) survives
    #     next to the rarer "Adventure stories" (316) instead of being eaten
    out = []
    for lab in kept:
        w = set(lab.lower().split())
        supersets = [o for o in kept if w < set(o.lower().split())]
        if supersets and (lab.lower() in GENRE_GENERIC
                          or any(counts.get(o.lower(), 0) >= counts.get(lab.lower(), 0)
                                 for o in supersets)):
            continue
        out.append(lab)
    return out[:limit]


# ===========================================================================
# SHARED LOADERS
# ===========================================================================
@st.cache_resource
def _dl(fname):
    return hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset")


class IndexVectors:
    """Row-vector accessor backed by a FAISS IndexFlat.

    `vectors[i]` reconstructs embedding row i straight from the index rather than
    holding a separate .npy copy in RAM. IndexFlatIP already stores the full,
    L2-normalized vectors, so reconstruct(i) == the old embeddings[i] exactly
    (both row-aligned to the same parquet). Dropping the .npy load removes the
    single biggest boot item — at 1024-dim that's ~1.8 GB of download AND ~1.8 GB
    of RAM for books alone — which is what pushed the Space over its limits when
    the corpora moved from MiniLM (384d) to BGE-M3 (1024d).

    Only the few liked-item vectors are reconstructed per request (candidates come
    from index.search), so the per-call cost is negligible. Rollback if ever
    needed: restore the load_*_emb() loaders and pass the .npy arrays instead.
    """
    def __init__(self, index):
        self._index = index

    def __getitem__(self, i):
        return self._index.reconstruct(int(i))


# ----- books -----
@st.cache_resource
def load_books_corpus():
    df = pd.read_parquet(_dl(B_CORPUS)).reset_index(drop=True)
    return df


@st.cache_resource
def load_books_index():
    return faiss.read_index(_dl(B_FAISS))


@st.cache_resource
def load_books_tags():
    try:
        tdf = pd.read_parquet(_dl(B_TAGS))
        tdf["book_id"] = tdf["book_id"].astype(str)
        return tdf.pivot_table(index="book_id", columns="tag", values="score", fill_value=0.0)
    except Exception as e:
        print(f"Book tags unavailable: {e}")
        return None


# ----- movies -----
@st.cache_resource
def load_movies_corpus():
    df = pd.read_parquet(_dl(M_CORPUS)).reset_index(drop=True)
    return df


@st.cache_resource
def load_movies_index():
    return faiss.read_index(_dl(M_FAISS))


@st.cache_resource
def load_movies_tags():
    try:
        tdf = pd.read_parquet(_dl(M_TAGS))
        tdf["movie_id"] = tdf["movie_id"].astype(str)
        return tdf.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)
    except Exception as e:
        print(f"Movie tags unavailable: {e}")
        return None


@st.cache_resource
def build_movie_search_index(_df):
    return {
        "titles": _df["title"].apply(_normalize).tolist(),
        "directors": _df["director"].apply(_normalize).tolist() if "director" in _df.columns else [""] * len(_df),
        "cast": _df["cast"].apply(
            lambda c: _normalize(", ".join(c) if c is not None and hasattr(c, "__iter__") and not isinstance(c, str) else "")
        ).tolist() if "cast" in _df.columns else [""] * len(_df),
    }



# ----- tv -----
@st.cache_resource
def load_tv_corpus():
    df = pd.read_parquet(_dl(T_CORPUS)).reset_index(drop=True)
    return df


@st.cache_resource
def load_tv_index():
    return faiss.read_index(_dl(T_FAISS))


@st.cache_resource
def load_tv_tags():
    try:
        tdf = pd.read_parquet(_dl(T_TAGS))
        tdf["tv_id"] = tdf["tv_id"].astype(str)
        return tdf.pivot_table(index="tv_id", columns="tag", values="score", fill_value=0.0)
    except Exception as e:
        print(f"TV tags unavailable: {e}")
        return None


@st.cache_resource
def build_tv_search_index(_df):
    return {
        "titles": _df["name"].apply(_normalize).tolist(),
        "creators": _df["created_by"].apply(_normalize).tolist() if "created_by" in _df.columns else [""] * len(_df),
    }


@st.cache_resource
def build_book_search_index(_df):
    # Normalized once at boot so book search accent-folds (Brontë, Murakami, etc.)
    # the same way movies/TV already do, instead of raw str.contains.
    return {
        "titles": _df["title"].apply(_normalize).tolist(),
        "authors": _df["primary_author"].apply(_normalize).tolist(),
    }

# ===========================================================================
# BOOKS RECOMMENDER (mirrors deployment build with tag re-ranking)
# ===========================================================================
# --- cross-modal: books from films and shows -------------------------------
# All three corpora were embedded with the same model (BAAI/bge-m3, 1024d), so
# a movie vector can query the book index directly. Measured before building
# (probe_cross_modal.py): 8 of 10 known adaptations retrieved their source
# novel in the top 10, five of them at rank 1 -- Blade Runner -> Do Androids
# Dream (0.761), The Shining -> The Shining (0.891), Dune -> Dune (0.809).
# The space is aligned.
#
# TWO THINGS THE PROBE SETTLED ABOUT THE DESIGN:
#
# 1. PER-ANCHOR, NOT CENTROID. Averaging 53 liked films produced a vector near
#    everything and specific to nothing: the top books were James Patterson,
#    Iris Johansen and a Warcraft tie-in at 0.79-0.81. Querying one film at a
#    time gave Knives Out -> Deja Dead and The Westing Game, True Romance ->
#    crime thrillers. Same lesson as series inflation in profiles -- averaging
#    over a taste loses the taste.
#
# 2. CROSS-MODAL SCORES RUN LOWER. Book->book top-5 averaged 0.710; movie->book
#    0.672. Small, but a threshold calibrated on within-modal similarity
#    systematically under-admits here. So candidates are POOL-CENTRED before
#    comparison, the same fix as the tag blend, rather than compared raw.
#
# A weak anchor is dropped rather than padded: Grand Budapest's best book was
# 0.645, well under the within-modal baseline, and forcing three books out of
# an anchor with no good neighbour is how Bourdain -> The Flight Attendant
# happens.
# Sized to match the main tab (20) rather than to fit a screen. At 4 anchors x
# 3 books the tab returned 7 rows after author dedup and the similarity floor,
# which read as a teaser next to a full list. More anchors rather than more
# books per anchor: 3 is enough to see what a film pulls, and a wider spread of
# films is what makes the tab feel like it knows the profile.
CROSS_MODAL_ANCHORS = 8        # films/shows to draw from
CROSS_MODAL_PER_ANCHOR = 3     # books each
CROSS_MODAL_N = 20             # matches the main tab, whichever limit binds first
# Without this the tab collapsed to 14 superhero graphic novels out of 20: the
# Spider-Man / Batman / Superman films sit in one tight region and all point at
# the same corner of the book corpus. The author cap stops one WRITER
# repeating; nothing stopped one GENRE taking the tab. recommend_books has
# genre_cap=6 for exactly this and it was not ported over.
CROSS_MODAL_GENRE_CAP = 5
CROSS_MODAL_MIN_SIM = 0.66     # below the movie->book mean; a floor, not a target


def _is_adaptation(row):
    """TMDB's 'based on novel or book' keyword.

    HIGH PRECISION, LOW RECALL -- verified by hand: Blade Runner, Fight Club
    and The Shining carry it, Knives Out and Collateral correctly do not, but
    DUNE DOES NOT either, despite being one of the most filmed novels there is.
    Community-applied keywords are power-law distributed and the tail is
    missing, the same folksonomy gap the movie tag work already documented.

    So this only ever ADDS a label. A film without the keyword whose novel
    comes back still shows the novel -- as a discovery, unlabelled. Inferring
    adaptation from similarity plus a title match instead would have labelled
    The Maltese Falcon -> Trouble Is My Business as a source novel, which is
    worse than missing Dune.
    """
    for col in ("keywords_tmdb", "keywords"):
        try:
            v = row[col]
        except (KeyError, TypeError):
            continue
        if v is None:
            continue
        text = " ".join(str(x) for x in v) if isinstance(v, (list, tuple, np.ndarray)) else str(v)
        t = text.lower()
        if "based on novel" in t or "based on book" in t:
            return True
    return False


def recommend_books_from_watching(ratings, src_df, src_emb, b_df, b_emb, b_index,
                                  kind="movie", n_anchors=CROSS_MODAL_ANCHORS,
                                  per_anchor=CROSS_MODAL_PER_ANCHOR,
                                  min_ratings=10, like_threshold=4.0,
                                  neutral=3.0, cross_weight=0.35, neg_scale=1.0,
                                  min_sim=CROSS_MODAL_MIN_SIM, retrieve_k=60,
                                  n=CROSS_MODAL_N, genre_cap=CROSS_MODAL_GENRE_CAP,
                                  skip_ids=None):
    """Books retrieved from film/TV vectors, scored against the WHOLE profile.

    The first version was pure per-anchor top-k: each film retrieved its own
    books independently, so a candidate knew about exactly one film and nothing
    about the other eighty-eight. recommend_books does not work that way -- it
    scores every candidate against every anchor, rewards books that answer to
    several (cross_weight) and penalises books close to what you disliked. That
    is what produces "because you liked X and Y" instead of a flat list of
    four unrelated shelves.

    So retrieval is still per-anchor (it is what surfaces candidates), but
    SCORING is whole-profile:

        final = home_sim + cross_weight * cross_sim - neg_scale * neg_pen

    home_sim  best match across all liked films -- the anchor shown
    cross_sim second-best match, if a different film also likes this book
    neg_pen   closest approach to anything rated below `neutral`

    Returns a DataFrame shaped like recommend_books' output.
    """
    title_col = "title" if kind == "movie" else "name"

    def _vec(store, i):
        # IndexVectors exposes __getitem__ only -- no __len__, no slicing.
        return np.asarray(store[int(i)], dtype="float32")

    def _valid(r):
        ci = r.get("corpus_index")
        return ci is not None and 0 <= int(ci) < len(src_df)

    liked = [r for r in ratings if _valid(r) and r.get("rating", 0) >= like_threshold]
    disliked = [r for r in ratings if _valid(r) and r.get("rating", 0) < neutral]
    if not liked:
        return pd.DataFrame()

    liked.sort(key=lambda r: -float(r.get("rating", 0)))
    L = np.vstack([_vec(src_emb, r["corpus_index"]) for r in liked])
    N = (np.vstack([_vec(src_emb, r["corpus_index"]) for r in disliked])
         if disliked else None)
    neg_w = (np.array([(neutral - float(r["rating"])) / neutral for r in disliked])
             if disliked else None)

    # Retrieve from the strongest anchors; over-sample because weak anchors and
    # duplicate books get dropped later.
    skip = {str(x) for x in (skip_ids or [])}
    rc = pd.to_numeric(b_df["ratings_count"], errors="coerce").fillna(0)
    cand = {}
    for r in liked[:max(n_anchors * 3, 12)]:
        q = _vec(src_emb, r["corpus_index"]).reshape(1, -1)
        _, I = b_index.search(q, retrieve_k)
        for bi in I[0]:
            if bi >= 0:
                cand.setdefault(int(bi), None)
    if not cand:
        return pd.DataFrame()

    # Score every candidate against the WHOLE profile, not just its finder.
    scored = []
    for bi in cand:
        if str(b_df.iloc[bi]["book_id"]) in skip or rc.iloc[bi] < min_ratings:
            continue
        bv = _vec(b_emb, bi)
        sims = L @ bv
        order = np.argsort(-sims)
        home_i = int(order[0])
        home = float(sims[home_i])
        if home < min_sim:
            continue
        cross_i = int(order[1]) if len(order) > 1 else None
        cross = float(sims[cross_i]) if cross_i is not None else 0.0
        neg = 0.0
        if N is not None:
            neg = float(np.max((N @ bv) * neg_w))
        scored.append((home + cross_weight * cross - neg_scale * neg,
                       bi, home_i, home, cross_i, cross))
    if not scored:
        return pd.DataFrame()
    scored.sort(key=lambda x: -x[0])

    out, used_books, used_authors, per_film = [], set(), set(), {}
    genre_counts, kept_tokens = {}, []

    # Bucket, not first-genre. A cap on the raw first genre did nothing: the
    # superhero books came back labelled "Comics & Graphic Novels", "Comics",
    # "Graphic Novels", "Fantasy", "Literature" and "Poetry" -- six buckets for
    # one shelf, each comfortably under the cap. Same shape as the movie
    # _genre_bucket merging Crime and Thriller so the cap cannot be laundered
    # by picking a different label for the same thing.
    FAMILIES = (
        ("comics", {"comics", "graphic novels", "comics & graphic novels",
                    "manga", "comic", "superheroes", "graphic novel"}),
        ("kids-ya", {"young adult", "young adult fiction", "middle grade",
                     "children's", "juvenile fiction"}),
        ("crime", {"crime", "mystery", "thriller", "murder", "police",
                   "detective", "suspense"}),
        ("sff", {"fantasy", "science fiction", "high fantasy", "space",
                 "dystopian", "paranormal"}),
    )

    def _first_genre(g):
        try:
            gs = [str(x).strip().lower() for x in (g if g is not None else [])]
        except TypeError:
            gs = []
        if not gs:
            return "_none"
        # A family match anywhere in the list wins over raw position: a comic
        # tagged Fantasy first is still a comic.
        for name, members in FAMILIES:
            if set(gs) & members:
                return name
        return gs[0]

    def _toks(t):
        stop = {"the", "a", "an", "of", "and", "vol", "volume", "book", "novel",
                "complete", "collection", "graphic"}
        return {w for w in "".join(c if c.isalnum() else " " for c in str(t).lower()).split()
                if len(w) >= 3 and w not in stop}
    for final, bi, home_i, home, cross_i, cross in scored:
        if len(out) >= n:
            break
        anchor = liked[home_i]
        film = anchor.get("matched_title", str(src_df.iloc[int(anchor["corpus_index"])][title_col]))
        if len(per_film) >= n_anchors and film not in per_film:
            continue
        if per_film.get(film, 0) >= per_anchor or bi in used_books:
            continue
        book = b_df.iloc[bi]
        auth = str(book.get("primary_author", "")).strip().lower()
        if auth and auth in used_authors:
            continue                # stops The Godfather -> three Corleone books
        pg = _first_genre(book["genres"])
        if pg != "_none" and genre_counts.get(pg, 0) >= genre_cap:
            continue                # stops one genre taking the whole tab
        # The Odyssey (Homer) and The Odyssey: A Graphic Novel (Hinds) are
        # different book_ids and different authors, so only a title check
        # catches them. Same rule the main ranker applies via _is_title_dup.
        tk = _toks(book["title"])
        if tk and any(tk <= k or k <= tk for k in kept_tokens):
            continue
        also = None
        if cross_i is not None and cross >= min_sim:
            cand_also = liked[cross_i].get("matched_title")
            if cand_also and cand_also != film:
                also = cand_also
        src_row = src_df.iloc[int(anchor["corpus_index"])]
        out.append({
            "title": book["title"], "author": book["primary_author"],
            "genres": list(book["genres"]) if book["genres"] is not None else [],
            "year": book["publication_year"],
            "ratings_count": int(book["ratings_count"]),
            "similarity": float(final), "because_of": film,
            "also_because": also, "because_kind": kind,
            "is_source_novel": bool(_is_adaptation(src_row) and per_film.get(film, 0) == 0),
            "source": book.get("source", "hardcover"),
            "tag_boost": False, "match_tags": [],
            "book_id": book["book_id"], "corpus_index": int(bi),
            "overview": book["description"] if "description" in b_df.columns else "",
            "image_url": book.get("image_url", ""),
        })
        used_books.add(bi)
        if auth:
            used_authors.add(auth)
        if pg != "_none":
            genre_counts[pg] = genre_counts.get(pg, 0) + 1
        if tk:
            kept_tokens.append(tk)
        per_film[film] = per_film.get(film, 0) + 1
    return pd.DataFrame(out)


def recommend_books(ratings, df, embeddings, index, tags_df=None, n=20,
                    min_ratings=10, like_threshold=3.5, per_anchor_cap=3,
                    exclude_rated_authors=True, per_author_cap=2,
                    tag_weight=0.25, title_token_penalty=0.2,
                    skip_ids=None, neutral=3.0, cross_weight=0.35, neg_scale=1.0,
                    genre_cap=6, retrieve_k=150, author_adjacent_penalty=0.30,
                    quality_weight=0.12, min_avg_rating=3.6, shrink_ratings=50,
                    hardcover_ratings_scale=1.0, untagged_as_zero=True):
    """Whole-profile ranking (same design as recommend_movies).

    Per-anchor retrieval preserves distinct taste modes; scoring uses the full
    signed profile: kinship-damped cross-pollination from OTHER anchors (max,
    not sum), global negative penalty from every sub-3-star rating, signed tag
    blend over ALL ratings (~33K tagged books). Anchors are diversity-ordered
    (greedy farthest-point) so minority modes get round-robin turns before the
    dominant cluster repeats. Genre-capped selection, water-filled.
    Quality gating (added after the movie ablation showed the vote_average floor
    was the single biggest contributor there — removing it cost 30% of
    recall@20, more than any other component). Books carry `average_rating`, so
    the same Bayesian-shrunk floor and prior apply:
        adj = (n*R + m*C) / (n + m),  m = shrink_ratings
    which stops a 4.6 at 2,000 ratings from outranking a 4.2 at 500,000.

    UNITS: the v2 corpus is Hardcover-only, so ratings_count is uniform and
    hardcover_ratings_scale is 1.0 (it existed to reconcile the old mixed
    corpus, where Goodreads counts ran ~100x larger). Hardcover distribution:
    median 14, p75 31, p90 81, max 10,262. min_ratings=10 leaves 49,817 books
    recommendable -- comparable to the ~58K that cleared the old Goodreads
    floor. shrink_ratings drops 5000 -> 50 for the same reason: the Bayesian
    prior has to be on the scale of the counts it is shrinking.

    untagged_as_zero is what actually rebalances the two sources. Goodreads
    rows are 96.9% tagged; Hardcover rows are 0.24% tagged and have NO genres
    at all. Under the centred tag adjustment an untagged candidate would sit at
    a neutral 0 and dodge the tag term entirely -- the same way untagged TV
    shows outranked tagged ones before the centring fix. Scoring them as
    tag_sim = 0 instead means "no evidence of taste match" reads as
    below-average rather than free, which is what it is. Set False to restore
    neutral handling.
    """
    liked = [r for r in ratings if r["rating"] >= like_threshold]
    liked = sorted(liked, key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame(), None

    anchor_vecs = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    anchor_w01 = np.array([(r["rating"] - neutral) / 2.0 for r in liked], dtype="float32")

    # diversity-order anchors (greedy farthest-point, small rating tiebreak)
    if len(liked) > 1:
        _order = [0]
        _rem = set(range(1, len(liked)))
        while _rem:
            best_i, best_s = None, -1e9
            for i in _rem:
                d = 1.0 - max(float(anchor_vecs[j] @ anchor_vecs[i]) for j in _order)
                s = d + 0.15 * float(anchor_w01[i])
                if s > best_s:
                    best_s, best_i = s, i
            _order.append(best_i)
            _rem.discard(best_i)
        liked = [liked[i] for i in _order]
        anchor_vecs = anchor_vecs[_order]
        anchor_w01 = anchor_w01[_order]

    _anchor_damp = np.clip(anchor_vecs @ anchor_vecs.T, 0, None) ** 2

    disliked = [r for r in ratings if r["rating"] < neutral]
    dis_vecs = (np.array([embeddings[r["corpus_index"]] for r in disliked]).astype("float32")
                if disliked else None)
    dis_w01 = (np.array([(neutral - r["rating"]) / 2.0 for r in disliked], dtype="float32")
               if disliked else None)

    rated_book_ids = {r["book_id"] for r in ratings}
    if skip_ids:
        rated_book_ids = rated_book_ids | set(skip_ids)
    rated_authors = {r["matched_author"].lower() for r in ratings if r.get("matched_author")}

    # --- Author-adjacency vocabulary -----------------------------------------
    # exclude_rated_authors above only catches books written BY a rated author.
    # It misses books written ABOUT them, which is how Ellmann's `Oscar Wilde`
    # (author: Richard Ellmann) surfaced off `The Picture of Dorian Gray`, a
    # Douglas Adams biography off Hitchhiker's, and `The Discworld Companion`.
    # Two signals catch those:
    #   A. the candidate TITLE contains a rated author's FULL name. Full name,
    #      not any token -- a lone surname match would penalise `The Once and
    #      Future King` for anyone who rated Stephen King.
    #   B. the title carries a companion/criticism marker AND shares a
    #      significant token with something already rated.
    _AUTHOR_STOP = {"the", "and", "van", "von", "der", "den", "del", "de",
                    "jr", "sir", "dr", "mrs", "saint", "st"}

    def _author_tokens(name):
        toks = set()
        for w in str(name).lower().replace(".", " ").replace(",", " ").split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) >= 4 and w not in _AUTHOR_STOP:
                toks.add(w)
        return toks

    rated_author_names = []
    rated_title_tokens = set()
    for r in ratings:
        if r.get("matched_author"):
            at = _author_tokens(r["matched_author"])
            if len(at) >= 2:          # need a full name to match safely
                rated_author_names.append(at)
        rated_title_tokens |= _title_tokens(r.get("matched_title", ""))

    # signed tag profile over ALL rated + tagged books
    tag_profile = None
    top_user_tags = None
    if tags_df is not None:
        t_vecs, t_w = [], []
        for r in ratings:
            w = r["rating"] - neutral
            if w == 0:
                continue
            bid = str(r["book_id"])
            if bid in tags_df.index:
                t_vecs.append(tags_df.loc[bid].values)
                t_w.append(w * neg_scale if w < 0 else w)
        if t_vecs:
            _tw = np.array(t_w, dtype="float32")
            tag_profile = (np.array(t_vecs, dtype="float32") * _tw[:, None]).sum(axis=0) / np.abs(_tw).sum()
            top_user_tags = pd.Series(tag_profile, index=tags_df.columns).nlargest(5)

    all_sims, all_idxs = index.search(anchor_vecs, retrieve_k)

    # corpus mean rating over the recommendable subset -- the C in the
    # Bayesian shrink. Computed once per call, not per candidate.
    if "average_rating" in df.columns:
        _C_books = float(df.loc[df["ratings_count"] >= min_ratings,
                                "average_rating"].mean())
        if not np.isfinite(_C_books):
            _C_books = 3.9
    else:
        _C_books = 3.9

    _cand_cache = {}

    def _cand_signals(idx):
        if idx in _cand_cache:
            return _cand_cache[idx]
        cvec = np.asarray(embeddings[idx], dtype="float32")
        cross_all = (anchor_vecs @ cvec) * anchor_w01
        neg_pen = 0.0
        if dis_vecs is not None:
            neg_pen = float(np.clip((dis_vecs @ cvec) * dis_w01, 0, None).max())
        _cand_cache[idx] = (cross_all, neg_pen)
        return _cand_cache[idx]

    _all_tag_sims = []
    per_anchor_queues = []
    for like_i, (sims, idxs) in enumerate(zip(all_sims, all_idxs)):
        seed_tokens = _title_tokens(liked[like_i].get("matched_title", ""))
        queue = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            book = df.iloc[idx]
            if book["book_id"] in rated_book_ids:
                continue
            if exclude_rated_authors and book["primary_author"].lower() in rated_authors:
                continue
            # source-aware popularity floor (see docstring)
            _src = book["source"] if "source" in df.columns else "goodreads"
            _floor = (min_ratings * hardcover_ratings_scale
                      if _src == "hardcover" else min_ratings)
            if book["ratings_count"] < _floor:
                continue

            # Bayesian-shrunk quality floor + prior
            _ar = book["average_rating"] if "average_rating" in df.columns else None
            if _ar is None or pd.isna(_ar):
                quality = 0.0
            else:
                _n = float(book["ratings_count"])
                ar_adj = (_n * float(_ar) + shrink_ratings * _C_books) / (_n + shrink_ratings)
                if ar_adj < min_avg_rating:
                    continue  # hard floor on the shrunk, trustworthy value
                quality = (ar_adj - min_avg_rating) / (5.0 - min_avg_rating)

            cross_all, neg_pen = _cand_signals(int(idx))
            if len(cross_all) > 1:
                damped = cross_all * _anchor_damp[like_i]
                others = np.delete(damped, like_i)
                cross_j = int(np.argmax(others))
                cross_val = float(others[cross_j])
                cross_anchor_i = cross_j if cross_j < like_i else cross_j + 1
            else:
                cross_val, cross_anchor_i = 0.0, None

            emb_score = (float(sim)
                         + cross_weight * max(cross_val, 0.0)
                         - neg_scale * neg_pen
                         + quality_weight * quality)

            tag_sim = None
            match_tags = []
            if tag_profile is not None and str(book["book_id"]) in tags_df.index:
                book_tags = tags_df.loc[str(book["book_id"])].values
                nu, nb = np.linalg.norm(tag_profile), np.linalg.norm(book_tags)
                if nu > 0 and nb > 0:
                    tag_sim = float(np.dot(tag_profile, book_tags) / (nu * nb))
                    _all_tag_sims.append(tag_sim)
                    match_tags = _matching_tags(tag_profile, book_tags, tags_df.columns)

            btoks = _title_tokens(book["title"])
            shared = seed_tokens & btoks
            title_pen = (title_token_penalty * min(len(shared), 3)
                         if (title_token_penalty and shared) else 0.0)

            # author-adjacency guard (see rated_author_names above).
            # btoks_n adds de-possessived forms so `Douglas Adams's Starship
            # Titanic` (which tokenises "Adams's" -> "adamss") still matches.
            # A full author-name match is treated as an EXCLUSION, not a
            # penalty: a book titled `Oscar Wilde` when Oscar Wilde is rated is
            # unambiguously about him, the same way exclude_rated_authors treats
            # books by him. A penalty alone left it at rank 2 in testing --
            # these titles sit so close to the anchor in embedding space that
            # any survivable penalty leaves them near the top.
            if author_adjacent_penalty:
                btoks_n = btoks | {w[:-1] for w in btoks if len(w) > 4 and w.endswith("s")}
                if any(nm <= btoks_n for nm in rated_author_names):
                    continue
                # The companion-marker signal is softer, so it only demotes. The
                # rated-title overlap must be something OTHER than the marker
                # itself, or `Life of Pi` trips on a rated `Life, the Universe
                # and Everything` purely through the shared word "life".
                if (btoks & _COMPANION_MARKERS) and (
                        (btoks & rated_title_tokens) - _COMPANION_MARKERS):
                    title_pen += author_adjacent_penalty

            queue.append([int(idx), emb_score, tag_sim, title_pen, cross_anchor_i, cross_val, match_tags])
        per_anchor_queues.append({"like": liked[like_i], "queue": queue})

    # --- PASS 2: centre the tag signal on the candidate pool ------------------
    # See recommend_tv for the measurement that motivated this.
    _mu = float(np.mean(_all_tag_sims)) if _all_tag_sims else 0.0
    for pbq in per_anchor_queues:
        scored = []
        for idx, emb_score, tag_sim, title_pen, ci, cv, mtags in pbq["queue"]:
            if tag_sim is not None:
                adj = tag_weight * (tag_sim - _mu)
            elif untagged_as_zero and _all_tag_sims:
                adj = tag_weight * (0.0 - _mu)   # untagged reads as below-average
            else:
                adj = 0.0
            final_sim = emb_score + adj - title_pen
            if final_sim < 0.0:
                continue  # pushed below zero by dislike/title penalties
            boosted = bool(tag_sim is not None and adj > 0)
            scored.append((idx, final_sim, boosted, ci, cv, mtags if boosted else []))
        scored.sort(key=lambda x: -x[1])
        pbq["queue"] = scored

    # selection: round-robin + genre cap (water-filled) + work dedup
    results, seen_idxs, seen_works, kept_tokens = [], set(), set(), []
    genre_counts = {}
    # per_author_cap MEASURED on goodbooks-10k (200 users, scoring protocol,
    # recall@20): cap=1 0.008, cap=2 0.010, cap=3 0.012, off 0.016. Monotonic,
    # no knee -- every relaxation buys recall, so there is no setting that
    # keeps the dedup for free.
    #
    # 2 rather than the 1 originally shipped: the complaint was three Martha
    # Wells books crowding one list, not two books by an author you liked.
    # Cap=1 was stricter than the report required and cost twice the recall.
    #
    # Read the cost with the metric in mind: goodbooks relevance is dominated
    # by series completion, so "another book by this author" is the most
    # predictable thing in the dataset and this cap refuses exactly that. The
    # gap is the metric disagreeing with a deliberate product choice.
    #
    # exclude_rated_authors only blocks books by authors already IN the profile.
    # It says nothing about how often one UNRATED author may appear in the
    # output, so a strong embedding match could return All Systems Red and
    # Rapport (both Martha Wells) adjacent in the top 20. Capped like genres,
    # and lifted in the water-fill pass so a narrow profile still fills.
    author_counts = {}
    _GENERIC_GENRES = {"fiction", "nonfiction", "novels", "audiobook", "book club",
                       "adult", "adult fiction", "literature"}

    def _genre_bucket(book):
        gl = list(book["genres"]) if book["genres"] is not None else []
        for g in gl:
            if isinstance(g, str) and g.strip() and g.strip().lower() not in _GENERIC_GENRES:
                return g.strip()
        return gl[0].strip() if gl and isinstance(gl[0], str) else "_none"

    def _fill(enforce_genre_cap):
        while len(results) < n:
            added = 0
            for pbq in per_anchor_queues:
                taken = 0
                qi = 0
                while qi < len(pbq["queue"]) and taken < per_anchor_cap and len(results) < n:
                    idx, sim, tag_boost, cross_i, cross_val, mtags = pbq["queue"][qi]
                    if idx in seen_idxs:
                        pbq["queue"].pop(qi)
                        continue
                    book = df.iloc[idx]
                    work_id = book["work_id"]
                    if work_id and work_id in seen_works:
                        pbq["queue"].pop(qi)
                        continue
                    btoks = _title_tokens(book["title"])
                    if _is_title_dup(btoks, kept_tokens):
                        pbq["queue"].pop(qi)
                        continue
                    pg = _genre_bucket(book)
                    if enforce_genre_cap and genre_counts.get(pg, 0) >= genre_cap:
                        qi += 1
                        continue
                    _auth = (book["primary_author"] or "").strip().lower()
                    if (enforce_genre_cap and per_author_cap and _auth
                            and author_counts.get(_auth, 0) >= per_author_cap):
                        qi += 1
                        continue
                    pbq["queue"].pop(qi)
                    seen_idxs.add(idx)
                    if work_id:
                        seen_works.add(work_id)
                    kept_tokens.append(btoks)
                    genre_counts[pg] = genre_counts.get(pg, 0) + 1
                    if _auth:
                        author_counts[_auth] = author_counts.get(_auth, 0) + 1
                    also = None
                    if cross_i is not None and cross_val >= 0.30:
                        also = liked[cross_i]["matched_title"]
                        if also == pbq["like"]["matched_title"]:
                            also = None
                    results.append({
                        "title": book["title"], "author": book["primary_author"],
                        "genres": list(book["genres"]) if book["genres"] is not None else [],
                        "year": book["publication_year"], "ratings_count": int(book["ratings_count"]),
                        "similarity": sim, "because_of": pbq["like"]["matched_title"],
                        "also_because": also,
                        "source": book.get("source", "goodreads"), "tag_boost": tag_boost, "match_tags": mtags,
                        "book_id": book["book_id"], "corpus_index": idx,
                        "overview": book["description"] if "description" in df.columns and isinstance(book["description"], str) else "",
                        "image_url": book["image_url"] if "image_url" in df.columns and isinstance(book["image_url"], str) else "",
                    })
                    taken += 1; added += 1
                if len(results) >= n:
                    break
            if added == 0:
                break

    _fill(enforce_genre_cap=True)
    if len(results) < n:
        _fill(enforce_genre_cap=False)
    return pd.DataFrame(results), top_user_tags


def search_books(query, df, search_idx, limit=12):
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    q = _normalize(query)
    if not q:
        return df.iloc[0:0]
    titles = pd.Series(search_idx["titles"], index=df.index)
    authors = pd.Series(search_idx["authors"], index=df.index)
    mask = (titles.str.contains(q, na=False, regex=False) |
            authors.str.contains(q, na=False, regex=False))
    return df[mask].nlargest(limit, "ratings_count")


# ===========================================================================
# MOVIES RECOMMENDER (mirrors app_movies with all fixes)
# ===========================================================================
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


_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with",
    "is", "it", "his", "her", "my", "our", "your", "part", "ii", "iii", "iv",
    "2", "3", "4", "movie", "film", "show", "series", "season",
}


# Words that mark a book as being ABOUT another work/author rather than being
# the work itself: biographies, companions, critical editions, letter
# collections. Used by the author-adjacency guard in recommend_books.
_COMPANION_MARKERS = {
    "companion", "biography", "annotated", "criticism", "letters", "essays",
    "memoir", "handbook", "casebook", "readers", "guide", "study", "studies",
    "interviews", "conversations", "unauthorized", "authorized", "life",
    "papers", "notebooks", "journals", "collected", "portrait", "making",
}


def _title_tokens(title):
    if not isinstance(title, str):
        return set()
    toks = set()
    for w in title.lower().replace(":", " ").replace("-", " ").split():
        w = "".join(ch for ch in w if ch.isalnum())
        if len(w) >= 3 and w not in _TITLE_STOPWORDS:
            toks.add(w)
    return toks


def _is_title_dup(tokens, kept_token_sets, min_shared=2):
    """True if a candidate title shares >= min_shared significant tokens with an
    already-selected result — i.e. it's a sequel/remake/franchise sibling
    (both Thomas Crown Affairs, the Marigold Hotel pair, multiple GoT titles)."""
    if not tokens:
        return False
    return any(len(tokens & k) >= min_shared for k in kept_token_sets)


def _creator_names(s):
    """Split a multi-valued created_by string ('A, B') into a name set."""
    if not isinstance(s, str) or not s.strip():
        return set()
    return {p.strip().lower() for p in s.split(",") if p.strip()}


def tag_badge(rec, lead=""):
    """'tag-matched: stylized-violence, tense' instead of a bare label.

    Two testers said the badge and the taste signature meant nothing to them --
    one asked outright for the tags to be listed. The tags are the reason the
    item was lifted, so naming them turns an opaque marker into an explanation.
    """
    if not rec.get("tag_boost"):
        return ""
    tags = [t for t in (rec.get("match_tags") or []) if t][:3]
    label = "tag-matched" + (": " + ", ".join(tags) if tags else "")
    return f"{lead}<span style='color:#7a9ec9'>{label}</span>"


def _matching_tags(tag_profile, film_tags, columns, k=3):
    """The tags actually responsible for a tag-matched badge.

    Two testers asked what "tag-matched" and "Inferred taste signature" mean.
    The badge only ever carried a boolean, so the answer was invisible. The
    contribution of each tag to the cosine is the elementwise product of the
    signed profile and the film's tag vector, so the largest positive terms
    ARE the explanation -- no extra model, just the arithmetic already done.
    """
    try:
        contrib = np.asarray(tag_profile, dtype="float32") * np.asarray(film_tags, dtype="float32")
        if not contrib.size:
            return []
        top = np.argsort(-contrib)[:k]
        return [str(columns[i]) for i in top if contrib[i] > 0]
    except Exception:
        return []


def recommend_movies(ratings, df, embeddings, index, tags_df=None, n=20,
                     min_votes=400, like_threshold=3.5, per_anchor_cap=3, tag_weight=0.25,
                     exclude_rated_directors=True, title_token_penalty=0.2, skip_ids=None,
                     neutral=3.0, cross_weight=0.35, neg_scale=1.0,
                     quality_weight=0.12, min_vote_average=6.0, genre_cap=6,
                     retrieve_k=150, flat_ranking=False, shrink_votes=1000,
                     min_runtime=MOVIE_MIN_RUNTIME, kids_filter=MOVIE_KIDS_FILTER,
                     pop_weight=MOVIE_POP_WEIGHT):
    """Whole-profile ranking with mode preservation.

    Retrieval stays PER-ANCHOR (each liked film pulls its own neighbourhood) so
    minority tastes — musicals, anime, Bollywood — survive next to a dominant
    crime cluster. Scoring then uses the FULL profile:
      score = home_sim                          (how close to its own anchor)
            + cross_weight * best cross signal  (max over OTHER anchors, weighted
                                                 by rating: cross-pollination —
                                                 a Bollywood pick rises if it also
                                                 resembles a loved anime. MAX, not
                                                 sum, so the biggest cluster can't
                                                 drown the small ones)
            - neg_scale * worst dislike signal  (max similarity to any <3-star
                                                 film, weighted: GLOBAL pruning)
            + quality_weight * quality prior    (soft lift for well-rated films)
    Hard floor: vote_average >= min_vote_average kills the Giglis outright.
    Tag blend (signed over ALL ratings) refines when the candidate is tagged.
    Selection: round-robin over anchors with a per-genre cap (water-filled:
    the cap relaxes only if we can't fill n), so no one genre floods the list.
    """
    liked = [r for r in ratings if r["rating"] >= like_threshold]
    liked = sorted(liked, key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame(), None

    anchor_vecs = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    anchor_w01 = np.array([(r["rating"] - neutral) / 2.0 for r in liked], dtype="float32")  # 5*->1.0, 4*->0.5

    # --- Diversity-order the anchors (greedy farthest-point) ------------------
    # Pure rating order lets the 15-strong crime cluster consume every slot
    # before a 4-star anime/musical anchor ever gets a turn. Reorder so each
    # successive anchor is maximally DISSIMILAR to those already placed (with a
    # small rating tiebreak): the round-robin then visits crime, musical, anime,
    # Bollywood... before circling back for a second crime anchor.
    if len(liked) > 1:
        _order = [0]
        _rem = set(range(1, len(liked)))
        while _rem:
            best_i, best_s = None, -1e9
            for i in _rem:
                d = 1.0 - max(float(anchor_vecs[j] @ anchor_vecs[i]) for j in _order)
                s = d + 0.15 * float(anchor_w01[i])
                if s > best_s:
                    best_s, best_i = s, i
            _order.append(best_i)
            _rem.discard(best_i)
        liked = [liked[i] for i in _order]
        anchor_vecs = anchor_vecs[_order]
        anchor_w01 = anchor_w01[_order]

    # anchor<->anchor affinity, used to DAMP cross-pollination between unrelated
    # modes: Collateral may boost candidates in True Romance's queue (kindred),
    # but its crime gravity shouldn't drag Chicago's musical queue toward
    # Scarface. Damping factor = clip(cos)^2.
    _anchor_damp = np.clip(anchor_vecs @ anchor_vecs.T, 0, None) ** 2

    disliked = [r for r in ratings if r["rating"] < neutral]
    dis_vecs = (np.array([embeddings[r["corpus_index"]] for r in disliked]).astype("float32")
                if disliked else None)
    dis_w01 = (np.array([(neutral - r["rating"]) / 2.0 for r in disliked], dtype="float32")
               if disliked else None)  # 1*->1.0, 2*->0.5

    rated_movie_ids = {str(m) for m in {r["movie_id"] for r in ratings}}
    if skip_ids:
        rated_movie_ids = rated_movie_ids | {str(s) for s in skip_ids}
    rated_directors = {r["matched_director"].lower() for r in ratings if r.get("matched_director")}

    # --- Kids-content suppression: decide ONCE per call ----------------------
    # Skipped entirely when the corpus predates the certification enrichment,
    # so this is inert against the old parquet and the constant flip can
    # happen in either order without a KeyError on the movies tab.
    _suppress_kids = False
    if kids_filter and "cert_tier" in df.columns:
        _kids_n = _liked_n = 0
        for r in ratings:
            ci = r.get("corpus_index")
            if ci is None or not (0 <= int(ci) < len(df)):
                continue
            if r["rating"] >= like_threshold:
                _liked_n += 1
                if _is_kids_film(df.iloc[int(ci)]):
                    _kids_n += 1
        # Someone who likes Pixar keeps getting Pixar. Count OR share, so one
        # stray nostalgia rating inside a large adult profile does not disable
        # the filter, while a genuinely kid-heavy profile is never touched.
        _ratio = _kids_n / _liked_n if _liked_n else 0.0
        _likes_kids = (_kids_n >= MOVIE_KIDS_LIKE_MIN
                       or _ratio >= MOVIE_KIDS_LIKE_RATIO)
        _suppress_kids = not _likes_kids

    # --- Signed tag profile over ALL rated + tagged items ---------------------
    tag_profile = None
    top_user_tags = None
    if tags_df is not None:
        t_vecs, t_w = [], []
        for r in ratings:
            w = r["rating"] - neutral
            if w == 0:
                continue
            mid = str(r["movie_id"])
            if mid in tags_df.index:
                t_vecs.append(tags_df.loc[mid].values)
                t_w.append(w * neg_scale if w < 0 else w)
        if t_vecs:
            _tw = np.array(t_w, dtype="float32")
            tag_profile = (np.array(t_vecs, dtype="float32") * _tw[:, None]).sum(axis=0) / np.abs(_tw).sum()
            top_user_tags = pd.Series(tag_profile, index=tags_df.columns).nlargest(5)

    all_sims, all_idxs = index.search(anchor_vecs, retrieve_k)
    has_director = "director" in df.columns

    # Bayesian shrink for vote_average (IMDb weighted-rating form):
    #   va_adj = (v*R + m*C) / (v + m)
    # Thin-vote averages are unreliable (an 8.6 at 158 votes is usually a ~7 film
    # with enthusiastic early raters), so both the hard floor and the quality
    # prior act on the shrunk value: low-count ratings collapse toward the
    # corpus mean C, well-voted ones barely move. m = shrink_votes.
    _C = float(df.loc[df["vote_count"] >= min_votes, "vote_average"].mean())

    def _va_shrunk(va, vc):
        return (vc * va + shrink_votes * _C) / (vc + shrink_votes)

    # memoize per-candidate cross/neg/quality signals (candidates repeat across queues)
    _cand_cache = {}

    def _cand_signals(idx):
        if idx in _cand_cache:
            return _cand_cache[idx]
        cvec = np.asarray(embeddings[idx], dtype="float32")
        cross_all = (anchor_vecs @ cvec) * anchor_w01          # weighted cos to every anchor
        neg_pen = 0.0
        if dis_vecs is not None:
            neg_pen = float(np.clip((dis_vecs @ cvec) * dis_w01, 0, None).max())
        _cand_cache[idx] = (cross_all, neg_pen)
        return _cand_cache[idx]

    _all_tag_sims = []
    _pool_votes = []          # popularity prior, centred in pass 2
    per_anchor_queues = []
    for like_i, (sims, idxs) in enumerate(zip(all_sims, all_idxs)):
        seed_tokens = _title_tokens(liked[like_i].get("matched_title", ""))
        queue = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            film = df.iloc[idx]
            if str(film["id"]) in rated_movie_ids:
                continue
            if int(film["vote_count"]) < min_votes:
                continue
            # Runtime floor: the corpus mixes features with promo shorts and
            # featurettes -- "Team Thor" is a 4-minute Marvel skit that was
            # recommended as a film. Nothing else in the data distinguishes
            # them. Applied only when runtime is present and non-zero, so a
            # missing value never silently drops a real film.
            if min_runtime and "runtime" in df.columns:
                _rt = film["runtime"]
                if pd.notna(_rt) and 0 < float(_rt) < min_runtime:
                    continue
            # Kids-content suppression. Only the explicitly all-ages tier is
            # dropped; unrated and missing certifications pass through, since
            # an absent certification carries no information about age
            # suitability and treating it as adult would silently remove
            # ~14% of the recommendable corpus.
            if _suppress_kids and _is_kids_film(film):
                continue
            va = float(film["vote_average"]) if pd.notna(film["vote_average"]) else None
            if va is None:
                continue
            va_adj = _va_shrunk(va, int(film["vote_count"]))
            if va_adj < min_vote_average:
                continue  # hard quality floor (on the shrunk, trustworthy value)
            if (exclude_rated_directors and has_director
                    and isinstance(film["director"], str) and film["director"]
                    and film["director"].lower() in rated_directors):
                continue

            cross_all, neg_pen = _cand_signals(int(idx))
            # best cross signal from any OTHER anchor (max, not sum), damped by
            # kinship to the home anchor so unrelated modes don't bleed
            if len(cross_all) > 1:
                damped = cross_all * _anchor_damp[like_i]
                others = np.delete(damped, like_i)
                cross_j = int(np.argmax(others))
                cross_val = float(others[cross_j])
                # map back to original anchor index for attribution
                cross_anchor_i = cross_j if cross_j < like_i else cross_j + 1
            else:
                cross_val, cross_anchor_i = 0.0, None

            quality = (va_adj - min_vote_average) / (10.0 - min_vote_average)
            emb_score = (float(sim)
                         + cross_weight * max(cross_val, 0.0)
                         - neg_scale * neg_pen
                         + quality_weight * quality)

            tag_sim = None
            match_tags = []
            if tag_profile is not None and str(film["id"]) in tags_df.index:
                film_tags = tags_df.loc[str(film["id"])].values
                nu, nf = np.linalg.norm(tag_profile), np.linalg.norm(film_tags)
                if nu > 0 and nf > 0:
                    tag_sim = float(np.dot(tag_profile, film_tags) / (nu * nf))
                    _all_tag_sims.append(tag_sim)
                    match_tags = _matching_tags(tag_profile, film_tags, tags_df.columns)

            shared = seed_tokens & _title_tokens(film["title"])
            title_pen = (title_token_penalty * min(len(shared), 3)
                         if (title_token_penalty and shared) else 0.0)
            _pool_votes.append(float(film["vote_count"]) if pd.notna(film["vote_count"]) else 0.0)
            queue.append([int(idx), emb_score, tag_sim, title_pen, cross_anchor_i, cross_val, match_tags])
        per_anchor_queues.append({"like": liked[like_i], "queue": queue})

    # --- PASS 2: centre the tag signal on the candidate pool ------------------
    # See recommend_tv for the measurement that motivated this. Blending
    # emb_score (~0.7-0.9) with tag_sim (~0.0-0.6) subtracted from every tagged
    # candidate; centring on the pool mean makes the term signed instead.
    _mu = float(np.mean(_all_tag_sims)) if _all_tag_sims else 0.0

    # Popularity as a pool percentile, centred like the tag term. log1p first
    # so the gap between 500 and 5,000 votes matters more than the gap between
    # 20,000 and 25,000 -- raw counts are heavily skewed and the top end is all
    # "famous" anyway.
    _pop_pct = {}
    if pop_weight and _pool_votes:
        _lv = np.log1p(np.array(_pool_votes, dtype="float64"))
        _lo, _hi = float(_lv.min()), float(_lv.max())
        _rng = (_hi - _lo) or 1.0
        _norm = (_lv - _lo) / _rng
        _pop_mu = float(_norm.mean())
        _pop_pct = {"lo": _lo, "rng": _rng, "mu": _pop_mu}

    def _pop_adj(vote_count):
        if not pop_weight or not _pop_pct:
            return 0.0
        v = float(vote_count) if pd.notna(vote_count) else 0.0
        n = (float(np.log1p(v)) - _pop_pct["lo"]) / _pop_pct["rng"]
        return pop_weight * (n - _pop_pct["mu"])

    for paq in per_anchor_queues:
        scored = []
        for idx, emb_score, tag_sim, title_pen, ci, cv, mtags in paq["queue"]:
            adj = tag_weight * (tag_sim - _mu) if tag_sim is not None else 0.0
            adj += _pop_adj(df.iloc[idx]["vote_count"])
            final_sim = emb_score + adj - title_pen
            if final_sim < 0.0:
                continue  # pushed below zero by dislike/title penalties
            boosted = bool(tag_sim is not None and adj > 0)
            scored.append((idx, final_sim, boosted, ci, cv, mtags if boosted else []))
        scored.sort(key=lambda x: -x[1])
        paq["queue"] = scored

    def _film_row(idx, sim, tag_boost, because, also, mtags=()):
        film = df.iloc[idx]
        genres_str = film["genres"] if isinstance(film["genres"], str) else ""
        return {
            "title": film["title"],
            "director": film["director"] if has_director and isinstance(film["director"], str) else "",
            "cast": list(film["cast"])[:4] if "cast" in df.columns and film["cast"] is not None else [],
            "studio": film["studio"] if "studio" in df.columns and isinstance(film["studio"], str) else "",
            "language": lang_label(film["original_language"]),
            "genres": [g.strip() for g in genres_str.split(",") if g.strip()],
            "year": film["release_year"],
            "vote_count": int(film["vote_count"]),
            "vote_average": float(film["vote_average"]) if pd.notna(film["vote_average"]) else None,
            "similarity": sim, "because_of": because, "also_because": also,
            "tag_boost": tag_boost, "match_tags": mtags, "id": film["id"], "corpus_index": idx,
            "poster_path": film["poster_path"] if "poster_path" in df.columns and isinstance(film["poster_path"], str) else "",
            "overview": film["overview"] if "overview" in df.columns and isinstance(film["overview"], str) else "",
        }

    if flat_ranking:
        # Pure score-ordered ranking: merge all queues, keep each candidate's best
        # score, sort globally. No round-robin, no genre cap, no title-dedup —
        # this is the SCORING alone, used by the eval harness to compare against
        # baselines apples-to-apples (and available as a low-diversity mode).
        best = {}
        for paq in per_anchor_queues:
            for idx, s, tb, ci, cv, mt in paq["queue"]:
                cur = best.get(idx)
                if cur is None or s > cur[0]:
                    best[idx] = (s, tb, paq["like"]["matched_title"], ci, cv, mt)
        ranked = sorted(best.items(), key=lambda x: -x[1][0])[:n]
        results = []
        for idx, (s, tb, because, ci, cv, mt) in ranked:
            also = None
            if ci is not None and cv >= 0.30:
                also = liked[ci]["matched_title"]
                if also == because:
                    also = None
            results.append(_film_row(idx, s, tb, because, also, mt))
        return pd.DataFrame(results), top_user_tags

    # --- Selection: round-robin + per-genre cap (water-filled) ----------------
    results, seen_idxs, kept_tokens = [], set(), []
    genre_counts = {}

    def _genre_bucket(film):
        gs = film["genres"] if isinstance(film["genres"], str) else ""
        gset = {g.strip() for g in gs.split(",") if g.strip()}
        # Crime/Thriller ANYWHERE in the genre list counts against one shared
        # bucket — otherwise "Action, Comedy, Crime" launders past a first-genre
        # cap and the list floods with crime-adjacent picks anyway.
        if gset & {"Crime", "Thriller"}:
            return "crime-thriller"
        first = gs.split(",")[0].strip() if gs else ""
        return first or "_none"

    def _fill(enforce_genre_cap):
        while len(results) < n:
            added = 0
            for paq in per_anchor_queues:
                taken = 0
                qi = 0
                while qi < len(paq["queue"]) and taken < per_anchor_cap and len(results) < n:
                    idx, sim, tag_boost, cross_i, cross_val, mtags = paq["queue"][qi]
                    if idx in seen_idxs:
                        paq["queue"].pop(qi)
                        continue
                    film = df.iloc[idx]
                    ftoks = _title_tokens(film["title"])
                    if _is_title_dup(ftoks, kept_tokens):
                        paq["queue"].pop(qi)
                        continue
                    pg = _genre_bucket(film)
                    if enforce_genre_cap and genre_counts.get(pg, 0) >= genre_cap:
                        qi += 1  # leave in queue for the relax pass
                        continue
                    paq["queue"].pop(qi)
                    seen_idxs.add(idx)
                    kept_tokens.append(ftoks)
                    genre_counts[pg] = genre_counts.get(pg, 0) + 1
                    genres_str = film["genres"] if isinstance(film["genres"], str) else ""
                    genres = [g.strip() for g in genres_str.split(",") if g.strip()]
                    also = None
                    # damped scale: kindred-anchor cross lands ~0.30-0.45, unrelated ~0.05-0.15
                    if cross_i is not None and cross_val >= 0.30:
                        also = liked[cross_i]["matched_title"]
                        if also == paq["like"]["matched_title"]:
                            also = None
                    results.append({
                        "title": film["title"],
                        "director": film["director"] if has_director and isinstance(film["director"], str) else "",
                        "cast": list(film["cast"])[:4] if "cast" in df.columns and film["cast"] is not None else [],
                        "studio": film["studio"] if "studio" in df.columns and isinstance(film["studio"], str) else "",
                        "language": lang_label(film["original_language"]),
                        "genres": genres, "year": film["release_year"],
                        "vote_count": int(film["vote_count"]),
                        "vote_average": float(film["vote_average"]) if pd.notna(film["vote_average"]) else None,
                        "similarity": sim, "because_of": paq["like"]["matched_title"],
                        "also_because": also,
                        "tag_boost": tag_boost, "match_tags": mtags,
                        "id": film["id"], "corpus_index": idx,
                        "poster_path": film["poster_path"] if "poster_path" in df.columns and isinstance(film["poster_path"], str) else "",
                        "overview": film["overview"] if "overview" in df.columns and isinstance(film["overview"], str) else "",
                    })
                    taken += 1; added += 1
                if len(results) >= n:
                    break
            if added == 0:
                break

    _fill(enforce_genre_cap=True)
    if len(results) < n:
        _fill(enforce_genre_cap=False)  # water-fill: relax the cap rather than under-deliver
    return pd.DataFrame(results), top_user_tags


def _fuzzy_score(q, title):
    """Typo tolerance that survives a PARTIAL title.

    SequenceMatcher against the whole title punishes length mismatch, so a typo
    in a partial query failed outright even though the intent was obvious:
        brekaing bad  vs breaking bad                     0.92  ok
        gilmore girsl vs gilmore girls a year in the life 0.58  MISSED
        haikyu        vs haikyu to the top                0.52  MISSED
    Comparing against a leading window the length of the query, and against
    each word, recovers those without loosening the threshold.
    """
    if not title:
        return 0.0
    best = SequenceMatcher(None, q, title).ratio()
    window = title[:len(q) + 4]
    if window:
        best = max(best, SequenceMatcher(None, q, window).ratio())
    for w in title.split()[:6]:
        best = max(best, SequenceMatcher(None, q, w).ratio())
    return best


def _strip_year(q):
    """Pull a trailing 4-digit year out of the query.

    A tester reported having to scroll because search "didn't take the year as
    an input" -- typing "fargo 1996" matched nothing, since no title contains
    the year. Returns (query_without_year, year_or_None).
    """
    m = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if not m:
        return q, None
    return re.sub(r"\s+", " ", (q[:m.start()] + q[m.end():])).strip(), int(m.group(1))


def search_movies(query, df, search_idx, limit=12):
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    q = _normalize(query)
    if not q:
        return df.iloc[0:0]
    q_words = q.split()

    titles = pd.Series(search_idx["titles"], index=df.index)
    directors = pd.Series(search_idx["directors"], index=df.index)
    cast = pd.Series(search_idx["cast"], index=df.index)
    
    # Exact substring match across title, director, or cast.
    # RANK BY WHERE IT MATCHED, then by popularity. Ranking purely on
    # vote_count let a cast match on a popular film outrank an exact title
    # match: searching "kiki" returned The Old Guard (KiKi Layne is in the
    # cast) above Kiki's Delivery Service.
    t_hit = titles.str.contains(q, na=False, regex=False)
    d_hit = directors.str.contains(q, na=False, regex=False)
    c_hit = cast.str.contains(q, na=False, regex=False)
    exact = t_hit | d_hit | c_hit
    if exact.any():
        hits = df[exact].copy()
        starts = titles.loc[hits.index].str.startswith(q, na=False)
        # 0 title-prefix, 1 title-substring, 2 director, 3 cast
        hits["_tier"] = np.where(starts, 0,
                         np.where(t_hit.loc[hits.index], 1,
                          np.where(d_hit.loc[hits.index], 2, 3)))
        hits = hits.sort_values(["_tier", "vote_count"], ascending=[True, False])
        return hits.drop(columns="_tier").head(limit)
    
    # All words present in title (multi-word query fallback)
    mask = pd.Series(True, index=df.index)
    for w in q_words:
        mask &= titles.str.contains(w, na=False, regex=False)
    if mask.any():
        return df[mask].nlargest(limit, "vote_count")
    
    # Fuzzy match against titles only (cast/director fuzzy is too noisy)
    # "fargo 1996": no title contains the year, so retry without it, filtered to
    # that year. A FALLBACK, never preprocessing -- stripping first would empty
    # the query for the film "1917" and mangle "Blade Runner 2049", both of
    # which match fine above.
    q_bare, q_year = _strip_year(q)
    if q_year is not None and q_bare and "release_year" in df.columns:
        yr = df[df["release_year"] == q_year]
        if len(yr):
            m = titles.loc[yr.index].str.contains(q_bare, na=False, regex=False)
            if m.any():
                return yr[m].nlargest(limit, "vote_count")

    shortlist = df.nlargest(8000, "vote_count")
    short_titles = titles.loc[shortlist.index]
    scores = short_titles.apply(lambda t: _fuzzy_score(q, t))
    good = scores[scores >= 0.7].sort_values(ascending=False)
    if len(good):
        return shortlist.loc[good.index[:limit]]
    return df.iloc[0:0]

# ===========================================================================
# TV RECOMMENDER (mirrors movies; no tag layer in v1, multi-valued creator)
# ===========================================================================
def recommend_tv(ratings, df, embeddings, index, tags_df=None, n=20,
                 min_votes=300, like_threshold=3.5, per_anchor_cap=3, tag_weight=0.25,
                 maturity_drop=TV_MATURITY_DROP,
                 exclude_rated_creators=True, title_token_penalty=0.2, skip_ids=None,
                 neutral=3.0, cross_weight=0.35, neg_scale=1.0,
                 quality_weight=0.12, min_vote_average=6.0, genre_cap=6,
                 retrieve_k=150, shrink_votes=500, flat_ranking=False):
    """Whole-profile ranking (same design as recommend_movies), with the tag
    layer applied as a POOL-CENTRED ADJUSTMENT rather than a blend.

    Why centred: tag_sim (cosine between sparse tag vectors, ~0.0-0.6) and
    emb_score (~0.7-0.9) are not on the same scale. The old form
        (1 - w) * emb_score + w * tag_sim
    therefore subtracted from every tagged candidate and left untagged ones
    untouched -- measured on one profile at 127/127 tagged candidates hurt,
    0 helped, dropping the tagged share of the top 15 to 27% out of a 91%
    tagged pool. Centring on the pool mean makes the tag term signed:
        emb_score + w * (tag_sim - pool_mean)
    Above-average tag match lifts, below-average pushes down, untagged sits
    at a neutral 0. Needs two passes: the mean isn't known until every
    candidate has been scored.

    TMDB TV has no 'Thriller' genre; the family bucket is Crime/Mystery.
    """
    liked = [r for r in ratings if r["rating"] >= like_threshold]
    liked = sorted(liked, key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame(), None

    anchor_vecs = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    anchor_w01 = np.array([(r["rating"] - neutral) / 2.0 for r in liked], dtype="float32")

    if len(liked) > 1:
        _order = [0]
        _rem = set(range(1, len(liked)))
        while _rem:
            best_i, best_s = None, -1e9
            for i in _rem:
                d = 1.0 - max(float(anchor_vecs[j] @ anchor_vecs[i]) for j in _order)
                s = d + 0.15 * float(anchor_w01[i])
                if s > best_s:
                    best_s, best_i = s, i
            _order.append(best_i)
            _rem.discard(best_i)
        liked = [liked[i] for i in _order]
        anchor_vecs = anchor_vecs[_order]
        anchor_w01 = anchor_w01[_order]

    _anchor_damp = np.clip(anchor_vecs @ anchor_vecs.T, 0, None) ** 2

    disliked = [r for r in ratings if r["rating"] < neutral]
    dis_vecs = (np.array([embeddings[r["corpus_index"]] for r in disliked]).astype("float32")
                if disliked else None)
    dis_w01 = (np.array([(neutral - r["rating"]) / 2.0 for r in disliked], dtype="float32")
               if disliked else None)

    rated_tv_ids = {str(t) for t in {r["tv_id"] for r in ratings}}
    if skip_ids:
        rated_tv_ids = rated_tv_ids | {str(s) for s in skip_ids}
    # --- viewer maturity baseline, computed once -------------------------
    # Median over positively-rated shows with a known level. Median not mean:
    # one Paw Patrol rating should not drag an adult profile down, and one
    # Breaking Bad should not push a children's profile up.
    # Inert when the corpus predates the enrichment, so the T_CORPUS flip and
    # this code can deploy in either order without a KeyError.
    _profile_maturity = None
    if maturity_drop and "maturity" in df.columns:
        _lv = []
        for r in ratings:
            ci = r.get("corpus_index")
            if ci is None or not (0 <= int(ci) < len(df)):
                continue
            if r["rating"] >= like_threshold:
                m = df.iloc[int(ci)]["maturity"]
                if pd.notna(m):
                    _lv.append(float(m))
        if len(_lv) >= TV_MATURITY_MIN_RATED:
            _profile_maturity = float(np.median(_lv))

    # --- same-title reboots ------------------------------------------------
    # title_token_penalty only compares a candidate against the ONE anchor that
    # retrieved it. Avatar: The Last Airbender (2024) came back through The
    # Legend of Korra -- tokens {legend, korra}, which share nothing with the
    # remake's title -- so the penalty was zero and the 2024 live-action reboot
    # was recommended to someone who had rated the 2005 original.
    #
    # Deliberately EXACT token-set equality, not overlap: that catches reboots
    # carrying the same name while leaving sequels and spin-offs to the
    # existing penalty, which is already the most expensive component in the
    # movie ablation (-22% recall@20). Widening it here would cost more.
    _rated_title_toks = set()
    for r in ratings:
        _tt = _title_tokens(r.get("matched_title", ""))
        if _tt:
            _rated_title_toks.add(frozenset(_tt))

    rated_creators = set()
    for r in ratings:
        rated_creators |= _creator_names(r.get("matched_creator"))

    # --- Signed tag profile over ALL rated + tagged series --------------------
    tag_profile = None
    top_user_tags = None
    if tags_df is not None:
        t_vecs, t_w = [], []
        for r in ratings:
            w = r["rating"] - neutral
            if w == 0:
                continue
            tid = str(r["tv_id"])
            if tid in tags_df.index:
                t_vecs.append(tags_df.loc[tid].values)
                t_w.append(w * neg_scale if w < 0 else w)
        if t_vecs:
            _tw = np.array(t_w, dtype="float32")
            tag_profile = (np.array(t_vecs, dtype="float32") * _tw[:, None]).sum(axis=0) / np.abs(_tw).sum()
            top_user_tags = pd.Series(tag_profile, index=tags_df.columns).nlargest(5)

    all_sims, all_idxs = index.search(anchor_vecs, retrieve_k)
    has_creator = "created_by" in df.columns

    # Bayesian shrink for vote_average -- see recommend_movies. TV vote counts
    # run smaller, so m is smaller too; a "8.6" at 158 votes collapses to ~7.
    _C = float(df.loc[df["vote_count"] >= min_votes, "vote_average"].mean())

    def _va_shrunk(va, vc):
        return (vc * va + shrink_votes * _C) / (vc + shrink_votes)

    _cand_cache = {}

    def _cand_signals(idx):
        if idx in _cand_cache:
            return _cand_cache[idx]
        cvec = np.asarray(embeddings[idx], dtype="float32")
        cross_all = (anchor_vecs @ cvec) * anchor_w01
        neg_pen = 0.0
        if dis_vecs is not None:
            neg_pen = float(np.clip((dis_vecs @ cvec) * dis_w01, 0, None).max())
        _cand_cache[idx] = (cross_all, neg_pen)
        return _cand_cache[idx]

    # --- PASS 1: embedding score + raw tag sims (no blending yet) -------------
    _all_tag_sims = []
    per_anchor_queues = []
    for like_i, (sims, idxs) in enumerate(zip(all_sims, all_idxs)):
        seed_tokens = _title_tokens(liked[like_i].get("matched_title", ""))
        queue = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            show = df.iloc[idx]
            if str(show["tmdb_id"]) in rated_tv_ids:
                continue
            if _rated_title_toks:
                _cand_toks = _title_tokens(show["name"])
                if _cand_toks and frozenset(_cand_toks) in _rated_title_toks:
                    continue
            if _profile_maturity is not None:
                _m = show["maturity"]
                # Unknown stays in. An absent rating carries no information
                # about age suitability and must not be read as "low".
                if pd.notna(_m) and (_profile_maturity - float(_m)) >= maturity_drop:
                    continue
            if int(show["vote_count"]) < min_votes:
                continue
            va = float(show["vote_average"]) if pd.notna(show["vote_average"]) else None
            if va is None:
                continue
            va_adj = _va_shrunk(va, int(show["vote_count"]))
            if va_adj < min_vote_average:
                continue  # hard quality floor (on the shrunk value)
            if exclude_rated_creators and has_creator and rated_creators:
                if _creator_names(show["created_by"]) & rated_creators:
                    continue

            cross_all, neg_pen = _cand_signals(int(idx))
            if len(cross_all) > 1:
                damped = cross_all * _anchor_damp[like_i]
                others = np.delete(damped, like_i)
                cross_j = int(np.argmax(others))
                cross_val = float(others[cross_j])
                cross_anchor_i = cross_j if cross_j < like_i else cross_j + 1
            else:
                cross_val, cross_anchor_i = 0.0, None

            quality = (va_adj - min_vote_average) / (10.0 - min_vote_average)
            emb_score = (float(sim)
                         + cross_weight * max(cross_val, 0.0)
                         - neg_scale * neg_pen
                         + quality_weight * quality)

            tag_sim = None
            match_tags = []
            if tag_profile is not None and str(show["tmdb_id"]) in tags_df.index:
                show_tags = tags_df.loc[str(show["tmdb_id"])].values
                nu, ns = np.linalg.norm(tag_profile), np.linalg.norm(show_tags)
                if nu > 0 and ns > 0:
                    tag_sim = float(np.dot(tag_profile, show_tags) / (nu * ns))
                    _all_tag_sims.append(tag_sim)
                    match_tags = _matching_tags(tag_profile, show_tags, tags_df.columns)

            shared = seed_tokens & _title_tokens(show["name"])
            title_pen = (title_token_penalty * min(len(shared), 3)
                         if (title_token_penalty and shared) else 0.0)
            queue.append([int(idx), emb_score, tag_sim, title_pen, cross_anchor_i, cross_val, match_tags])
        per_anchor_queues.append({"like": liked[like_i], "queue": queue})

    # --- PASS 2: centre the tag signal on the candidate pool ------------------
    _mu = float(np.mean(_all_tag_sims)) if _all_tag_sims else 0.0
    for paq in per_anchor_queues:
        scored = []
        for idx, emb_score, tag_sim, title_pen, ci, cv, mtags in paq["queue"]:
            adj = tag_weight * (tag_sim - _mu) if tag_sim is not None else 0.0
            final_sim = emb_score + adj - title_pen
            if final_sim < 0.0:
                continue  # pushed below zero by dislike/title penalties
            # badge means "tags LIFTED this", not merely "is tagged" -- candidate
            # pools run ~90% tagged, so a tagged-ness badge would be noise.
            boosted = bool(tag_sim is not None and adj > 0)
            scored.append((idx, final_sim, boosted, ci, cv, mtags if boosted else []))
        scored.sort(key=lambda x: -x[1])
        paq["queue"] = scored

    results, seen_idxs, kept_tokens = [], set(), []
    genre_counts = {}

    def _genre_bucket(show):
        gs = show["genres"] if isinstance(show["genres"], str) else ""
        gset = {g.strip() for g in gs.split(",") if g.strip()}
        if gset & {"Crime", "Mystery"}:
            return "crime-mystery"
        first = gs.split(",")[0].strip() if gs else ""
        return first or "_none"

    def _fill(enforce_genre_cap):
        while len(results) < n:
            added = 0
            for psq in per_anchor_queues:
                taken = 0
                qi = 0
                while qi < len(psq["queue"]) and taken < per_anchor_cap and len(results) < n:
                    idx, sim, tag_boost, cross_i, cross_val, mtags = psq["queue"][qi]
                    if idx in seen_idxs:
                        psq["queue"].pop(qi)
                        continue
                    show = df.iloc[idx]
                    stoks = _title_tokens(show["name"])
                    if _is_title_dup(stoks, kept_tokens):
                        psq["queue"].pop(qi)
                        continue
                    pg = _genre_bucket(show)
                    if enforce_genre_cap and genre_counts.get(pg, 0) >= genre_cap:
                        qi += 1
                        continue
                    psq["queue"].pop(qi)
                    seen_idxs.add(idx)
                    kept_tokens.append(stoks)
                    genre_counts[pg] = genre_counts.get(pg, 0) + 1
                    genres_str = show["genres"] if isinstance(show["genres"], str) else ""
                    genres = [g.strip() for g in genres_str.split(",") if g.strip()]
                    seasons = (int(show["number_of_seasons"])
                               if "number_of_seasons" in df.columns and pd.notna(show["number_of_seasons"]) else 0)
                    also = None
                    if cross_i is not None and cross_val >= 0.30:
                        also = liked[cross_i]["matched_title"]
                        if also == psq["like"]["matched_title"]:
                            also = None
                    results.append({
                        "name": show["name"],
                        "creator": show["created_by"] if has_creator and isinstance(show["created_by"], str) else "",
                        "network": show["networks"] if "networks" in df.columns and isinstance(show["networks"], str) else "",
                        "language": lang_label(show["original_language"]),
                        "genres": genres, "year": show["start_year"],
                        "seasons": seasons,
                        "vote_count": int(show["vote_count"]),
                        "vote_average": float(show["vote_average"]) if pd.notna(show["vote_average"]) else None,
                        "similarity": sim, "because_of": psq["like"]["matched_title"],
                        "also_because": also,
                        "tag_boost": tag_boost, "match_tags": mtags,
                        "tmdb_id": show["tmdb_id"], "corpus_index": idx,
                        "poster_path": show["poster_path"] if "poster_path" in df.columns and isinstance(show["poster_path"], str) else "",
                        "overview": show["overview"] if "overview" in df.columns and isinstance(show["overview"], str) else "",
                    })
                    taken += 1; added += 1
                if len(results) >= n:
                    break
            if added == 0:
                break

    if flat_ranking:
        # Pure score-ordered ranking: merge all queues, keep each candidate's
        # best score, sort globally. No round-robin, no genre cap, no
        # title-dedup. Mirrors recommend_movies' flat_ranking so probes and
        # eval harnesses can examine the SCORING alone. The movie ranker gained
        # this when the genre tabs were built; TV never did, which is why
        # probe_genre_tabs --kind tv failed on an unexpected keyword.
        best = {}
        for paq in per_anchor_queues:
            for idx, final_sim, boosted, ci, cv, mtags in paq["queue"]:
                cur = best.get(idx)
                if cur is None or final_sim > cur[0]:
                    best[idx] = (final_sim, boosted,
                                 paq["like"]["matched_title"], ci, cv, mtags)
        ranked = sorted(best.items(), key=lambda x: -x[1][0])[:n]
        flat = []
        for idx, (sim, tag_boost, because, ci, cv, mtags) in ranked:
            show = df.iloc[idx]
            also = None
            if ci is not None and cv >= 0.30:
                also = liked[ci]["matched_title"]
                if also == because:
                    also = None
            _seasons = show["number_of_seasons"] if "number_of_seasons" in df.columns else None
            flat.append({
                "name": show["name"],
                "creator": show["created_by"] if isinstance(show["created_by"], str) else "",
                "network": show["networks"] if "networks" in df.columns and isinstance(show["networks"], str) else "",
                "language": lang_label(show["original_language"]),
                # A LIST, matching the normal path (line ~2047). The card does
                # ", ".join(rec["genres"][:3]) -- handing it a string slices
                # three CHARACTERS and renders "A, n, i".
                "genres": [g.strip() for g in
                           (show["genres"] if isinstance(show["genres"], str) else "").split(",")
                           if g.strip()],
                "year": show["start_year"],
                "seasons": int(_seasons) if pd.notna(_seasons) else None,
                "vote_count": int(show["vote_count"]),
                "vote_average": float(show["vote_average"]) if pd.notna(show["vote_average"]) else None,
                "similarity": sim, "because_of": because, "also_because": also,
                "tag_boost": tag_boost, "match_tags": mtags,
                "tmdb_id": show["tmdb_id"], "corpus_index": idx,
                "poster_path": show["poster_path"] if "poster_path" in df.columns and isinstance(show["poster_path"], str) else "",
                "overview": show["overview"] if "overview" in df.columns and isinstance(show["overview"], str) else "",
            })
        return pd.DataFrame(flat), top_user_tags

    _fill(enforce_genre_cap=True)
    if len(results) < n:
        _fill(enforce_genre_cap=False)
    return pd.DataFrame(results), top_user_tags


def search_tv(query, df, search_idx, limit=12):
    if not query or len(query.strip()) < 2:
        return df.iloc[0:0]
    q = _normalize(query)
    if not q:
        return df.iloc[0:0]
    q_words = q.split()

    titles = pd.Series(search_idx["titles"], index=df.index)
    creators = pd.Series(search_idx["creators"], index=df.index)
    
    # Exact substring match across name or creator, ranked by WHERE it
    # matched before popularity (see search_movies for the "kiki" case).
    t_hit = titles.str.contains(q, na=False, regex=False)
    c_hit = creators.str.contains(q, na=False, regex=False)
    exact = t_hit | c_hit
    if exact.any():
        hits = df[exact].copy()
        starts = titles.loc[hits.index].str.startswith(q, na=False)
        hits["_tier"] = np.where(starts, 0, np.where(t_hit.loc[hits.index], 1, 2))
        hits = hits.sort_values(["_tier", "vote_count"], ascending=[True, False])
        return hits.drop(columns="_tier").head(limit)
    
    mask = pd.Series(True, index=df.index)
    for w in q_words:
        mask &= titles.str.contains(w, na=False, regex=False)
    if mask.any():
        return df[mask].nlargest(limit, "vote_count")
    
    # "fargo 1996": no title contains the year, so retry without it, filtered to
    # that year. A FALLBACK, never preprocessing -- stripping first would empty
    # the query for the film "1917" and mangle "Blade Runner 2049", both of
    # which match fine above.
    q_bare, q_year = _strip_year(q)
    if q_year is not None and q_bare and "start_year" in df.columns:
        yr = df[df["start_year"] == q_year]
        if len(yr):
            m = titles.loc[yr.index].str.contains(q_bare, na=False, regex=False)
            if m.any():
                return yr[m].nlargest(limit, "vote_count")

    shortlist = df.nlargest(8000, "vote_count")
    short_titles = titles.loc[shortlist.index]
    scores = short_titles.apply(lambda t: _fuzzy_score(q, t))
    good = scores[scores >= 0.7].sort_values(ascending=False)
    if len(good):
        return shortlist.loc[good.index[:limit]]
    return df.iloc[0:0]


# ===========================================================================
# CALLBACKS (namespaced)
# ===========================================================================
def b_add(book_id, title, author, corpus_index):
    stars = st.session_state.get(f"b_rate_{book_id}")   # st.feedback stars: 0-4 -> 1-5
    if stars is None or any(r["book_id"] == book_id for r in st.session_state.ratings):
        return
    st.session_state.ratings.append({
        "book_id": book_id, "matched_title": title, "matched_author": author,
        "corpus_index": int(corpus_index), "rating": float(stars + 1)})


def b_remove(book_id):
    st.session_state.ratings = [r for r in st.session_state.ratings if r["book_id"] != book_id]


def b_clear():
    st.session_state.ratings = []
    st.session_state.skipped = []


def b_add_watchlist(item):
    if not any(w["book_id"] == item["book_id"] for w in st.session_state.watchlist):
        st.session_state.watchlist.append(item)


def b_remove_watchlist(book_id):
    st.session_state.watchlist = [w for w in st.session_state.watchlist if w["book_id"] != book_id]


def b_skip(book_id):
    if book_id not in st.session_state.skipped:
        st.session_state.skipped.append(book_id)


def m_add(movie_id, title, director, corpus_index):
    stars = st.session_state.get(f"m_rate_{movie_id}")   # st.feedback stars: 0-4 -> 1-5
    if stars is None or any(r["movie_id"] == movie_id for r in st.session_state.movie_ratings):
        return
    st.session_state.movie_ratings.append({
        "movie_id": int(movie_id), "matched_title": title, "matched_director": director,
        "corpus_index": int(corpus_index), "rating": float(stars + 1)})


def m_remove(movie_id):
    st.session_state.movie_ratings = [r for r in st.session_state.movie_ratings if r["movie_id"] != movie_id]


def m_clear():
    st.session_state.movie_ratings = []
    st.session_state.movie_skipped = []


def m_add_watchlist(item):
    if not any(w["movie_id"] == item["movie_id"] for w in st.session_state.movie_watchlist):
        st.session_state.movie_watchlist.append(item)


def m_remove_watchlist(movie_id):
    st.session_state.movie_watchlist = [w for w in st.session_state.movie_watchlist if w["movie_id"] != movie_id]


def m_skip(movie_id):
    if movie_id not in st.session_state.movie_skipped:
        st.session_state.movie_skipped.append(movie_id)


def t_add(tv_id, name, creator, corpus_index):
    stars = st.session_state.get(f"t_rate_{tv_id}")   # st.feedback stars: 0-4 -> 1-5
    if stars is None or any(r["tv_id"] == tv_id for r in st.session_state.tv_ratings):
        return
    st.session_state.tv_ratings.append({
        "tv_id": int(tv_id), "matched_title": name, "matched_creator": creator,
        "corpus_index": int(corpus_index), "rating": float(stars + 1)})


def t_remove(tv_id):
    st.session_state.tv_ratings = [r for r in st.session_state.tv_ratings if r["tv_id"] != tv_id]


def t_clear():
    st.session_state.tv_ratings = []
    st.session_state.tv_skipped = []


def t_add_watchlist(item):
    if not any(w["tv_id"] == item["tv_id"] for w in st.session_state.tv_watchlist):
        st.session_state.tv_watchlist.append(item)


def t_remove_watchlist(tv_id):
    st.session_state.tv_watchlist = [w for w in st.session_state.tv_watchlist if w["tv_id"] != tv_id]


def t_skip(tv_id):
    if tv_id not in st.session_state.tv_skipped:
        st.session_state.tv_skipped.append(tv_id)


# ----- rate-from-recommendation-card (the "already watched? -> rate" loop) -----
# These mirror the search-add callbacks but read a distinct widget key
# (*_recrate_*) so a rec card's rater never collides with the search rater.
def b_rate_rec(book_id, title, author, corpus_index, widget_key=None):
    stars = st.session_state.get(widget_key or f"b_recrate_{book_id}")   # st.feedback stars: 0-4 -> 1-5
    if stars is None or any(r["book_id"] == book_id for r in st.session_state.ratings):
        return
    st.session_state.ratings.append({
        "book_id": book_id, "matched_title": title, "matched_author": author or "",
        "corpus_index": int(corpus_index), "rating": float(stars + 1)})
    st.session_state["_b_rated_toast"] = title


def m_rate_rec(movie_id, title, director, corpus_index, widget_key=None):
    # widget_key is passed explicitly: rec cards appear in several tabs, so the
    # star widget's key is namespaced (m_recrate_<tab>_<id>). Rebuilding the key
    # from movie_id alone silently read None and the rating vanished.
    key = widget_key or f"m_recrate_{movie_id}"
    stars = st.session_state.get(key)                       # st.feedback stars: 0-4 -> 1-5
    if stars is None or any(r["movie_id"] == movie_id for r in st.session_state.movie_ratings):
        return
    st.session_state.movie_ratings.append({
        "movie_id": int(movie_id), "matched_title": title, "matched_director": director or "",
        "corpus_index": int(corpus_index), "rating": float(stars + 1)})
    # Heaton: "I don't get a new recommendation to replace that one." It IS
    # replaced on the next run, but the list re-ranks and the new entry lands at
    # the bottom, so nothing visibly happens where you clicked.
    st.session_state["_m_rated_toast"] = title


def t_rate_rec(tv_id, name, creator, corpus_index, widget_key=None):
    # widget_key is passed explicitly for the same reason as m_rate_rec: once
    # rec cards appear in several tabs the star key is namespaced
    # (t_recrate_<tab>_<id>), and rebuilding it from tv_id alone reads None,
    # returns early, and silently discards the rating. py_compile passes, no
    # exception is raised, and the star even fills in.
    stars = st.session_state.get(widget_key or f"t_recrate_{tv_id}")
    if stars is None or any(r["tv_id"] == tv_id for r in st.session_state.tv_ratings):
        return
    st.session_state.tv_ratings.append({
        "tv_id": int(tv_id), "matched_title": name, "matched_creator": creator or "",
        "corpus_index": int(corpus_index), "rating": float(stars + 1)})
    st.session_state["_t_rated_toast"] = name


# ===========================================================================
# LOCAL-STORAGE PERSISTENCE (per-browser; survives refresh/close — no login)
# ===========================================================================
# Isolated and defensive. If the component fails to install/load on the Space,
# flip ENABLE_PERSISTENCE to False and the app reverts to in-session behaviour
# (the previous working build). Everything below is wrapped so a storage failure
# degrades to "ratings don't persist", never "app breaks" — EXCEPT the library's
# own blocking init (see get_local_storage), which the kill switch guards.
ENABLE_PERSISTENCE = True

# Temporary: shows what localStorage actually contains, so we can tell apart
# "component never loaded" from "data saved but not restoring". Set False once
# persistence is confirmed working.
SHOW_LS_DEBUG = False

_LS_AVAILABLE = False
if ENABLE_PERSISTENCE:
    try:
        from streamlit_local_storage import LocalStorage
        _LS_AVAILABLE = True
    except Exception as _e:
        print(f"streamlit-local-storage unavailable, persistence off: {_e}")

# session-state ratings key -> localStorage item key
LS_KEYS = {
    "ratings": "re_books", "movie_ratings": "re_movies", "tv_ratings": "re_tv",
    "watchlist": "re_books_wl", "movie_watchlist": "re_movies_wl", "tv_watchlist": "re_tv_wl",
    "skipped": "re_books_skip", "movie_skipped": "re_movies_skip", "tv_skipped": "re_tv_skip",
}


def _ls_norm(value):
    return json.dumps(value, sort_keys=True)


def get_local_storage():
    if not (ENABLE_PERSISTENCE and _LS_AVAILABLE):
        return None
    try:
        return LocalStorage(key="re_localstorage")
    except Exception as e:
        print(f"LocalStorage init failed, persistence off this run: {e}")
        return None


_VOL_WORD = re.compile(r"\b(vol|volume|bk|book|no|number)\b\.?\s*(\d+)")
_SERIES_SUFFIX = re.compile(r"\(([^)]*#\d+[^)]*)\)")


def _core_title(t):
    """Strip series suffix, subtitle, and volume-word variance.

    Both of these collapse to 'scott pilgrim v1':
        'Scott Pilgrim, Volume 1: Scott Pilgrim's Precious Little Life' (stored)
        'Scott Pilgrim Vol. 1: Scott's Precious Little Life'            (corpus)

    The raw title is split BEFORE normalising -- _normalize turns ':' into a
    space, so subtitle stripping has to happen first or it never fires.
    """
    if not isinstance(t, str):
        return ""
    t = _SERIES_SUFFIX.sub(" ", t)
    head_raw = re.split(r":| - ", t)[0]
    n_full = _VOL_WORD.sub(r"v\2", _normalize(t))
    vol = re.search(r"\bv\d+\b", n_full)
    head = _VOL_WORD.sub(r"v\2", _normalize(head_raw))
    if vol and vol.group(0) not in head:
        head = f"{head} {vol.group(0)}"
    return head.strip()


def _surname(a):
    n = _normalize(a)
    return n.split()[-1] if n else ""


# A tester rated 43 films and had to scroll to the bottom of the page to reach
# the last one, then all the way back up to the search box. A fixed-height
# scroll box only moved the problem inside a smaller window; sorting plus
# pages makes a long list navigable instead of merely contained. Heavy raters
# are the users the ranker serves best, so the UI should not punish them.
RATED_PAGE_SIZE = 15

# Ratings carry no timestamp, but session-state order IS insertion order, so
# first/most-recent come free without touching the stored schema.
RATED_SORTS = {
    "Recently rated": lambda xs: list(reversed(xs)),
    "First rated": lambda xs: list(xs),
    "Rating: high to low": lambda xs: sorted(xs, key=lambda r: -float(r.get("rating", 0))),
    "Rating: low to high": lambda xs: sorted(xs, key=lambda r: float(r.get("rating", 0))),
    "Title A-Z": lambda xs: sorted(xs, key=lambda r: str(r.get("matched_title", "")).lower()),
    "Title Z-A": lambda xs: sorted(xs, key=lambda r: str(r.get("matched_title", "")).lower(),
                                   reverse=True),
}


def rated_list_page(items, key_prefix, default_sort="Rating: high to low"):
    """Header row + sort control. Returns the rows for the current page.

    Page buttons are NOT drawn here -- call rated_list_pager() after the rows,
    so the control sits under the list it pages through rather than above it.
    """
    n = len(items)
    sort_key, page_key = f"{key_prefix}_ratesort", f"{key_prefix}_ratepage"
    if sort_key not in st.session_state:
        st.session_state[sort_key] = default_sort

    def _reset_page():
        st.session_state[page_key] = 0

    pages = max(1, (n + RATED_PAGE_SIZE - 1) // RATED_PAGE_SIZE)
    page = min(int(st.session_state.get(page_key, 0)), pages - 1)
    st.session_state[page_key] = page
    st.session_state[f"{key_prefix}_ratepages"] = pages

    if n > 1:
        left, right = st.columns([1, 1.1])
        with left:
            start = page * RATED_PAGE_SIZE
            end = min(start + RATED_PAGE_SIZE, n)
            st.markdown(f"<div class='list-count'>Showing {start + 1}-{end} of {n}</div>",
                        unsafe_allow_html=True)
        with right:
            st.selectbox("Sort by", list(RATED_SORTS), key=sort_key,
                         on_change=_reset_page, label_visibility="collapsed")

    ordered = RATED_SORTS.get(st.session_state.get(sort_key, default_sort),
                              RATED_SORTS[default_sort])(items)
    start = page * RATED_PAGE_SIZE
    return ordered[start:start + RATED_PAGE_SIZE]


def rated_list_pager(key_prefix):
    """Numbered page buttons, rendered BELOW the rows."""
    pages = int(st.session_state.get(f"{key_prefix}_ratepages", 1))
    if pages < 2:
        return
    page_key = f"{key_prefix}_ratepage"
    page = int(st.session_state.get(page_key, 0))
    shown = min(pages, 8)
    # trailing spacer keeps the buttons compact instead of stretched edge to edge
    cols = st.columns([1] * shown + [max(1.0, 9.0 - shown)])
    for i in range(shown):
        cols[i].button(
            f"{i + 1}", key=f"{key_prefix}_pg_{i}",
            disabled=(i == page),          # current page reads as "you are here"
            use_container_width=True,
            on_click=lambda i=i: st.session_state.update({page_key: i}))


def export_payload():
    """Everything the user has entered, in one JSON blob."""
    return json.dumps({
        "app": "recommendation-engine",
        "version": 1,
        "exported": pd.Timestamp.now("UTC").isoformat(),
        "data": {k: st.session_state.get(k, []) for k in LS_KEYS},
    }, ensure_ascii=False, indent=2)


def import_payload(raw):
    """Load an exported blob back into session state.

    Ratings live only in localStorage: clearing browser data destroys them,
    and two testers asked for this after losing (or fearing losing) profiles
    they had spent real time building. Row positions differ between corpus
    versions, so imported book ratings go back through the same migration
    used at hydrate rather than being trusted as-is.
    """
    try:
        payload = json.loads(raw)
    except Exception:
        return False, "That file isn't valid JSON."
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return False, "That doesn't look like an export from this app."

    loaded, notes = 0, []
    for ss_key in LS_KEYS:
        items = data.get(ss_key)
        if not isinstance(items, list):
            continue
        items = [r for r in items if isinstance(r, dict)]
        try:
            if ss_key == "ratings" and b_df is not None:
                items, _m, _d, _mg = _migrate_book_ratings(items, b_df)
                if _d:
                    notes.append(f"{_d} book rating(s) no longer in the catalogue")
            elif ss_key == "movie_ratings" and m_df is not None:
                items, _d = _drop_stale_indices(items, len(m_df))
            elif ss_key == "tv_ratings" and t_df is not None:
                items, _d = _drop_stale_indices(items, len(t_df))
        except Exception as e:
            # A validation failure must not cost the user the import.
            print(f"import validation for {ss_key} failed: {e}")
        st.session_state[ss_key] = items
        loaded += len(items)
    msg = f"Imported {loaded} item(s)."
    if notes:
        msg += " " + "; ".join(notes) + "."
    return True, msg


def backup_controls(key_prefix):
    with st.expander("Back up or restore your ratings"):
        st.caption("Your ratings live in this browser only — clearing site data "
                   "wipes them. Download a copy, or restore one here.")
        st.download_button("Download my ratings (JSON)", data=export_payload(),
                           file_name="my-recommendation-profile.json",
                           mime="application/json", use_container_width=True,
                           key=f"{key_prefix}_dl")
        up = st.file_uploader("Restore from a file", type=["json"],
                              key=f"{key_prefix}_up", label_visibility="collapsed")
        if up is not None and not st.session_state.get(f"_imported_{up.name}_{up.size}"):
            ok, msg = import_payload(up.getvalue().decode("utf-8", "replace"))
            st.session_state[f"_imported_{up.name}_{up.size}"] = True
            if ok:
                persist_to_ls()
                st.success(msg)
            else:
                st.error(msg)


def profile_coach(items, noun, verb):
    """Onboarding copy + a live nudge toward a USABLE profile.

    The old caption said "Aim for 3+", which testers read as a target rather
    than a floor -- one rated exactly three, got weak results, and concluded
    the engine was bad. Worse, it never asked for low ratings, and the
    ablation says removing negatives HALVES recall: dislikes are the second
    most load-bearing signal in the ranker after the quality floor.
    """
    n = len(items)
    lows = sum(1 for r in items if float(r.get("rating", 0)) <= 2)
    highs = sum(1 for r in items if float(r.get("rating", 0)) >= 4)

    st.caption(f"Search, score, search again. **The more you {verb}, the better this gets** "
               f"— 15-20 is where it starts working properly, and rating things you "
               f"*disliked* matters as much as things you loved.")

    if n == 0:
        return
    if n < 8:
        msg, icon = f"{n} rated — keep going, results are rough below ~10.", "🔴"
    elif n < 15:
        msg, icon = f"{n} rated — getting there.", "🟡"
    else:
        msg, icon = f"{n} rated — good profile size.", "🟢"
    if n >= 5 and lows == 0:
        msg += f" Add a few 1-2★ {noun} you didn't like — that's what stops "
        msg += "recommendations drifting."
        icon = "🟡"
    elif n >= 5 and highs == 0:
        msg += " Add some 4-5★ favourites so it knows what to aim at."
        icon = "🟡"
    st.caption(f"{icon} {msg}")


def clear_all_button(ss_flag, on_confirm, key_prefix, noun):
    """Two-step Clear all.

    A tester lost 21 ratings to a single stray click: the button sat directly
    under the search box, full width, styled like every other control, and
    fired immediately. Ratings live only in localStorage, so there is nothing
    to recover them from. Arming first, then confirming, costs one extra click
    on a destructive action nobody performs often.
    """
    if st.session_state.get(ss_flag):
        st.warning(f"Delete all your {noun}? This cannot be undone.")
        c1, c2 = st.columns(2)
        c1.button("Yes, delete", key=f"{key_prefix}_yes", type="primary",
                  use_container_width=True, on_click=on_confirm)
        c2.button("Cancel", key=f"{key_prefix}_no", use_container_width=True,
                  on_click=lambda: st.session_state.update({ss_flag: False}))
    else:
        st.button("Clear all", key=f"{key_prefix}_arm", use_container_width=True,
                  on_click=lambda: st.session_state.update({ss_flag: True}))


@st.cache_resource
def _book_match_index(_df):
    """Lookup tables for re-matching stored ratings across a corpus swap.

    Exact title+author was the only tier originally, which dropped 18 of 50
    ratings on the Goodreads->Hardcover swap and reported them as "not in the
    catalogue". 14 of the 18 were in fact present under a different title form
    -- omnibus editions, "Volume" vs "Vol.", added subtitles. Only 4 were
    genuinely absent. Hence the extra tiers in _match_book.
    """
    by_pair, by_title = {}, {}
    by_author, core_by_author = {}, {}
    titles = _df["title"].tolist()
    authors = (_df["primary_author"].tolist() if "primary_author" in _df.columns
               else [""] * len(_df))
    ids = _df["book_id"].astype(str).tolist()
    ratings = (_df["ratings_count"].tolist() if "ratings_count" in _df.columns
               else [0] * len(_df))
    for pos, (t, a, bid, rc) in enumerate(zip(titles, authors, ids, ratings)):
        nt, na = _normalize(t), _normalize(a)
        if not nt:
            continue
        by_pair.setdefault(f"{nt}|{na}", (bid, pos))
        by_title.setdefault(nt, (bid, pos))   # first wins; df is rank-ordered
        sn = _surname(a)
        if sn:
            ct = _core_title(t)
            by_author.setdefault(sn, []).append((pos, ct, bid, rc))
            core_by_author.setdefault(sn, {}).setdefault(ct, (bid, pos))
    return by_pair, by_title, by_author, core_by_author


def _match_book(title, author, idx):
    """(book_id, position) or None.

    Never matches across authors -- that constraint is what stops the fuzzy
    tier pairing 'The Maltese Falcon' with 'The Thin Man'.
    """
    by_pair, by_title, by_author, core_by_author = idx
    nt, na = _normalize(title), _normalize(author)
    sn = _surname(author)

    hit = by_pair.get(f"{nt}|{na}") or by_title.get(nt)
    if hit:
        return hit

    core = _core_title(title)
    hit = core_by_author.get(sn, {}).get(core)
    if hit:
        return hit

    # Leading whole-word prefix, same author: catches comma-subtitles such as
    # "Babel" -> "Babel, or The Necessity of Violence: ...", which the
    # colon/dash split cannot see.
    if core and len(core) >= 4:
        pref = [(pos, bid, rc) for pos, ct, bid, rc in by_author.get(sn, [])
                if ct == core or ct.startswith(core + " ")]
        if pref:
            best = max(pref, key=lambda x: x[2])
            return (best[1], best[0])

    best, best_r = None, 0.0
    for pos, ct, bid, _rc in by_author.get(sn, []):
        r = SequenceMatcher(None, core, ct).ratio()
        if r > best_r:
            best, best_r = (bid, pos), r
    if best and best_r >= 0.72:
        return best

    # Omnibus / boxset -> its most-read component work.
    parts = re.split(r"[;&]| and (?=[A-Z])", title)
    expanded = []
    for p in parts:
        p = p.strip()
        if len(p) > 3:
            expanded.append(p)
            for tail in re.split(r":| - ", p)[1:]:
                if len(tail.strip()) > 3:
                    expanded.append(tail.strip())
    if len(parts) > 1:
        cands = [core_by_author.get(sn, {}).get(_core_title(p)) for p in expanded]
        cands = [c for c in cands if c]
        if cands:
            rc_by_pos = {pos: rc for pos, _ct, _bid, rc in by_author.get(sn, [])}
            return max(cands, key=lambda h: rc_by_pos.get(h[1], 0))
    return None


def _migrate_book_ratings(items, b_df):
    """Re-point stored book ratings at the current corpus.

    corpus_index is a ROW POSITION, meaningful only against the corpus it was
    captured from. After a corpus swap a stale index either addresses the wrong
    book or, if it is past the end, makes FAISS reconstruct() raise
    'key < ntotal' — which kills the whole Streamlit run and blanks all three
    tabs, not just books. Ratings do carry matched_title/matched_author, so most
    can be recovered by name instead of discarded: exact title+author first,
    then title alone.

    Returns (kept, n_migrated, n_dropped, n_merged). Entries already valid keep
    their data; duplicates that resolve to one book are merged, not duplicated.
    """
    if not isinstance(items, list) or b_df is None or not len(b_df):
        return items, 0, 0, 0
    n_rows = len(b_df)
    valid_ids = set(b_df["book_id"].astype(str))
    idx_tables = _book_match_index(b_df)

    kept, migrated, dropped, merged = [], 0, 0, 0
    seen = {}     # book_id -> position in kept

    def _place(rec):
        """Append, or merge into an existing entry for the same book.

        Distinct stored titles can legitimately resolve to ONE corpus book --
        an omnibus and its component ('Walkabout: A Walk in the Woods & Down
        Under' and 'A Walk in the Woods' are both A Walk in the Woods). Two
        entries sharing a book_id crash Streamlit on duplicate widget keys, and
        before that they silently double-weight the book in the taste vector.
        Keep the higher rating: the omnibus rating and the single rating are
        both the user's opinion of that text.
        """
        nonlocal merged
        bid = str(rec.get("book_id", ""))
        if bid in seen:
            prior = kept[seen[bid]]
            if float(rec.get("rating", 0)) > float(prior.get("rating", 0)):
                kept[seen[bid]] = rec
            merged += 1
            return
        seen[bid] = len(kept)
        kept.append(rec)

    for r in items:
        if not isinstance(r, dict):
            dropped += 1
            continue
        idx = r.get("corpus_index")
        bid = str(r.get("book_id", ""))
        if isinstance(idx, int) and 0 <= idx < n_rows and bid in valid_ids:
            _place(r)
            continue
        hit = _match_book(r.get("matched_title", ""),
                          r.get("matched_author", ""), idx_tables)
        if hit:
            new_r = dict(r)
            new_r["book_id"], new_r["corpus_index"] = hit[0], hit[1]
            _place(new_r)
            migrated += 1
        else:
            dropped += 1
    return kept, migrated, dropped, merged
    return kept, migrated, dropped


def _drop_stale_indices(items, n_rows, id_key=None, valid_ids=None):
    """Generic bounds guard for movies/TV (and books post-migration).

    Same failure as above: an out-of-range corpus_index takes down every tab.
    Movies and TV have not changed corpora, but a future expansion that SHRINKS
    one would hit this, and a stack trace is a bad way to find out.
    """
    if not isinstance(items, list):
        return items, 0
    kept, dropped = [], 0
    for r in items:
        if not isinstance(r, dict):
            dropped += 1
            continue
        idx = r.get("corpus_index")
        ok = isinstance(idx, int) and 0 <= idx < n_rows
        if ok and valid_ids is not None and id_key:
            ok = str(r.get(id_key, "")) in valid_ids
        if ok:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def hydrate_from_ls():
    """Restore rated lists from localStorage. Retries across reruns because the
    browser delivers localStorage asynchronously — the first read can be empty,
    with real data arriving a rerun or two later. Per-key 'synced' flag locks a
    list once we've definitively read it, so Clear/add stay authoritative after."""
    pending = [k for k in LS_KEYS if not st.session_state.get(f"_ls_synced_{k}")]
    if not pending:
        return
    ls = get_local_storage()
    if ls is None:
        return
    # Re-pull from the browser on subsequent runs (bounded) in case data arrived
    # after the first mount. getItem reads a cached dict, so without this a late
    # delivery would never be seen.
    attempts = st.session_state.get("_ls_attempts", 0)
    if attempts and attempts < 6:
        try:
            ls.refreshItems()
        except Exception as e:
            print(f"refreshItems failed: {e}")
    st.session_state["_ls_attempts"] = attempts + 1
    for ss_key in pending:
        ls_key = LS_KEYS[ss_key]
        try:
            raw = ls.getItem(ls_key)
        except Exception as e:
            print(f"hydrate {ss_key} failed: {e}")
            raw = None
        if raw is None:
            continue  # not in storage yet (never saved, or not delivered) — retry next run
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = None
        if isinstance(data, list):
            # Validate against the CURRENT corpus before anything reads it.
            if ss_key == "ratings":
                data, _mig, _drop, _merge = _migrate_book_ratings(data, b_df)
                if _mig or _drop or _merge:
                    st.session_state["_book_migration"] = (_mig, _drop, _merge)
                    print(f"hydrate: book ratings migrated={_mig} "
                          f"dropped={_drop} merged={_merge}")
            elif ss_key == "movie_ratings":
                data, _drop = _drop_stale_indices(data, len(m_df))
            elif ss_key == "tv_ratings":
                data, _drop = _drop_stale_indices(data, len(t_df))
            if not st.session_state.get(ss_key):  # don't clobber a profile started this session
                st.session_state[ss_key] = data
            st.session_state[f"_ls_last_{ss_key}"] = _ls_norm(st.session_state.get(ss_key, []))
            st.session_state[f"_ls_synced_{ss_key}"] = True


def persist_to_ls():
    """Write any changed rated list back to localStorage (unique key per list)."""
    ls = get_local_storage()
    if ls is None:
        return
    for ss_key, ls_key in LS_KEYS.items():
        try:
            cur = _ls_norm(st.session_state.get(ss_key, []))
            if cur != st.session_state.get(f"_ls_last_{ss_key}"):
                ls.setItem(ls_key, cur, key=f"ls_set_{ss_key}")
                st.session_state[f"_ls_last_{ss_key}"] = cur
        except Exception as e:
            print(f"persist {ss_key} failed: {e}")


# ===========================================================================
# LOAD EVERYTHING
# ===========================================================================
with st.spinner("Loading the recommendation engine (first load downloads data, ~2-3 min)..."):
    b_df = load_books_corpus()
    b_norm = build_book_search_index(b_df)
    b_index = load_books_index()
    b_emb = IndexVectors(b_index)
    b_tags = load_books_tags()
    m_df = load_movies_corpus()
    m_index = load_movies_index()
    m_emb = IndexVectors(m_index)
    m_tags = load_movies_tags()
    m_norm = build_movie_search_index(m_df)
    t_df = load_tv_corpus()
    t_index = load_tv_index()
    t_emb = IndexVectors(t_index)
    t_norm = build_tv_search_index(t_df)
    t_tags = load_tv_tags()

if "ratings" not in st.session_state:
    st.session_state.ratings = []
if "movie_ratings" not in st.session_state:
    st.session_state.movie_ratings = []
if "tv_ratings" not in st.session_state:
    st.session_state.tv_ratings = []
# watchlists (saved for later) + skipped (dismissed recs) — same persistence path
for _k in ("watchlist", "movie_watchlist", "tv_watchlist",
           "skipped", "movie_skipped", "tv_skipped"):
    if _k not in st.session_state:
        st.session_state[_k] = []

# Restore rated lists from the browser (once per session), before tabs read them.
hydrate_from_ls()

st.title("Recommendation Engine")
st.caption("Search a title, tap the stars to rate it, and recommendations appear on the right — across books, films, and TV.")

books_tab, movies_tab, tv_tab = st.tabs(["Books", "Movies", "TV"])


# ===========================================================================
# BOOKS TAB
# ===========================================================================
with books_tab:
    b_tagstat = f" · {len(b_tags):,} tagged" if b_tags is not None else ""
    st.caption(f"{len(b_df):,} books{b_tagstat}")
    _mig = st.session_state.get("_book_migration")
    if _mig:
        # tolerate a 2-tuple left in session state by an older run
        _m, _d, _mrg = (list(_mig) + [0, 0, 0])[:3]
        _bits = []
        if _m:
            _bits.append(f"{_m} rating{'s' if _m != 1 else ''} re-matched")
        if _mrg:
            _bits.append(f"{_mrg} duplicate{'s' if _mrg != 1 else ''} merged")
        if _d:
            _bits.append(f"{_d} not found in the updated catalogue")
        if _bits:
            st.info("Book catalogue was updated: " + ", ".join(_bits) + ".")
    bl, br = st.columns([1, 1.3], gap="large")
    with bl:
        st.subheader("Your taste profile")
        backup_controls("b")
        clear_all_button("_b_confirm_clear", b_clear, "b_clear", "book ratings")
        st.markdown("**Add books you've read:**")
        profile_coach(st.session_state.ratings, "books", "rate")
        _toast = st.session_state.pop("_b_rated_toast", None)
        if _toast:
            st.success(f"Added **{_toast}** to your profile — recommendations updated.")
        _sc1, _sc2 = st.columns([6, 1])
        with _sc1:
            bq = st.text_input("Search by title", key="b_search", label_visibility="collapsed",
                               placeholder="e.g. The Hobbit or Brandon Sanderson")
        with _sc2:
            st.button("✕", key="b_clearsearch", use_container_width=True,
                      help="Clear search",
                      on_click=lambda: st.session_state.update({"b_search": ""}))
        if bq:
            matches = search_books(bq, b_df, b_norm)
            if len(matches) == 0:
                st.info("No matches found.")
            else:
                rated = {r["book_id"]: r["rating"] for r in st.session_state.ratings}
                on_wl_s = {w["book_id"] for w in st.session_state.watchlist}
                for pos, book in matches.iterrows():
                    yr = year_label(book['publication_year'])
                    # Cover in the result row: two testers could not tell
                    # same-titled editions apart from text alone.
                    _sc = book_img_tag(book["image_url"] if "image_url" in b_df.columns else "", "wl-poster")
                    st.markdown(f"<div class='item-card wl-row'>{_sc}<div>"
                                f"<span class='item-title'>{book['title']}</span>{yr}<br>"
                                f"<span class='item-sub'>{book['primary_author']}</span></div></div>",
                                unsafe_allow_html=True)
                    # Already in the profile: show the standing score instead of
                    # the star widget. Previously this row was skipped outright,
                    # which made a rated title indistinguishable from one missing
                    # from the catalogue.
                    if book["book_id"] in rated:
                        st.markdown(f"<div class='rated-flag'>Already rated "
                                    f"<b>{int(rated[book['book_id']])}/5</b> — in your profile below</div>",
                                    unsafe_allow_html=True)
                        st.markdown("<hr class='row-div'>", unsafe_allow_html=True)
                        continue
                    x1, x2 = st.columns([1.4, 1])
                    with x1:
                        st.feedback("stars", key=f"b_rate_{book['book_id']}", on_change=b_add,
                                    args=(book["book_id"], book["title"], book["primary_author"], pos))
                    with x2:
                        wl_saved = book["book_id"] in on_wl_s
                        st.button("Saved" if wl_saved else "+ Watchlist", key=f"b_wls_{book['book_id']}",
                                  disabled=wl_saved, use_container_width=True, on_click=b_add_watchlist,
                                  args=({"book_id": book["book_id"], "title": book["title"],
                                         "author": book["primary_author"],
                                         "year": int(book["publication_year"]) if pd.notna(book["publication_year"]) else None,
                                         "image_url": book["image_url"] if "image_url" in b_df.columns and isinstance(book["image_url"], str) else ""},))
                    st.markdown("<hr class='row-div'>", unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state.ratings:
            nb = len(st.session_state.ratings)
            st.markdown(f"**{nb} book{'s' if nb != 1 else ''} rated:**")
            if nb < 3:
                st.caption("Add a few more for noticeably better recommendations.")
            for r in rated_list_page(st.session_state.ratings, "b"):
                y1, y2 = st.columns([5, 1])
                with y1:
                    st.markdown(f"<div class='item-card'><span class='item-title'>{r['matched_title']}</span><br>"
                                f"<span class='item-sub'>{r['matched_author']}</span> &middot; "
                                f"<span style='color:#c9a86a'>{int(r['rating'])}/5</span></div>", unsafe_allow_html=True)
                with y2:
                    st.button("X", key=f"b_rm_{r['book_id']}", on_click=b_remove, args=(r["book_id"],))
            rated_list_pager("b")
        else:
            st.info("No books rated yet. Search above to add some.")
    with br:
        st.subheader("Recommended for you")
        if st.session_state.watchlist:
            with st.expander(f"Watchlist — {len(st.session_state.watchlist)} saved",
                             expanded=st.session_state.get("_b_wl_open", False)):
                for w in st.session_state.watchlist:
                    wc1, wc2 = st.columns([5, 1])
                    with wc1:
                        wyr = year_label(w.get("year"))
                        wimg = book_img_tag(w.get("image_url"), "wl-poster")
                        st.markdown(f"<div class='item-card wl-row'>{wimg}<div>"
                                    f"<span class='item-title'>{w['title']}</span>{wyr}<br>"
                                    f"<span class='item-sub'>{w.get('author', '')}</span></div></div>",
                                    unsafe_allow_html=True)
                    with wc2:
                        st.button("X", key=f"b_wl_rm_{w['book_id']}",
                                  on_click=lambda i: (st.session_state.update({'_b_wl_open': True}), b_remove_watchlist(i)), args=(w["book_id"],))
        # --- From your films and shows ------------------------------------
        # Rendered BEFORE the empty-state check on purpose: a user with no
        # books rated but films rated still gets something here, which is the
        # cold-start case the feedback round exposed (almost everyone rated
        # 5-10 items and stopped). Kept as a separate strip rather than blended
        # into the main ranking so a weak cross-modal result cannot contaminate
        # the tab that already works, and so the provenance stays legible.
        _xm = pd.DataFrame()
        _xm_ratings = list(st.session_state.get("movie_ratings", []))
        _xm_err = None
        try:
            if _xm_ratings:
                _xm = recommend_books_from_watching(
                    _xm_ratings, m_df, m_emb, b_df, b_emb, b_index,
                    kind="movie", skip_ids=st.session_state.skipped)
        except Exception as _e:
            # A bare `except: pass` here hid the first failure completely and
            # cost a debugging round. Swallow it for the USER -- the books tab
            # must not die for a secondary strip -- but never for the developer.
            _xm_err = f"{type(_e).__name__}: {_e}"
            _xm = pd.DataFrame()

        if st.query_params.get("debug") and (_xm_err or not len(_xm)):
            _liked_n = sum(1 for r in _xm_ratings if r.get("rating", 0) >= 4.0)
            _idx_ok = sum(1 for r in _xm_ratings
                          if r.get("corpus_index") is not None
                          and 0 <= int(r["corpus_index"]) < len(m_df))
            st.warning(
                f"cross-modal produced nothing — movie ratings: "
                f"{len(_xm_ratings)}, liked (>=4.0): {_liked_n}, "
                f"resolvable corpus_index: {_idx_ok}, "
                f"error: {_xm_err or 'none — retrieval returned 0 rows'}. "
                f"If there is no error and liked > 0, CROSS_MODAL_MIN_SIM "
                f"({CROSS_MODAL_MIN_SIM}) or the pool-mean gate is filtering "
                f"everything.")

        # Tabs, matching movies and TV. "From your films" appears only when
        # cross-modal actually returned something -- an empty tab advertising a
        # feature that produced nothing is worse than no tab at all.
        # One card renderer for BOTH tabs. The cross-modal tab first shipped
        # with its own minimal card -- no genres, no tag badge, no overview, no
        # rating stars -- so the same book looked like a different, poorer
        # object depending on which tab you found it in. keyns namespaces every
        # widget key: a book can appear in both tabs and duplicate Streamlit
        # keys raise. The star key is passed to the callback explicitly, for
        # the same reason m_rate_rec needs it.
        def _book_card(rec, keyns, on_wl, _g_allow, _g_counts):
            yr = year_label(rec['year'])
            genres = ", ".join(clean_genres(rec["genres"], _g_allow, _g_counts))
            # The old corpus mixed Goodreads and Hardcover counts, which were
            # ~100x apart, so hardcover rows were divided by 25 for display.
            # The v2 corpus is uniformly Hardcover with genuine counts
            # (median 14) -- dividing now would render "0 ratings" on most
            # cards. Show the real number.
            dc = int(rec['ratings_count'])
            lbl = "ratings"
            badge = tag_badge(rec)
            # Join only the parts that exist: ~7.5K books have no usable
            # genres, and an unconditional separator left those cards
            # opening with a bare "· 19 ratings".
            meta = " &middot; ".join(b for b in [genres, f"{dc:,} {lbl}", badge] if b)
            b_also = rec.get("also_because")
            b_because = f"because you liked <b>{rec['because_of']}</b>"
            if b_also and isinstance(b_also, str):
                b_because += f" and <b>{b_also}</b>"
            b_img_html = book_img_tag(rec.get("image_url"), "rec-poster")
            with st.container(border=True):
                st.markdown(f"<div class='rec-row'>{b_img_html}<div class='rec-body'>"
                            f"<span class='item-title'>{rec['title']}</span>{yr}<br>"
                            f"<span class='item-sub'>{rec['author']}</span>"
                            f"<div class='item-meta'>{meta}</div>"
                            f"<div class='because'>{b_because}</div></div></div>",
                            unsafe_allow_html=True)
                saved = rec["book_id"] in on_wl
                wcol, scol = st.columns(2)
                wcol.button("On watchlist" if saved else "+ Watchlist", key=f"b_wl_{keyns}_{rec['book_id']}",
                            disabled=saved, use_container_width=True, on_click=b_add_watchlist,
                            args=({"book_id": rec["book_id"], "title": rec["title"], "author": rec["author"],
                                   "year": int(rec["year"]) if pd.notna(rec["year"]) else None,
                                   "image_url": rec.get("image_url", "")},))
                scol.button("Not interested", key=f"b_skip_{keyns}_{rec['book_id']}", use_container_width=True,
                            on_click=b_skip, args=(rec["book_id"],))
                with st.expander("Details & rate"):
                    if rec.get("overview"):
                        st.markdown(f"<div class='blurb'>{rec['overview']}</div>", unsafe_allow_html=True)
                    st.markdown("<span class='rate-prompt'>Already read it? Rate it to sharpen your recommendations:</span>",
                                unsafe_allow_html=True)
                    # Namespaced, and the key is handed to the callback rather
                    # than rebuilt from book_id. Rebuilding it is what silently
                    # discarded every rating made from a movie genre-tab card:
                    # py_compile passed, no exception, the star even filled in.
                    _bkey = f"b_recrate_{keyns}_{rec['book_id']}"
                    st.feedback("stars", key=_bkey, on_change=b_rate_rec,
                                args=(rec["book_id"], rec["title"], rec["author"],
                                      rec["corpus_index"], _bkey))

        _b_labels = ["For you"] + (["From your films"] if len(_xm) else [])
        _b_tabs = st.tabs(_b_labels)

        with _b_tabs[0]:
            if not st.session_state.ratings:
                st.markdown("_Rate a few books to see recommendations._")
            elif len([r for r in st.session_state.ratings if r["rating"] >= 3.5]) == 0:
                st.warning("Rate at least one book 4 or higher.")
            else:
                recs, tags = recommend_books(st.session_state.ratings, b_df, b_emb, b_index,
                                             tags_df=b_tags, n=20, skip_ids=st.session_state.skipped)
                if tags is not None and len(tags) > 0:
                    st.markdown(f"<div class='tag-hint'>Inferred taste signature: {', '.join(tags.index[:5])}</div>",
                                unsafe_allow_html=True)
                if len(recs) == 0:
                    st.info("No recommendations found.")
                on_wl = {w["book_id"] for w in st.session_state.watchlist}
                _g_allow, _g_counts = _book_genre_allow(b_df)
                for _, rec in recs.iterrows():
                    _book_card(rec, "all", on_wl, _g_allow, _g_counts)
        if len(_xm):
            with _b_tabs[1]:
                st.caption("Your film ratings, searched against the book corpus — "
                           "all three corpora use the same embedding model, so a "
                           "film vector can query the book index directly.")
                _xm_wl = {w["book_id"] for w in st.session_state.watchlist}
                _g_allow2, _g_counts2 = _book_genre_allow(b_df)
                # No group headers: the card already carries "because you
                # liked X and Y", and printing the anchor twice per card was
                # just noise. Whole-profile scoring means a book can answer to
                # two films anyway, so grouping by one of them was misleading.
                for _, rec in _xm.iterrows():
                    _book_card(rec, "xm", _xm_wl, _g_allow2, _g_counts2)
# ===========================================================================
# MOVIES TAB
# ===========================================================================
with movies_tab:
    m_tagstat = f" · {len(m_tags):,} tagged" if m_tags is not None else ""
    st.caption(f"{len(m_df):,} films{m_tagstat}")
    ml, mr = st.columns([1, 1.3], gap="large")
    with ml:
        st.subheader("Your taste profile")
        backup_controls("m")
        clear_all_button("_m_confirm_clear", m_clear, "m_clear", "film ratings")
        st.markdown("**Add films you've seen:**")
        profile_coach(st.session_state.movie_ratings, "films", "rate")
        _toast = st.session_state.pop("_m_rated_toast", None)
        if _toast:
            st.success(f"Added **{_toast}** to your profile — recommendations updated.")
        _sc1, _sc2 = st.columns([6, 1])
        with _sc1:
            mq = st.text_input("Search by title, director, or cast", key="m_search", label_visibility="collapsed",
                       placeholder="e.g. Inception, Nolan, or Bong Joon-ho")
        with _sc2:
            st.button("✕", key="m_clearsearch", use_container_width=True,
                      help="Clear search",
                      on_click=lambda: st.session_state.update({"m_search": ""}))
        if mq:
            matches = search_movies(mq, m_df, m_norm)
            if len(matches) == 0:
                st.info("No matches found.")
            else:
                rated = {r["movie_id"]: r["rating"] for r in st.session_state.movie_ratings}
                on_wl_s = {w["movie_id"] for w in st.session_state.movie_watchlist}
                for pos, film in matches.iterrows():
                    yr = f" ({int(film['release_year'])})" if pd.notna(film['release_year']) else ""
                    director = film["director"] if "director" in m_df.columns and isinstance(film["director"], str) and film["director"] else lang_label(film["original_language"])
                    _sp = poster_url(film.get("poster_path"))
                    _si = f"<img class='wl-poster' loading='lazy' src='{_sp}'>" if _sp else "<div class='wl-poster cover-blank'></div>"
                    st.markdown(f"<div class='item-card wl-row'>{_si}<div>"
                                f"<span class='item-title'>{film['title']}</span>{yr}<br>"
                                f"<span class='item-sub'>{director}</span></div></div>",
                                unsafe_allow_html=True)
                    if int(film["id"]) in rated:
                        st.markdown(f"<div class='rated-flag'>Already rated "
                                    f"<b>{int(rated[int(film['id'])])}/5</b> — in your profile below</div>",
                                    unsafe_allow_html=True)
                        st.markdown("<hr class='row-div'>", unsafe_allow_html=True)
                        continue
                    x1, x2 = st.columns([1.4, 1])
                    with x1:
                        st.feedback("stars", key=f"m_rate_{film['id']}", on_change=m_add,
                                    args=(film["id"], film["title"],
                                          film["director"] if "director" in m_df.columns and isinstance(film["director"], str) else "", pos))
                    with x2:
                        wl_saved = int(film["id"]) in on_wl_s
                        st.button("Saved" if wl_saved else "+ Watchlist", key=f"m_wls_{film['id']}",
                                  disabled=wl_saved, use_container_width=True, on_click=m_add_watchlist,
                                  args=({"movie_id": int(film["id"]), "name": film["title"],
                                         "director": film["director"] if "director" in m_df.columns and isinstance(film["director"], str) else "",
                                         "year": int(film["release_year"]) if pd.notna(film["release_year"]) else None,
                                         "poster_path": film["poster_path"] if "poster_path" in m_df.columns and isinstance(film["poster_path"], str) else ""},))
                    st.markdown("<hr class='row-div'>", unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state.movie_ratings:
            nm = len(st.session_state.movie_ratings)
            st.markdown(f"**{nm} film{'s' if nm != 1 else ''} rated:**")
            if nm < 3:
                st.caption("Add a few more for noticeably better recommendations.")
            for r in rated_list_page(st.session_state.movie_ratings, "m"):
                y1, y2 = st.columns([5, 1])
                with y1:
                    dline = f"<span class='item-sub'>{r['matched_director']}</span><br>" if r.get("matched_director") else ""
                    st.markdown(f"<div class='item-card'><span class='item-title'>{r['matched_title']}</span><br>"
                                f"{dline}<span style='color:#c9a86a'>{int(r['rating'])}/5</span></div>", unsafe_allow_html=True)
                with y2:
                    st.button("X", key=f"m_rm_{r['movie_id']}", on_click=m_remove, args=(r["movie_id"],))
            rated_list_pager("m")
        else:
            st.info("No films rated yet. Search above to add some.")
    with mr:
        st.subheader("Recommended for you")
        if st.session_state.movie_watchlist:
            with st.expander(f"Watchlist — {len(st.session_state.movie_watchlist)} saved",
                             expanded=st.session_state.get("_m_wl_open", False)):
                for w in st.session_state.movie_watchlist:
                    wc1, wc2 = st.columns([5, 1])
                    with wc1:
                        wyr = f" ({int(w['year'])})" if w.get("year") else ""
                        wpurl = poster_url(w.get("poster_path"))
                        wimg = f"<img class='wl-poster' loading='lazy' src='{wpurl}'>" if wpurl else ""
                        st.markdown(f"<div class='item-card wl-row'>{wimg}<div>"
                                    f"<span class='item-title'>{w['name']}</span>{wyr}<br>"
                                    f"<span class='item-sub'>{w.get('director', '')}</span></div></div>",
                                    unsafe_allow_html=True)
                    with wc2:
                        st.button("X", key=f"m_wl_rm_{w['movie_id']}",
                                  on_click=lambda i: (st.session_state.update({'_m_wl_open': True}), m_remove_watchlist(i)), args=(w["movie_id"],))
        if not st.session_state.movie_ratings:
            st.markdown("_Rate a few films to see recommendations._")
        elif len([r for r in st.session_state.movie_ratings if r["rating"] >= 3.5]) == 0:
            st.warning("Rate at least one film 4 or higher.")
        else:
            recs, tags = recommend_movies(st.session_state.movie_ratings, m_df, m_emb, m_index,
                                          tags_df=m_tags, n=20, skip_ids=st.session_state.movie_skipped)
            if tags is not None and len(tags) > 0:
                st.markdown(f"<div class='tag-hint'>Inferred taste signature: {', '.join(tags.index[:5])}</div>",
                            unsafe_allow_html=True)
            if len(recs) == 0:
                st.info("No recommendations found.")
            on_wl = {w["movie_id"] for w in st.session_state.movie_watchlist}

            def _movie_card(rec, keyns):
                """One rec card. keyns namespaces the widget keys: the same film
                can appear in the overall tab and a genre tab, and Streamlit
                raises DuplicateElementKey on a repeated key even across tabs."""
                yr = f" ({int(rec['year'])})" if pd.notna(rec['year']) else ""
                genres = ", ".join(rec["genres"][:3]) if len(rec["genres"]) else ""
                rating_str = f"{rec['vote_average']:.1f}/10" if rec["vote_average"] else ""
                headline = rec["director"] if rec.get("director") else rec["language"]
                meta_bits = " &middot; ".join(b for b in [rec.get("studio", ""), genres, f"{rec['vote_count']:,} votes", rating_str] if b)
                badge = tag_badge(rec, " &middot; ")
                cast_line = f"<div class='item-cast'>with {', '.join(rec['cast'][:4])}</div>" if rec.get("cast") else ""
                purl = poster_url(rec.get("poster_path"))
                img_html = f"<img class='rec-poster' loading='lazy' src='{purl}'>" if purl else ""
                also = rec.get("also_because")
                because_line = f"because you liked <b>{rec['because_of']}</b>"
                if also and isinstance(also, str):
                    because_line += f" and <b>{also}</b>"
                with st.container(border=True):
                    st.markdown(
                        f"<div class='rec-row'>{img_html}<div class='rec-body'>"
                        f"<span class='item-title'>{rec['title']}</span>{yr}<br>"
                        f"<span class='item-sub'>{headline}</span>"
                        f"<div class='item-meta'>{meta_bits}{badge}</div>{cast_line}"
                        f"<div class='because'>{because_line}</div>"
                        f"{watch_link_html('movie', rec.get('id'))}"
                        f"</div></div>",
                        unsafe_allow_html=True)
                    mid = int(rec["id"])
                    saved = mid in on_wl
                    wcol, scol = st.columns(2)
                    wcol.button("On watchlist" if saved else "+ Watchlist", key=f"m_wl_{keyns}_{mid}",
                                disabled=saved, use_container_width=True, on_click=m_add_watchlist,
                                args=({"movie_id": mid, "name": rec["title"], "director": rec.get("director", ""),
                                       "year": int(rec["year"]) if pd.notna(rec["year"]) else None,
                                       "poster_path": rec.get("poster_path", "")},))
                    scol.button("Not interested", key=f"m_skip_{keyns}_{mid}", use_container_width=True,
                                on_click=m_skip, args=(mid,))
                    with st.expander("Details & rate"):
                        if rec.get("overview"):
                            st.markdown(f"<div class='blurb'>{rec['overview']}</div>", unsafe_allow_html=True)
                        st.markdown("<span class='rate-prompt'>Already watched? Rate it to sharpen your recommendations:</span>",
                                    unsafe_allow_html=True)
                        st.feedback("stars", key=f"m_recrate_{keyns}_{mid}", on_change=m_rate_rec,
                                    args=(rec["id"], rec["title"], rec.get("director", ""),
                                          rec["corpus_index"], f"m_recrate_{keyns}_{mid}"))

            # --- genre tabs ---------------------------------------------------
            # Deep pool, flat and uncapped: the diversity round-robin and the
            # genre-family cap both exist to make ONE list varied, which is the
            # opposite of what bucketing needs -- capping per genre would thin
            # exactly the buckets we are about to split on.
            #
            # Buckets use the FIRST genre only, and deliberately NOT the
            # ranker's _genre_bucket: that rule merges Crime and Thriller to
            # stop the diversity cap being laundered, which collapsed 232 of
            # 400 candidates into one blob. Split, the same pool gives five
            # balanced buckets (Comedy 86, Action 80, Drama 69, Crime 66,
            # Thriller 29) whose contents check out by eye.
            pool, _ = recommend_movies(st.session_state.movie_ratings, m_df, m_emb, m_index,
                                       tags_df=m_tags, n=MOVIE_POOL_N, flat_ranking=True,
                                       genre_cap=10_000, per_anchor_cap=10_000,
                                       skip_ids=st.session_state.movie_skipped)
            shown = set(recs["id"]) if len(recs) else set()
            genre_tabs = []
            if len(pool):
                rest = pool[~pool["id"].isin(shown)]
                primary = [(g[0] if isinstance(g, list) and len(g) else "") for g in rest["genres"]]
                rest = rest.assign(_primary=primary)
                counts = rest["_primary"].value_counts()
                # 2x depth: a bucket that can only just fill its tab is padding
                # the bottom half with whatever was left, not with good matches.
                for g, c in counts.items():
                    if g and c >= MOVIE_TAB_N * 2 and len(genre_tabs) < MOVIE_MAX_TABS:
                        bucket = rest[rest["_primary"] == g]
                        if MOVIE_DEEP_CUTS and "vote_count" in bucket.columns:
                            quiet = bucket[bucket["vote_count"] <= MOVIE_TAB_VOTE_CEILING]
                            loud = bucket[bucket["vote_count"] > MOVIE_TAB_VOTE_CEILING]
                            # Backfill rather than ship a short tab. Unlike the
                            # median cut this usually removes nothing at all,
                            # which is the point.
                            sub = pd.concat([quiet, loud]).head(MOVIE_TAB_N)
                        else:
                            sub = bucket.head(MOVIE_TAB_N)
                        genre_tabs.append((g, sub))

            if genre_tabs:
                labels = ["For you"] + [g for g, _ in genre_tabs]
                for tab, label in zip(st.tabs(labels), labels):
                    with tab:
                        if label == "For you":
                            for _, rec in recs.iterrows():
                                _movie_card(rec, "all")
                        else:
                            sub = dict(genre_tabs)[label]
                            _note = ("lesser-known " if MOVIE_DEEP_CUTS else "")
                            st.caption(f"{len(sub)} {_note}{label.lower()} picks, "
                                       f"none repeated from For you.")
                            if MOVIE_DEEP_CUTS:
                                st.caption("These skip the most-watched titles on "
                                           "purpose — anything great and famous still "
                                           "shows up under **For you**.")
                            for _, rec in sub.iterrows():
                                _movie_card(rec, label.lower().replace(" ", ""))
            else:
                for _, rec in recs.iterrows():
                    _movie_card(rec, "all")


# ===========================================================================
# TV TAB
# ===========================================================================
with tv_tab:
    t_tagstat = f" · {len(t_tags):,} tagged" if t_tags is not None else ""
    st.caption(f"{len(t_df):,} series{t_tagstat}")
    tl, tr = st.columns([1, 1.3], gap="large")
    with tl:
        st.subheader("Your taste profile")
        backup_controls("t")
        clear_all_button("_t_confirm_clear", t_clear, "t_clear", "TV ratings")
        st.markdown("**Add shows you've seen:**")
        profile_coach(st.session_state.tv_ratings, "shows", "rate")
        _toast = st.session_state.pop("_t_rated_toast", None)
        if _toast:
            st.success(f"Added **{_toast}** to your profile — recommendations updated.")
        _sc1, _sc2 = st.columns([6, 1])
        with _sc1:
            tq = st.text_input("Search by title or creator", key="t_search", label_visibility="collapsed",
                       placeholder="e.g. Breaking Bad or Vince Gilligan")
        with _sc2:
            st.button("✕", key="t_clearsearch", use_container_width=True,
                      help="Clear search",
                      on_click=lambda: st.session_state.update({"t_search": ""}))
        if tq:
            matches = search_tv(tq, t_df, t_norm)
            if len(matches) == 0:
                st.info("No matches found.")
            else:
                rated = {r["tv_id"]: r["rating"] for r in st.session_state.tv_ratings}
                on_wl_s = {w["tv_id"] for w in st.session_state.tv_watchlist}
                for pos, show in matches.iterrows():
                    yr = f" ({int(show['start_year'])})" if pd.notna(show['start_year']) else ""
                    subtitle = show["created_by"] if "created_by" in t_df.columns and isinstance(show["created_by"], str) and show["created_by"] else lang_label(show["original_language"])
                    _sp = poster_url(show.get("poster_path"))
                    _si = f"<img class='wl-poster' loading='lazy' src='{_sp}'>" if _sp else "<div class='wl-poster cover-blank'></div>"
                    st.markdown(f"<div class='item-card wl-row'>{_si}<div>"
                                f"<span class='item-title'>{show['name']}</span>{yr}<br>"
                                f"<span class='item-sub'>{subtitle}</span></div></div>",
                                unsafe_allow_html=True)
                    if int(show["tmdb_id"]) in rated:
                        st.markdown(f"<div class='rated-flag'>Already rated "
                                    f"<b>{int(rated[int(show['tmdb_id'])])}/5</b> — in your profile below</div>",
                                    unsafe_allow_html=True)
                        st.markdown("<hr class='row-div'>", unsafe_allow_html=True)
                        continue
                    x1, x2 = st.columns([1.4, 1])
                    with x1:
                        st.feedback("stars", key=f"t_rate_{show['tmdb_id']}", on_change=t_add,
                                    args=(show["tmdb_id"], show["name"],
                                          show["created_by"] if "created_by" in t_df.columns and isinstance(show["created_by"], str) else "", pos))
                    with x2:
                        wl_saved = int(show["tmdb_id"]) in on_wl_s
                        st.button("Saved" if wl_saved else "+ Watchlist", key=f"t_wls_{show['tmdb_id']}",
                                  disabled=wl_saved, use_container_width=True, on_click=t_add_watchlist,
                                  args=({"tv_id": int(show["tmdb_id"]), "name": show["name"],
                                         "creator": show["created_by"] if "created_by" in t_df.columns and isinstance(show["created_by"], str) else "",
                                         "year": int(show["start_year"]) if pd.notna(show["start_year"]) else None,
                                         "poster_path": show["poster_path"] if "poster_path" in t_df.columns and isinstance(show["poster_path"], str) else ""},))
                    st.markdown("<hr class='row-div'>", unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state.tv_ratings:
            nt = len(st.session_state.tv_ratings)
            st.markdown(f"**{nt} show{'s' if nt != 1 else ''} rated:**")
            if nt < 3:
                st.caption("Add a few more for noticeably better recommendations.")
            for r in rated_list_page(st.session_state.tv_ratings, "t"):
                y1, y2 = st.columns([5, 1])
                with y1:
                    cline = f"<span class='item-sub'>{r['matched_creator']}</span><br>" if r.get("matched_creator") else ""
                    st.markdown(f"<div class='item-card'><span class='item-title'>{r['matched_title']}</span><br>"
                                f"{cline}<span style='color:#c9a86a'>{int(r['rating'])}/5</span></div>", unsafe_allow_html=True)
                with y2:
                    st.button("X", key=f"t_rm_{r['tv_id']}", on_click=t_remove, args=(r["tv_id"],))
            rated_list_pager("t")
        else:
            st.info("No shows rated yet. Search above to add some.")
    with tr:
        st.subheader("Recommended for you")
        if st.session_state.tv_watchlist:
            with st.expander(f"Watchlist — {len(st.session_state.tv_watchlist)} saved",
                             expanded=st.session_state.get("_t_wl_open", False)):
                for w in st.session_state.tv_watchlist:
                    wc1, wc2 = st.columns([5, 1])
                    with wc1:
                        wyr = f" ({int(w['year'])})" if w.get("year") else ""
                        wpurl = poster_url(w.get("poster_path"))
                        wimg = f"<img class='wl-poster' loading='lazy' src='{wpurl}'>" if wpurl else ""
                        st.markdown(f"<div class='item-card wl-row'>{wimg}<div>"
                                    f"<span class='item-title'>{w['name']}</span>{wyr}<br>"
                                    f"<span class='item-sub'>{w.get('creator', '')}</span></div></div>",
                                    unsafe_allow_html=True)
                    with wc2:
                        st.button("X", key=f"t_wl_rm_{w['tv_id']}",
                                  on_click=lambda i: (st.session_state.update({'_t_wl_open': True}), t_remove_watchlist(i)), args=(w["tv_id"],))
        if not st.session_state.tv_ratings:
            st.markdown("_Rate a few shows to see recommendations._")
        elif len([r for r in st.session_state.tv_ratings if r["rating"] >= 3.5]) == 0:
            st.warning("Rate at least one show 4 or higher.")
        else:
            recs, tags = recommend_tv(st.session_state.tv_ratings, t_df, t_emb, t_index,
                                      tags_df=t_tags, n=20, skip_ids=st.session_state.tv_skipped)
            if tags is not None and len(tags) > 0:
                st.markdown(f"<div class='tag-hint'>Inferred taste signature: {', '.join(tags.index[:5])}</div>",
                            unsafe_allow_html=True)
            if len(recs) == 0:
                st.info("No recommendations found.")
            on_wl = {w["tv_id"] for w in st.session_state.tv_watchlist}

            # Extracted from the render loop so genre tabs can reuse it.
            # keyns namespaces EVERY per-card widget key: the same show can
            # appear under For you and under a genre tab, and duplicate
            # Streamlit keys raise. The star key is also passed to the callback
            # explicitly -- see t_rate_rec for why rebuilding it loses ratings.
            def _tv_card(rec, keyns):
                yr = f" ({int(rec['year'])})" if pd.notna(rec['year']) else ""
                genres = ", ".join(rec["genres"][:3]) if len(rec["genres"]) else ""
                rating_str = f"{rec['vote_average']:.1f}/10" if rec["vote_average"] else ""
                seasons_str = (f"{rec['seasons']} season" + ("s" if rec['seasons'] != 1 else "")) if rec["seasons"] else ""
                # created_by is sparsely populated on TMDB -- anime and
                # documentaries almost never carry it. The old fallback put
                # rec["language"] in the same unlabelled slot as the creator
                # name, so a card read "Inuyashiki: Last Hero / Japanese" and
                # looked like a data error rather than an absent field.
                # Language belongs with the other metadata, not in the byline.
                headline = rec["creator"] if rec.get("creator") else ""
                _lang = "" if rec.get("creator") else (rec.get("language") or "")
                meta_bits = " &middot; ".join(b for b in [network_label(rec.get("network", "")), _lang, genres, seasons_str, f"{rec['vote_count']:,} votes", rating_str] if b)
                badge = tag_badge(rec, " &middot; ")
                purl = poster_url(rec.get("poster_path"))
                img_html = f"<img class='rec-poster' loading='lazy' src='{purl}'>" if purl else ""
                t_also = rec.get("also_because")
                t_because = f"because you liked <b>{rec['because_of']}</b>"
                if t_also and isinstance(t_also, str):
                    t_because += f" and <b>{t_also}</b>"
                # Built outside the f-string on purpose: the Space runs Python
                # < 3.12, where a backslash inside an f-string expression is a
                # SyntaxError. Nested quotes in an inline conditional need
                # escaping, so the whole span is assembled here instead.
                _sub_html = (f"<span class='item-sub'>{headline}</span>"
                             if headline else "")
                with st.container(border=True):
                    st.markdown(
                        f"<div class='rec-row'>{img_html}<div class='rec-body'>"
                        f"<span class='item-title'>{rec['name']}</span>{yr}<br>"
                        f"{_sub_html}"
                        f"<div class='item-meta'>{meta_bits}{badge}</div>"
                        f"<div class='because'>{t_because}</div>"
                        f"{watch_link_html('tv', rec.get('tmdb_id'))}"
                        f"</div></div>",
                        unsafe_allow_html=True)
                    tid = int(rec["tmdb_id"])
                    saved = tid in on_wl
                    wcol, scol = st.columns(2)
                    wcol.button("On watchlist" if saved else "+ Watchlist", key=f"t_wl_{keyns}_{rec['tmdb_id']}",
                                disabled=saved, use_container_width=True, on_click=t_add_watchlist,
                                args=({"tv_id": tid, "name": rec["name"], "creator": rec.get("creator", ""),
                                       "year": int(rec["year"]) if pd.notna(rec["year"]) else None,
                                       "poster_path": rec.get("poster_path", "")},))
                    scol.button("Not interested", key=f"t_skip_{keyns}_{rec['tmdb_id']}", use_container_width=True,
                                on_click=t_skip, args=(tid,))
                    with st.expander("Details & rate"):
                        if rec.get("overview"):
                            st.markdown(f"<div class='blurb'>{rec['overview']}</div>", unsafe_allow_html=True)
                        st.markdown("<span class='rate-prompt'>Already watched? Rate it to sharpen your recommendations:</span>",
                                    unsafe_allow_html=True)
                        _wkey = f"t_recrate_{keyns}_{rec['tmdb_id']}"
                        st.feedback("stars", key=_wkey, on_change=t_rate_rec,
                                    args=(rec["tmdb_id"], rec["name"], rec.get("creator", ""),
                                          rec["corpus_index"], _wkey))

            # --- genre tabs ---------------------------------------------------
            # First genre, no merge. The ranker's _genre_bucket merges Crime and
            # Mystery to stop the diversity cap being laundered, which is a
            # different job from labelling a tab: on an 86-show profile that
            # merge put 55 of 300 candidates in one bucket.
            t_pool, _ = recommend_tv(st.session_state.tv_ratings, t_df, t_emb, t_index,
                                     tags_df=t_tags, n=TV_POOL_N, flat_ranking=True,
                                     genre_cap=10_000, per_anchor_cap=10_000,
                                     skip_ids=st.session_state.tv_skipped)
            t_shown = set(recs["tmdb_id"]) if len(recs) else set()
            tv_tabs = []
            if len(t_pool):
                t_rest = t_pool[~t_pool["tmdb_id"].isin(t_shown)]
                # TV genres arrive as a comma string, not a list like movies.
                # recommend_tv returns genres as a LIST in both paths, but be
                # tolerant of either so this does not silently bucket everything
                # into "" if the row builder changes again.
                def _first_genre(g):
                    if isinstance(g, (list, tuple)) and len(g):
                        return str(g[0]).strip()
                    if isinstance(g, str) and g.strip():
                        return g.split(",")[0].strip()
                    return ""
                t_primary = [_first_genre(g) for g in t_rest["genres"]]
                t_rest = t_rest.assign(_primary=t_primary)
                t_counts = t_rest["_primary"].value_counts()
                for g, c in t_counts.items():
                    if g and c >= TV_TAB_N * 2 and len(tv_tabs) < TV_MAX_TABS:
                        bucket = t_rest[t_rest["_primary"] == g]
                        if TV_DEEP_CUTS and "vote_count" in bucket.columns:
                            quiet = bucket[bucket["vote_count"] <= TV_TAB_VOTE_CEILING]
                            loud = bucket[bucket["vote_count"] > TV_TAB_VOTE_CEILING]
                            sub = pd.concat([quiet, loud]).head(TV_TAB_N)
                        else:
                            sub = bucket.head(TV_TAB_N)
                        tv_tabs.append((g, sub))

            if tv_tabs:
                t_labels = ["For you"] + [g for g, _ in tv_tabs]
                for tab, label in zip(st.tabs(t_labels), t_labels):
                    with tab:
                        if label == "For you":
                            for _, rec in recs.iterrows():
                                _tv_card(rec, "all")
                        else:
                            sub = dict(tv_tabs)[label]
                            _note = ("lesser-known " if TV_DEEP_CUTS else "")
                            st.caption(f"{len(sub)} {_note}{label.lower()} picks, "
                                       f"none repeated from For you.")
                            if TV_DEEP_CUTS:
                                st.caption("These skip the most-watched titles on "
                                           "purpose — anything great and famous still "
                                           "shows up under **For you**.")
                            for _, rec in sub.iterrows():
                                _tv_card(rec, label.lower().replace(" ", "").replace("&", ""))
            else:
                for _, rec in recs.iterrows():
                    _tv_card(rec, "all")

# ===========================================================================
# PERSIST (after all tabs/callbacks have settled this run)
# ===========================================================================
persist_to_ls()


# ===========================================================================
# STORAGE DEBUG (temporary — set SHOW_LS_DEBUG=False once persistence is confirmed)
# ===========================================================================
if SHOW_LS_DEBUG and st.query_params.get("debug") == "1":
    with st.expander("storage debug", expanded=True):
        info = {"ENABLE_PERSISTENCE": ENABLE_PERSISTENCE,
                "component_imported": _LS_AVAILABLE,
                "attempts": st.session_state.get("_ls_attempts"),
                "synced": {k: st.session_state.get(f"_ls_synced_{k}") for k in LS_KEYS}}
        st.write(info)
        if not _LS_AVAILABLE:
            st.error("Component NOT imported — is `streamlit-local-storage` in requirements.txt and did the Space rebuild?")
        else:
            _dbg_ls = get_local_storage()
            if _dbg_ls is None:
                st.warning("get_local_storage() returned None (init failed this run).")
            else:
                try:
                    _allitems = _dbg_ls.getAll()
                    st.write("getAll() keys:", list(_allitems.keys()) if isinstance(_allitems, dict) else _allitems)
                    st.write("raw stored values:", _allitems)
                except Exception as _e:
                    st.write(f"getAll() failed: {_e}")


st.divider()

# Feedback link. The first round produced the three most-corroborated defects
# in the project -- kids' content reaching adult profiles, search returning the
# wrong Kiki, ratings lost to a single stray click -- and the phrasing did the
# work: asking for ONE recommendation that felt wrong AND what was rated just
# before it is what made the reports diagnosable. A generic "how were the
# recs?" returned nothing actionable.
#
# Global rather than per-tab: a reader who bounces off one tab should still be
# able to say so, and a second link inside each tab reads as nagging.
FEEDBACK_URL = "https://forms.gle/REPLACE_ME"   # <- paste the Google Form link

if "REPLACE_ME" not in FEEDBACK_URL:
    st.markdown(
        f"<div class='feedback-line'>Tried it? "
        f"<a class='watch-link' href='{FEEDBACK_URL}' target='_blank' "
        f"rel='noopener noreferrer'>Tell me what it got wrong &rarr;</a> "
        f"— name one recommendation that felt off and what you'd rated just "
        f"before it. That's the version I can actually act on.</div>",
        unsafe_allow_html=True)

st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB. "
    "Film and TV data and images courtesy of The Movie Database. "
    "Book data from Goodreads and Hardcover."
)