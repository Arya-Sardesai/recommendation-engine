"""
embed_movies.py  (v2 embed-text + Lever B multilingual model)

Embeds the movie corpus and builds a FAISS IndexFlatIP, mirroring the books
pipeline conventions (normalize_embeddings=True, float32, IndexFlatIP,
row-position alignment to the parquet, .npy + .index out).

WHAT'S SET HERE
---------------
- TEXT recipe = "v2_kw": title DROPPED, overview leads, TMDB keywords + genres
  folded in. (Lever A, already validated against v1 on the title-leakage cases.)
- MODEL = BAAI/bge-m3 (Lever B). 568M params, 1024d, multilingual. Targets the
  residual non-English failures MiniLM-v2_kw left behind (Om Shanti Om -> English
  revenge films; Oldboy -> exploitation). Embeds symmetrically, no query/passage
  prefix needed -- every film is embedded as plain text, which is the item<->item
  case. Normalizes cleanly, so IndexFlatIP == cosine still holds.

*** KEYWORD COLUMN (changed -- read this) ***
The TMDB bulk enrichment wrote fresh keywords to `keywords_tmdb`. The legacy
Kaggle `keywords` column still exists in the parquet and is STALE (2023 dump,
pre-rename location vocabulary: "london" vs TMDB's "london, england").
This script previously read `keywords` and would have silently embedded stale
tags under a `v2_kw` artifact name -- no error, no warning. KW_COL is now
explicit and MISSING IT IS A HARD FAILURE. Do not add a fallback.

Union with the legacy column was measured and rejected: only 767 films (0.7% of
corpus) have keywords in the Kaggle column but not in TMDB's, and the two
vocabularies disagree on location naming, so a union would split concepts across
two tokens.

Stoplist lives in kw_common.py (shared with embed_tv.py) so movies and TV cannot
drift apart. See that file for why "based on novel or book" is KEPT and
"woman director" is STRIPPED.

8GB-CARD NOTES (why these aren't MiniLM's numbers)
--------------------------------------------------
BGE-M3 is ~15x heavier than MiniLM. To fit the RTX 4060 (8GB):
  - ENCODE_BATCH dropped 128 -> 32   (drop to 16 if you still OOM)
  - MAX_SEQ_LEN capped at 512        (film text is short; BGE-M3 defaults to 8192)
  - USE_FP16=True halves weight memory and speeds encode. If the post-encode norm
    check isn't ~1.0000 or you see NaNs, set USE_FP16=False and lower ENCODE_BATCH.
Expect movies (~110K) to take ~15-30 min instead of ~60s. That's the model, fine.

Artifact names auto-version off MODEL_NAME, so this writes
  movie_embeddings_bgem3_v2_kw.npy  /  movie_faiss_bgem3_v2_kw.index
with NO collision against MiniLM v1 or v2_kw. v1 stays live; rollback is just the
filename constants in app.py.

Run from REPO ROOT:
    python notebooks/pipeline/embed_movies.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

from kw_common import KW_STOP, to_terms as _to_terms

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CORPUS = PROCESSED_DIR / "movies.parquet"

# ---- experiment knobs --------------------------------------------------------
MODEL_NAME = "BAAI/bge-m3"          # Lever B multilingual; was all-MiniLM-L6-v2
EMBED_VARIANT = "v2_kw2"             # overview | rich_v1 | v2_kw | v2_kw_people
KW_COL = "keywords_tmdb"            # HARD REQUIREMENT -- see docstring. No fallback.
KW_MAX = 20                         # cap keywords folded in (p90 is 13, so rarely binds)
CAST_MAX = 4                        # _people arm only: top-billed cast to append
ENCODE_BATCH = 32                   # BGE-M3 on 8GB; drop to 16 if you OOM
MAX_SEQ_LEN = 512                   # film text is short; cap BGE-M3's 8192 default
USE_FP16 = True                     # halve weight memory on the 4060; flip off if norms drift
SHARD = 20_000                      # rows per encode() call; caps GPU memory

# Measured coverage on the enriched corpus: 66.7% of films have >=1 keyword.
# If the observed rate lands far below this, the wrong column is being read.
KW_COVERAGE_FLOOR = 0.55

# Artifact names are VERSIONED by variant+model so v1 stays live and rollback is
# just flipping the filename constants in app.py (M_EMB / M_FAISS).
_MODEL_TAG = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:12]
OUT_EMB = PROCESSED_DIR / f"movie_embeddings_{_MODEL_TAG}_{EMBED_VARIANT}.npy"
OUT_FAISS = PROCESSED_DIR / f"movie_faiss_{_MODEL_TAG}_{EMBED_VARIANT}.index"


def build_embed_text(df: pd.DataFrame) -> list[str]:
    """Build the per-film text fed to the encoder, per EMBED_VARIANT."""
    if EMBED_VARIANT == "overview":
        return df["overview"].fillna("").astype(str).tolist()

    if EMBED_VARIANT == "rich_v1":
        title = df["title"].fillna("").astype(str)
        year = df["release_year"].astype("Int64").astype(str).replace("<NA>", "")
        genres = df["genres"].fillna("").astype(str)
        overview = df["overview"].fillna("").astype(str)
        texts = []
        for t, y, g, o in zip(title, year, genres, overview):
            head = f"{t} ({y})." if y else f"{t}."
            mid = f" {g}." if g and g.lower() != "nan" else ""
            tail = f" {o}" if o and o.lower() != "nan" else ""
            texts.append((head + mid + tail).strip())
        return texts

    if EMBED_VARIANT not in ("v2_kw", "v2_kw2", "v2_kw_people"):
        raise SystemExit(f"Unknown EMBED_VARIANT: {EMBED_VARIANT!r}")

    # --- fail loud on the wrong keyword column ------------------------------
    if KW_COL not in df.columns:
        raise SystemExit(
            f"ERROR: required keyword column {KW_COL!r} not in corpus.\n"
            f"  Present columns: {list(df.columns)}\n"
            f"  Run the TMDB merge before embedding. Do NOT fall back to the\n"
            f"  legacy 'keywords' column -- it is stale and uses pre-rename\n"
            f"  location vocabulary."
        )

    # ---- v2: title dropped, overview leads, keywords + genres carry theme ----
    want_people = EMBED_VARIANT == "v2_kw_people"
    has = lambda c: c in df.columns

    overview = df["overview"].fillna("").astype(str).tolist()
    keywords = df[KW_COL].tolist()
    genres = df["genres"].tolist() if has("genres") else [None] * len(df)
    directors = df["directors_all"].tolist() if has("directors_all") else (
        df["director"].tolist() if has("director") else [None] * len(df))
    cast = df["cast"].tolist() if has("cast") else [None] * len(df)

    texts, n_with_kw = [], 0
    for i in range(len(df)):
        parts = []
        ov = overview[i].strip()
        if ov and ov.lower() != "nan":
            parts.append(ov)
        kw = _to_terms(keywords[i], max_n=KW_MAX, stop=KW_STOP)
        if kw:
            parts.append(f"Themes: {', '.join(kw)}.")
            n_with_kw += 1
        gens = _to_terms(genres[i])
        if gens:
            parts.append(f"Genres: {', '.join(gens)}.")
        if want_people:
            dirs = _to_terms(directors[i], max_n=2)
            if dirs:
                parts.append(f"Directed by {', '.join(dirs)}.")
            cst = _to_terms(cast[i], max_n=CAST_MAX)
            if cst:
                parts.append(f"Starring {', '.join(cst)}.")
        texts.append(" ".join(parts).strip())

    # --- coverage guard: catches a wrong-but-present column, and catches an
    # over-aggressive stoplist, before 25 minutes of GPU time are spent.
    rate = n_with_kw / max(len(df), 1)
    print(f"Keyword coverage in embed text: {n_with_kw:,}/{len(df):,} ({rate:.1%}) "
          f"from column {KW_COL!r}")
    if rate < KW_COVERAGE_FLOOR:
        raise SystemExit(
            f"ERROR: keyword coverage {rate:.1%} is below the {KW_COVERAGE_FLOOR:.0%} "
            f"floor.\n  Either the wrong column is being read or KW_STOP is eating "
            f"real signal.\n  Investigate before burning GPU time."
        )
    return texts


def main():
    if not CORPUS.exists():
        raise SystemExit(f"ERROR: {CORPUS} not found. Build the corpus first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")
    print(f"Model: {MODEL_NAME}  |  Variant: {EMBED_VARIANT}  |  kw_col: {KW_COL}")
    print(f"batch={ENCODE_BATCH}  max_seq_len={MAX_SEQ_LEN}  fp16={USE_FP16 and device=='cuda'}")

    df = pd.read_parquet(CORPUS)
    print(f"Corpus rows: {len(df):,}")

    texts = build_embed_text(df)
    empties = sum(1 for t in texts if not t)
    print(f"Empty embed texts: {empties:,} (will embed as '')")
    for k in (0, len(texts) // 2):
        print(f"Sample [{k}]:\n  {texts[k][:240]!r}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if USE_FP16 and device == "cuda":
        model.half()   # fp16 weights -> fits 8GB with headroom; outputs still normalize

    embs = []
    for start in range(0, len(texts), SHARD):
        shard = texts[start:start + SHARD]
        emb = model.encode(
            shard,
            batch_size=ENCODE_BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # IP == cosine; required for IndexFlatIP
        ).astype("float32")              # store float32 regardless of fp16 compute
        embs.append(emb)
        print(f"  shard {start:>7,}-{start + len(shard):>7,}  ({emb.shape[0]:,} rows)")
        if device == "cuda":
            torch.cuda.empty_cache()

    combined_emb = np.vstack(embs).astype("float32")
    print(f"\nEmbeddings: {combined_emb.shape}  dtype={combined_emb.dtype}")
    assert combined_emb.shape[0] == len(df), "row count mismatch emb vs corpus!"

    # fp16 can nudge norms slightly off 1.0; verify before trusting IP==cosine
    norms = np.linalg.norm(combined_emb[:1000], axis=1)
    print(f"norm check (first 1k): min={norms.min():.4f} max={norms.max():.4f}")
    if not (0.99 <= norms.min() and norms.max() <= 1.01):
        print("  WARNING: norms drifted from 1.0 -- set USE_FP16=False and re-run.")

    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    print(f"FAISS index: {index.ntotal:,} vectors, dim {combined_emb.shape[1]}")

    np.save(OUT_EMB, combined_emb)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"\nWrote:\n  {OUT_EMB}\n  {OUT_FAISS}")

    D, I = index.search(combined_emb[:1], 4)
    print("\n--- sanity: nearest neighbours of film 0 ---")
    print(f"query: {df.iloc[0]['title']}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"  {rank}: [{score:.3f}] {df.iloc[idx]['title']}")


if __name__ == "__main__":
    main()
