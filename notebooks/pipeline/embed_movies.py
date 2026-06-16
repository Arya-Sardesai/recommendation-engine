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

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CORPUS = PROCESSED_DIR / "movies.parquet"

# ---- experiment knobs --------------------------------------------------------
MODEL_NAME = "BAAI/bge-m3"          # Lever B multilingual; was all-MiniLM-L6-v2
EMBED_VARIANT = "v2_kw"             # overview | rich_v1 | v2_kw | v2_kw_people
KW_MAX = 20                         # cap keywords folded in (TMDB ~orders by relevance)
CAST_MAX = 4                        # _people arm only: top-billed cast to append
ENCODE_BATCH = 32                   # BGE-M3 on 8GB; drop to 16 if you OOM
MAX_SEQ_LEN = 512                   # film text is short; cap BGE-M3's 8192 default
USE_FP16 = True                     # halve weight memory on the 4060; flip off if norms drift
SHARD = 20_000                      # rows per encode() call; caps GPU memory

# Production-meta keyword tokens that carry no thematic meaning -> strip.
KW_STOP = {
    "aftercreditsstinger", "duringcreditsstinger", "based on novel or book",
    "based on comic", "based on young adult novel", "based on true story",
    "woman director", "sequel", "prequel", "remake", "reboot",
}

# Artifact names are VERSIONED by variant+model so v1 stays live and rollback is
# just flipping the filename constants in app.py (M_EMB / M_FAISS).
_MODEL_TAG = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:12]
OUT_EMB = PROCESSED_DIR / f"movie_embeddings_{_MODEL_TAG}_{EMBED_VARIANT}.npy"
OUT_FAISS = PROCESSED_DIR / f"movie_faiss_{_MODEL_TAG}_{EMBED_VARIANT}.index"


def _to_terms(val, max_n=None, stop=None):
    """Coerce a comma-string OR an array/list column value into clean terms.

    Handles both the str columns (keywords, genres) and the ndarray columns
    (cast, directors_all) uniformly. Drops empties / 'nan' / stoplisted tokens.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        terms = [str(x).strip() for x in val]
    else:
        s = str(val)
        if s.strip().lower() in ("", "nan"):
            return []
        terms = [t.strip() for t in s.split(",")]
    out = []
    for t in terms:
        if not t or t.lower() == "nan":
            continue
        if stop and t.lower() in stop:
            continue
        out.append(t)
    return out[:max_n] if max_n else out


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

    if EMBED_VARIANT not in ("v2_kw", "v2_kw_people"):
        raise SystemExit(f"Unknown EMBED_VARIANT: {EMBED_VARIANT!r}")

    # ---- v2: title dropped, overview leads, keywords + genres carry theme ----
    want_people = EMBED_VARIANT == "v2_kw_people"
    has = lambda c: c in df.columns

    overview = df["overview"].fillna("").astype(str).tolist()
    keywords = df["keywords"].tolist() if has("keywords") else [None] * len(df)
    genres = df["genres"].tolist() if has("genres") else [None] * len(df)
    directors = df["directors_all"].tolist() if has("directors_all") else (
        df["director"].tolist() if has("director") else [None] * len(df))
    cast = df["cast"].tolist() if has("cast") else [None] * len(df)

    texts = []
    for i in range(len(df)):
        parts = []
        ov = overview[i].strip()
        if ov and ov.lower() != "nan":
            parts.append(ov)
        kw = _to_terms(keywords[i], max_n=KW_MAX, stop=KW_STOP)
        if kw:
            parts.append(f"Themes: {', '.join(kw)}.")
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
    return texts


def main():
    if not CORPUS.exists():
        raise SystemExit(f"ERROR: {CORPUS} not found. Build the corpus first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")
    print(f"Model: {MODEL_NAME}  |  Variant: {EMBED_VARIANT}")
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