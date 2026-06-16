"""
embed_tv.py  (v2 title-drop + Lever B multilingual model)

Embeds the TV corpus and builds a FAISS IndexFlatIP, mirroring embed_movies.py
(normalize_embeddings=True, float32, IndexFlatIP, row-position alignment, .npy +
.index out).

WHAT'S SET HERE
---------------
- TEXT recipe = "v2_notitle": series NAME dropped, overview leads, genres folded
  in. This is TV's ONLY Lever-A move -- the asaniczka TV dump has no keywords
  column, so the keyword enrichment that movies got is not available here. (When
  the TMDB API is reachable from Canada, TV keywords become a later upgrade.)
  Dropping the name is the same title-leakage fix validated on movies.
- MODEL = BAAI/bge-m3, the multilingual model that won the movies pilot
  (fixed Oldboy, improved Parasite/Bollywood cases). TV is ~57% non-English, so
  it benefits at least as much as movies.

8GB-CARD NOTES: identical to embed_movies.py -- batch 32, max_seq_len 512, fp16.
If the norm check isn't ~1.0, set USE_FP16=False. If you OOM, drop batch to 16.

Artifact names auto-version off MODEL_NAME + variant:
  tv_embeddings_bgem3_v2_notitle.npy  /  tv_faiss_bgem3_v2_notitle.index
No collision with the MiniLM v1 TV artifacts; v1 stays live.

Run from REPO ROOT:
    python notebooks/pipeline/embed_tv.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CORPUS = PROCESSED_DIR / "tv_corpus.parquet"

# ---- experiment knobs --------------------------------------------------------
MODEL_NAME = "BAAI/bge-m3"          # multilingual; won the movies pilot
EMBED_VARIANT = "v2_notitle"        # overview | rich_v1 | v2_notitle
ENCODE_BATCH = 32                   # BGE-M3 on 8GB; drop to 16 if you OOM
MAX_SEQ_LEN = 512
USE_FP16 = True                     # flip off if the norm check drifts from 1.0
SHARD = 20_000


_MODEL_TAG = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:12]
OUT_EMB = PROCESSED_DIR / f"tv_embeddings_{_MODEL_TAG}_{EMBED_VARIANT}.npy"
OUT_FAISS = PROCESSED_DIR / f"tv_faiss_{_MODEL_TAG}_{EMBED_VARIANT}.index"


def _clean(s) -> str:
    s = "" if s is None else str(s).strip()
    return "" if s.lower() == "nan" else s


def build_embed_text(df: pd.DataFrame) -> list[str]:
    """Per-series text fed to the encoder, per EMBED_VARIANT.

    TV column names: name / start_year / genres / overview (genres is a
    comma-joined string, same as movies).
    """
    if EMBED_VARIANT == "overview":
        return df["overview"].fillna("").astype(str).tolist()

    if EMBED_VARIANT == "rich_v1":
        name = df["name"].fillna("").astype(str)
        year = df["start_year"].astype("Int64").astype(str).replace("<NA>", "")
        genres = df["genres"].fillna("").astype(str)
        overview = df["overview"].fillna("").astype(str)
        texts = []
        for n, y, g, o in zip(name, year, genres, overview):
            head = f"{n} ({y})." if y else f"{n}."
            mid = f" {g}." if g and g.lower() != "nan" else ""
            tail = f" {o}" if o and o.lower() != "nan" else ""
            texts.append((head + mid + tail).strip())
        return texts

    if EMBED_VARIANT != "v2_notitle":
        raise SystemExit(f"Unknown EMBED_VARIANT: {EMBED_VARIANT!r}")

    # ---- v2: name dropped, overview leads, genres carry the rest ----
    overview = df["overview"].tolist()
    genres = df["genres"].tolist() if "genres" in df.columns else [None] * len(df)
    texts = []
    for ov, g in zip(overview, genres):
        parts = []
        ov = _clean(ov)
        if ov:
            parts.append(ov)
        gens = [t.strip() for t in _clean(g).split(",") if t.strip()]
        if gens:
            parts.append(f"Genres: {', '.join(gens)}.")
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
    print(f"Empty embed texts: {sum(1 for t in texts if not t):,}")
    print(f"Sample [0]:\n  {texts[0][:240]!r}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if USE_FP16 and device == "cuda":
        model.half()

    embs = []
    for start in range(0, len(texts), SHARD):
        shard = texts[start:start + SHARD]
        emb = model.encode(
            shard, batch_size=ENCODE_BATCH, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        embs.append(emb)
        print(f"  shard {start:>7,}-{start + len(shard):>7,}  ({emb.shape[0]:,} rows)")
        if device == "cuda":
            torch.cuda.empty_cache()

    combined_emb = np.vstack(embs).astype("float32")
    print(f"\nEmbeddings: {combined_emb.shape}  dtype={combined_emb.dtype}")
    assert combined_emb.shape[0] == len(df), "row count mismatch emb vs corpus!"
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
    print("\n--- sanity: nearest neighbours of series 0 ---")
    print(f"query: {df.iloc[0]['name']}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"  {rank}: [{score:.3f}] {df.iloc[idx]['name']}")


if __name__ == "__main__":
    main()