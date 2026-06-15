"""
embed_tv.py

Embeds the TV corpus with MiniLM and builds a FAISS IndexFlatIP, mirroring
embed_movies.py exactly (all-MiniLM-L6-v2, normalize_embeddings=True, float32,
IndexFlatIP, row-position alignment to the parquet, .npy + .index out).

Differences from embed_movies.py are ONLY schema-driven (TV corpus columns):
  1. Reads tv_corpus.parquet and uses the TV column names:
       title   -> name
       release_year -> start_year   (Int64 year derived in the corpus build)
     genres / overview are the same comma-joined-string + plain-text fields.
  2. The corpus build already wrote an `embed_text` column. We REBUILD it here
     anyway (same rich format) so the embedder is self-contained and you can
     A/B overview-only via USE_RICH_TEXT without rerunning the corpus step.
  3. Output names namespaced for TV so books/movies/TV indexes never collide.

FAISS row i  <->  tv_corpus.parquet df.iloc[i]  (implicit id, no separate map),
exactly like books and movies.

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

OUT_EMB = PROCESSED_DIR / "tv_embeddings_minilm_v1.npy"
OUT_FAISS = PROCESSED_DIR / "tv_faiss_minilm_v1.index"

MODEL_NAME = "all-MiniLM-L6-v2"
ENCODE_BATCH = 128          # same as movies/books
SHARD = 20_000              # rows per encode() call; caps GPU memory on the 4060
USE_RICH_TEXT = True        # True: name+genres+overview ; False: overview only


def build_embed_text(df: pd.DataFrame) -> list[str]:
    """Build the per-series text string fed to MiniLM.

    Rich format:  "<name> (<start_year>). <genres>. <overview>"
    Mirrors the movies embedder; only the column names differ (name/start_year).
    """
    if not USE_RICH_TEXT:
        return df["overview"].fillna("").astype(str).tolist()

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


def main():
    if not CORPUS.exists():
        raise SystemExit(f"ERROR: {CORPUS} not found. Build the corpus first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")

    df = pd.read_parquet(CORPUS)
    print(f"Corpus rows: {len(df):,}")

    texts = build_embed_text(df)
    print(f"Text format: {'rich (name+genres+overview)' if USE_RICH_TEXT else 'overview only'}")
    print(f"Sample embed text:\n  {texts[0][:160]!r}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    # sharded encode to keep GPU memory bounded on 8GB; concatenate after
    embs = []
    for start in range(0, len(texts), SHARD):
        shard = texts[start:start + SHARD]
        emb = model.encode(
            shard,
            batch_size=ENCODE_BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # -> IP == cosine, same as books/movies
        ).astype("float32")
        embs.append(emb)
        print(f"  shard {start:>7,}-{start + len(shard):>7,}  ({emb.shape[0]:,} rows)")
        if device == "cuda":
            torch.cuda.empty_cache()

    combined_emb = np.vstack(embs).astype("float32")
    print(f"\nEmbeddings: {combined_emb.shape}  dtype={combined_emb.dtype}")
    assert combined_emb.shape[0] == len(df), "row count mismatch emb vs corpus!"

    # FAISS IndexFlatIP, plain add(), row-aligned to df (same as books/movies)
    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    print(f"FAISS index: {index.ntotal:,} vectors, dim {combined_emb.shape[1]}")

    np.save(OUT_EMB, combined_emb)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"\nWrote:\n  {OUT_EMB}\n  {OUT_FAISS}")

    # quick self-check: nearest neighbours of row 0 should include row 0 itself
    D, I = index.search(combined_emb[:1], 4)
    print("\n--- sanity: nearest neighbours of series 0 ---")
    print(f"query: {df.iloc[0]['name']}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"  {rank}: [{score:.3f}] {df.iloc[idx]['name']}")


if __name__ == "__main__":
    main()