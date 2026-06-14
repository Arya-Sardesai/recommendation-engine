"""
embed_movies.py

Embeds the movie corpus with MiniLM and builds a FAISS IndexFlatIP, mirroring
the books pipeline conventions (all-MiniLM-L6-v2, normalize_embeddings=True,
float32, IndexFlatIP, row-position alignment to the parquet, .npy + .index out).

Two DELIBERATE differences from the books embedder, both intentional:
  1. Embeds  title + overview + genres  (not overview alone). TMDB overviews
     are often a single short sentence; adding title+genres gives the model
     more signal. Books embedded long descriptions, so didn't need this.
     -> see build_embed_text(); flip USE_RICH_TEXT to A/B against overview-only.
  2. No row-alignment assertions. Books had a goodreads+hardcover two-block
     structure; movies are one source in parquet order, so alignment is trivial.

FAISS row i  <->  movies.parquet df.iloc[i]  (implicit id, no separate map file),
exactly like books.

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

# Output names mirror the books pattern (embeddings_minilm_v1 / faiss_minilm_v1)
# but namespaced for movies so the two never collide.
OUT_EMB = PROCESSED_DIR / "movie_embeddings_minilm_v1.npy"
OUT_FAISS = PROCESSED_DIR / "movie_faiss_minilm_v1.index"

MODEL_NAME = "all-MiniLM-L6-v2"
ENCODE_BATCH = 128          # same as books
SHARD = 20_000              # rows per encode() call; caps GPU memory on the 4060
USE_RICH_TEXT = True        # True: title+overview+genres ; False: overview only


def build_embed_text(df: pd.DataFrame) -> list[str]:
    """Build the per-film text string fed to MiniLM.

    Rich format:  "<title> (<year>). <genres>. <overview>"
    Genres in the corpus are a comma-joined string already (e.g. "Drama, Crime").
    Nulls are handled so a missing field just drops out cleanly.
    """
    if not USE_RICH_TEXT:
        return df["overview"].fillna("").astype(str).tolist()

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


def main():
    if not CORPUS.exists():
        raise SystemExit(f"ERROR: {CORPUS} not found. Build the corpus first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")

    df = pd.read_parquet(CORPUS)
    print(f"Corpus rows: {len(df):,}")

    texts = build_embed_text(df)
    print(f"Text format: {'rich (title+genres+overview)' if USE_RICH_TEXT else 'overview only'}")
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
            normalize_embeddings=True,   # -> IP == cosine, same as books
        ).astype("float32")
        embs.append(emb)
        print(f"  shard {start:>7,}-{start + len(shard):>7,}  ({emb.shape[0]:,} rows)")
        if device == "cuda":
            torch.cuda.empty_cache()

    combined_emb = np.vstack(embs).astype("float32")
    print(f"\nEmbeddings: {combined_emb.shape}  dtype={combined_emb.dtype}")
    assert combined_emb.shape[0] == len(df), "row count mismatch emb vs corpus!"

    # FAISS IndexFlatIP, plain add(), row-aligned to df (same as books)
    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    print(f"FAISS index: {index.ntotal:,} vectors, dim {combined_emb.shape[1]}")

    np.save(OUT_EMB, combined_emb)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"\nWrote:\n  {OUT_EMB}\n  {OUT_FAISS}")

    # quick self-check: nearest neighbours of row 0 should include row 0 itself
    D, I = index.search(combined_emb[:1], 4)
    print("\n--- sanity: nearest neighbours of film 0 ---")
    print(f"query: {df.iloc[0]['title']}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"  {rank}: [{score:.3f}] {df.iloc[idx]['title']}")


if __name__ == "__main__":
    main()