"""
embed_v1.py  (books -- BGE-M3 full re-embed)

Embeds the FULL book corpus (books_v1.parquet) with BGE-M3 and rebuilds the
FAISS index.

WHY THIS IS NO LONGER INCREMENTAL
---------------------------------
The old version reused the 409K Goodreads MiniLM vectors and embedded only the
~9.7K new Hardcover rows. That shortcut is impossible across a model change:
MiniLM is 384d and BGE-M3 is 1024d, so the old vectors can't be stacked with new
ones. Books therefore get a clean full re-embed of all ~432K descriptions.

NO TEXT CHANGE. Books were always embedded on `description` only (no title), so
Lever A never applied to them -- this is a pure Lever-B model swap. ~432K rows at
batch 32 on the 4060 is ~20-25 min.

Row order is preserved (Goodreads block first, then Hardcover, as written in
books_v1.parquet), so FAISS row i <-> df.iloc[i] still holds with no id-map file.

ARTIFACT SIZE: 432,715 x 1024 x 4B ~= 1.8GB for the .npy and again for the index.
This is the heaviest corpus -- the one to watch on HF Space boot RAM. If boot
memory is tight, drop the .npy load in app.py and use index.reconstruct(i)
instead (IndexFlatIP already stores the full vectors).

Run from REPO ROOT (GPU strongly preferred):
    python notebooks/pipeline/embed_v1.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

PROCESSED_DIR = Path("data/processed")
V1_CORPUS = PROCESSED_DIR / "books_v1.parquet"

# ---- knobs (mirror movies/TV) -----------------------------------------------
MODEL_NAME = "BAAI/bge-m3"
ENCODE_BATCH = 32
MAX_SEQ_LEN = 512                   # book descriptions are short-to-medium; cap BGE-M3's 8192
USE_FP16 = True
SHARD = 20_000

_MODEL_TAG = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:12]
V1_EMBEDDINGS = PROCESSED_DIR / f"embeddings_{_MODEL_TAG}.npy"      # embeddings_bgem3.npy
V1_FAISS = PROCESSED_DIR / f"faiss_{_MODEL_TAG}.index"             # faiss_bgem3.index


def main():
    if not V1_CORPUS.exists():
        raise SystemExit(f"ERROR: {V1_CORPUS} not found.")

    df = pd.read_parquet(V1_CORPUS)
    print(f"Combined corpus: {len(df):,} books (full re-embed, not incremental)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}  |  model={MODEL_NAME}")
    print(f"batch={ENCODE_BATCH}  max_seq_len={MAX_SEQ_LEN}  fp16={USE_FP16 and device=='cuda'}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if USE_FP16 and device == "cuda":
        model.half()

    descriptions = df["description"].fillna("").astype(str).tolist()

    embs = []
    for start in range(0, len(descriptions), SHARD):
        shard = descriptions[start:start + SHARD]
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
    assert combined_emb.shape[0] == len(df), "Embedding count != corpus count!"
    norms = np.linalg.norm(combined_emb[:1000], axis=1)
    print(f"norm check (first 1k): min={norms.min():.4f} max={norms.max():.4f}")
    if not (0.99 <= norms.min() and norms.max() <= 1.01):
        print("  WARNING: norms drifted from 1.0 -- set USE_FP16=False and re-run.")

    np.save(V1_EMBEDDINGS, combined_emb)
    print(f"Saved embeddings to {V1_EMBEDDINGS}")

    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    faiss.write_index(index, str(V1_FAISS))
    print(f"FAISS index rebuilt: {index.ntotal:,} vectors -> {V1_FAISS}")

    D, I = index.search(combined_emb[:1], 4)
    print("\n--- sanity: nearest neighbours of book 0 ---")
    title_col = "title" if "title" in df.columns else df.columns[0]
    print(f"query: {df.iloc[0][title_col]}")
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        print(f"  {rank}: [{score:.3f}] {df.iloc[idx][title_col]}")


if __name__ == "__main__":
    main()