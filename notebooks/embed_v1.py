"""
Embed the new Hardcover books and rebuild the v1 FAISS index.

Strategy: reuse the existing 409K Goodreads embeddings, embed ONLY the ~9.7K
new Hardcover rows, stack them, rebuild the index. The combined parquet
(books_v1.parquet) preserves row order: Goodreads rows first, then Hardcover,
so we can align embeddings by position.

Run with GPU (Turbo mode). Embedding ~10K books takes ~1 min.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

PROCESSED_DIR = Path("data/processed")

V1_CORPUS = PROCESSED_DIR / "books_v1.parquet"
OLD_EMBEDDINGS = PROCESSED_DIR / "embeddings_minilm_v0_deduped.npy"  # 409K Goodreads
V1_EMBEDDINGS = PROCESSED_DIR / "embeddings_minilm_v1.npy"
V1_FAISS = PROCESSED_DIR / "faiss_minilm_v1.index"

def main():
    df = pd.read_parquet(V1_CORPUS)
    print(f"Combined corpus: {len(df):,} books")

    old_emb = np.load(OLD_EMBEDDINGS).astype("float32")
    print(f"Existing Goodreads embeddings: {old_emb.shape}")

    # the new books are the Hardcover rows (source == 'hardcover'), appended after GR
    n_old = old_emb.shape[0]
    new_df = df.iloc[n_old:]   # rows after the Goodreads block
    print(f"New rows to embed: {len(new_df):,}")

    # sanity check: the rows after n_old should all be hardcover
    assert (new_df["source"] == "hardcover").all(), "Row alignment off - new rows aren't all hardcover!"
    # and the rows before should all be goodreads
    assert (df.iloc[:n_old]["source"] == "goodreads").all(), "Row alignment off - old block isn't all goodreads!"
    print("Row alignment verified (GR block first, HC block after)")

    # embed the new descriptions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    new_descriptions = new_df["description"].tolist()
    new_emb = model.encode(
        new_descriptions, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    print(f"New embeddings: {new_emb.shape}")

    # stack: old + new, aligned to df row order
    combined_emb = np.vstack([old_emb, new_emb])
    print(f"Combined embeddings: {combined_emb.shape}")
    assert combined_emb.shape[0] == len(df), "Embedding count != corpus count!"

    np.save(V1_EMBEDDINGS, combined_emb)
    print(f"Saved embeddings to {V1_EMBEDDINGS}")

    # rebuild FAISS
    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    faiss.write_index(index, str(V1_FAISS))
    print(f"FAISS index rebuilt: {index.ntotal:,} vectors -> {V1_FAISS}")

if __name__ == "__main__":
    main()