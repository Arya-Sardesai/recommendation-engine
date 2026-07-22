"""
convert_embeddings_fp16.py — halve embedding artifacts for the free HF Space.

The Space was hanging on cold boot downloading the 1.7 GB faiss_bgem3.index.
Fix: don't ship indexes at all — ship fp16 vectors (half the size) and let
app.py rebuild IndexFlatIP in memory at startup (a brute-force index is just
the vectors plus a header). fp16 is lossless enough for cosine ranking on
normalized vectors.

Run once from REPO ROOT after any re-embed, then upload via scripts/upload_data.py:
    python scripts/convert_embeddings_fp16.py
"""
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")

PAIRS = [
    ("embeddings_bgem3.npy",               "embeddings_bgem3_fp16.npy"),
    ("movie_embeddings_bgem3_v2_kw.npy",    "movie_embeddings_bgem3_v2_kw_fp16.npy"),
    ("tv_embeddings_bgem3_v2_notitle.npy",  "tv_embeddings_bgem3_v2_notitle_fp16.npy"),
]

for src, dst in PAIRS:
    p = PROCESSED / src
    if not p.exists():
        print(f"skip (missing): {src}")
        continue
    v = np.load(p)
    np.save(PROCESSED / dst, v.astype(np.float16))
    print(f"{src} -> {dst}  ({v.shape}, fp32 -> fp16)")