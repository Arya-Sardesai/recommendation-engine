"""
embed_books.py  (v2 Hardcover corpus: title-drop + genres + moods, BGE-M3)

Embeds books_v2_hc.parquet and builds a FAISS IndexFlatIP, mirroring
embed_tv.py / embed_movies.py exactly: normalize_embeddings=True, float32,
IndexFlatIP, row-position alignment, .npy + .index out.

WHY TITLE-DROP
--------------
Same reason as movies and TV: including the title leaks title tokens into the
vector, so `Dune Messiah` retrieves `Dune` and the list fills with sequels. The
app applies a separate title-token penalty at rank time; embedding the title
too would double-count the effect. Costs ~50% of hit@20 on leave-one-out eval
and is kept deliberately -- the diversity is the product.

WHY GENRES + MOODS IN THE TEXT
------------------------------
Mirrors the movies `_kw` variant (v2_kw2), which is the configuration currently
producing the best-looking tab. Hardcover's community tags carry signal the
description does not:
  - Genre  90.5% coverage, median 2-10 tags
  - Mood   35.7% coverage  <- tone: "dark", "tense", "reflective", "funny"
Mood is the important one. Tonal register is invisible in plot summaries (the
known residual that keyword tagging fixed for movies), and this is the first
book corpus that has it at all. The old Goodreads tags were 99.3% kNN-PROPAGATED
-- inferred from these same embeddings, so they largely agreed with whatever the
embeddings already believed, which is why they could not correct
`Love in the Time of Cholera` -> generic romance.

These fields ALSO feed a separate tag parquet (build_book_tags_hc.py) used for
re-rank. Two different jobs: text here affects RETRIEVAL (which candidates come
back), the tag layer affects RANKING (how they are ordered). Movies do both.

ARTIFACT NAMES auto-version off MODEL_NAME + variant:
  book_embeddings_bgem3_v2_hc_notitle.npy
  book_faiss_bgem3_v2_hc_notitle.index

ALIGNMENT LAW: these are NEW filenames. The live corpus (books_v1.parquet,
embeddings_bgem3.npy, faiss_bgem3.index) is untouched until app.py names the
new files -- embeddings + index uploaded FIRST, parquet second, constants last.

Run from REPO ROOT:
    python notebooks\\pipeline\\embed_books.py
    python notebooks\\pipeline\\embed_books.py --limit 500     # smoke test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
CORPUS = PROCESSED_DIR / "books_v2_hc.parquet"

MODEL_NAME = "BAAI/bge-m3"           # multilingual; won the movies pilot
EMBED_VARIANT = "v2_hc_notitle"      # description | v2_hc_notitle
GENRE_MAX = 12                       # p90 of genre-tag count
MOOD_MAX = 8
ENCODE_BATCH = 32                    # BGE-M3 on 8GB; drop to 16 if you OOM
MAX_SEQ_LEN = 512
USE_FP16 = True                      # flip off if the norm check drifts from 1.0
SHARD_SIZE = 20_000                  # encode in shards so a crash loses less

_MODEL_TAG = MODEL_NAME.split("/")[-1].replace("-", "").lower()[:12]
OUT_EMB = PROCESSED_DIR / f"book_embeddings_{_MODEL_TAG}_{EMBED_VARIANT}.npy"
OUT_FAISS = PROCESSED_DIR / f"book_faiss_{_MODEL_TAG}_{EMBED_VARIANT}.index"

# Genres so common they carry no discriminative signal. Kept in the embed text
# (harmless there) but this list is shared with build_book_tags_hc.py, which
# drops them from the tag matrix -- a tag on 55% of the corpus is noise.
GENERIC_GENRES = {
    "fiction", "general", "nonfiction", "non-fiction", "juvenile fiction",
    "literature", "books", "ebook", "audiobook",
}


def _clean(s) -> str:
    s = "" if s is None else str(s).strip()
    return "" if s.lower() == "nan" else s


def _listcol(v, limit):
    """Corpus list columns arrive as numpy arrays or lists; normalise + cap."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        out = [_clean(x) for x in list(v)[:limit]]
        return [x for x in out if x]
    s = _clean(v)
    return [s] if s else []


def build_embed_text(df: pd.DataFrame) -> list[str]:
    """Per-book text fed to the encoder, per EMBED_VARIANT.

    Book column names: title / primary_author / description / genres / moods
    NOTE the title and author are deliberately absent from v2_hc_notitle.
    """
    if EMBED_VARIANT == "description":
        return df["description"].fillna("").astype(str).tolist()

    if EMBED_VARIANT != "v2_hc_notitle":
        raise SystemExit(f"Unknown EMBED_VARIANT: {EMBED_VARIANT!r}")

    for col in ("description", "genres", "moods"):
        if col not in df.columns:
            raise SystemExit(
                f"ERROR: column {col!r} missing from {CORPUS}.\n"
                f"  Run fetch_hardcover_v2.py --stage merge --write first."
            )

    texts = []
    for desc, genres, moods in zip(df["description"], df["genres"], df["moods"]):
        parts = [_clean(desc)]
        g = _listcol(genres, GENRE_MAX)
        m = _listcol(moods, MOOD_MAX)
        if g:
            parts.append("Genres: " + ", ".join(g) + ".")
        if m:
            parts.append("Mood: " + ", ".join(m) + ".")
        texts.append(" ".join(p for p in parts if p).strip())
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N rows (smoke test)")
    args = ap.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"ERROR: {CORPUS} not found. Build the corpus first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {MODEL_NAME}  |  Variant: {EMBED_VARIANT}  |  device: {device}")
    print(f"batch={ENCODE_BATCH}  max_seq_len={MAX_SEQ_LEN}  fp16={USE_FP16 and device == 'cuda'}")

    df = pd.read_parquet(CORPUS)
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"SMOKE TEST: {len(df):,} rows only -- artifacts will NOT be written")
    print(f"Corpus: {len(df):,} books")

    texts = build_embed_text(df)
    empty = sum(1 for t in texts if not t)
    print(f"Empty embed texts: {empty:,}")
    if empty:
        print("  WARNING: empty texts embed to a near-arbitrary vector and will")
        print("  retrieve unpredictably. They passed the description filter, so")
        print("  this should be 0 -- investigate if it is not.")

    print("\n--- sample embed texts ---")
    for k in (0, len(texts) // 2, len(texts) - 1):
        t = texts[k]
        print(f"[{k}] {df.iloc[k]['title'][:45]!r}")
        print(f"    {t[:240]}{'...' if len(t) > 240 else ''}\n")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if USE_FP16 and device == "cuda":
        model.half()

    embs = []
    for start in range(0, len(texts), SHARD_SIZE):
        shard = texts[start:start + SHARD_SIZE]
        print(f"\nshard {start:,}..{start + len(shard):,}")
        emb = model.encode(
            shard, batch_size=ENCODE_BATCH, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        embs.append(emb)

    combined_emb = np.vstack(embs)
    print(f"\nEmbeddings: {combined_emb.shape}")

    norms = np.linalg.norm(combined_emb, axis=1)
    print(f"Norm check: min {norms.min():.5f}  max {norms.max():.5f}  (want ~1.0)")
    if abs(norms.mean() - 1.0) > 1e-3:
        print("  WARNING: norms drifted from 1.0 -- set USE_FP16=False and re-run.")

    if len(combined_emb) != len(df):
        raise SystemExit(f"ALIGNMENT FAILURE: {len(combined_emb)} vectors vs {len(df)} rows")

    index = faiss.IndexFlatIP(combined_emb.shape[1])
    index.add(combined_emb)
    print(f"FAISS index: {index.ntotal:,} vectors, dim {combined_emb.shape[1]}")

    # self-match probes at scattered positions, including the last row
    print("\n--- self-match probes (each row must retrieve itself at rank 1) ---")
    ok = True
    probe_positions = [0, len(df) // 4, len(df) // 2, (3 * len(df)) // 4, len(df) - 1]
    for pos in probe_positions:
        _, idxs = index.search(combined_emb[pos:pos + 1], 1)
        hit = int(idxs[0][0])
        flag = "OK " if hit == pos else "FAIL"
        if hit != pos:
            ok = False
        print(f"  {flag} row {pos:>6,} -> {hit:>6,}   {df.iloc[pos]['title'][:40]}")
    if not ok:
        raise SystemExit("Self-match probe failed -- do NOT upload these artifacts.")

    if args.limit:
        print("\nSMOKE TEST complete -- nothing written. Re-run without --limit.")
        return

    np.save(OUT_EMB, combined_emb)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"\nwrote {OUT_EMB}")
    print(f"wrote {OUT_FAISS}")
    print(f"\nALIGNMENT: {len(df):,} rows == {combined_emb.shape[0]:,} vectors "
          f"== {index.ntotal:,} index entries")
    print("\nUpload order: embeddings + index FIRST (inert under new names),")
    print("parquet SECOND, then flip B_EMB/B_FAISS/B_CORPUS in app.py and push.")


if __name__ == "__main__":
    main()
