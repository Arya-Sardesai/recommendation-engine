"""
embed_books_append.py — embed ONLY the newly appended rows and concatenate.

WHY NOT JUST RERUN embed_books.py
It has no start offset (--limit truncates, it does not skip), so a rerun
re-embeds all 74,054 existing rows. That costs an hour and, worse, replaces
vectors that are already live. Even if the model is deterministic, fp16
kernels and batch composition can shift a vector in the last decimal, which
silently breaks the property the append-only design exists to guarantee:
rows 0..74,053 are byte-identical before and after.

WHY IT IMPORTS RATHER THAN COPIES
The embed text for v2_hc_notitle is description + capped genres + capped moods,
assembled by build_embed_text with _clean, _listcol, GENRE_MAX, MOOD_MAX and a
stopword list. Retyping any of that risks 20 vectors built from a slightly
different recipe than the other 74,054 -- they would land in a subtly wrong
region and retrieve oddly, and NOTHING downstream would catch it, because the
alignment checks only compare counts.

So this imports the real functions and calls them on the sliced frame.

USAGE
    python notebooks\\pipeline\\embed_books_append.py
    python notebooks\\pipeline\\embed_books_append.py --dry-run
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import embed_books as EB  # noqa: E402  — the source of truth for the recipe

PROCESSED = Path("data/processed")
# Nov 2026: hc3 (live) -> hc4 (merge_new_books.py: Google-blurb + award books).
# Next time, move each name up one: OLD_EMB = the live vectors, the rest new.
NEW_CORPUS = PROCESSED / "books_v2_hc4.parquet"
OLD_EMB = PROCESSED / "book_embeddings_bgem3_v2_hc3_notitle.npy"   # the LIVE vectors
OUT_EMB = PROCESSED / "book_embeddings_bgem3_v2_hc4_notitle.npy"
OUT_FAISS = PROCESSED / "book_faiss_bgem3_v2_hc4_notitle.index"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show the embed texts and stop before loading the model")
    # 1,000 was the sweep's size. The November merge is ~10,000 rows: still an
    # append (existing rows untouched), just a bigger one -- ~10 min on the 4060.
    ap.add_argument("--max-new", type=int, default=20000,
                    help="refuse above this; more is a rebuild, not an append")
    args = ap.parse_args()

    df = pd.read_parquet(NEW_CORPUS)
    old = np.load(OLD_EMB)
    start = old.shape[0]
    n_new = len(df) - start
    print(f"existing vectors : {start:,}")
    print(f"new corpus rows  : {len(df):,}")
    print(f"to embed         : {n_new}")
    if n_new <= 0:
        raise SystemExit("nothing to embed — corpus is not longer than the .npy")
    if n_new > args.max_new:
        raise SystemExit(f"{n_new} new rows is beyond --max-new={args.max_new}; use "
                         f"embed_books.py for a full rebuild")

    new_rows = df.iloc[start:].reset_index(drop=True)
    texts = EB.build_embed_text(new_rows)      # SAME recipe, imported not copied
    assert len(texts) == n_new

    empty = sum(1 for t in texts if not t or not t.strip())
    print(f"empty embed texts: {empty}")
    if empty:
        raise SystemExit("an appended row has no embed text — it would embed to "
                         "a near-arbitrary vector and retrieve unpredictably")

    print("\nembed text preview:")
    for i, t in enumerate(texts[:3]):
        print(f"  [{start + i}] {new_rows.iloc[i]['title'][:40]!r}")
        print(f"        {t[:160].replace(chr(10), ' ')}...")
    print(f"  mean length {np.mean([len(t) for t in texts]):.0f} chars "
          f"(min {min(len(t) for t in texts)}, max {max(len(t) for t in texts)})")

    if args.dry_run:
        print("\nDRY RUN — model not loaded, nothing written.")
        return

    from sentence_transformers import SentenceTransformer
    print(f"\nloading {EB.MODEL_NAME} ...")
    model = SentenceTransformer(EB.MODEL_NAME, device="cuda")
    model.max_seq_length = EB.MAX_SEQ_LEN

    # same precision as the live vectors: embed_books.py encoded them in fp16
    if EB.USE_FP16:
        model.half()
    vecs = model.encode(
        texts, batch_size=EB.ENCODE_BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    norms = np.linalg.norm(vecs, axis=1)
    print(f"norms: min {norms.min():.4f} max {norms.max():.4f} "
          f"(should be ~1.0 — a drift here means fp16 trouble)")
    assert vecs.shape[1] == old.shape[1], \
        f"dim mismatch: new {vecs.shape[1]} vs existing {old.shape[1]}"

    combined = np.vstack([old, vecs])
    assert combined.shape[0] == len(df), "row count mismatch after concat"
    # the whole point of appending: nothing before `start` may move
    assert np.array_equal(combined[:start], old), "EXISTING VECTORS CHANGED"
    print(f"existing {start:,} vectors byte-identical after concat")

    np.save(OUT_EMB, combined)
    print(f"wrote {OUT_EMB}  {combined.shape}")

    import faiss
    index = faiss.IndexFlatIP(combined.shape[1])
    index.add(combined)
    faiss.write_index(index, str(OUT_FAISS))
    print(f"wrote {OUT_FAISS}  ntotal={index.ntotal:,}")

    # self-match probe on appended rows: each must retrieve itself first
    probe_positions = [start, len(df) - 1]
    D, I = index.search(combined[probe_positions], 1)
    for pos, hit, score in zip(probe_positions, I[:, 0], D[:, 0]):
        flag = "OK " if hit == pos else "BAD"
        print(f"  {flag} row {pos:>6,} -> {hit:>6,}  sim={score:.4f}  "
              f"{df.iloc[pos]['title'][:40]}")
    assert all(I[i, 0] == p for i, p in enumerate(probe_positions)), \
        "a probed row did not self-match — index is misaligned"

    print("\nNEXT:")
    print("  1. run build_book_tags_hc.py (already points at hc4) and")
    print("     build_content_warnings.py --stage build")
    print("  2. upload embeddings + index FIRST, parquet + tags SECOND")
    print("  3. flip B_EMB / B_FAISS / B_CORPUS / B_TAGS LAST")
    print("  4. re-run eval_goodbooks (baseline: recall@20 0.016, scoring protocol)")


if __name__ == "__main__":
    main()
