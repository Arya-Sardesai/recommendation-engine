"""
embed_screen_append.py — embed ONLY newly appended movie/TV rows and concatenate.

WHY THIS EXISTS
Books have embed_books_append.py. Movies and TV never did, so the only way to
pick up expanded rows was a full re-embed: ~25 minutes of GPU for movies, and
it REPLACES vectors that are currently live. Even with a deterministic model,
fp16 kernels and batch composition can move a vector in the last decimal, which
breaks the one property an append is supposed to guarantee -- that rows
0..N-1 are byte-identical before and after.

expand_movies.py and expand_tv.py both merge with
    pd.concat([corpus, new_df]).drop_duplicates(subset=id, keep="first")
so existing rows keep their positions and new rows land at the end. That is
what makes an append valid here. stage_refresh only rewrites vote_count and
vote_average, and neither is in the embed text, so the existing vectors still
describe their rows.

WHY IT IMPORTS RATHER THAN COPIES
The embed recipe (overview + KW_STOP-filtered keywords + genres, with the
coverage guard) lives in embed_movies.build_embed_text / embed_tv.build_embed_text.
Retyping it risks the appended rows being built from a slightly different recipe
than the other 121,000 -- they would land in a subtly wrong region and retrieve
oddly, and nothing downstream would catch it, because every alignment check
compares counts.

WHICH CORPUS TO READ
Pass the parquet the app actually serves (movies_v2_meta.parquet /
tv_v2_meta.parquet). Its rows and order are identical to movies.parquet /
tv_corpus.parquet, but reading the live one means the length check is against
what will actually ship. Run the enrichment merge BEFORE this, so appended rows
carry cert_tier / maturity; otherwise the age filters treat every new title as
unrated, which is a silent no-op by design.

USAGE
    python notebooks\\pipeline\\embed_screen_append.py --kind movie --dry-run
    python notebooks\\pipeline\\embed_screen_append.py --kind movie --tag v3_kw2
    python notebooks\\pipeline\\embed_screen_append.py --kind tv --tag v3_kw_notitle
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROCESSED = Path("data/processed")

CONFIG = {
    "movie": {
        "module": "embed_movies",
        "corpus": PROCESSED / "movies_v2_meta.parquet",
        "old_emb": PROCESSED / "movie_embeddings_bgem3_v2_kw2.npy",
        "emb_fmt": "movie_embeddings_bgem3_{tag}.npy",
        "faiss_fmt": "movie_faiss_bgem3_{tag}.index",
        "title_col": "title",
    },
    "tv": {
        "module": "embed_tv",
        "corpus": PROCESSED / "tv_v2_meta.parquet",
        "old_emb": PROCESSED / "tv_embeddings_bgem3_v2_kw_notitle.npy",
        "emb_fmt": "tv_embeddings_bgem3_{tag}.npy",
        "faiss_fmt": "tv_faiss_bgem3_{tag}.index",
        "title_col": "name",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["movie", "tv"], required=True)
    ap.add_argument("--tag", default=None,
                    help="artifact suffix for the NEW files, e.g. v3_kw2. "
                         "Must differ from the live one so the upload is inert.")
    ap.add_argument("--corpus", default=None, help="override the corpus parquet")
    ap.add_argument("--old-emb", default=None, help="override the existing .npy")
    ap.add_argument("--max-new", type=int, default=20000,
                    help="refuse above this; that many rows is a rebuild, not an append")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the embed texts and stop before loading the model")
    args = ap.parse_args()

    cfg = CONFIG[args.kind]
    corpus_path = Path(args.corpus) if args.corpus else cfg["corpus"]
    old_emb_path = Path(args.old_emb) if args.old_emb else cfg["old_emb"]

    for p in (corpus_path, old_emb_path):
        if not p.exists():
            sys.exit(f"{p} not found -- run from the repo root")

    mod = __import__(cfg["module"])
    title_col = cfg["title_col"]

    df = pd.read_parquet(corpus_path)
    old = np.load(old_emb_path)
    start = old.shape[0]
    n_new = len(df) - start

    print(f"corpus           : {corpus_path.name}  ({len(df):,} rows)")
    print(f"existing vectors : {old.shape[0]:,} x {old.shape[1]}  ({old_emb_path.name})")
    print(f"to embed         : {n_new}")

    if n_new == 0:
        sys.exit("nothing to embed -- corpus is the same length as the .npy. "
                 "Did the expansion merge run with --write?")
    if n_new < 0:
        sys.exit(f"corpus is SHORTER than the .npy by {-n_new}. Rows were removed "
                 f"or the wrong corpus was passed; an append cannot fix that.")
    if n_new > args.max_new:
        sys.exit(f"{n_new:,} new rows is beyond --max-new={args.max_new:,}. "
                 f"Use the full embed script and treat it as a migration.")

    # The rows the old vectors describe must not have moved. Nothing here can
    # prove that for certain, but a changed title at a sampled position means
    # the expansion reordered and the whole append is invalid.
    print("\nspot-checking that existing rows did not move:")
    for pos in (0, start // 2, start - 1):
        print(f"  row {pos:>7,}  {str(df.iloc[pos][title_col])[:52]}")
    print("  ^ these must match what those positions held before the expansion.")

    new_rows = df.iloc[start:].reset_index(drop=True)
    texts = mod.build_embed_text(new_rows)      # SAME recipe, imported not copied
    assert len(texts) == n_new, f"recipe returned {len(texts)} texts for {n_new} rows"

    empty = sum(1 for t in texts if not t or not t.strip())
    print(f"\nempty embed texts: {empty}")
    if empty:
        sys.exit("an appended row has no embed text -- it would embed to a "
                 "near-arbitrary vector and retrieve unpredictably. Fix the "
                 "corpus filter before embedding.")

    print("\nembed text preview:")
    for i, t in enumerate(texts[:3]):
        print(f"  [{start + i}] {str(new_rows.iloc[i][title_col])[:44]!r}")
        print(f"        {t[:160].replace(chr(10), ' ')}...")
    lens = [len(t) for t in texts]
    print(f"  mean length {np.mean(lens):.0f} chars "
          f"(min {min(lens)}, max {max(lens)})")

    if args.dry_run:
        print("\nDRY RUN -- model not loaded, nothing written.")
        return

    if not args.tag:
        sys.exit("--tag is required for a real run. Give the new artifacts a name "
                 "the app does not yet reference, so the upload is inert.")
    out_emb = PROCESSED / cfg["emb_fmt"].format(tag=args.tag)
    out_faiss = PROCESSED / cfg["faiss_fmt"].format(tag=args.tag)
    if out_emb.resolve() == old_emb_path.resolve():
        sys.exit("--tag matches the LIVE artifact name. Pick a new one; "
                 "overwriting the live .npy is the thing this script exists to avoid.")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nloading {mod.MODEL_NAME} on {device} ...")
    model = SentenceTransformer(mod.MODEL_NAME, device=device)
    model.max_seq_length = mod.MAX_SEQ_LEN
    if getattr(mod, "USE_FP16", False) and device == "cuda":
        model.half()

    vecs = model.encode(
        texts, batch_size=mod.ENCODE_BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    norms = np.linalg.norm(vecs, axis=1)
    print(f"norms: min {norms.min():.4f} max {norms.max():.4f} "
          f"(a drift from 1.0 means fp16 trouble -- set USE_FP16=False and re-run)")
    assert vecs.shape[1] == old.shape[1], \
        f"dim mismatch: new {vecs.shape[1]} vs existing {old.shape[1]}"

    combined = np.vstack([old, vecs])
    assert combined.shape[0] == len(df), "row count mismatch after concat"
    # the whole point of appending: nothing before `start` may move
    assert np.array_equal(combined[:start], old), "EXISTING VECTORS CHANGED"
    print(f"existing {start:,} vectors byte-identical after concat")

    np.save(out_emb, combined)
    print(f"wrote {out_emb}  {combined.shape}")

    import faiss
    index = faiss.IndexFlatIP(combined.shape[1])
    index.add(combined)
    faiss.write_index(index, str(out_faiss))
    print(f"wrote {out_faiss}  ntotal={index.ntotal:,}")

    # self-match probes: first appended row, last row, and one old row to prove
    # the concatenation did not shift the existing block
    probes = [0, start, len(df) - 1]
    D, I = index.search(combined[probes], 1)
    ok = True
    print("\nself-match probes (each row must retrieve itself at rank 1):")
    for pos, hit, score in zip(probes, I[:, 0], D[:, 0]):
        flag = "OK " if hit == pos else "FAIL"
        if hit != pos:
            ok = False
        print(f"  {flag} row {pos:>7,} -> {hit:>7,}  sim={score:.4f}  "
              f"{str(df.iloc[pos][title_col])[:40]}")
    if not ok:
        sys.exit("a probed row did not self-match -- index is misaligned, "
                 "do NOT upload these artifacts.")

    print(f"\nALIGNMENT: {len(df):,} rows == {combined.shape[0]:,} vectors "
          f"== {index.ntotal:,} index entries")
    print("\nNEXT, IN THIS ORDER:")
    print("  1. rebuild the tag layer over the expanded corpus")
    print("  2. upload embeddings + index FIRST (inert under the new names)")
    print("  3. upload the parquet + tags SECOND")
    print("  4. probe a known-changed value through the live load path")
    print(f"  5. flip the {args.kind.upper()[0]}_* constants in app.py LAST")


if __name__ == "__main__":
    main()
