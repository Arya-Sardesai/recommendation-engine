"""
probe_cross_modal.py — do movie and book embeddings live in the same usable
space, or only nominally the same space?

THE CLAIM BEING TESTED
All three corpora were embedded with BAAI/bge-m3 at 1024d, so the vectors are
dimensionally compatible and cosine is defined between them. That is not the
same as being COMPARABLE. The embed text recipes differ:

    movies : overview + keywords
    books  : description + capped genres + capped moods

Different text shapes occupy different regions of a sentence-embedding space.
If cross-modal cosines sit in a systematically lower band than within-modal
ones, then every threshold in the ranker calibrated on within-modal scores --
cross_val >= 0.30, the pool-centred tag adjustment, the quality floor -- is
wrong by an unknown amount when applied across modalities.

WHAT IT MEASURES
  1. SCORE DISTRIBUTION: movie->book cosines vs book->book cosines. A large
     shift means the two corpora sit apart and raw thresholds will not carry.
  2. ADAPTATIONS: films with a known novel behind them. This is the sharpest
     test available -- if Fight Club the film does not retrieve Fight Club the
     novel, the space is not aligned in any useful sense.
  3. TASTE: a real movie profile, queried against the book index, printed for
     reading. The number can look fine while the titles are nonsense.

READ-ONLY. No writes.

USAGE
    python notebooks\\pipeline\\probe_cross_modal.py
    python notebooks\\pipeline\\probe_cross_modal.py --ratings tests\\movies_ratings.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"

# Films whose source novel should be in a 74K book corpus. If the space is
# aligned, each film's vector should retrieve its novel highly.
ADAPTATIONS = [
    ("Fight Club", "Fight Club"),
    ("The Godfather", "The Godfather"),
    ("Blade Runner", "Do Androids Dream of Electric Sheep"),
    ("The Maltese Falcon", "The Maltese Falcon"),
    ("No Country for Old Men", "No Country for Old Men"),
    ("The Shining", "The Shining"),
    ("Jurassic Park", "Jurassic Park"),
    ("The Martian", "The Martian"),
    ("Gone Girl", "Gone Girl"),
    ("Dune", "Dune"),
]


def _norm_title(s):
    return "".join(c for c in str(s).lower() if c.isalnum() or c == " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=None,
                    help="localStorage dump; adds the real-profile section")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    ns = load_ranker(SPACE_APP, require=("recommend_books",))
    m_art = resolve_artifacts(SPACE_APP, PROCESSED, kind="movie", verbose=False)
    b_art = resolve_artifacts(SPACE_APP, PROCESSED, kind="book", verbose=False)

    import faiss
    mdf = pd.read_parquet(m_art["CORPUS"]).reset_index(drop=True)
    memb = np.load(m_art["EMB"]).astype("float32")
    bdf = pd.read_parquet(b_art["CORPUS"]).reset_index(drop=True)
    bemb = np.load(b_art["EMB"]).astype("float32")
    bix = faiss.read_index(str(b_art["FAISS"]))

    print(f"movies {len(mdf):,} x {memb.shape[1]}   books {len(bdf):,} x {bemb.shape[1]}")
    if memb.shape[1] != bemb.shape[1]:
        sys.exit("dimension mismatch — not the same model, nothing else matters")

    # ---- 1. score distributions -------------------------------------------
    rng = np.random.default_rng(0)
    b_sample = bemb[rng.choice(len(bemb), 300, replace=False)]
    m_sample = memb[rng.choice(len(memb), 300, replace=False)]

    D_bb, _ = bix.search(b_sample, 6)          # book -> book, skip self
    within = D_bb[:, 1:].mean(axis=1)
    D_mb, _ = bix.search(m_sample, 5)          # movie -> book
    cross = D_mb.mean(axis=1)

    print("\n" + "=" * 68)
    print("1. SCORE DISTRIBUTION")
    print("=" * 68)
    print(f"  book -> book  (top-5 mean)  {within.mean():.3f}  "
          f"[p10 {np.percentile(within,10):.3f}, p90 {np.percentile(within,90):.3f}]")
    print(f"  movie -> book (top-5 mean)  {cross.mean():.3f}  "
          f"[p10 {np.percentile(cross,10):.3f}, p90 {np.percentile(cross,90):.3f}]")
    shift = within.mean() - cross.mean()
    print(f"  shift: {shift:+.3f}")
    if abs(shift) > 0.10:
        print("  >> LARGE SHIFT. Cross-modal scores are not on the same scale as")
        print("     within-modal ones, so any threshold calibrated on book->book")
        print("     similarity is wrong here. This is the tag-blend bug's shape:")
        print("     two ranges compared as if they matched. Pool-centre before use.")
    else:
        print("  >> comparable ranges; thresholds should carry over roughly")

    # ---- 2. adaptations ----------------------------------------------------
    print("\n" + "=" * 68)
    print("2. ADAPTATIONS  (the sharp test)")
    print("=" * 68)
    mtitles = {_norm_title(t): i for i, t in enumerate(mdf["title"])}
    btitles = [_norm_title(t) for t in bdf["title"]]

    found = hits = 0
    for film, novel in ADAPTATIONS:
        mi = mtitles.get(_norm_title(film))
        if mi is None:
            print(f"  ??  {film:28s} film not in corpus")
            continue
        found += 1
        D, I = bix.search(memb[mi:mi + 1], args.top)
        want = _norm_title(novel)
        rank = None
        for r, bi in enumerate(I[0], 1):
            if want in btitles[bi] or btitles[bi] in want:
                rank = r
                break
        top1 = bdf.iloc[I[0][0]]["title"]
        if rank:
            hits += 1
            print(f"  OK  {film:28s} -> novel at rank {rank} (sim {D[0][rank-1]:.3f})")
        else:
            print(f"  --  {film:28s} novel NOT in top {args.top}")
        print(f"      top book: {str(top1)[:52]} (sim {D[0][0]:.3f})")

    if found:
        print(f"\n  {hits}/{found} adaptations retrieved their novel in the top "
              f"{args.top}")
        if hits == 0:
            print("  >> the space is NOT aligned for this purpose. Stop here.")
        elif hits < found / 2:
            print("  >> weak. Usable as a soft signal at best, not as a ranking.")
        else:
            print("  >> aligned well enough to build on.")

    # ---- 3. a real taste profile -------------------------------------------
    if args.ratings:
        raw = json.loads(Path(args.ratings).read_text(encoding="utf-8"))
        v = raw.get("re_movies") if isinstance(raw, dict) else raw
        if isinstance(v, str):
            v = json.loads(v)
        liked = [r for r in (v or []) if r.get("rating", 0) >= 4.0
                 and r.get("corpus_index") is not None
                 and 0 <= int(r["corpus_index"]) < len(memb)]
        print("\n" + "=" * 68)
        print(f"3. REAL PROFILE  ({len(liked)} films rated >= 4)")
        print("=" * 68)
        if not liked:
            print("  no usable movie ratings in that dump")
            return
        vecs = np.array([memb[int(r["corpus_index"])] for r in liked], dtype="float32")
        prof = vecs.mean(axis=0)
        prof /= np.linalg.norm(prof)
        D, I = bix.search(prof.reshape(1, -1).astype("float32"), args.top)
        print("  profile-centroid -> books:")
        for r, (bi, s) in enumerate(zip(I[0], D[0]), 1):
            b = bdf.iloc[bi]
            print(f"    {r:>2}. {str(b['title'])[:44]:46s} {str(b['primary_author'])[:20]:22s} {s:.3f}")
        print("\n  READ THESE. A centroid of 100+ films is a blurry query and may")
        print("  return generic literary fiction that matches nothing in")
        print("  particular. Per-anchor retrieval (one film -> its books) is the")
        print("  alternative if this looks vague.")

        print("\n  per-anchor, 3 strongest films:")
        for r in liked[:3]:
            mi = int(r["corpus_index"])
            D1, I1 = bix.search(memb[mi:mi + 1], 3)
            print(f"    {str(mdf.iloc[mi]['title'])[:32]:34s} ->")
            for bi, s in zip(I1[0], D1[0]):
                print(f"        {str(bdf.iloc[bi]['title'])[:44]:46s} {s:.3f}")


if __name__ == "__main__":
    main()
