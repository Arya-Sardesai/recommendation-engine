"""
probe_floor_ab.py — what does lowering the books serving floor actually surface?

THE ARGUMENT THIS SETTLES
"An obscure book won't get recommended anyway, so the floor costs nothing."
Half true. Retrieval is pure cosine -- FAISS returns nearest neighbours with no
notion of popularity -- and the Bayesian prior shrinks a zero-rating book toward
the corpus mean rather than penalising it. So nothing stops an obscure book from
surfacing once the floor is gone. Whether that is good depends entirely on what
those books ARE, and 24,255 of the corpus's 74,074 rows sit under the current
floor of 10.

So: run the same profile at several floors and read what appears. The counts
tell you how much changes; the titles tell you whether it is an improvement.

READ-ONLY. Changes nothing.

USAGE
    python notebooks\\pipeline\\probe_floor_ab.py --ratings tests\\movies_ratings.json
    python notebooks\\pipeline\\probe_floor_ab.py --ratings tests\\movies_ratings.json --floors 0 5 10 25
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


def load_book_ratings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        v = raw.get("re_books")
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return v
    return raw if isinstance(raw, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--floors", type=int, nargs="+", default=[0, 2, 5, 10])
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    ns = load_ranker(SPACE_APP, require=("recommend_books", "_migrate_book_ratings"))
    art = resolve_artifacts(SPACE_APP, PROCESSED, kind="book", verbose=False)

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, "ALIGNMENT"

    tags_df = None
    if "TAGS" in art and Path(art["TAGS"]).exists():
        t = pd.read_parquet(art["TAGS"])
        t["book_id"] = t["book_id"].astype(str)
        tags_df = t.pivot_table(index="book_id", columns="tag", values="score",
                                fill_value=0.0)

    ratings = load_book_ratings(Path(args.ratings))
    ratings, _m, _d, _g = ns["_migrate_book_ratings"](ratings, df)
    print(f"corpus {len(df):,} | profile {len(ratings)} rated\n")

    rc_all = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    print("how much of the corpus each floor admits:")
    for f in sorted(args.floors):
        n = int((rc_all >= f).sum())
        print(f"  floor {f:>3}: {n:>7,} servable  ({n/len(df):.0%} of corpus)")

    results = {}
    for f in sorted(args.floors):
        recs, _ = ns["recommend_books"](ratings, df, emb, index,
                                        tags_df=tags_df, n=args.n,
                                        min_ratings=f)
        results[f] = recs

    base = max(args.floors)
    base_ids = set(results[base].get("book_id", pd.Series(dtype=str)).astype(str))

    for f in sorted(args.floors):
        recs = results[f]
        rc = pd.to_numeric(recs.get("ratings_count", pd.Series(dtype=float)),
                           errors="coerce").fillna(0)
        ids = set(recs.get("book_id", pd.Series(dtype=str)).astype(str))
        new = ids - base_ids
        print("\n" + "=" * 72)
        print(f"FLOOR {f}   ({len(new)} of {len(recs)} not in the floor-{base} list)")
        print("=" * 72)
        if len(rc):
            print(f"  ratings_count of the recs: median {int(rc.median())}, "
                  f"min {int(rc.min())}, max {int(rc.max())}")
        for _, r in recs.iterrows():
            bid = str(r.get("book_id", ""))
            mark = "NEW" if bid in new else "   "
            n_rc = int(r["ratings_count"]) if pd.notna(r.get("ratings_count")) else 0
            print(f"  {mark} {str(r.get('title',''))[:40]:42s} "
                  f"{str(r.get('author',''))[:20]:22s} rc={n_rc}")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  Look ONLY at the NEW rows at the lower floors. If they are books")
    print("  you would want, the floor is throwing away good material and it")
    print("  should come down. If they are duplicate editions, untranslated")
    print("  records, or titles nobody has heard of for a reason, the floor is")
    print("  doing its job and the targeted authors need an exemption instead")
    print("  of a blanket change.")
    print("\n  Whatever this shows, re-run eval_goodbooks at the new floor before")
    print("  shipping it -- 0.016 is the number to beat, and a floor change moves")
    print("  the candidate pool for every user, not just this profile.")


if __name__ == "__main__":
    main()
