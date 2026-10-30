"""
eval_untagged_anchor.py -- does turning the tag layer off under UNTAGGED
anchors cost recall?

untagged_anchor_tags = tag weight under an anchor with no Hardcover tags, as a
multiple of tag_weight. 1.0 = what ships now, 0.0 = no tag term there.
Tagged anchors are identical in every row, so any difference comes only from
users whose liked books include untagged ones -- the script prints how many
that is, because if goodbooks' popular books are nearly all tagged, the eval
barely sees the change and a tie is the expected result.

Ship rule (same as every books change): ship 0.0 if it is NOT significantly
worse (CI not entirely below 0). goodbooks can't see "a romance under a chef
memoir"; the probe on your own profile can.

USAGE (repo root, after add_untagged_anchor.py --write):
    python tests\\eval_untagged_anchor.py --gb-dir data\\raw\\books\\goodbooks-10k
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_goodbooks as eg  # noqa: E402
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

GRID = [1.0, 0.5, 0.25, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-dir", required=True)
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    ns = load_ranker(eg.SPACE_APP, require=("recommend_books",))
    if "untagged_anchor_tags" not in ns["recommend_books"].__code__.co_varnames:
        sys.exit("recommend_books has no untagged_anchor_tags: run add_untagged_anchor.py --write")
    art = resolve_artifacts(eg.SPACE_APP, eg.PROCESSED, kind="book")

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, "ALIGNMENT broken"
    tags_df = eg.load_tag_matrix(art["TAGS"])
    kw = {}
    if "AWARDS" in art and art["AWARDS"].exists() and "book_award_summaries" in ns:
        kw["awards"] = ns["book_award_summaries"](pd.read_parquet(art["AWARDS"]))
    if "FORMS" in art and art["FORMS"].exists():
        f = pd.read_parquet(art["FORMS"])
        kw["forms"] = dict(zip(f["book_id"].astype(str), f["form"].astype(str)))

    eg.LINKS = eg.build_links(Path(args.gb_dir) / "books.csv", df)
    users = eg.load_users(Path(args.gb_dir) / "ratings.csv", eg.LINKS, args.users)

    tagged = set(tags_df.index)
    linked = set(eg.LINKS.values())
    unt = sum(1 for ci in linked if str(df.iloc[ci]["book_id"]) not in tagged)
    has_unt = sum(1 for g in users.values()
                  if any(str(df.iloc[eg.LINKS[int(b)]]["book_id"]) not in tagged
                         for b, r in zip(g["book_id"], g["rating"]) if r >= 4))
    print(f"  untagged among goodbooks-linked books: {unt:,} of {len(linked):,} "
          f"({unt / max(len(linked), 1):.1%})")
    print(f"  sampled users with at least one untagged liked book: {has_unt} of "
          f"{len(users)} -- only these can move\n")

    rec = ns["recommend_books"]
    rows, per = [], {}
    for v in GRID:
        params = dict(eg.SCORING_PROTOCOL, untagged_anchor_tags=v, **kw)
        s, _ = eg.eval_users(users, df, emb, index, tags_df, rec, params=params,
                             label=f"untagged x{v:.2f}: ", folds=args.folds)
        per[v] = s.pop("_per_user")
        rows.append((v, s))

    print("\n" + "=" * 86)
    print(f"  {args.users} users, {args.folds}-fold; tag weight under untagged anchors")
    print(f"{'setting':<22}{'recall@20':>14}{'d vs 1.0':>11}{'95% CI':>24}{'P(worse)':>10}")
    print("-" * 86)
    for v, s in rows:
        lab = f"x{v:.2f}" + ("  (live)" if v == 1.0 else "  (off)" if v == 0.0 else "")
        line = f"{lab:<22}{s['r20'][0]:>9.4f}±{s['r20'][1]:<4.3f}"
        if v != 1.0:
            d, lo, hi, pw = eg.paired_bootstrap(per[1.0], per[v])
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            flag = " <<" if (lo > 0 or hi < 0) else ""
            line += f"{d:>+11.4f}{ci:>24}{pw:>9.0%}{flag}"
        print(line)
    print("=" * 86)
    d, lo, hi, _ = eg.paired_bootstrap(per[1.0], per[0.0])
    if hi < 0:
        print("x0.00 is significantly WORSE: keep 1.0, or try the smallest row that ties.")
    else:
        print("x0.00 is not significantly worse: set BOOK_UNTAGGED_ANCHOR_TAGS = 0.0, "
              "then re-run the Bourdain probe.")


if __name__ == "__main__":
    main()
