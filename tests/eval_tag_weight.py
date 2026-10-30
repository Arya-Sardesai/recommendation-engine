"""
eval_tag_weight.py -- how much should the reader-mood tags count at all?

Under Kitchen Confidential the raw neighbours were chef memoirs, and with the
tag layer OFF the picks were Humble Pie, Dirt, Cork Dork. With it on, the
profile-wide moods (funny, emotional, adventurous -- on nearly every book)
gave a romance +0.10 and it won. Same pattern as The Ax -> romances.

This sweeps tag_weight (live = 0.25) the elbow way, like the award sweep:
recall@20 on goodbooks, 300 users x 3 folds, paired bootstrap vs the live
value. Ship rule: the SMALLEST weight that is not significantly worse than
live -- goodbooks can't see "romance under a chef memoir", your probe can,
so a tie goes to the weight that lets the blurb match decide.

USAGE (repo root):
    python tests\\eval_tag_weight.py --gb-dir data\\raw\\books\\goodbooks-10k
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_goodbooks as eg  # noqa: E402
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

LIVE = 0.25
GRID = [0.25, 0.15, 0.10, 0.05, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-dir", required=True)
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    ns = load_ranker(eg.SPACE_APP, require=("recommend_books",))
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

    rec = ns["recommend_books"]
    rows, per = [], {}
    for w in GRID:
        params = dict(eg.SCORING_PROTOCOL, tag_weight=w, **kw)
        s, _ = eg.eval_users(users, df, emb, index, tags_df, rec, params=params,
                             label=f"tag_weight {w:.2f}: ", folds=args.folds)
        per[w] = s.pop("_per_user")
        rows.append((w, s))

    print("\n" + "=" * 90)
    print(f"  {args.users} users, {args.folds}-fold; paired vs live tag_weight {LIVE}")
    print(f"{'tag_weight':<18}{'recall@20':>14}{'d vs live':>11}{'95% CI':>24}{'P(worse)':>10}")
    print("-" * 90)
    ok = []
    for w, s in rows:
        line = f"{w:<6.2f}{'(live)' if w == LIVE else '':<12}{s['r20'][0]:>9.4f}±{s['r20'][1]:<4.3f}"
        if w != LIVE:
            d, lo, hi, pw = eg.paired_bootstrap(per[LIVE], per[w])
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            line += f"{d:>+11.4f}{ci:>24}{pw:>9.0%}{' <<' if (lo > 0 or hi < 0) else ''}"
            if hi >= 0:
                ok.append(w)
        print(line)
    print("=" * 90)
    if ok:
        print(f"smallest weight not significantly worse than live: {min(ok):.2f}")
        print("set it as the tag_weight default in recommend_books, then re-run the")
        print("Bourdain probe and probe_award_lists.py and judge the lists by eye.")
    else:
        print("every lower weight is significantly worse: keep 0.25.")


if __name__ == "__main__":
    main()
