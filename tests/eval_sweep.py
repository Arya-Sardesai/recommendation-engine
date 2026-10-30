"""
eval_sweep.py -- sweep ANY one recommend_books setting, the elbow way.

One script instead of a new eval file per knob. Same protocol as the others:
goodbooks-10k, 300 users x 3 folds, every value paired per user against the
live value, 95% bootstrap CI and P(worse). Forms (shelf gate) and awards are
passed exactly as the Space passes them.

USAGE (repo root):
    python tests\\eval_sweep.py --gb-dir data\\raw\\books\\goodbooks-10k --param shelf_penalty --live 0.10 --grid 0.10,0.15,0.20,0.30
    python tests\\eval_sweep.py --gb-dir ... --param tag_weight --live 0.25 --grid 0.25,0.35,0.50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_goodbooks as eg  # noqa: E402
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-dir", required=True)
    ap.add_argument("--param", required=True, help="a recommend_books keyword")
    ap.add_argument("--live", type=float, required=True, help="the value shipping now")
    ap.add_argument("--grid", required=True, help="comma-separated values; include --live")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",")]
    if args.live not in grid:
        grid = [args.live] + grid

    ns = load_ranker(eg.SPACE_APP, require=("recommend_books",))
    if args.param not in ns["recommend_books"].__code__.co_varnames:
        sys.exit(f"recommend_books has no setting called {args.param}")
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
    for v in grid:
        params = dict(eg.SCORING_PROTOCOL, **kw)
        params[args.param] = v
        s, _ = eg.eval_users(users, df, emb, index, tags_df, rec, params=params,
                             label=f"{args.param} {v:g}: ", folds=args.folds)
        per[v] = s.pop("_per_user")
        rows.append((v, s))

    print("\n" + "=" * 90)
    print(f"  {args.param}: {args.users} users, {args.folds}-fold; paired vs live {args.live:g}")
    print(f"{'value':<18}{'recall@20':>14}{'d vs live':>11}{'95% CI':>24}{'P(worse)':>10}")
    print("-" * 90)
    tied, better = [], []
    for v, s in rows:
        line = f"{v:<6g}{'(live)' if v == args.live else '':<12}{s['r20'][0]:>9.4f}±{s['r20'][1]:<4.3f}"
        if v != args.live:
            d, lo, hi, pw = eg.paired_bootstrap(per[args.live], per[v])
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            line += f"{d:>+11.4f}{ci:>24}{pw:>9.0%}{' <<' if (lo > 0 or hi < 0) else ''}"
            if lo > 0:
                better.append(v)
            elif hi >= 0:
                tied.append(v)
        print(line)
    print("=" * 90)
    print(f"significantly better: {', '.join(f'{v:g}' for v in better) or 'none'}")
    print(f"tied with live:       {', '.join(f'{v:g}' for v in tied) or 'none'}")
    print("anything else is significantly worse. Pick among better/tied by the probe.")


if __name__ == "__main__":
    main()
