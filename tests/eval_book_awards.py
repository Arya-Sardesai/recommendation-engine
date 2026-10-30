"""
eval_book_awards.py -- sweep the BOOK award bonus and find the elbow.

Like choosing k for k-means: score a range of values, plot them, and take the
point where the curve stops climbing -- not simply the highest score. Past the
elbow a bigger bonus buys no more recall, only more award books crowding
closer matches out of the list.

For each weight w (win = w, shortlisted = w * --nom-ratio):
  recall@20 on goodbooks-10k, 300 users x 3 folds (same users and splits for
  every weight, so the per-user scores are paired), bootstrap CI vs w = 0,
  and the AWARD SHARE: what fraction of each top-20 is a major-award book.

THE PICK
  1. best = the weight with the highest mean recall
  2. pick = the SMALLEST weight whose paired CI against `best` still includes
     0 -- statistically as good as the best, with the least distortion
  3. reject it if the award share passes --max-share (default: twice the w=0
     share); then take the next smaller weight
  If no weight beats w = 0 (CI excludes 0), the honest answer is "ship it for
  the badges and leave the bonus at a small nudge or off".

READ WITH CARE
  goodbooks readers rated popular books, and famous prize winners ARE popular
  books, so part of any gain here is a popularity effect. The award share
  column is the guard against that: if recall rises only as award books take
  over the list, that's the popularity effect, not better taste matching.
  Then check the pick by eye on your own profile (probe_why_books.py).

USAGE (repo root, after build_book_awards.py and add_book_awards.py --write):
    python tests\\eval_book_awards.py --gb-dir data\\raw\\books\\goodbooks-10k
    python tests\\eval_book_awards.py --gb-dir ... --grid 0,0.04,0.08 --users 100 --folds 1   # quick look
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_goodbooks as eg  # noqa: E402
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

OUT_CSV = eg.REPO_ROOT / "tests" / "award_sweep_books.csv"
OUT_PNG = eg.REPO_ROOT / "tests" / "award_sweep_books.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-dir", required=True)
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--grid", default="0,0.02,0.04,0.06,0.08,0.12,0.16,0.24",
                    help="win weights to try; 0 must be first (the baseline)")
    ap.add_argument("--nom-ratio", type=float, default=0.5,
                    help="shortlisted bonus = win * this (films use 0.5)")
    ap.add_argument("--max-share", type=float, default=None,
                    help="reject weights whose top-20 award share exceeds this "
                         "(default: 2x the share at w=0)")
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",")]
    if grid[0] != 0.0:
        sys.exit("--grid must start with 0 (the baseline)")

    ns = load_ranker(eg.SPACE_APP, require=("recommend_books", "book_award_summaries"))
    art = resolve_artifacts(eg.SPACE_APP, eg.PROCESSED, kind="book")
    if "AWARDS" not in art or not art["AWARDS"].exists():
        sys.exit("book_awards.parquet missing: run notebooks\\pipeline\\build_book_awards.py "
                 "and add_book_awards.py --write first")

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, "ALIGNMENT broken"
    tags_df = None
    if "TAGS" in art and art["TAGS"].exists():
        tags_df = eg.load_tag_matrix(art["TAGS"])

    awards = ns["book_award_summaries"](pd.read_parquet(art["AWARDS"]))
    is_major = {k for k, v in awards.items() if v["major"]}
    ids = pd.to_numeric(df["book_id"], errors="coerce")
    cat_share = float(ids.isin(is_major).mean())
    print(f"  awards: {len(awards):,} books badged, {len(is_major):,} with a MAJOR "
          f"award ({cat_share:.1%} of the catalogue)")

    eg.LINKS = eg.build_links(Path(args.gb_dir) / "books.csv", df)
    users = eg.load_users(Path(args.gb_dir) / "ratings.csv", eg.LINKS, args.users)
    linked_major = sum(1 for ci in set(eg.LINKS.values())
                       if pd.notna(ids.iloc[ci]) and int(ids.iloc[ci]) in is_major)
    print(f"  {linked_major:,} major-award books are among the goodbooks-linked books "
          f"(the only ones this eval can see)\n")

    rec = ns["recommend_books"]
    shares = []

    def rec_counting(*a, **k):
        recs, t = rec(*a, **k)
        top = pd.to_numeric(recs["book_id"].head(20), errors="coerce") if len(recs) else []
        if len(top):
            shares.append(float(pd.Series(top).isin(is_major).mean()))
        return recs, t

    rows, per = [], {}
    for w in grid:
        params = dict(eg.SCORING_PROTOCOL, awards=awards, award_win=w,
                      award_nom=w * args.nom_ratio)
        shares.clear()
        s, _ = eg.eval_users(users, df, emb, index, tags_df, rec_counting,
                             params=params, label=f"w={w:.2f}: ", folds=args.folds)
        per[w] = s.pop("_per_user")
        rows.append({"w": w, "r20": s["r20"][0], "se": s["r20"][1],
                     "share": float(np.mean(shares)) if shares else 0.0})

    base = rows[0]
    for r in rows[1:]:
        d, lo, hi, pw = eg.paired_bootstrap(per[0.0], per[r["w"]])
        r.update(d=d, lo=lo, hi=hi, pw=pw)
    best = max(rows, key=lambda r: r["r20"])
    cap = args.max_share if args.max_share is not None else 2 * base["share"]

    pick, why = None, ""
    for r in rows:
        if r["w"] == 0.0:
            continue
        _, lo_b, hi_b, _ = (eg.paired_bootstrap(per[best["w"]], per[r["w"]])
                            if r is not best else (0, 0, 0, 0))
        if hi_b >= 0 and r["share"] <= cap:     # not significantly worse than best
            pick = r
            break
    if pick is not None and best.get("lo", -1) <= 0:
        why = ("no weight is significantly better than 0 -- a tie. Keep the "
               "badges; the bonus is optional (a small nudge is harmless).")
    elif pick is not None:
        why = "smallest weight statistically tied with the best, award share under the cap."

    print("\n" + "=" * 96)
    print(f"  {args.users} users, {args.folds}-fold. win = w, shortlisted = w x {args.nom_ratio}")
    print(f"{'w (win/nom)':<16}{'recall@20':>15}{'d vs 0':>10}{'95% CI':>22}"
          f"{'P(worse)':>10}{'award share':>14}")
    print("-" * 96)
    top = max(r["r20"] for r in rows) or 1
    for r in rows:
        lab = f"{r['w']:.2f} / {r['w'] * args.nom_ratio:.2f}"
        line = f"{lab:<16}{r['r20']:>9.4f}±{r['se']:<5.3f}"
        if r["w"]:
            flag = " <<" if (r["lo"] > 0 or r["hi"] < 0) else "   "
            ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
            line += f"{r['d']:>+10.4f}{ci:>22}{r['pw']:>9.0%}{flag[1:]}"
        else:
            line += f"{'(baseline)':>32}{'':>10}"
        line += f"{r['share']:>11.1%}"
        bar = "#" * int(round(30 * r["r20"] / top))
        print(line + "   " + bar)
    print("=" * 96)
    print(f"award share at w=0 is {base['share']:.1%} (catalogue: {cat_share:.1%}); cap {cap:.1%}")
    if pick:
        print(f"\nELBOW PICK: win {pick['w']:.2f} / shortlisted {pick['w'] * args.nom_ratio:.2f}"
              f"  -- {why}")
        print(f"best mean was w={best['w']:.2f}; anything above the pick buys nothing measurable.")
    else:
        print("\nNo weight passed: every nonzero weight is worse than the best or over the "
              "award-share cap. Leave BOOK_AWARD_WIN at 0 and keep the badges.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["w", "r20", "se", "d", "lo", "hi", "pw", "share"])
        w_.writeheader()
        for r in rows:
            w_.writerow({k: r.get(k, "") for k in w_.fieldnames})
    print(f"\nwrote {OUT_CSV}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [r["w"] for r in rows]
        ax.errorbar(xs, [r["r20"] for r in rows], yerr=[1.96 * r["se"] for r in rows],
                    marker="o", capsize=3, color="#2a6fdb", label="recall@20")
        if pick:
            ax.axvline(pick["w"], color="#999", ls="--", lw=1)
            ax.annotate("elbow pick", (pick["w"], pick["r20"]), textcoords="offset points",
                        xytext=(6, -14), fontsize=9, color="#555")
        ax.set_xlabel("award bonus for a win (shortlist = half)")
        ax.set_ylabel("recall@20 (goodbooks-10k)")
        ax2 = ax.twinx()
        ax2.plot(xs, [r["share"] for r in rows], marker="s", color="#d9822b",
                 label="award share of top 20")
        ax2.set_ylabel("award books in top 20")
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for s in ("top",):
            ax.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88), frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=150)
        print(f"wrote {OUT_PNG}")
    except ImportError:
        print("(matplotlib not installed: no plot; the CSV has the numbers)")


if __name__ == "__main__":
    main()
