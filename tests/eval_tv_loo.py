"""
eval_tv_loo.py — leave-one-out evaluation + paired ablation for the TV ranker.

WHY THIS EXISTS
TV is the only tab with no benchmark. Movies have MovieLens (100 users,
recall@k, an ablation table, a popularity baseline). Books have goodbooks-10k.
The handoff's state table says, for TV: "Benchmark: none".

Consequences of that gap, all live right now:
  - TV_MATURITY_DROP=3 is reasoned, not measured
  - the TV tag layer has never been ablated, so nobody knows whether it works.
    The same mechanism is -31% on films and inside noise on books; TV could be
    either and there is no evidence
  - genre tabs, the vote ceiling of 2000, per_anchor_cap, retrieve_k: all
    hand-set
  - every TV change ships on eyeballing

WHY LEAVE-ONE-OUT AND NOT A MULTI-USER HARNESS
There is no public TV ratings dataset keyed to TMDB ids. The Kaggle "TV ratings"
sets are aggregate IMDb scores, not per-user interactions, and aggregate scores
cannot produce a held-out relevance judgement. The one real multi-user option is
the MyAnimeList database (57M ratings, 310K users, 16,872 titles), which would
cover the ~11.8% of this corpus that is anime and needs a goodbooks-style title
linking stage. That is a bigger job; this is the cheap tier that works today.

READ THE ERROR BARS. One profile and a few dozen holdouts is a wide interval.
The paired statistics below are what make the ablation readable anyway: each
configuration is scored on the SAME holdouts, so the per-holdout difference has
far less variance than either number alone. That is the correction that
reversed the movie ablation's conclusions -- the diversity round-robin looked
expensive unpaired and was noise (t=0.4) once paired.

PROTOCOL NOTE
exclude_rated_creators is turned OFF during evaluation. Holding out a show
while a sibling by the same creator stays rated auto-excludes that creator, and
the miss is the protocol's fault rather than the ranker's. Same adjustment
eval_loo and eval_movielens already make for directors.

USAGE
    python tests\\eval_tv_loo.py tests\\movies_ratings.json
    python tests\\eval_tv_loo.py tests\\movies_ratings.json --ablate
    python tests\\eval_tv_loo.py tests\\movies_ratings.json --ablate --max-k 50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"

LIKE_HOLDOUT = 4.0
MAX_K = 50

# Each row turns ONE component off. The comparison is always against `full`,
# on the same holdouts, so the delta is paired.
ABLATIONS = [
    ("full",                 {}),
    ("no tag layer",         {"tag_weight": 0.0}),
    ("no quality floor",     {"min_vote_average": 0.0, "quality_weight": 0.0}),
    ("no maturity filter",   {"maturity_drop": 99}),
    ("no title penalty",     {"title_token_penalty": 0.0}),
    ("no negatives",         {"neg_scale": 0.0}),
    ("no cross-pollination", {"cross_weight": 0.0}),
    ("flat (no diversity)",  {"flat_ranking": True}),
]


def load_tv_ratings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        v = raw.get("re_tv")
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return v
        sys.exit(f"no 're_tv' key in {path} -- keys present: {list(raw.keys())}")
    return raw if isinstance(raw, list) else []


def rank_popularity(rest, df, k, min_votes=0):
    """Top-k unrated series by vote_count, for ONE holdout.

    Takes `rest` -- the profile WITHOUT the held-out show -- not the full
    ratings list. Excluding every rated id including the holdout removes the
    answer from the candidate pool before the baseline can find it, which
    scores a guaranteed zero and looks exactly like a real finding. The first
    version of this function did that.
    """
    rated = {str(r["tv_id"]) for r in rest if r.get("tv_id") is not None}
    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    pool = df[vc >= min_votes]
    top = pool[~pool["tmdb_id"].astype(str).isin(rated)].nlargest(k * 2, "vote_count")
    return [str(i) for i in top["tmdb_id"].tolist()[:k]]


def evaluate(ratings, df, emb, index, tags_df, rec, params, max_k, label=""):
    """-> list of ranks (None = miss), one per holdout, IN HOLDOUT ORDER.

    Order matters: the ablation pairs configurations by position, so every run
    must evaluate the same holdouts in the same sequence.
    """
    holdouts = [r for r in ratings if r.get("rating", 0) >= LIKE_HOLDOUT]
    ranks = []
    for i, held in enumerate(holdouts, 1):
        rest = [r for r in ratings if r is not held]
        target = str(held["tv_id"])
        try:
            # recommend_tv returns (DataFrame, top_user_tags), like the other
            # two rankers. Indexing the tuple is a TypeError, not an empty result.
            recs, _ = rec(rest, df, emb, index, tags_df=tags_df, n=max_k,
                          exclude_rated_creators=False, genre_cap=10 ** 9,
                          **params)
        except Exception as e:
            print(f"    holdout {i} failed: {type(e).__name__}: {e}")
            ranks.append(None)
            continue
        ids = [str(x) for x in recs["tmdb_id"].tolist()] if len(recs) else []
        ranks.append(ids.index(target) + 1 if target in ids else None)
        if i % 10 == 0:
            print(f"  {label}[{i}/{len(holdouts)}]")
    return ranks


def metrics(ranks):
    n = max(len(ranks), 1)
    return {
        "hit@10": sum(1 for r in ranks if r and r <= 10) / n,
        "hit@20": sum(1 for r in ranks if r and r <= 20) / n,
        "mrr": sum(1.0 / r for r in ranks if r) / n,
    }


def paired_t(full_ranks, abl_ranks):
    """t on the PER-HOLDOUT difference in reciprocal rank.

    Unpaired comparison of two hit rates over a few dozen holdouts is dominated
    by which shows happened to be held out. Pairing removes that entirely: each
    holdout is its own control. |t| >= 2 is a real effect.
    """
    d = []
    for a, b in zip(full_ranks, abl_ranks):
        ra = 1.0 / a if a else 0.0
        rb = 1.0 / b if b else 0.0
        d.append(ra - rb)          # positive = removing it HURT
    d = np.array(d, dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float(d.mean()), 0.0
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--max-k", type=int, default=MAX_K)
    ap.add_argument("--min-votes", type=int, default=0,
                    help="0 (default) measures the SCORING; 300 is the deployed "
                         "floor and measures what a user sees")
    args = ap.parse_args()

    ratings = load_tv_ratings(Path(args.ratings))
    if not ratings:
        sys.exit("no TV ratings found. Export from the Space with ?debug=1 and "
                 "copy the whole stored-values blob.")

    ns = load_ranker(SPACE_APP, require=("recommend_tv",))
    art = resolve_artifacts(SPACE_APP, PROCESSED, kind="tv", verbose=False)

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, (
        f"ALIGNMENT: parquet {len(df)}, npy {emb.shape[0]}, index {index.ntotal}")

    tags_df = None
    if "TAGS" in art and Path(art["TAGS"]).exists():
        t = pd.read_parquet(art["TAGS"])
        t["tv_id"] = t["tv_id"].astype(str)
        tags_df = t.pivot_table(index="tv_id", columns="tag", values="score",
                                fill_value=0.0)
        print(f"  tags: {tags_df.shape[0]:,} series")
    else:
        print("  NO TAG FILE -- the tag ablation row will be meaningless")

    # RE-RESOLVE corpus_index FROM tv_id. Do not trust the stored value.
    #
    # corpus_index is a ROW POSITION, and the TV corpus went from 23,464 rows
    # (tv_corpus) to 30,235 (tv_v2_meta). Every stale index is still in bounds
    # against the larger corpus, so a bounds check passes all of them while they
    # point at the WRONG SHOWS -- the ranker then builds a taste profile out of
    # arbitrary series and scores at chance. The first run of this harness did
    # exactly that: 2 hits in 70, which is 50/1750 to two decimal places.
    #
    # Books have _migrate_book_ratings in app.py for this. TV has no equivalent,
    # so the harness re-resolves here. tv_id is a stable TMDB id; corpus_index
    # never is.
    pos = {str(t): i for i, t in enumerate(df["tmdb_id"])}
    usable, moved, lost = [], 0, 0
    for r in ratings:
        tid = str(r.get("tv_id", ""))
        i = pos.get(tid)
        if i is None:
            lost += 1
            continue
        if r.get("corpus_index") != i:
            moved += 1
        r = dict(r)
        r["corpus_index"] = i
        usable.append(r)
    print(f"  re-resolved corpus_index from tv_id: {moved} moved, "
          f"{lost} no longer in the corpus")
    if moved > len(ratings) * 0.2:
        print("  >> most indices were stale. Any earlier run of this harness")
        print("     scored a profile built from the wrong shows.")
    ratings = usable

    holdouts = [r for r in ratings if r.get("rating", 0) >= LIKE_HOLDOUT]
    print(f"  corpus {len(df):,} series | {len(ratings)} rated | "
          f"{len(holdouts)} holdouts (>= {LIKE_HOLDOUT})")

    # THE CHECK THAT SAVES A WASTED RUN.
    # recommend_tv applies min_votes; anything below it can never be returned,
    # so a holdout under the floor is a guaranteed miss that says nothing about
    # the ranker. The first version of this harness did not check, scored
    # 2 hits in 70, and the popularity baseline scored ZERO -- which is the
    # signature of a floor problem, not a quality finding. Same protocol bug
    # eval_goodbooks documents for the books author suppressors.
    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    servable = int((vc >= args.min_votes).sum())
    reach = [r for r in holdouts
             if vc.iloc[int(r["corpus_index"])] >= args.min_votes]
    print(f"  serving floor min_votes={args.min_votes}: {servable:,} of "
          f"{len(df):,} series servable ({servable/len(df):.1%})")
    print(f"  holdouts ABOVE the floor (the only ones that can be hit): "
          f"{len(reach)} of {len(holdouts)}")
    if len(reach) < len(holdouts) * 0.5:
        print()
        print("  >> MORE THAN HALF THE HOLDOUTS ARE UNREACHABLE. Every number")
        print("     below would be dominated by the floor rather than the")
        print("     ranking. Re-run with --min-votes 0 to measure the SCORING,")
        print("     which is what an ablation is for. Then run once at the")
        print("     deployed floor to see what a user actually experiences --")
        print("     the two answer different questions and both are worth")
        print("     having.")
    if len(reach) < 10:
        print("\n  Refusing to report an ablation on fewer than 10 reachable")
        print("  holdouts. Lower --min-votes or rate more shows.")
        sys.exit(1)
    print()
    if len(holdouts) < 10:
        print("  WARNING: fewer than 10 holdouts. Every number below is")
        print("  directional at best. Rate more shows before drawing a")
        print("  conclusion from this.\n")

    rec = ns["recommend_tv"]
    configs = ABLATIONS if args.ablate else [("full", {})]

    results = {}
    for name, params in configs:
        print(f"running: {name}")
        merged = dict(params)
        merged.setdefault("min_votes", args.min_votes)
        results[name] = evaluate(ratings, df, emb, index, tags_df, rec,
                                 merged, args.max_k, label=f"{name}: ")

    pop_ranks = []
    for held in holdouts:
        rest = [r for r in ratings if r is not held]
        ids = rank_popularity(rest, df, args.max_k, args.min_votes)
        t = str(held["tv_id"])
        pop_ranks.append(ids.index(t) + 1 if t in ids else None)

    print("\n" + "=" * 74)
    print(f"{'method':<24}{'hit@10':>9}{'hit@20':>9}{'MRR':>9}"
          f"{'dMRR':>9}{'t':>7}")
    print("-" * 74)
    full = results["full"]
    for name, _ in configs:
        m = metrics(results[name])
        if name == "full":
            print(f"{name:<24}{m['hit@10']:>9.3f}{m['hit@20']:>9.3f}"
                  f"{m['mrr']:>9.3f}{'':>9}{'':>7}")
        else:
            d, t = paired_t(full, results[name])
            flag = "  <<" if abs(t) >= 2 else ""
            print(f"{name:<24}{m['hit@10']:>9.3f}{m['hit@20']:>9.3f}"
                  f"{m['mrr']:>9.3f}{d:>+9.4f}{t:>7.1f}{flag}")
    pm = metrics(pop_ranks)
    print("-" * 74)
    print(f"{'popularity baseline':<24}{pm['hit@10']:>9.3f}{pm['hit@20']:>9.3f}"
          f"{pm['mrr']:>9.3f}")
    print("=" * 74)
    print("dMRR is the PAIRED mean difference in reciprocal rank vs `full`.")
    print("Positive means removing that component HURT, so it is earning its")
    print("place. |t| >= 2 marked <<. Anything below that is noise, however")
    print("large the hit-rate gap looks -- that is the correction that")
    print("reversed the movie ablation's conclusions.")
    print()
    print("Then compare `full` against the popularity baseline. Movies lose to")
    print("it (0.043 vs 0.080) and books lose worse (0.016 vs 0.056). If TV")
    print("also loses, that is the expected result for a content-only ranker,")
    print("not a bug -- publish it the way the other two are published.")


if __name__ == "__main__":
    main()
