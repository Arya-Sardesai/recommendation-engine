"""
MovieLens multi-user evaluation & tuning harness for the movies recommender.

WHAT THIS IS
The statistically meaningful version of eval_loo.py. Instead of 53 holdouts
from one person, it evaluates on hundreds of real MovieLens users whose rated
films exist in your corpus, so differences between methods (and between
hyperparameter configs) are measurable instead of anecdotal.

PROTOCOL (per user)
  - Map the user's MovieLens ratings to your corpus via links.csv (tmdbId).
  - Random 80/20 split of their ratings (seeded, reproducible).
  - TRAIN (80%) -> feed to the recommender exactly like app ratings.
  - TEST relevant set = films in the held-out 20% the user rated >= 4.0,
    restricted to recommendable films (in corpus, vote_count >= MIN_VOTES).
    NOTE: the new ranker's vote_average >= 6.0 floor is NOT applied to the
    relevant set on purpose — if the floor hides a film a user loved, that is
    a real cost of the floor and should count against the method.
  - Run each method ONCE on TRAIN, take top-K, measure recall@10/20/50
    (fraction of the relevant set that appears) and hit@20 (did >= 1 appear).

METHODS
  new : recommend_movies from app.py (whole-profile ranker), diversity
        selection lifted (genre_cap huge, exclusions off) — we measure scoring.
  old : v1-style liked-only max-similarity baseline.
  pop : top-K unrated by vote_count.

MODES
  evaluate (default):
      python eval_movielens.py --ml-dir path/to/ml-25m --users 300
  tune (grid-search on tune users, validate best config on unseen test users):
      python eval_movielens.py --ml-dir path/to/ml-25m --tune --users 200 --test-users 150

SETUP
  1. Download ml-25m.zip from https://grouplens.org/datasets/movielens/25m/
     (~250 MB) and extract; you need ratings.csv and links.csv.
  2. Set artifact paths below (same as eval_loo.py).
Runtime: ~10-20 min for 300 users on CPU; tuning multiplies by grid size.

INTERPRETING
  - recall@20 is the headline. new > old by more than the printed stderr band
    = real improvement across many users, not one taste profile.
  - Tuning results report tune-set best AND unseen-test performance; only the
    latter is the claimable number (the lecture's train/test discipline,
    applied at the user level).

HONEST LIMITS
  - MovieLens users skew film-buff and pre-2019 heavy; absolute recall numbers
    aren't comparable across papers, only across methods on THIS setup.
  - Tag coverage: only ~13.5K films have tags, so the tag term is muted here
    exactly as it is in production.
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

# ---------------------------------------------------------------------------
# CONFIG — artifact paths (same layout as eval_loo.py; run from repo root)
# ---------------------------------------------------------------------------
PROCESSED = Path("data/processed")
CORPUS_PATH = PROCESSED / "movies.parquet"
EMB_PATH = PROCESSED / "movie_embeddings_bgem3_v2_kw.npy"
FAISS_PATH = PROCESSED / "movie_faiss_bgem3_v2_kw.index"
TAGS_PATH = PROCESSED / "movie_tags.parquet"
APP_PY = Path(__file__).resolve().parents[1].parent / "hf-space" / "app.py"

MIN_VOTES = 400
MAX_K = 50
SEED = 42

# user-sampling constraints: enough mapped ratings to have signal, not so many
# that they're a bot/completionist profile
MIN_MAPPED_RATINGS = 25
MAX_MAPPED_RATINGS = 200

# tuning grid (kept small on purpose; expand once runtime is known)
TUNE_GRID = [
    {"cross_weight": cw, "neg_scale": ns, "tag_weight": tw}
    for cw in (0.2, 0.35, 0.5)
    for ns in (0.5, 1.0, 1.5)
    for tw in (0.15, 0.25)
]


def load_app_functions(app_path):
    src = open(app_path).read()
    tree = ast.parse(src)
    wanted_assign = {"LANG_NAMES", "_TITLE_STOPWORDS"}
    wanted_funcs = {"lang_label", "_title_tokens", "_is_title_dup", "recommend_movies"}
    ns = {"np": np, "pd": pd, "faiss": faiss}
    for node in tree.body:
        take = False
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            take = True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in wanted_assign:
                    take = True
        if take:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(app_path), "exec"), ns)
    return ns["recommend_movies"]


# ---------------------------------------------------------------------------
# MovieLens loading + mapping
# ---------------------------------------------------------------------------
def load_movielens(ml_dir, df, n_users, seed, exclude_seeds=()):
    """Return {userId: [rating dicts in app format]} for n_users sampled users
    whose mapped-into-corpus rating count is within bounds."""
    ml = Path(ml_dir)
    links = pd.read_csv(ml / "links.csv", usecols=["movieId", "tmdbId"]).dropna()
    links["tmdbId"] = links["tmdbId"].astype("int64")

    # corpus lookup: tmdb id -> (row index, title, director) ; also recommendable set
    ids = pd.to_numeric(df["id"], errors="coerce")
    id_to_row = {int(v): i for i, v in ids.items() if pd.notna(v)}
    links = links[links["tmdbId"].isin(id_to_row.keys())]
    mid_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))
    print(f"links: {len(mid_to_tmdb):,} MovieLens films map into the corpus")

    ratings = pd.read_csv(ml / "ratings.csv", usecols=["userId", "movieId", "rating"])
    ratings = ratings[ratings["movieId"].isin(mid_to_tmdb.keys())]
    counts = ratings.groupby("userId").size()
    eligible = counts[(counts >= MIN_MAPPED_RATINGS) & (counts <= MAX_MAPPED_RATINGS)].index
    eligible = [u for u in eligible if u not in set(exclude_seeds)]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(eligible), size=min(n_users, len(eligible)), replace=False)
    print(f"users: {len(eligible):,} eligible ({MIN_MAPPED_RATINGS}-{MAX_MAPPED_RATINGS} mapped ratings); sampled {len(chosen)}")

    has_director = "director" in df.columns
    users = {}
    sub = ratings[ratings["userId"].isin(chosen)]
    for uid, grp in sub.groupby("userId"):
        rl = []
        for _, row in grp.iterrows():
            tmdb = mid_to_tmdb[row["movieId"]]
            ridx = id_to_row[tmdb]
            film = df.iloc[ridx]
            rl.append({
                "movie_id": int(tmdb),
                "matched_title": film["title"],
                "matched_director": film["director"] if has_director and isinstance(film["director"], str) else "",
                "corpus_index": int(ridx),
                "rating": float(row["rating"]),
            })
        users[int(uid)] = rl
    return users, list(chosen)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def rank_old_style(train, df, embeddings, index, k=MAX_K):
    liked = [r for r in train if r["rating"] >= 3.5]
    if not liked:
        return []
    anchor_vecs = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    sims, idxs = index.search(anchor_vecs, 100)
    rated = {str(r["movie_id"]) for r in train}
    best = {}
    for srow, irow in zip(sims, idxs):
        for s, ix in zip(srow, irow):
            if ix < 0:
                continue
            film = df.iloc[ix]
            if str(film["id"]) in rated or int(film["vote_count"]) < MIN_VOTES:
                continue
            if s > best.get(int(ix), -1e9):
                best[int(ix)] = float(s)
    ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
    return [str(df.iloc[ix]["id"]) for ix, _ in ranked]


def rank_popularity(train, df, k=MAX_K):
    rated = {str(r["movie_id"]) for r in train}
    top = df[~df["id"].astype(str).isin(rated)].nlargest(k * 2, "vote_count")
    return [str(i) for i in top["id"].tolist()[:k]]


def rank_new(train, df, embeddings, index, tags_df, recommend_movies, params=None, k=MAX_K):
    kwargs = dict(n=k, exclude_rated_directors=False, genre_cap=10 ** 9)
    if params:
        kwargs.update(params)
    recs, _ = recommend_movies(train, df, embeddings, index, tags_df=tags_df, **kwargs)
    return [str(x) for x in recs["id"].tolist()] if len(recs) else []


# ---------------------------------------------------------------------------
# Per-user evaluation
# ---------------------------------------------------------------------------
def split_user(ratings, rng):
    idx = rng.permutation(len(ratings))
    cut = max(1, int(round(len(ratings) * 0.2)))
    test_i = set(idx[:cut].tolist())
    train = [r for i, r in enumerate(ratings) if i not in test_i]
    test = [r for i, r in enumerate(ratings) if i in test_i]
    return train, test


def eval_users(users, df, embeddings, index, tags_df, recommend_movies,
               params=None, methods=("new", "old", "pop"), seed=SEED, label=""):
    rng = np.random.default_rng(seed)
    vote_ok = dict(zip(df["id"].astype(str), df["vote_count"] >= MIN_VOTES))
    per_user = {m: {"r10": [], "r20": [], "r50": [], "hit20": []} for m in methods}
    skipped = 0

    for n_done, (uid, ratings) in enumerate(users.items(), 1):
        train, test = split_user(ratings, rng)
        relevant = {str(r["movie_id"]) for r in test
                    if r["rating"] >= 4.0 and vote_ok.get(str(r["movie_id"]), False)}
        if not relevant or not any(r["rating"] >= 3.5 for r in train):
            skipped += 1
            continue

        ranked = {}
        if "new" in methods:
            ranked["new"] = rank_new(train, df, embeddings, index, tags_df,
                                     recommend_movies, params=params)
        if "old" in methods:
            ranked["old"] = rank_old_style(train, df, embeddings, index)
        if "pop" in methods:
            ranked["pop"] = rank_popularity(train, df)

        for m, ids in ranked.items():
            for K, key in ((10, "r10"), (20, "r20"), (50, "r50")):
                topk = set(ids[:K])
                per_user[m][key].append(len(relevant & topk) / len(relevant))
            per_user[m]["hit20"].append(1.0 if relevant & set(ids[:20]) else 0.0)

        if n_done % 25 == 0:
            print(f"  {label}[{n_done}/{len(users)}] users evaluated...")

    n = len(per_user[methods[0]]["r20"])
    print(f"\n{label}evaluated {n} users ({skipped} skipped: no relevant test items)")
    summary = {}
    for m in methods:
        row = {}
        for key in ("r10", "r20", "r50", "hit20"):
            a = np.array(per_user[m][key])
            row[key] = (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)
        summary[m] = row
    return summary, n


def print_summary(summary, n):
    print("=" * 74)
    print(f"{'method':<8}{'recall@10':>14}{'recall@20':>14}{'recall@50':>14}{'hit@20':>12}   (n={n})")
    print("-" * 74)
    for m, row in summary.items():
        print(f"{m:<8}"
              f"{row['r10'][0]:>9.3f}±{row['r10'][1]:.3f}"
              f"{row['r20'][0]:>9.3f}±{row['r20'][1]:.3f}"
              f"{row['r50'][0]:>9.3f}±{row['r50'][1]:.3f}"
              f"{row['hit20'][0]:>8.3f}±{row['hit20'][1]:.3f}")
    print("=" * 74)
    print("± is the standard error; gaps larger than ~2x the combined stderr are real.")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune(users_tune, users_test, df, embeddings, index, tags_df, recommend_movies):
    print(f"\nTUNING on {len(users_tune)} users, grid of {len(TUNE_GRID)} configs "
          f"(new method only)...")
    results = []
    for gi, params in enumerate(TUNE_GRID, 1):
        s, n = eval_users(users_tune, df, embeddings, index, tags_df,
                          recommend_movies, params=params, methods=("new",),
                          label=f"cfg {gi}/{len(TUNE_GRID)} ")
        r20 = s["new"]["r20"][0]
        results.append((r20, params))
        print(f"  cfg {gi}: {params} -> recall@20 = {r20:.4f}")
    results.sort(key=lambda x: -x[0])
    best_r20, best = results[0]
    print(f"\nBEST on tune set: {best} (recall@20 = {best_r20:.4f})")

    print(f"\nVALIDATING best config on {len(users_test)} UNSEEN test users "
          f"(vs default + baselines)...")
    s_best, n = eval_users(users_test, df, embeddings, index, tags_df,
                           recommend_movies, params=best, methods=("new",), label="best ")
    s_all, n2 = eval_users(users_test, df, embeddings, index, tags_df,
                           recommend_movies, params=None, label="default ")
    print("\n--- default config + baselines on test users ---")
    print_summary(s_all, n2)
    print("\n--- TUNED config on test users ---")
    print_summary(s_best, n)
    print("\nThe claimable number is the tuned 'new' row on TEST users. If it is")
    print("within stderr of the default, the defaults were already near-optimal —")
    print("also a legitimate finding.")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Ablation — switch one component off at a time, same users, and see what hurts
# ---------------------------------------------------------------------------
ABLATIONS = [
    # round 1 settled: negatives HELP (removal halved recall), quality prior
    # helps, cross-pollination & tags neutral on recall, flat > full (the
    # diversity round-robin costs recall — deliberate product trade).
    # Round 2: the components round 1 never switched off — both deliberately
    # anti-"obvious" product choices that leave-out eval punishes.
    ("full (default)",        None),
    ("flat (no diversity RR)", {"flat_ranking": True}),
    ("no title-token penalty", {"title_token_penalty": 0.0}),
    ("no vote_avg floor",      {"min_vote_average": 0.0}),
    ("flat + no penalty/floor", {"flat_ranking": True, "title_token_penalty": 0.0,
                                 "min_vote_average": 0.0}),
]


def ablate(users, df, embeddings, index, tags_df, recommend_movies):
    print(f"\nABLATION on {len(users)} users — one component off per row; same "
          f"users, same splits (seeded), so rows are directly comparable.\n")
    rows = []
    for name, params in ABLATIONS:
        s, n = eval_users(users, df, embeddings, index, tags_df,
                          recommend_movies, params=params, methods=("new",),
                          label=f"{name}: ")
        r = s["new"]
        rows.append((name, r))
        print(f"  {name:<24} recall@20 = {r['r20'][0]:.4f}±{r['r20'][1]:.3f}   "
              f"hit@20 = {r['hit20'][0]:.3f}")
    # baselines on the same users for reference
    s_base, n = eval_users(users, df, embeddings, index, tags_df,
                           recommend_movies, methods=("old", "pop"), label="baselines: ")
    print("\n" + "=" * 74)
    print(f"{'variant':<26}{'recall@10':>12}{'recall@20':>12}{'recall@50':>12}{'hit@20':>10}")
    print("-" * 74)
    for name, r in rows:
        print(f"{name:<26}{r['r10'][0]:>12.4f}{r['r20'][0]:>12.4f}{r['r50'][0]:>12.4f}{r['hit20'][0]:>10.3f}")
    for m in ("old", "pop"):
        r = s_base[m]
        print(f"{m:<26}{r['r10'][0]:>12.4f}{r['r20'][0]:>12.4f}{r['r50'][0]:>12.4f}{r['hit20'][0]:>10.3f}")
    print("=" * 74)
    print("Read: whichever off-switch RAISES recall names the component that hurts")
    print("at this rating scale. 'flat' isolates scoring from diversity selection.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-dir", required=True, help="folder containing ratings.csv and links.csv")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--test-users", type=int, default=150, help="tuning mode: unseen validation users")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    recommend_movies = load_app_functions(APP_PY)
    print("loading corpus + artifacts...")
    df = pd.read_parquet(CORPUS_PATH).reset_index(drop=True)
    embeddings = np.load(EMB_PATH).astype("float32")
    index = faiss.read_index(str(FAISS_PATH))
    tags_df = None
    if TAGS_PATH.exists():
        t = pd.read_parquet(TAGS_PATH)
        t["movie_id"] = t["movie_id"].astype(str)
        tags_df = t.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)
        print(f"tags loaded: {tags_df.shape[0]:,} films")

    users, chosen = load_movielens(args.ml_dir, df, args.users, args.seed)
    if args.ablate:
        ablate(users, df, embeddings, index, tags_df, recommend_movies)
    elif args.tune:
        users_test, _ = load_movielens(args.ml_dir, df, args.test_users,
                                       args.seed + 1, exclude_seeds=chosen)
        tune(users, users_test, df, embeddings, index, tags_df, recommend_movies)
    else:
        summary, n = eval_users(users, df, embeddings, index, tags_df, recommend_movies)
        print_summary(summary, n)


if __name__ == "__main__":
    main()