"""
Leave-one-out evaluation for the movies recommender (lecture-style, adapted to ranking).

WHAT IT MEASURES
For every film you rated >= LIKE_HOLDOUT, we pretend you never rated it, run the
recommender on your remaining ratings, and check where the held-out film lands
in the ranked list. Reported per method:
  hit@10 / hit@20 : fraction of held-out films that appeared in the top 10/20
  MRR             : mean reciprocal rank (1/rank, 0 if not in top MAX_K)

METHODS COMPARED (the lecture's evaluate() pattern, for ranking)
  new  : the current whole-profile ranker (recommend_movies from app.py)
  old  : v1-style baseline — liked-only anchors, max cosine similarity,
         no negatives, no cross-pollination, no quality signal
  pop  : popularity floor — top-K unrated films by vote_count

PROTOCOL ADJUSTMENTS (important for honest numbers)
  - exclude_rated_directors=False : otherwise holding out Spider-Man 2 while
    Spider-Man stays rated auto-excludes Raimi -> a miss that isn't the
    ranker's fault.
  - genre_cap lifted, n=MAX_K : we evaluate the SCORING, not the diversity
    selection layer (which deliberately trades accuracy for variety).
  - Known residual undercount: franchise holdouts can still be dropped by the
    title-dedup inside selection (Spider-Man 2 vs Spider-Man). Affects a
    handful of items, and affects 'new' only — so 'new' numbers are, if
    anything, slightly UNDERstated. Noted, accepted.

CAVEAT ON INTERPRETATION
One user, ~40-55 holdouts -> wide error bars. Treat differences under ~5
percentage points as noise. This is the cheap directional tier; the MovieLens
multi-user harness is the definitive one (separate chapter).

HOW TO RUN (locally, from the repo)
  1. Set the artifact paths below to your data/processed/ files.
  2. Export your ratings: open the Space with ?debug=1 and copy the ENTIRE
     "raw stored values" blob into movies_ratings.json — the script extracts
     re_movies from it automatically (a bare re_movies value also works).
  3. python eval_loo.py movies_ratings.json
Runtime: a few minutes (one full recommend per holdout).
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG — point these at your local artifacts
# ---------------------------------------------------------------------------
PROCESSED = Path("data/processed")
# Resolved from app.py at runtime by resolve_artifacts() -- see main().
# These remain only as the fallback if a constant is missing from app.py.
CORPUS_PATH = PROCESSED / "movies.parquet"
EMB_PATH = PROCESSED / "movie_embeddings_bgem3_v2_kw2.npy"
FAISS_PATH = PROCESSED / "movie_faiss_bgem3_v2_kw2.index"
TAGS_PATH = PROCESSED / "movie_tags_v2.parquet"          # optional; skipped if absent
APP_PY = Path(__file__).resolve().parents[1].parent / "hf-space" / "app.py"   # main/tests -> Poject/hf-space/app.py

LIKE_HOLDOUT = 4.0   # hold out every film rated >= this
MAX_K = 50           # rank horizon; beyond this counts as a miss
MIN_VOTES = 400      # same floor as the app


# ---------------------------------------------------------------------------
# Pull the ranker + helpers out of app.py (same AST pattern as the smoke tests)
# ---------------------------------------------------------------------------
# The hand-maintained whitelist that used to live here broke every time a new
# module-level constant appeared in app.py (MOVIE_MIN_RUNTIME and
# MOVIE_KIDS_FILTER as defaulted parameters killed both harnesses at once with
# a NameError). ranker_loader takes every top-level definition instead, so it
# tracks app.py without edits here.
def load_app_functions(app_path):
    return load_ranker(app_path)["recommend_movies"]


# ---------------------------------------------------------------------------
# Baseline rankers
# ---------------------------------------------------------------------------
def rank_old_style(ratings, df, embeddings, index, k=MAX_K):
    """v1-style: liked-only anchors, candidate score = max cosine over anchors.
    No negatives, no cross bonus, no quality. Returns ranked list of movie ids."""
    liked = [r for r in ratings if r["rating"] >= 3.5]
    if not liked:
        return []
    anchor_vecs = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    sims, idxs = index.search(anchor_vecs, 100)
    rated = {str(r["movie_id"]) for r in ratings}
    best = {}
    for srow, irow in zip(sims, idxs):
        for s, ix in zip(srow, irow):
            if ix < 0:
                continue
            film = df.iloc[ix]
            if str(film["id"]) in rated or int(film["vote_count"]) < MIN_VOTES:
                continue
            key = int(ix)
            if s > best.get(key, -1e9):
                best[key] = float(s)
    ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
    return [str(df.iloc[ix]["id"]) for ix, _ in ranked]


def rank_popularity(ratings, df, k=MAX_K):
    rated = {str(r["movie_id"]) for r in ratings}
    top = df[~df["id"].astype(str).isin(rated)].nlargest(k * 2, "vote_count")
    return [str(i) for i in top["id"].tolist()[:k]]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(ratings, df, embeddings, index, tags_df, recommend_movies):
    holdouts = [r for r in ratings if r["rating"] >= LIKE_HOLDOUT]
    print(f"{len(ratings)} ratings, {len(holdouts)} holdouts (rated >= {LIKE_HOLDOUT})\n")

    methods = {"new": [], "old": [], "pop": []}   # per-holdout rank (None = miss)
    for i, held in enumerate(holdouts):
        rest = [r for r in ratings if r is not held]
        target = str(held["movie_id"])

        # NEW — scoring-focused settings (see protocol notes in docstring)
        recs, _ = recommend_movies(
            rest, df, embeddings, index, tags_df=tags_df, n=MAX_K,
            exclude_rated_directors=False, genre_cap=10 ** 9,
        )
        new_ids = [str(x) for x in recs["id"].tolist()] if len(recs) else []

        old_ids = rank_old_style(rest, df, embeddings, index)
        pop_ids = rank_popularity(rest, df)

        for name, ranked in (("new", new_ids), ("old", old_ids), ("pop", pop_ids)):
            rank = ranked.index(target) + 1 if target in ranked else None
            methods[name].append(rank)
        print(f"  [{i+1}/{len(holdouts)}] {held['matched_title'][:44]:<46}"
              f" new:{methods['new'][-1] or '-':>4}  old:{methods['old'][-1] or '-':>4}"
              f"  pop:{methods['pop'][-1] or '-':>4}")

    print("\n" + "=" * 58)
    print(f"{'method':<8}{'hit@10':>10}{'hit@20':>10}{'MRR':>10}   (n={len(holdouts)})")
    print("-" * 58)
    for name, ranks in methods.items():
        n = len(ranks)
        h10 = sum(1 for r in ranks if r and r <= 10) / n
        h20 = sum(1 for r in ranks if r and r <= 20) / n
        mrr = sum(1.0 / r for r in ranks if r) / n
        print(f"{name:<8}{h10:>10.3f}{h20:>10.3f}{mrr:>10.3f}")
    print("=" * 58)
    print("Read: higher is better. Differences under ~0.05 are noise at this n.")
    print("'new' beating 'old' on hit@20 + MRR = the whole-profile rewrite is")
    print("empirically better, not just prettier. 'pop' is the floor any real")
    print("recommender must clear.")


def parse_ratings(path):
    """Accept any of the shapes the debug panel produces:
    - the full getAll() dump: {"re_movies": "[...]", "re_books": ...}  <- most common
    - the re_movies value alone, double-encoded ("[{...}]") or as a list
    """
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        if "re_movies" not in raw or not raw["re_movies"]:
            sys.exit(f"error: no 're_movies' key found in {path} — "
                     f"keys present: {list(raw.keys())}")
        raw = raw["re_movies"]
    if isinstance(raw, str):                  # localStorage double-encodes
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        sys.exit(f"error: {path} did not yield a list of ratings")
    return raw


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit("usage: python eval_loo.py movies_ratings.json")
    ratings_raw = parse_ratings(sys.argv[1])
    print(f"loaded {len(ratings_raw)} ratings from {sys.argv[1]}")

    recommend_movies = load_app_functions(APP_PY)

    # Follow app.py rather than the constants above. Hardcoding movies.parquet
    # meant this harness evaluated the PRE-ENRICHMENT corpus: no cert_tier, so
    # the kids-content filter was inert and the numbers described a ranker that
    # was never deployed.
    art = resolve_artifacts(APP_PY, PROCESSED, kind="movie")
    corpus_path = art.get("CORPUS", CORPUS_PATH)
    emb_path = art.get("EMB", EMB_PATH)
    faiss_path = art.get("FAISS", FAISS_PATH)
    tags_path = art.get("TAGS", TAGS_PATH)

    df = pd.read_parquet(corpus_path).reset_index(drop=True)
    embeddings = np.load(emb_path).astype("float32")
    index = faiss.read_index(str(faiss_path))
    assert len(df) == embeddings.shape[0] == index.ntotal, (
        f"ALIGNMENT: parquet {len(df)}, npy {embeddings.shape[0]}, "
        f"index {index.ntotal}")
    print(f"  corpus {len(df):,} rows, aligned"
          f"{'  [cert_tier present]' if 'cert_tier' in df.columns else ''}")
    tags_df = None
    if Path(tags_path).exists():
        t = pd.read_parquet(tags_path)
        t["movie_id"] = t["movie_id"].astype(str)
        tags_df = t.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)
        print(f"tags loaded: {tags_df.shape[0]:,} films")
    evaluate(ratings_raw, df, embeddings, index, tags_df, recommend_movies)


if __name__ == "__main__":
    main()
