"""
eval_goodbooks.py — multi-user recall evaluation for the BOOKS ranker.

WHY THIS EXISTS
Movies has MovieLens: 100 users, recall@k, an ablation table, a popularity
baseline. Books has had nothing at all. Every books decision so far -- the
author cap, the work_id investigation, the YA question -- was made by reading
one list and forming an opinion. This is the harness that makes books
decisions measurable the way movie decisions already are.

THE JOIN PROBLEM
goodbooks-10k carries Goodreads ids; the corpus is Hardcover. There is no
shared key, so books are matched on normalised TITLE + AUTHOR. That matching is
lossy and the loss is NOT random -- popular English-language fiction matches
far better than translated or regional titles -- so the reported numbers
describe the matched subset, not the corpus. The link stage prints its own
match rate for exactly that reason, and it is cached because it is the slow
part.

    goodbooks-10k: 6M ratings, 10k books, 53,424 users
    https://github.com/zygmuntz/goodbooks-10k  (books.csv + ratings.csv)

PROTOCOL
Per user: split rated books into train/test (seeded), rank from train, measure
whether held-out liked books appear in the top k. Same shape as
eval_movielens.py so the two are readable side by side.

METHODS
  new : the deployed recommend_books
  pop : top-k unrated books by ratings_count -- the floor any recommender must
        clear, and on MovieLens the one the movie ranker does NOT clear

USAGE
    python tests\\eval_goodbooks.py --gb-dir data\\raw\\books\\goodbooks-10k
    python tests\\eval_goodbooks.py --gb-dir ... --users 100
    python tests\\eval_goodbooks.py --gb-dir ... --relink     # rebuild the join
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"
LINK_CACHE = REPO_ROOT / "tests" / "goodbooks_links.json"

# --- PROTOCOL --------------------------------------------------------------
# eval_movielens passes exclude_rated_directors=False and lifts the genre cap
# because holding out Spider-Man 2 while Spider-Man stays rated auto-excludes
# Raimi: a miss that is not the ranker's fault. Books needs the same treatment
# and MORE of it, because the books ranker has three author-level suppressors
# the movie ranker does not:
#
#   exclude_rated_authors=True   drops every book by an author already rated
#   per_author_cap=1             at most one book per author in the output
#   author_adjacent_penalty=0.3  demotes near-neighbours of rated authors
#
# goodbooks-10k relevance is dominated by SERIES -- Harry Potter 1-7, Hunger
# Games, Lord of the Rings. With the suppressors on, a user who rated book 1
# can never be credited for book 2, and the harness measures the diversity
# layer instead of the scoring. First run this way scored recall@20 = 0.002
# against a popularity baseline of 0.057, which is not a quality finding, it
# is a protocol bug.
#
# Pass --as-deployed to measure the shipped configuration instead. That number
# is the honest one for "what does a user actually see", and it will be much
# lower -- the two answer different questions and both are worth having.
SCORING_PROTOCOL = dict(
    exclude_rated_authors=False,
    per_author_cap=0,
    author_adjacent_penalty=0.0,
    genre_cap=10 ** 9,
    per_anchor_cap=10 ** 9,
)

SEED = 17
MIN_RATED = 15          # users below this have too little train signal
MAX_RATED = 200         # and above this are outliers, not readers
LIKE = 4.0              # a held-out book counts as relevant at >= this


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(title):
    """Drop subtitle and series parenthetical: goodbooks titles carry both
    ('The Fellowship of the Ring (The Lord of the Rings, #1)')."""
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":", t)[0]
    return re.sub(r"^(the|a|an) ", "", _norm(t)).strip()


def _first_author(a):
    return _norm(str(a).split(",")[0])


def build_links(books_csv, df, relink=False):
    """Map goodbooks book_id -> corpus row index on title + author.

    Two passes, strict first. Requiring the author to match as well as the
    title is what keeps 'The Gift' (there are dozens) from linking to the wrong
    book -- title-only matching looked fine in aggregate and was wrong on
    exactly the common titles that carry the most ratings.
    """
    if LINK_CACHE.exists() and not relink:
        cached = json.loads(LINK_CACHE.read_text(encoding="utf-8"))
        print(f"  link cache: {len(cached):,} books "
              f"(--relink to rebuild)")
        return {int(k): int(v) for k, v in cached.items()}

    gb = pd.read_csv(books_csv)
    print(f"  goodbooks: {len(gb):,} books")

    by_pair, by_title = {}, {}
    for i, (t, a) in enumerate(zip(df["title"], df["primary_author"])):
        ct, ca = _core(t), _first_author(a)
        by_pair.setdefault((ct, ca), i)
        by_title.setdefault(ct, i)

    links, strict, loose = {}, 0, 0
    for r in gb.itertuples():
        ct, ca = _core(r.title), _first_author(r.authors)
        hit = by_pair.get((ct, ca))
        if hit is not None:
            links[int(r.book_id)] = int(hit); strict += 1
            continue
        hit = by_title.get(ct)
        if hit is not None:
            links[int(r.book_id)] = int(hit); loose += 1

    LINK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LINK_CACHE.write_text(json.dumps(links), encoding="utf-8")
    print(f"  linked {len(links):,} of {len(gb):,} ({len(links)/len(gb):.1%})"
          f"   [{strict:,} title+author, {loose:,} title-only]")
    print("  NOTE: unmatched books are dropped from evaluation. The loss is not")
    print("  random — translated and regional titles match worst — so these")
    print("  numbers describe the matched subset, not the whole corpus.")
    return links


def load_users(ratings_csv, links, n_users, seed=SEED):
    rt = pd.read_csv(ratings_csv)
    rt = rt[rt["book_id"].isin(links)]
    sizes = rt.groupby("user_id").size()
    eligible = sizes[(sizes >= MIN_RATED) & (sizes <= MAX_RATED)].index
    print(f"  users: {len(eligible):,} eligible "
          f"({MIN_RATED}-{MAX_RATED} mapped ratings)")
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.array(eligible), size=min(n_users, len(eligible)),
                      replace=False)
    sub = rt[rt["user_id"].isin(pick)]
    return {u: g for u, g in sub.groupby("user_id")}


def eval_users(users, df, emb, index, tags_df, recommend_books, params=None,
               k_list=(10, 20, 50), seed=SEED, label="", folds=1):
    """folds=1: one random 70/30 split per user (the original protocol).
    folds=K: each user's ratings are cut into K parts and each part is the
    test set once, train on the rest; the user's score is the mean over
    folds. Same users and same splits for every config (seeded), so the
    per-user scores stay paired."""
    params = params or {}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in k_list}
    hit20, n_ok, skipped = [], 0, 0
    per_user = []        # recall@20 per user, None if skipped -- for pairing
    maxk = max(k_list)

    for i, (uid, g) in enumerate(users.items(), 1):
        rows = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        if folds <= 1:
            cut = int(len(rows) * 0.7)
            splits = [(rows.iloc[:cut], rows.iloc[cut:])]
        else:
            parts = np.array_split(np.arange(len(rows)), folds)
            splits = [(rows.drop(rows.index[p]), rows.iloc[p]) for p in parts]
        u_acc = {k: [] for k in k_list}
        u_hit, u_r20 = [], []
        failed = False
        for train_r, test_r in splits:
            relevant = {int(b) for b, r in zip(test_r["book_id"], test_r["rating"])
                        if r >= LIKE}
            if not relevant or len(train_r) < 5:
                continue
            res = _score_split(train_r, relevant, df, emb, index, tags_df,
                               recommend_books, params, k_list, maxk, uid)
            if res is None:
                failed = True
                break
            for k in k_list:
                u_acc[k].append(res[k])
            u_hit.append(res["hit"])
            u_r20.append(res[20] if 20 in res else res[k_list[0]])
        if failed or not u_r20:
            skipped += 1
            per_user.append(None)
            continue
        for k in k_list:
            acc[k].append(float(np.mean(u_acc[k])))
        hit20.append(float(np.mean(u_hit)))
        per_user.append(float(np.mean(u_r20)))
        n_ok += 1
        if i % 25 == 0:
            print(f"  {label}[{i}/{len(users)}] evaluated...")

    def ms(v):
        a = np.array(v) if v else np.array([0.0])
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0

    out = {f"r{k}": ms(acc[k]) for k in k_list}
    out["hit20"] = ms(hit20)
    print(f"  {label}evaluated {n_ok} users ({skipped} skipped)")
    out["_per_user"] = per_user
    return out, n_ok


def _score_split(train_r, relevant, df, emb, index, tags_df, recommend_books,
             params, k_list, maxk, uid):
    ratings = []
    for b, r in zip(train_r["book_id"], train_r["rating"]):
        ci = LINKS[int(b)]
        row = df.iloc[ci]
        ratings.append({"book_id": str(row["book_id"]),
                        "matched_title": row["title"],
                        "matched_author": row.get("primary_author", ""),
                        "corpus_index": int(ci), "rating": float(r)})
    try:
        recs, _ = recommend_books(ratings, df, emb, index,
                                  tags_df=tags_df, n=maxk, **params)
    except Exception as e:
        print(f"    user {uid} failed: {type(e).__name__}: {e}")
        return None

    got_ci = list(recs["corpus_index"]) if "corpus_index" in recs.columns else []
    rel_ci = {LINKS[b] for b in relevant if b in LINKS}
    res = {k: len(set(got_ci[:k]) & rel_ci) / max(len(rel_ci), 1) for k in k_list}
    res["hit"] = 1.0 if set(got_ci[:20]) & rel_ci else 0.0
    return res


def eval_pop(users, df, k_list=(10, 20, 50)):
    rc = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    order = list(np.argsort(-rc.values))
    acc = {k: [] for k in k_list}
    hit20 = []
    rng = np.random.default_rng(SEED)
    for uid, g in users.items():
        rows = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        cut = int(len(rows) * 0.7)
        train_r, test_r = rows.iloc[:cut], rows.iloc[cut:]
        relevant = {int(b) for b, r in zip(test_r["book_id"], test_r["rating"])
                    if r >= LIKE}
        if not relevant or len(train_r) < 5:
            continue
        seen = {LINKS[int(b)] for b in train_r["book_id"]}
        top = [i for i in order if i not in seen][:max(k_list)]
        rel_ci = {LINKS[b] for b in relevant if b in LINKS}
        for k in k_list:
            acc[k].append(len(set(top[:k]) & rel_ci) / max(len(rel_ci), 1))
        hit20.append(1.0 if set(top[:20]) & rel_ci else 0.0)

    def ms(v):
        a = np.array(v) if v else np.array([0.0])
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {f"r{k}": ms(acc[k]) for k in k_list} | {"hit20": ms(hit20)}


def load_tag_matrix(path):
    t = pd.read_parquet(path)
    t["book_id"] = t["book_id"].astype(str)
    return t.pivot_table(index="book_id", columns="tag", values="score",
                         fill_value=0.0)


def paired_t(base, other):
    """Paired per user: the correction that reversed the movie ablation.
    Positive delta = `other` beats `base`."""
    d = np.array([b - a for a, b in zip(base, other)
                  if a is not None and b is not None], dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return (float(d.mean()) if len(d) else 0.0), 0.0
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def paired_bootstrap(a, b, n_boot=10000, seed=SEED):
    """95% CI of mean(b - a) over users, resampling users with replacement,
    and the share of resamples where b is worse than a."""
    d = np.array([y - x for x, y in zip(a, b) if x is not None and y is not None])
    if len(d) < 2:
        return 0.0, 0.0, 0.0, 0.5
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((means < 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-dir", required=True,
                    help="folder with books.csv and ratings.csv")
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--folds", type=int, default=1,
                    help="K-fold per user (each part held out once); 1 = the "
                         "original single 70/30 split. Runtime scales with K.")
    ap.add_argument("--relink", action="store_true")
    ap.add_argument("--as-deployed", action="store_true",
                    help="measure the shipped config, suppressors and all, "
                         "instead of the scoring-only protocol")
    ap.add_argument("--ablate", action="store_true",
                    help="one component off per row")
    ap.add_argument("--compare-tags", default=None, metavar="PARQUET",
                    help="A/B the live tag layer against this tag file, paired "
                         "per user, e.g. data\\processed\\book_tags_v3_llm.parquet")
    ap.add_argument("--anchor-mix", action="store_true",
                    help="A/B the per-anchor tag profile (anchor_tag_mix) "
                         "against the old whole-profile tags, paired per user")
    ap.add_argument("--shelf-idf", action="store_true",
                    help="A/B tag IDF weighting and the neighbourhood genre "
                         "bonus against the current default ranker")
    ap.add_argument("--neg-floor", action="store_true",
                    help="A/B the dislike floor, similarity stretch and "
                         "per-anchor tag mix against the old shipped ranker")
    args = ap.parse_args()

    gb = Path(args.gb_dir)
    ns = load_ranker(SPACE_APP, require=("recommend_books",))
    art = resolve_artifacts(SPACE_APP, PROCESSED, kind="book")

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, (
        f"ALIGNMENT: parquet {len(df)}, npy {emb.shape[0]}, index {index.ntotal}")
    print(f"  corpus {len(df):,} rows, aligned")

    tags_df = None
    if "TAGS" in art and Path(art["TAGS"]).exists():
        t = pd.read_parquet(art["TAGS"])
        t["book_id"] = t["book_id"].astype(str)
        tags_df = t.pivot_table(index="book_id", columns="tag", values="score",
                                fill_value=0.0)
        print(f"  tags: {tags_df.shape[0]:,} books")

    global LINKS
    LINKS = build_links(gb / "books.csv", df, relink=args.relink)
    users = load_users(gb / "ratings.csv", LINKS, args.users)
    print(f"  sampled {len(users)} users\n")

    rec = ns["recommend_books"]
    base_params = {} if args.as_deployed else dict(SCORING_PROTOCOL)
    print("  protocol:", "AS DEPLOYED (diversity layer on)" if args.as_deployed
          else "SCORING ONLY (author suppressors + caps lifted)")
    rows = []
    if args.shelf_idf:
        CONFIGS = [
            ("current default",        {}),
            ("tag idf 0.5",            {"tag_idf": 0.5}),
            ("tag idf 1.0",            {"tag_idf": 1.0}),
            ("genre +0.10",            {"genre_weight": 0.10}),
            ("genre +0.20",            {"genre_weight": 0.20}),
            ("idf 1.0 + genre 0.10",   {"tag_idf": 1.0, "genre_weight": 0.10}),
        ]
        base = "current default"
    elif args.neg_floor:
        OLD = {"anchor_tag_mix": 0.0, "neg_floor_k": 0, "sim_scale": 1.0}
        F = {"anchor_tag_mix": 0.0, "neg_floor_k": 200}
        CONFIGS = [
            ("old (shipped)",          OLD),
            ("floor only",             {**F, "sim_scale": 1.0}),
            ("floor + sim x2",         {**F, "sim_scale": 2.0}),
            ("floor + sim x3 (new)",   {**F, "sim_scale": 3.0}),
            ("floor + sim x5",         {**F, "sim_scale": 5.0}),
            ("floor + sim x3 + mix.5", {**F, "sim_scale": 3.0, "anchor_tag_mix": 0.5}),
        ]
        base = "old (shipped)"
    elif args.anchor_mix:
        base = "old (mix 0)"
        CONFIGS = [
            ("old (mix 0)",   {"anchor_tag_mix": 0.0}),
            ("no tag layer",  {"tag_weight": 0.0}),
            ("mix 0.25",      {"anchor_tag_mix": 0.25}),
            ("mix 0.50",      {"anchor_tag_mix": 0.50}),
            ("mix 0.75",      {"anchor_tag_mix": 0.75}),
            ("mix 1.0",       {"anchor_tag_mix": 1.0}),
        ]
    if args.neg_floor or args.anchor_mix or args.shelf_idf:
        per = {}
        for name, params in CONFIGS:
            merged = dict(base_params)
            merged.update(params)
            s_, _ = eval_users(users, df, emb, index, tags_df, rec,
                               params=merged, label=f"{name}: ", folds=args.folds)
            per[name] = s_.pop("_per_user")
            rows.append((name, s_))
        print("\n" + "=" * 92)
        print(f"  {args.users} users, {args.folds}-fold"
              + (" (single 70/30 split)" if args.folds <= 1 else ""))
        print(f"{'method':<24}{'recall@20':>14}{'d r@20':>10}{'t':>6}"
              f"{'95% CI (bootstrap)':>24}{'P(worse)':>10}")
        print("-" * 92)
        for name, r in rows:
            line = f"{name:<24}{r['r20'][0]:>9.4f}±{r['r20'][1]:<4.3f}"
            if name != base:
                d, t = paired_t(per[base], per[name])
                _, lo, hi, pw = paired_bootstrap(per[base], per[name])
                flag = " <<" if (lo > 0 or hi < 0) else ""
                line += (f"{d:>+10.4f}{t:>6.1f}"
                         f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}{pw:>9.0%}{flag}")
            print(line)
        print("=" * 92)
        print("CI excludes 0 (<<) = a real difference. P(worse) = share of the")
        print("10,000 user-resamples where that row loses to the baseline.")
        print(f"Paired vs `{base}`. Ship the new config if it is not")
        print("significantly WORSE (t > -2): goodbooks can't see the romance-")
        print("under-a-thriller failure, so a tie is a win for the live app.")
        return
    if args.compare_tags:
        alt_path = Path(args.compare_tags)
        if not alt_path.exists():
            raise SystemExit(f"{alt_path} not found")
        alt = load_tag_matrix(alt_path)
        print(f"  candidate tags: {alt.shape[0]:,} books x {alt.shape[1]} tags "
              f"({alt_path.name})")
        CONFIGS = [
            ("live tags",           tags_df, {}),
            ("no tag layer",        tags_df, {"tag_weight": 0.0}),
            ("LLM tags",            alt,     {}),
            ("LLM tags, w=0.40",    alt,     {"tag_weight": 0.40}),
            ("LLM tags, w=0.60",    alt,     {"tag_weight": 0.60}),
        ]
        per = {}
        for name, tmat, params in CONFIGS:
            merged = dict(base_params)
            merged.update(params)
            s_, _ = eval_users(users, df, emb, index, tmat, rec,
                               params=merged, label=f"{name}: ", folds=args.folds)
            per[name] = s_.pop("_per_user")
            rows.append((name, s_))
        rows.append(("pop", eval_pop(users, df)))

        print("\n" + "=" * 88)
        print(f"{'method':<22}{'recall@10':>14}{'recall@20':>14}{'recall@50':>14}"
              f"{'d r@20':>11}{'t':>7}")
        print("-" * 88)
        for name, r in rows:
            line = (f"{name:<22}"
                    f"{r['r10'][0]:>9.3f}±{r['r10'][1]:<4.3f}"
                    f"{r['r20'][0]:>9.3f}±{r['r20'][1]:<4.3f}"
                    f"{r['r50'][0]:>9.3f}±{r['r50'][1]:<4.3f}")
            if name in per and name != "live tags":
                d, t = paired_t(per["live tags"], per[name])
                line += f"{d:>+11.4f}{t:>7.1f}{' <<' if abs(t) >= 2 else ''}"
            print(line)
        print("=" * 88)
        print("d r@20 is the PAIRED per-user difference vs `live tags`.")
        print("Positive = better than what ships now. |t| >= 2 is real; below")
        print("that is noise however big the gap looks.")
        print("\nShip the LLM tags only if an LLM row is >= live with |t| >= 2,")
        print("or statistically tied AND the shelf gate needs them anyway.")
        return
    if args.ablate:
        # The author cap is a LIVE setting, shipped at 1 since this morning to
        # stop two Martha Wells books appearing adjacent. The first ablation
        # measured it at roughly half the recall (0.016 -> 0.008), so the
        # question is no longer whether it costs anything but where the knee
        # is. Sweeping it alongside the other components puts a number on
        # every value rather than only the shipped one.
        #
        # Read with care: goodbooks rewards series-completion, so "another
        # book by an author you liked" is the single most predictable thing in
        # the dataset. The cap refuses exactly that. A high cost here is the
        # metric disagreeing with a deliberate product choice, not evidence
        # the choice is wrong.
        CONFIGS = [
            ("full (protocol)", None),
            ("author cap = 1", {"per_author_cap": 1}),
            ("author cap = 2", {"per_author_cap": 2}),
            ("author cap = 3", {"per_author_cap": 3}),
            ("author-adj penalty on", {"author_adjacent_penalty": 0.30}),
            ("no tag layer", {"tag_weight": 0.0}),
            ("no title penalty", {"title_token_penalty": 0.0}),
        ]
        for name, params in CONFIGS:
            merged = dict(base_params)
            merged.update(params or {})
            s, _ = eval_users(users, df, emb, index, tags_df, rec,
                              params=merged, label=f"{name}: ", folds=args.folds)
            s.pop("_per_user", None)
            rows.append((name, s))
    else:
        s, _ = eval_users(users, df, emb, index, tags_df, rec,
                          params=base_params, label="new: ", folds=args.folds)
        s.pop("_per_user", None)
        rows.append(("new", s))
    rows.append(("pop", eval_pop(users, df)))

    print("\n" + "=" * 78)
    print(f"{'method':<22}{'recall@10':>14}{'recall@20':>14}{'recall@50':>14}{'hit@20':>12}")
    print("-" * 78)
    for name, r in rows:
        print(f"{name:<22}"
              f"{r['r10'][0]:>9.3f}±{r['r10'][1]:<4.3f}"
              f"{r['r20'][0]:>9.3f}±{r['r20'][1]:<4.3f}"
              f"{r['r50'][0]:>9.3f}±{r['r50'][1]:<4.3f}"
              f"{r['hit20'][0]:>12.3f}")
    print("=" * 78)
    print("± is the standard error; gaps larger than ~2x the combined stderr")
    print("are real. 'pop' is the floor — on MovieLens the movie ranker does")
    print("not clear it, and there is no reason to assume books will either.")


if __name__ == "__main__":
    main()
