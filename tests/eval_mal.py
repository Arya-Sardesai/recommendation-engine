"""
eval_mal.py — multi-user recall evaluation for the TV ranker, over the anime
slice, using the MyAnimeList ratings dump.

WHY THIS EXISTS
TV was the only tab with no benchmark. eval_tv_loo.py closed half the gap:
leave-one-out over one profile, 70 holdouts, which produced a headline number
(hit@20 0.029 vs a popularity baseline of 0.143) but an unreadable ablation --
two hits cannot support a t-test. Movies have 100 MovieLens users; books have
200 goodbooks users. This is TV's equivalent.

WHY ANIME AND NOT ALL TV
There is no public per-user ratings dataset keyed to TMDB tv ids. The Kaggle
"TV ratings" sets are aggregate IMDb scores, which cannot produce a held-out
relevance judgement. MAL is the one large interaction dataset in this space:
~57M ratings, ~310K users, ~16.9K titles. It covers the 11.8% of this corpus
that is anime -- the single largest keyword in the TV vocabulary, and the slice
where the corpus is richest (shounen, seinen, mecha, slice-of-life are all
declared tags).

READ THE NUMBER AS WHAT IT IS: anime recall, not TV recall. The two are not
interchangeable, and a ranker that does well here may do nothing for a viewer
whose taste is prestige drama.

THE LINK PROBLEM
MAL carries its own ids; the corpus is TMDB. There is no shared key, so titles
are matched on normalised name, trying MAL's English title, main title and
synonyms in turn. That matching is lossy and the loss is NOT random -- long
shounen franchises with many aliases match better than one-season originals --
so the reported numbers describe the matched subset. The link stage prints its
own match rate for exactly that reason.

THE CACHE IS STAMPED. goodbooks_links.json caches goodbooks ids against corpus
ROW INDICES with no validity check, which means it silently describes the wrong
titles the moment the corpus changes -- and the books corpus just gained 882
rows. This cache stores TMDB IDS, not positions, and stamps the corpus length
so a mismatch rebuilds instead of lying.

DATA (download once, put under data/raw/tv/mal/)
    https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020
  or the 2023 refresh. Either works; the column names differ and are detected.
  Needs: an anime metadata csv (ids + titles) and a ratings csv
  (user_id, anime_id, rating).

USAGE
    python tests\\eval_mal.py --mal-dir data\\raw\\tv\\mal --stage inspect
    python tests\\eval_mal.py --mal-dir data\\raw\\tv\\mal --stage link
    python tests\\eval_mal.py --mal-dir data\\raw\\tv\\mal --users 200
    python tests\\eval_mal.py --mal-dir data\\raw\\tv\\mal --users 200 --ablate
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
LINK_CACHE = REPO_ROOT / "tests" / "mal_links.json"

SEED = 17
MIN_RATED = 15      # below this there is too little train signal
MAX_RATED = 400     # above this is a cataloguer, not a viewer
LIKE = 8.0          # MAL is a 1-10 scale; 8+ is "liked"

# --- PROTOCOL --------------------------------------------------------------
# Same reasoning as eval_goodbooks' SCORING_PROTOCOL. The deployed ranker has
# suppressors that a held-out-title metric reads as failures:
#   exclude_rated_creators  drops everything by a studio/creator already rated
#   genre_cap               caps one genre family in the output
#   min_votes               most of the corpus is below the serving floor, and
#                           an unreachable holdout measures the floor, not the
#                           ranking (eval_tv_loo scored 2/70 before this)
# Pass --as-deployed to measure what a user actually sees instead.
SCORING_PROTOCOL = dict(
    exclude_rated_creators=False,
    genre_cap=10 ** 9,
    min_votes=0,
)

ABLATIONS = [
    ("full",                {}),
    ("no tag layer",        {"tag_weight": 0.0}),
    ("no quality floor",    {"min_vote_average": 0.0, "quality_weight": 0.0}),
    ("no maturity filter",  {"maturity_drop": 99}),
    ("no title penalty",    {"title_token_penalty": 0.0}),
    ("no negatives",        {"neg_scale": 0.0}),
    ("flat (no diversity)", {"flat_ranking": True}),
]


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _core(t):
    """Drop season/part markers so 'Attack on Titan Season 3' meets 'Attack on
    Titan'. Aggressive on purpose: TMDB holds one row per SERIES while MAL holds
    one per SEASON, so without this the two catalogues barely intersect."""
    t = _norm(t)
    t = re.sub(r"\b(season|series|part|cour)\s*\d+\b", " ", t)
    t = re.sub(r"\b(2nd|3rd|\d+th)\s+season\b", " ", t)
    t = re.sub(r"\bs\d+\b", " ", t)
    t = re.sub(r"\b(tv|ova|ona|special|movie|film)\b", " ", t)
    return re.sub(r"^(the|a|an) ", "", re.sub(r"\s+", " ", t)).strip()


def find_files(mal_dir):
    """Locate the metadata and ratings csvs without hardcoding a release."""
    mal_dir = Path(mal_dir)
    if not mal_dir.exists():
        sys.exit(f"{mal_dir} not found -- download the MAL dump there first")
    csvs = sorted(mal_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"no csv files in {mal_dir}")
    meta = ratings = None
    for p in csvs:
        head = pd.read_csv(p, nrows=5)
        cols = {c.lower().strip() for c in head.columns}
        has_user = any("user" in c for c in cols)
        has_rating = any(c in ("rating", "score", "my_score") for c in cols)
        has_title = any(c in ("name", "title", "english name", "anime title",
                              "english_name") for c in cols)
        if has_user and has_rating and ratings is None:
            ratings = p
        elif has_title and meta is None:
            meta = p
    return meta, ratings, csvs


def stage_inspect(mal_dir):
    meta, ratings, csvs = find_files(mal_dir)
    print(f"csv files found: {len(csvs)}\n")
    for p in csvs:
        head = pd.read_csv(p, nrows=3)
        role = ("METADATA" if p == meta else
                "RATINGS" if p == ratings else "")
        print(f"  {p.name:<42} {role}")
        print(f"    columns: {list(head.columns)[:12]}")
    print()
    if not meta or not ratings:
        print("  Could not identify both files automatically. Pass them")
        print("  explicitly with --meta and --ratings-csv.")
    else:
        print(f"  will link on : {meta.name}")
        print(f"  will eval on : {ratings.name}")


def pick(df, *names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def build_links(meta_csv, df, relink=False):
    """MAL anime_id -> TMDB tmdb_id, on normalised title.

    Stores TMDB IDS, never row positions, and stamps the corpus length. A cache
    keyed to positions is wrong the moment a row is appended and says nothing
    about it -- which is the live defect in goodbooks_links.json.
    """
    if LINK_CACHE.exists() and not relink:
        c = json.loads(LINK_CACHE.read_text(encoding="utf-8"))
        if c.get("_corpus_rows") == len(df):
            links = {int(k): int(v) for k, v in c["links"].items()}
            print(f"  link cache: {len(links):,} titles (--relink to rebuild)")
            return links
        print(f"  link cache stale (built against {c.get('_corpus_rows')} rows, "
              f"corpus now {len(df)}) -- rebuilding")

    m = pd.read_csv(meta_csv)
    id_col = pick(m, "mal_id", "anime_id", "id")
    # The 2018 dump names these title_english / title_synonyms; the 2020 Kaggle
    # re-upload uses English name / Other name. Looking for only one set silently
    # falls back to the romaji title alone, which is how the first run linked
    # 16.6% -- MAL leads with Shingeki no Kyojin while the corpus holds
    # Attack on Titan, and they never meet.
    title_cols = [c for c in (
        pick(m, "title_english", "english name", "english_name"),
        pick(m, "title", "name", "anime title"),
        pick(m, "title_synonyms", "other name", "synonyms"),
        pick(m, "title_japanese", "japanese name"),
    ) if c]
    if not id_col or not title_cols:
        sys.exit(f"could not find id/title columns in {meta_csv.name}: "
                 f"{list(m.columns)[:12]}")
    # A Movie, OVA, ONA, Special or Music entry cannot match a TV corpus, so
    # counting them in the denominator understates the link rate on the only
    # population that matters here.
    type_col = pick(m, "type")
    if type_col:
        before = len(m)
        kinds = m[type_col].astype(str).str.strip().str.lower()
        m = m[kinds == "tv"]
        print(f"  MAL metadata: {before:,} entries -> {len(m):,} TV series "
              f"(dropped {before - len(m):,} Movie/OVA/ONA/Special/Music)")
    else:
        print(f"  MAL metadata: {len(m):,} entries (no type column to filter on)")
    print(f"  id={id_col} | title fields={title_cols}")

    # _norm strips every non-ASCII character, so a title in Japanese, Korean,
    # Chinese or Arabic script normalises to an EMPTY STRING. The corpus is ~57%
    # non-English, so many series share that empty key -- and every MAL
    # title_japanese then "matched" whichever show claimed it first. The first
    # run of this fix reported 93.9% linked with 1,072 matches from
    # title_japanese: the Wikipedia failure again, a hit rate that was one
    # document. Keys shorter than MIN_KEY carry no identity and are never used.
    MIN_KEY = 3

    def _key(t):
        k = _core(t)
        return k if len(k) >= MIN_KEY else None

    by_core = {}
    for tid, name in zip(df["tmdb_id"], df["name"]):
        k = _key(name)
        if k:
            by_core.setdefault(k, int(tid))
    if "original_name" in df.columns:
        for tid, name in zip(df["tmdb_id"], df["original_name"]):
            k = _key(name)
            if k:
                by_core.setdefault(k, int(tid))

    links, by_field = {}, {c: 0 for c in title_cols}
    for row in m.itertuples(index=False):
        rec = row._asdict()
        mal_id = rec.get(id_col)
        if pd.isna(mal_id):
            continue
        for c in title_cols:
            v = rec.get(c)
            if not isinstance(v, str):
                continue
            for part in v.split(","):
                k = _key(part)
                if k is None:
                    continue
                hit = by_core.get(k)
                if hit is not None:
                    links[int(mal_id)] = hit
                    by_field[c] += 1
                    break
            if int(mal_id) in links:
                break

    LINK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LINK_CACHE.write_text(json.dumps(
        {"_corpus_rows": len(df), "links": {str(k): v for k, v in links.items()}}),
        encoding="utf-8")
    print(f"  linked {len(links):,} of {len(m):,} ({len(links)/max(len(m),1):.1%})"
          f"  by field: { {k: v for k, v in by_field.items() if v} }")

    # THE CHECK. A link rate says nothing about whether the links are distinct.
    # If many MAL titles resolve to one TMDB id, the rate is inflated by
    # collapse, not coverage -- exactly how Wikipedia returned 19/19 "hits"
    # that were three documents.
    from collections import Counter as _C
    fan_in = _C(links.values())
    distinct = len(fan_in)
    worst = fan_in.most_common(5)
    print(f"  distinct TMDB targets: {distinct:,} for {len(links):,} links")
    # Calibrated against a real run. MAL splits seasons into separate entries
    # and TMDB does not, so N-to-1 is EXPECTED for long shows: Major's six
    # seasons correctly map to one series. The threshold of 5 fired on exactly
    # that. What the check exists to catch is the empty-key collapse, where a
    # thousand unrelated titles land on one show -- a different order of
    # magnitude. A warning that fires on correct data teaches you to ignore it.
    COLLAPSE_WARN = 15
    if worst and worst[0][1] > COLLAPSE_WARN:
        name_of = dict(zip(df["tmdb_id"].astype(int), df["name"]))
        print(f"  WARNING: more than {COLLAPSE_WARN} MAL titles on one show:")
        for tid, n in worst:
            if n > COLLAPSE_WARN:
                print(f"    {n:>5} -> {str(name_of.get(tid, tid))[:50]}")
        print("  Season splits never get this large. This is a broken key,")
        print("  not a long-running series -- fix it before trusting recall.")
    elif worst:
        print(f"  largest fan-in: {worst[0][1]} (season splits; expected)")

    # print real pairs, because this project's label bugs were all caught by
    # reading examples and never by reading a rate
    name_of = dict(zip(df["tmdb_id"].astype(int), df["name"]))
    title_of = dict(zip(m[id_col], m[title_cols[0]]))
    print("  sample links (read these):")
    for mal_id, tid in list(links.items())[:: max(1, len(links) // 10)][:10]:
        print(f"    {str(title_of.get(mal_id, mal_id))[:38]:40s} -> "
              f"{str(name_of.get(tid, tid))[:38]}")
    print("  NOTE: unmatched titles drop out of evaluation, and the loss is not")
    print("  random -- long franchises with many aliases match better than")
    print("  one-season originals. These numbers describe the matched subset.")
    return links


def load_users(ratings_csv, links, n_users, seed=SEED):
    print(f"  reading {Path(ratings_csv).name} (this is the big one) ...")
    rt = pd.read_csv(ratings_csv)
    u = pick(rt, "user_id", "username", "user")
    a = pick(rt, "anime_id", "mal_id", "id")
    s = pick(rt, "rating", "score", "my_score")
    if not all((u, a, s)):
        sys.exit(f"could not find user/anime/rating columns: {list(rt.columns)[:12]}")
    rt = rt[[u, a, s]].rename(columns={u: "user", a: "anime", s: "rating"})
    rt = rt[rt["anime"].isin(links)]
    rt = rt[rt["rating"] > 0]          # MAL uses 0 for "watched, not scored"
    sizes = rt.groupby("user").size()
    eligible = sizes[(sizes >= MIN_RATED) & (sizes <= MAX_RATED)].index
    print(f"  {len(rt):,} ratings on linked titles | {len(eligible):,} eligible "
          f"users ({MIN_RATED}-{MAX_RATED} ratings)")
    if not len(eligible):
        sys.exit("no eligible users -- the link rate is probably too low")
    rng = np.random.default_rng(seed)
    pick_users = rng.choice(np.array(eligible),
                            size=min(n_users, len(eligible)), replace=False)
    sub = rt[rt["user"].isin(pick_users)]
    return {uid: g for uid, g in sub.groupby("user")}


def collapse_seasons(g, links):
    """One row per TMDB SERIES, not per MAL entry.

    MAL splits a show into one entry per season; TMDB keeps one series with the
    seasons nested. Major's six seasons are six MAL ids and one TMDB id, which
    is a CORRECT link -- 160 of the 1,784 links are this kind of N-to-1.

    But splitting train/test on raw MAL rows breaks the evaluation two ways:
      1. season 1 lands in train and season 3 in test. Both are the same TMDB
         id, the ranker excludes it as already-rated, and the test item is a
         guaranteed miss that says nothing about the ranker.
      2. six seasons in train become six anchors on one show -- the series
         inflation defect the books backlog describes, reproduced inside the
         harness that is meant to measure it.

    Collapsing first, averaging the scores, fixes both.
    """
    g = g.copy()
    g["tmdb"] = [links.get(int(a)) for a in g["anime"]]
    g = g[g["tmdb"].notna()]
    out = (g.groupby("tmdb", as_index=False)
             .agg(rating=("rating", "mean"), anime=("anime", "first")))
    return out


def eval_users(users, df, emb, index, tags_df, rec, links, pos,
               params=None, k_list=(10, 20, 50), seed=SEED, label=""):
    params = params or {}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in k_list}
    per_user_r20 = []
    n_ok = skipped = 0
    maxk = max(k_list)

    for i, (uid, g) in enumerate(users.items(), 1):
        g = collapse_seasons(g, links)
        rows = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        cut = int(len(rows) * 0.7)
        train, test = rows.iloc[:cut], rows.iloc[cut:]
        rel = {links[int(a)] for a, r in zip(test["anime"], test["rating"])
               if r >= LIKE and int(a) in links}
        if not rel or len(train) < 5:
            skipped += 1
            per_user_r20.append(np.nan)
            continue

        ratings = []
        for a, r in zip(train["anime"], train["rating"]):
            tid = links.get(int(a))
            ci = pos.get(tid)
            if ci is None:
                continue
            ratings.append({"tv_id": tid, "corpus_index": ci,
                            "matched_title": str(df.iloc[ci]["name"]),
                            "matched_creator": "",
                            # MAL 1-10 -> the app's 1-5 scale
                            "rating": float(r) / 2.0})
        if len(ratings) < 5:
            skipped += 1
            per_user_r20.append(np.nan)
            continue

        try:
            recs, _ = rec(ratings, df, emb, index, tags_df=tags_df, n=maxk,
                          **params)
        except Exception as e:
            print(f"    user {uid} failed: {type(e).__name__}: {e}")
            skipped += 1
            per_user_r20.append(np.nan)
            continue

        got = [int(x) for x in recs["tmdb_id"].tolist()] if len(recs) else []
        for k in k_list:
            acc[k].append(len(set(got[:k]) & rel) / max(len(rel), 1))
        per_user_r20.append(len(set(got[:20]) & rel) / max(len(rel), 1))
        n_ok += 1
        if i % 25 == 0:
            print(f"  {label}[{i}/{len(users)}]")

    def ms(v):
        a = np.array(v) if v else np.array([0.0])
        return (float(a.mean()),
                float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)

    out = {f"r{k}": ms(acc[k]) for k in k_list}
    print(f"  {label}evaluated {n_ok} users ({skipped} skipped)")
    return out, np.array(per_user_r20, dtype=float)


def eval_pop(users, df, links, k_list=(10, 20, 50), within_linked=True):
    """The floor any recommender must clear -- but it must be a FAIR floor.

    Ranked over the whole TV corpus, the top 20 by vote_count are Breaking Bad,
    Game of Thrones and Friends. Every held-out MAL item is anime. So a
    whole-corpus popularity list scores 0.000 at recall@20 by construction, and
    beating it proves nothing. The first run of this harness reported exactly
    that: 0.000 against the ranker's 0.030, which reads as the ranker winning
    and is actually the baseline being unable to compete.

    within_linked=True ranks popularity inside the MAL-linked set only -- "the
    most popular anime" -- which is the honest comparison: the same candidate
    universe the held-out items come from.
    """
    vc = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    order = [int(t) for t in df.iloc[np.argsort(-vc.values)]["tmdb_id"]]
    if within_linked:
        linked = set(links.values())
        order = [t for t in order if t in linked]
    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in k_list}
    for uid, g in users.items():
        g = collapse_seasons(g, links)
        rows = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        cut = int(len(rows) * 0.7)
        train, test = rows.iloc[:cut], rows.iloc[cut:]
        rel = {links[int(a)] for a, r in zip(test["anime"], test["rating"])
               if r >= LIKE and int(a) in links}
        if not rel or len(train) < 5:
            continue
        seen = {links[int(a)] for a in train["anime"] if int(a) in links}
        top = [t for t in order if t not in seen][:max(k_list)]
        for k in k_list:
            acc[k].append(len(set(top[:k]) & rel) / max(len(rel), 1))

    def ms(v):
        a = np.array(v) if v else np.array([0.0])
        return (float(a.mean()),
                float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)
    return {f"r{k}": ms(acc[k]) for k in k_list}


def paired_t(full, abl):
    """Paired per USER, which is the correction that reversed the movie
    ablation's conclusions. Unpaired, the diversity round-robin looked
    expensive; paired it was noise at t=0.4."""
    m = ~(np.isnan(full) | np.isnan(abl))
    d = full[m] - abl[m]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float(d.mean()) if len(d) else 0.0, 0.0
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mal-dir", required=True)
    ap.add_argument("--stage", choices=["inspect", "link", "eval"], default="eval")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--ratings-csv", default=None)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--relink", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--as-deployed", action="store_true",
                    help="measure the shipped config instead of the scoring")
    args = ap.parse_args()

    if args.stage == "inspect":
        stage_inspect(args.mal_dir)
        return

    ns = load_ranker(SPACE_APP, require=("recommend_tv",))
    art = resolve_artifacts(SPACE_APP, PROCESSED, kind="tv", verbose=False)

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, (
        f"ALIGNMENT: parquet {len(df)}, npy {emb.shape[0]}, index {index.ntotal}")
    print(f"  corpus {len(df):,} series, aligned")

    tags_df = None
    if "TAGS" in art and Path(art["TAGS"]).exists():
        t = pd.read_parquet(art["TAGS"])
        t["tv_id"] = t["tv_id"].astype(str)
        tags_df = t.pivot_table(index="tv_id", columns="tag", values="score",
                                fill_value=0.0)
        print(f"  tags: {tags_df.shape[0]:,} series")

    meta, ratings_csv, _ = find_files(args.mal_dir)
    if args.meta:
        meta = Path(args.meta)
    if args.ratings_csv:
        ratings_csv = Path(args.ratings_csv)
    if not meta or not ratings_csv:
        sys.exit("could not identify the csvs -- run --stage inspect")

    links = build_links(meta, df, relink=args.relink)
    if args.stage == "link":
        return

    pos = {int(t): i for i, t in enumerate(df["tmdb_id"])}
    users = load_users(ratings_csv, links, args.users)
    print(f"  sampled {len(users)} users\n")

    base = {} if args.as_deployed else dict(SCORING_PROTOCOL)
    print("  protocol:", "AS DEPLOYED" if args.as_deployed
          else "SCORING ONLY (suppressors and serving floor lifted)")

    rec = ns["recommend_tv"]
    configs = ABLATIONS if args.ablate else [("full", {})]
    rows, vectors = [], {}
    for name, p in configs:
        merged = dict(base)
        merged.update(p)
        print(f"\nrunning: {name}")
        s, v = eval_users(users, df, emb, index, tags_df, rec, links, pos,
                          params=merged, label=f"{name}: ")
        rows.append((name, s))
        vectors[name] = v

    pop = eval_pop(users, df, links, within_linked=True)
    pop_all = eval_pop(users, df, links, within_linked=False)

    print("\n" + "=" * 82)
    print(f"{'method':<24}{'recall@10':>13}{'recall@20':>13}{'recall@50':>13}"
          f"{'dR20':>10}{'t':>7}")
    print("-" * 82)
    for name, r in rows:
        line = (f"{name:<24}"
                f"{r['r10'][0]:>8.3f}±{r['r10'][1]:<4.3f}"
                f"{r['r20'][0]:>8.3f}±{r['r20'][1]:<4.3f}"
                f"{r['r50'][0]:>8.3f}±{r['r50'][1]:<4.3f}")
        if name != "full" and "full" in vectors:
            d, t = paired_t(vectors["full"], vectors[name])
            line += f"{d:>+10.4f}{t:>7.1f}" + ("  <<" if abs(t) >= 2 else "")
        print(line)
    print("-" * 82)
    print(f"{'pop: most popular anime':<24}"
          f"{pop['r10'][0]:>8.3f}±{pop['r10'][1]:<4.3f}"
          f"{pop['r20'][0]:>8.3f}±{pop['r20'][1]:<4.3f}"
          f"{pop['r50'][0]:>8.3f}±{pop['r50'][1]:<4.3f}"
          f"   <- the fair baseline")
    print(f"{'pop: whole corpus':<24}"
          f"{pop_all['r10'][0]:>8.3f}±{pop_all['r10'][1]:<4.3f}"
          f"{pop_all['r20'][0]:>8.3f}±{pop_all['r20'][1]:<4.3f}"
          f"{pop_all['r50'][0]:>8.3f}±{pop_all['r50'][1]:<4.3f}"
          f"   <- strawman, shown so it is not mistaken for a win")
    print("=" * 82)
    print("dR20 is the PAIRED per-user difference vs full. Positive = removing")
    print("it hurt. |t| >= 2 marked <<; below that is noise however large the")
    print("recall gap looks.")
    print()
    print("This is ANIME recall, not TV recall. Movies lose to popularity")
    print("(0.043 vs 0.080), books lose worse (0.016 vs 0.056), and TV's")
    print("single-profile harness lost 5x. If this loses too, that is the")
    print("expected shape for a content-only ranker -- publish it.")


if __name__ == "__main__":
    main()
