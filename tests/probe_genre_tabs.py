"""
probe_genre_tabs.py — does bucketing recommendations by genre produce tabs that
are actually DIFFERENT, or four names for the same list?

THE QUESTION
A coherent taste produces correlated genres. If a profile's dominant buckets are
crime / thriller / action, their candidate sets may be near-identical, in which
case genre tabs are theatre: either they overlap heavily (same list four times)
or you force worse matches into tabs 3 and 4 to keep them disjoint. Neither is
worth building. Measure before designing.

WHAT IT DOES
  1. pulls a DEEP ranked candidate pool from the live ranker (flat, caps off)
  2. runs the FALSIFIER first: is "first genre" real, or just TMDB id order?
  3. buckets candidates multi-label and by primary genre
  4. reports pairwise Jaccard overlap between buckets
  5. greedily selects buckets that are distinct enough -> "how many tabs does
     this profile actually support?"

READS THE LIVE RANKER. recommend_movies is AST-extracted from $space/app.py, so
this stays in sync with whatever is deployed (same trick as the eval harnesses).

TWO THINGS TO KNOW BEFORE READING THE OUTPUT
  * The deployed tabs apply MOVIE_TAB_VOTE_CEILING (8000) and this probe does
    not, unless you pass --vote-ceiling. Without it, bucket sizes here are
    larger than what ships and the viability test measures a pool the app does
    not serve. Pass --vote-ceiling 8000 to match production.
  * Any change to MOVIE_MIN_RUNTIME changes the candidate set, so numbers from
    this probe are not comparable across a runtime-floor change. Decide the
    floor first, then run this once.

USAGE
    python tests/probe_genre_tabs.py tests/movies_ratings.json
    python tests/probe_genre_tabs.py tests/movies_ratings.json --per-tab 10 --pool 400
    python tests/probe_genre_tabs.py tests/movies_ratings.json --vote-ceiling 8000
"""
import argparse
import ast
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker_loader import load_ranker as _load_ranker  # noqa: E402
from ranker_loader import resolve_artifacts as _resolve  # noqa: E402

# Per-modality column names and ranker entry points. The three corpora use
# different id/title columns, which is the only thing that differs here.
KINDS = {
    "movie": {"ranker": "recommend_movies", "id": "id", "title": "title",
              "prefix": "M_", "label": "films"},
    "tv":    {"ranker": "recommend_tv", "id": "tmdb_id", "title": "name",
              "prefix": "T_", "label": "series"},
}

# TMDB's canonical genre ids. Used to test whether the first genre in a film's
# list carries information or is just ascending id order.
# TMDB uses a SEPARATE genre vocabulary for TV: merged buckets
# ("Action & Adventure", "Sci-Fi & Fantasy") and ids that do not overlap the
# film list. Testing TV genre order against film ids would silently score 0%
# ascending and wrongly declare the order meaningful.
TMDB_ID_TV = {"Action & Adventure": 10759, "Animation": 16, "Comedy": 35,
              "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
              "Kids": 10762, "Mystery": 9648, "News": 10763, "Reality": 10764,
              "Sci-Fi & Fantasy": 10765, "Soap": 10766, "Talk": 10767,
              "War & Politics": 10768, "Western": 37}

TMDB_ID = {"Adventure": 12, "Fantasy": 14, "Animation": 16, "Drama": 18,
           "Horror": 27, "Action": 28, "Comedy": 35, "History": 36,
           "Western": 37, "Thriller": 53, "Crime": 80, "Documentary": 99,
           "Science Fiction": 878, "Mystery": 9648, "Music": 10402,
           "Romance": 10749, "Family": 10751, "War": 10752, "TV Movie": 10770}


# Mirrors _genre_bucket inside recommend_movies. Kept in step with it: Crime and
# Thriller collapse to one bucket because "Action, Comedy, Crime" would
# otherwise launder past a first-genre cap.
def genre_list(g):
    """Normalise a genres cell to a list.

    _film_row returns a LIST for movies; the TV row builder passes the raw
    comma-separated STRING straight through. The multi-label section used to
    test isinstance(list) only, so every TV show read as having zero genres
    and the probe reported "not enough multi-label buckets" on a corpus where
    most shows carry two or three.
    """
    if isinstance(g, (list, tuple, set)):
        return [str(x).strip() for x in g if str(x).strip()]
    if isinstance(g, str):
        return [x.strip() for x in g.split(",") if x.strip()]
    return []


def genre_bucket(genres_str):
    gs = genres_str if isinstance(genres_str, str) else ""
    gset = {g.strip() for g in gs.split(",") if g.strip()}
    if gset & {"Crime", "Thriller"}:
        return "crime-thriller"
    first = gs.split(",")[0].strip() if gs else ""
    return first or "_none"


class _StubST:
    """Streamlit stand-in so app.py's top-level defs can be exec'd headlessly.

    Only the cache decorators matter here; they appear both bare
    (@st.cache_resource) and called (@st.cache_data(show_spinner=False)), so
    each form has to work. Everything else becomes a no-op.
    """
    def _cache(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]                      # bare decorator
        return lambda f: f                   # called decorator
    cache_data = cache_resource = property(lambda self: self._cache)

    def __getattr__(self, name):
        return lambda *a, **k: None


def load_ranker(kind="movie"):
    """Thin wrapper over ranker_loader so all four harnesses share one loader."""
    need = (KINDS[kind]["ranker"], "IndexVectors")
    return _load_ranker(SPACE_APP, require=need)


def load_artifacts(ns, kind="movie"):
    """Local data/processed first; fall back to the HF dataset repo."""
    art = _resolve(SPACE_APP, PROCESSED, kind=kind)

    def path_for(p):
        if p.exists():
            return str(p)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id="Arya2305/recommendation-engine-data",
                               filename=p.name, repo_type="dataset")

    import faiss
    df = pd.read_parquet(path_for(art["CORPUS"])).reset_index(drop=True)
    index = faiss.read_index(path_for(art["FAISS"]))
    tags = None
    if "TAGS" in art:
        try:
            t = pd.read_parquet(path_for(art["TAGS"]))
            key = "movie_id" if kind == "movie" else "tv_id"
            if key not in t.columns:
                key = t.columns[0]
            tags = t.pivot_table(index=key, columns="tag", values="score",
                                 fill_value=0.0)
        except Exception as e:
            print(f"  (no tags: {e})")
    assert len(df) == index.ntotal, f"ALIGNMENT: {len(df)} rows vs {index.ntotal} index"
    print(f"  corpus {len(df):,}  index {index.ntotal:,}  aligned")
    return df, index, ns["IndexVectors"](index), tags


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def load_ratings(path, ls_key="re_movies", kind="movie"):
    """Accept a plain ratings list OR a raw localStorage dump.

    The app persists through streamlit-local-storage, so an exported dump is
    {"re_movies": "<json string>", "re_books": "...", ...} -- keys from
    LS_KEYS in app.py, and each VALUE is itself a JSON string needing a second
    parse. Silently returning [] here produced an empty pool and a confusing
    KeyError three steps later, so validate up front instead.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = None

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        # localStorage dump: prefer the movie-ratings key
        _alts = ((ls_key, "ratings", "tv_ratings", "tv", "shows", "items", "data")
                 if kind == "tv" else
                 (ls_key, "ratings", "movie_ratings", "movies", "items", "data"))
        for key in _alts:
            v = raw.get(key)
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    v = None
            if isinstance(v, list) and v:
                items = v
                break
        if items is None:
            for v in raw.values():
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except json.JSONDecodeError:
                        continue
                if isinstance(v, list) and v and isinstance(v[0], dict) and "rating" in v[0]:
                    items = v
                    break

    if not items:
        keys = list(raw) if isinstance(raw, dict) else type(raw).__name__
        sys.exit(f"no ratings found in {path} under {ls_key!r}.\n"
                 f"Keys present: {keys}\n"
                 f"Pass --ls-key to name the right one.")

    need = ({"rating", "corpus_index", "tv_id"} if kind == "tv"
            else {"rating", "corpus_index", "movie_id"})
    missing = need - set(items[0])
    if missing:
        sys.exit(f"ratings entries are missing {sorted(missing)}.\n"
                 f"First entry has: {sorted(items[0])}\n"
                 f"(If these are BOOK ratings, pass --ls-key re_movies.)")
    return items


def report_id_order(genre_lists, id_map=None):
    """THE FALSIFIER — run this before reading anything else.

    Primary-genre bucketing assumes the first genre in a film's list is the
    most relevant one. If TMDB simply returns genres sorted by id, "first
    genre" means Adventure(12) always beats Comedy(35) beats Crime(80), and
    every primary-genre bucket is an artefact of the id numbering rather than
    a statement about the film.
    """
    id_map = id_map or TMDB_ID
    asc = tot = 0
    for g in genre_lists:
        gl = genre_list(g)
        if len(gl) > 1:
            ids = [id_map.get(x, 0) for x in gl]
            tot += 1
            asc += int(ids == sorted(ids))
    print("=" * 68)
    print("FALSIFIER: is 'first genre' meaningful, or just TMDB id order?")
    if not tot:
        print("  no multi-genre films in the pool — cannot test.")
        print("=" * 68 + "\n")
        return None
    frac = asc / tot
    print(f"  multi-genre films whose order is ascending by TMDB id: "
          f"{asc}/{tot} ({frac:.0%})")
    if frac >= 0.90:
        print("  >> ~100%: 'first genre' IS id order. Primary-genre bucketing is")
        print("     arbitrary and the tabs below are labelling an artefact.")
        print("     STOP AND READ THIS BEFORE THE VERDICT AT THE BOTTOM.")
    elif frac < 0.60:
        print("  >> below 60%: the order carries real information.")
        print("     Primary-genre bucketing is sound.")
    else:
        print("  >> ambiguous (60-90%). Partly id order, partly not. Spot-check")
        print("     the sample titles below before trusting the buckets.")
    print("=" * 68 + "\n")
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings", help="ratings JSON (same file eval_loo uses)")
    ap.add_argument("--kind", choices=["movie", "tv"], default="movie",
                    help="which tab to probe")
    ap.add_argument("--per-tab", type=int, default=10)
    ap.add_argument("--overall", type=int, default=20)
    ap.add_argument("--pool", type=int, default=400, help="deep candidate pool size")
    ap.add_argument("--max-tabs", type=int, default=4)
    ap.add_argument("--ls-key", default="re_movies",
                    help="key to read from a localStorage dump")
    ap.add_argument("--distinct-at", type=float, default=0.35,
                    help="max Jaccard vs an already-chosen tab to still count as distinct")
    ap.add_argument("--vote-ceiling", type=int, default=0,
                    help="drop candidates at/above this vote_count, matching "
                         "MOVIE_TAB_VOTE_CEILING in app.py (8000). 0 = off, "
                         "which measures a LARGER pool than the app serves.")
    args = ap.parse_args()

    K = KINDS[args.kind]
    ID, TITLE = K["id"], K["title"]
    if args.kind == "tv" and args.ls_key == "re_movies":
        args.ls_key = "re_tv"

    ratings = load_ratings(Path(args.ratings), args.ls_key, kind=args.kind)
    print(f"profile: {len(ratings)} rated {K['label']}\n")

    ns = load_ranker(args.kind)
    df, index, emb, tags = load_artifacts(ns, args.kind)
    rec = ns[K["ranker"]]

    # TV has a far smaller servable pool than movies: min_votes=300 leaves
    # ~1,750 of 30,235 series. A 400-candidate pool is a quarter of everything
    # that can surface, so bucket sizes here are near the ceiling of what
    # exists rather than a sample of a deep corpus. Say so before the numbers
    # get read as if they were comparable to the movie probe.
    if "vote_count" in df.columns:
        floor = 300 if args.kind == "tv" else 400
        servable = int((df["vote_count"] >= floor).sum())
        print(f"  servable corpus (votes >= {floor}): {servable:,} of {len(df):,}"
              f"  ({servable/len(df):.1%})")
        if servable < args.pool * 5:
            print(f"  NOTE: pool of {args.pool} is {args.pool/servable:.0%} of the")
            print("  servable corpus. Tabs drawn from this are not deep cuts —")
            print("  they are most of what exists.")
    print()

    # --- Tab 1: exactly what the app shows today -----------------------------
    overall, _ = rec(ratings, df, emb, index, tags_df=tags, n=args.overall)
    overall_ids = set(overall[ID])
    print(f"Tab 1 (overall): {len(overall)} films")

    # --- Deep pool: caps off, flat, so bucketing sees the real ranking --------
    pool, _ = rec(ratings, df, emb, index, tags_df=tags, n=args.pool,
                  flat_ranking=True, genre_cap=10_000, per_anchor_cap=10_000)
    print(f"deep pool     : {len(pool)} films")

    if not len(pool):
        sys.exit("deep pool is empty — the ranker returned nothing for this profile.")

    # --- Deep-cut ceiling: match the deployed tabs ---------------------------
    # An absolute ceiling, not a per-bucket median. A relative cut removes half
    # of EVERY bucket including ones already obscure, which is how Action's
    # best pick fell from Hard Boiled to Stolen. "Already famous" is absolute.
    if args.vote_ceiling:
        if "vote_count" not in pool.columns:
            print("  WARNING: --vote-ceiling requested but the ranker did not "
                  "return vote_count; ceiling NOT applied.")
        else:
            before = len(pool)
            pool = pool[pool["vote_count"] < args.vote_ceiling].reset_index(drop=True)
            print(f"vote ceiling  : {before} -> {len(pool)} "
                  f"(dropped {before - len(pool)} at >= {args.vote_ceiling:,} votes)")
    else:
        print("vote ceiling  : OFF — this pool is LARGER than the app's tabs use.")
        print("                pass --vote-ceiling 8000 to match production.")
    print()

    # _film_row returns genres as a LIST; genre_bucket wants the raw string.
    pool = pool.assign(_bucket=[genre_bucket(", ".join(g) if isinstance(g, list) else (g or ""))
                                for g in pool["genres"]])

    pool_ex = pool[~pool[ID].isin(overall_ids)].reset_index(drop=True)
    genre_sets = [set(genre_list(g)) for g in pool_ex["genres"]]

    # --- FALSIFIER FIRST -----------------------------------------------------
    # Hoisted above the bucket output on purpose. Reading it last means every
    # encouraging number has already landed before the one that can invalidate
    # the whole design.
    report_id_order(pool_ex["genres"],
                    TMDB_ID_TV if args.kind == "tv" else TMDB_ID)

    counts = pool["_bucket"].value_counts()
    print("SINGLE-ASSIGNMENT buckets (ranker's own rule, one bucket per film):")
    for b, c in counts.head(12).items():
        print(f"   {b:22s} {c}")
    print("   ^ these are disjoint BY CONSTRUCTION, so their overlap is always 0.")
    print("     Useful for sizing, not for answering the overlap question.\n")

    # --- MULTI-LABEL: the design question actually being asked ---------------
    # A film carries several genres. If the Crime tab and the Action tab both
    # draw from the full genre list, do they surface the same films?
    all_genres = Counter(g for gs in genre_sets for g in gs)

    print(f"MULTI-LABEL genre mass in the pool (minus Tab 1, n={len(pool_ex)}):")
    for g, c in all_genres.most_common(12):
        print(f"   {g:22s} {c}")
    print()

    ml_tabs = {}
    for g, c in all_genres.most_common(10):
        if c < args.per_tab:
            continue
        mask = [g in gs for gs in genre_sets]
        ml_tabs[g] = list(pool_ex[mask][ID].head(args.per_tab))

    if len(ml_tabs) < 2:
        print("not enough multi-label buckets to compare.")
        return

    names = list(ml_tabs)[:8]
    print("MULTI-LABEL pairwise Jaccard (this is the number that matters):")
    print(f"{'':20s}" + "".join(f"{n[:9]:>11s}" for n in names))
    for a in names:
        row = "".join(f"{jaccard(ml_tabs[a], ml_tabs[b]):>11.2f}" if a != b else f"{'-':>11s}"
                      for b in names)
        print(f"{a[:20]:20s}{row}")
    print()

    # --- PRIMARY-GENRE: one film, one tab, using TMDB's FIRST genre only -----
    # No crime/thriller merge: that merge exists to stop the diversity cap
    # being laundered, which is a different job from labelling a tab.
    prim = [(genre_list(g)[0] if genre_list(g) else "_none") for g in pool_ex["genres"]]
    pool_ex = pool_ex.assign(_primary=prim)
    pcounts = pool_ex["_primary"].value_counts()
    print("PRIMARY-GENRE buckets (first genre, no merge, disjoint by design):")
    for g, c in pcounts.head(10).items():
        ok = "ok" if c >= args.per_tab * 2 else "thin"
        print(f"   {g:22s} {c:4d}  {ok}")
    viable = [g for g, c in pcounts.items() if c >= args.per_tab * 2 and g != "_none"]
    print(f"   -> {len(viable)} bucket(s) with >= {args.per_tab*2} films: {', '.join(viable[:6])}")
    if not args.vote_ceiling:
        print("   (measured WITHOUT the vote ceiling — the app's buckets are smaller)")
    print()
    for g in viable[:4]:
        t = pool_ex[pool_ex["_primary"] == g][TITLE].head(5).tolist()
        print(f"   {g:14s} {', '.join(x[:22] for x in t)}")
    print()

    pairs = [(jaccard(ml_tabs[a], ml_tabs[b]), a, b) for a, b in combinations(names, 2)]
    if pairs:
        worst = max(pairs)
        mean_ov = sum(x[0] for x in pairs) / len(pairs)
        print(f"highest overlap: {worst[1]} vs {worst[2]} = {worst[0]:.2f}")
        print(f"mean overlap   : {mean_ov:.2f}")

    chosen = []
    for g in names:
        if all(jaccard(ml_tabs[g], ml_tabs[c]) <= args.distinct_at for c in chosen):
            chosen.append(g)
        if len(chosen) == args.max_tabs:
            break
    print(f"\nVERDICT: {len(chosen)} distinct multi-label tab(s) at Jaccard <= {args.distinct_at}")
    for g in chosen:
        titles = pool_ex[pool_ex[ID].isin(ml_tabs[g])][TITLE].head(4).tolist()
        print(f"   {g:20s} e.g. {', '.join(t[:24] for t in titles)}")
    rejected = [g for g in names[:args.max_tabs + 2] if g not in chosen]
    if rejected:
        print(f"   too similar to a chosen tab: {', '.join(rejected)}")

    print("\nIf this comes back 1-2 tabs, genre tabs are theatre for this taste.")
    print("If 3-4, they are real. Run it on a friend's profile too — a broad")
    print("taste will support more tabs than a narrow one, and the feature")
    print("should adapt to that rather than always showing four.")
    print("\nAnd re-read the FALSIFIER block at the top before acting on this.")


if __name__ == "__main__":
    main()
