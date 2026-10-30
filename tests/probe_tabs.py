"""
probe_tabs.py -- which genre tabs the Films / TV page would show, under the old
ranker settings and the new ones, on your real profile.

    python tests\\probe_tabs.py
    python tests\\probe_tabs.py --kinds movie

The page builds its tabs like this: rank a big pool (400 films / 300 shows) in
flat mode, drop what is already on "For you", bucket the rest by FIRST genre,
and show the biggest buckets (up to MAX_TABS) that hold at least 2 x TAB_N
titles. This script repeats that for each setting and prints the bucket sizes,
so you can see why a tab appeared or disappeared.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import src.space_ranker as sr  # noqa: E402

CONFIGS = [
    ("old (floor off, idf off)", {"neg_floor_k": 0, "tag_idf": 0.0}),
    ("floor only", {"neg_floor_k": 200, "tag_idf": 0.0}),
    ("idf only", {"neg_floor_k": 0, "tag_idf": 1.0}),
    ("new (floor 200, idf 1)", {"neg_floor_k": 200, "tag_idf": 1.0}),
]
PAGE = {  # pool size, tab size, max tabs, id column -- as in app.py
    "movie": ("MOVIE_POOL_N", "MOVIE_TAB_N", "MOVIE_MAX_TABS", "id"),
    "tv": ("TV_POOL_N", "TV_TAB_N", "TV_MAX_TABS", "tmdb_id"),
}


def first_genre(g):
    if isinstance(g, (list, tuple)) and len(g):
        return str(g[0]).strip()
    if isinstance(g, str) and g.strip():
        return g.split(",")[0].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(sr.DEFAULT_PROFILE))
    ap.add_argument("--kinds", nargs="+", default=["movie", "tv"])
    args = ap.parse_args()
    raw = Path(args.profile).read_text(encoding="utf-8")

    for kind in args.kinds:
        b = sr.load(kind)
        ns = b["ns"]
        fn = ns[b["spec"]["rank"]]
        pool_c, tab_c, max_c, idc = PAGE[kind]
        pool_n, tab_n, max_tabs = ns[pool_c], ns[tab_c], ns[max_c]
        ratings, skipped = sr.read_profile(raw, kind)
        ratings, _ = sr.normalize_ratings(b, ratings)
        common = dict(tags_df=b["tags"], skip_ids=list(skipped or []),
                      awards=b.get("awards") or {})

        print(f"\n=== {kind.upper()}  pool {pool_n}, a tab needs >= {tab_n * 2}, "
              f"max {max_tabs} tabs " + "=" * 30)
        for label, ov in CONFIGS:
            recs, _ = fn(ratings, b["df"], b["emb"], b["index"], n=20, **common, **ov)
            pool, _ = fn(ratings, b["df"], b["emb"], b["index"], n=pool_n, flat_ranking=True,
                         genre_cap=10_000, per_anchor_cap=10_000, **common, **ov)
            shown = set(recs[idc]) if len(recs) else set()
            rest = pool[~pool[idc].isin(shown)] if len(pool) else pool
            counts = (rest["genres"].map(first_genre).value_counts()
                      if len(rest) else None)
            tabs = []
            if counts is not None:
                for g, c in counts.items():
                    if g and c >= tab_n * 2 and len(tabs) < max_tabs:
                        tabs.append(g)
            print(f"\n  {label}")
            print(f"    tabs:    {', '.join(tabs) or '(none)'}")
            if counts is not None:
                top = [f"{g or '(none)'} {c}" for g, c in counts.head(8).items()]
                print(f"    buckets: {', '.join(top)}")


if __name__ == "__main__":
    main()
