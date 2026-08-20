"""
probe_kids_filter.py — is the certification filter actually doing anything?

THE PROBLEM THIS SOLVES
The filter is a deliberate no-op when the corpus has no cert_tier column, so
that the parquet upload and the constant flip can happen in either order
without crashing the movies tab. That safety property has a cost: "M_CORPUS
was never flipped", "the CDN is still serving the old parquet" and "the filter
ran correctly and this profile has no kids films to remove" all look EXACTLY
the same from the app. You cannot tell by looking at recommendations.

So ask the ranker directly: run the same profile twice, once with the filter
on and once off, and diff.

Reuses probe_genre_tabs' loaders so it reads the same deployed app.py.

USAGE
    python tests/probe_kids_filter.py tests/movies_ratings.json
    python tests/probe_kids_filter.py tests/movies_ratings.json --n 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_genre_tabs import load_artifacts, load_ranker, load_ratings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ratings")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ls-key", default="re_movies")
    ap.add_argument("--drop-kids-ratings", action="store_true",
                    help="strip positively-rated children's films from the "
                         "profile before ranking. Proves the ON path against "
                         "your real profile instead of needing a second one.")
    args = ap.parse_args()

    ratings = load_ratings(Path(args.ratings), args.ls_key)
    ns = load_ranker()
    df, index, emb, tags = load_artifacts(ns)
    rec = ns["recommend_movies"]

    # --- CHECK 1: is the enriched corpus the one being served? ---------------
    print("\n" + "=" * 66)
    print("CHECK 1 — corpus")
    print("=" * 66)
    if "cert_tier" not in df.columns:
        print("  cert_tier ABSENT from the loaded corpus.")
        print("  The filter is inert. Either M_CORPUS still names the old")
        print("  parquet, or the flip is not deployed, or the CDN is serving a")
        print("  cached copy. Nothing below will show a difference.")
        sys.exit(1)
    vc = df["cert_tier"].value_counts()
    print(f"  cert_tier present. Distribution over {len(df):,} rows:")
    for t, c in vc.items():
        print(f"    {t:10s} {c:7,}")
    n_kids = int((df["cert_tier"] == "all_ages").sum())
    print(f"  all_ages films in corpus: {n_kids:,}")

    # cert_tier alone over-counts: G predates PG-13, so 2001: A Space Odyssey
    # and much of the 60s-70s catalogue are US:G without being children's
    # films. The suppressible class is certification AND genre.
    is_kids = ns["_is_kids_film"]
    kids_mask = df.apply(is_kids, axis=1)
    print(f"  of those, ALSO Animation/Family: {int(kids_mask.sum()):,}"
          f"  <- the actually-suppressible set")
    excluded = df[(df["cert_tier"] == "all_ages") & (~kids_mask)]
    if len(excluded):
        print(f"  G-rated but NOT children's films ({len(excluded):,}), e.g.:")
        for t in excluded.nlargest(6, "vote_count")["title"]:
            print(f"    {str(t)[:50]}")

    # --- CHECK 2: does this profile trigger suppression? ---------------------
    print("\n" + "=" * 66)
    print("CHECK 2 — profile")
    print("=" * 66)
    like_threshold = 3.5
    liked_tiers = {}
    liked_kids = []
    for r in ratings:
        ci = r.get("corpus_index")
        if ci is None or not (0 <= int(ci) < len(df)):
            continue
        if r["rating"] >= like_threshold:
            row = df.iloc[int(ci)]
            t = str(row["cert_tier"])
            liked_tiers[t] = liked_tiers.get(t, 0) + 1
            if is_kids(row):
                liked_kids.append(str(row["title"]))
    print(f"  positively-rated films by tier: {liked_tiers}")
    print(f"  positively-rated CHILDREN'S films: {len(liked_kids)}"
          f"{'  ' + str(liked_kids[:5]) if liked_kids else ''}")
    liked_n = sum(1 for r in ratings
                  if r.get("corpus_index") is not None
                  and 0 <= int(r["corpus_index"]) < len(df)
                  and r["rating"] >= like_threshold)
    ratio = len(liked_kids) / liked_n if liked_n else 0.0
    kmin = ns.get("MOVIE_KIDS_LIKE_MIN", 1)
    kratio = ns.get("MOVIE_KIDS_LIKE_RATIO", 0.0)
    print(f"  children's share of liked set: {len(liked_kids)}/{liked_n} = {ratio:.1%}")
    print(f"  gate: >= {kmin} films OR >= {kratio:.0%} share")
    will_suppress = not (len(liked_kids) >= kmin or ratio >= kratio)
    print(f"  -> suppression {'ON' if will_suppress else 'OFF'}")
    if not will_suppress:
        print("     This profile likes all-ages films, so the filter correctly")
        print("     stands down. To test the ON path, use a profile with no")
        print("     positively-rated G/U/TV-Y film.")

    # --- CHECK 3: A/B the ranker --------------------------------------------
    print("\n" + "=" * 66)
    print("CHECK 3 — ranker A/B")
    print("=" * 66)
    run_ratings = ratings
    if args.drop_kids_ratings and liked_kids:
        titles = set(liked_kids)
        run_ratings = [r for r in ratings
                       if not (r.get("corpus_index") is not None
                               and 0 <= int(r["corpus_index"]) < len(df)
                               and str(df.iloc[int(r["corpus_index"])]["title"]) in titles
                               and r["rating"] >= like_threshold)]
        print(f"  --drop-kids-ratings: removed {len(ratings) - len(run_ratings)} "
              f"rating(s) -> suppression forced ON")

    on, _ = rec(run_ratings, df, emb, index, tags_df=tags, n=args.n, kids_filter=True)
    off, _ = rec(run_ratings, df, emb, index, tags_df=tags, n=args.n, kids_filter=False)

    on_ids, off_ids = list(on["id"]), list(off["id"])
    removed = [i for i in off_ids if i not in set(on_ids)]
    added = [i for i in on_ids if i not in set(off_ids)]

    tier_of = dict(zip(df["id"], df["cert_tier"]))
    kids_of = dict(zip(df["id"], kids_mask))
    title_of = dict(zip(df["id"], df["title"]))

    print(f"  filter ON : {len(on_ids)} recs")
    print(f"  filter OFF: {len(off_ids)} recs")
    print(f"  removed by the filter: {len(removed)}")
    for i in removed[:12]:
        print(f"    - {str(title_of.get(i, i))[:44]:46s} [{tier_of.get(i)}]")
    print(f"  promoted into the gap : {len(added)}")
    for i in added[:12]:
        print(f"    + {str(title_of.get(i, i))[:44]:46s} [{tier_of.get(i)}]")

    kids_in_on = [i for i in on_ids if kids_of.get(i)]
    print(f"\n  all_ages films still in the ON list: {len(kids_in_on)}")
    for i in kids_in_on[:8]:
        print(f"    ! {str(title_of.get(i, i))[:44]}")

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    if not will_suppress:
        print("  Filter stood down for this profile (by design). Not a failure —")
        print("  but this run does NOT prove the ON path works. Re-run against a")
        print("  profile with no positively-rated all-ages film.")
    elif removed:
        print(f"  WORKING. {len(removed)} all-ages film(s) removed from the top "
              f"{args.n}.")
        if kids_in_on:
            print("  BUT some all_ages films survived — check the tier values above.")
    else:
        print("  Suppression was ON but removed NOTHING. Either this profile's")
        print(f"  top {args.n} contained no all-ages films anyway (likely for an")
        print("  adult-heavy profile — try --n 100), or the filter is not wired in.")
        print("  Re-run with a larger --n before concluding it is broken.")


if __name__ == "__main__":
    main()
