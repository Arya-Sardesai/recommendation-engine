"""
probe_books_profile.py — why is the books tab the weak one?

Three candidate causes, and they need completely different fixes. Guessing
between them is how a week gets spent on the wrong one.

  1. SERIES INFLATION (profile side). 40 rated volumes of one manga are 40
     anchors pointing at one neighbourhood, against 1 anchor each for ten other
     books. The list goes ~80% pirate manga and every other taste is drowned.
     Fix is profile-side, no re-embed: collapse liked books to one anchor per
     series, weighted by volumes rated.

  2. CORPUS THINNESS. The recommendations are reasonable given what is in the
     corpus, and the corpus simply lacks what this reader wants. Hardcover's
     coverage of Indian regional non-fiction is known thin. Fix is a harvest,
     not a ranker change.

  3. RANKING. Neither of the above; the candidates are there and well spread,
     and they are simply ordered badly.

This measures all three against a real profile and PRINTS THE TITLES, because
every label bug in this project was caught by reading examples and none were
visible in a distribution table.

READ-ONLY. No writes, no API.

USAGE
    python notebooks\\pipeline\\probe_books_profile.py --ratings tests\\movies_ratings.json
    python notebooks\\pipeline\\probe_books_profile.py --ratings tests\\movies_ratings.json --n 40
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"

# Volume markers. Unlike diagnose_book_works._core_title, which PRESERVES the
# volume number so Vol.1 and Vol.2 stay distinct works, this strips it: the
# whole point here is to collapse a series into one key and count how many of
# its volumes a profile carries.
_VOLUME = re.compile(r"\b(?:vol|volume|book|part|no|number)\s*\.?\s*\d+\b")
_TRAILING_NUM = re.compile(r",\s*\d+\s*$")
_HASH_NUM = re.compile(r"#\s*\d+")


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def series_key(title, author):
    """Collapse every volume of a series onto one key.

    Deliberately aggressive in the opposite direction from the work-id key:
    false merges here overstate series concentration, which is the safer error
    for a diagnostic. Verify by reading the grouped titles it prints.
    """
    t = re.sub(r"\([^)]*\)", " ", str(title))
    t = re.split(r":| - ", t)[0]
    t = _HASH_NUM.sub(" ", t)
    t = _VOLUME.sub(" ", _norm(t))
    t = _TRAILING_NUM.sub("", t).strip()
    t = re.sub(r"^(the|a|an) ", "", t).strip()
    return f"{t}||{_norm(author)}" if t else None


_NONFIC = {
    "nonfiction", "non-fiction", "memoir", "biography", "autobiography",
    "history", "true crime", "essays", "journalism", "travel", "science",
    "politics", "philosophy", "self help", "self-help", "business",
    "true story", "reference",
}


def is_nonfiction(genres):
    """True / False / None. Hardcover genres are a folksonomy, so this is a
    signal rather than a fact -- but a non-fiction anchor returning fiction is
    worth flagging even on a noisy label."""
    if genres is None:
        return None
    if isinstance(genres, str):
        gs = {g.strip().lower() for g in genres.split(",") if g.strip()}
    else:
        try:
            gs = {str(g).strip().lower() for g in genres}
        except TypeError:
            return None
    if not gs:
        return None
    if gs & _NONFIC:
        return True
    if "fiction" in gs or "novel" in gs or "novels" in gs:
        return False
    return None


def load_book_ratings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        v = raw.get("re_books")
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return v
    if isinstance(raw, list):
        return raw
    return []


def share(counter, total):
    if not counter or not total:
        return 0.0, None
    key, n = counter.most_common(1)[0]
    return n / total, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    ratings = load_book_ratings(Path(args.ratings))
    if not ratings:
        sys.exit("no re_books ratings in that dump. Export from the Space with "
                 "?debug=1 and copy the whole stored-values blob.")

    ns = load_ranker(SPACE_APP, require=("recommend_books", "_migrate_book_ratings"))
    art = resolve_artifacts(SPACE_APP, PROCESSED, kind="book", verbose=False)

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, (
        f"ALIGNMENT: parquet {len(df)}, npy {emb.shape[0]}, index {index.ntotal}")

    # The app runs stored ratings through this on hydrate and on import, because
    # corpus_index is a row position and the books corpus was swapped under it.
    # A harness that skips the migration is not measuring the deployed profile:
    # it either drops most of the ratings or dies on an out-of-range index.
    raw_n = len(ratings)
    ratings, _mig, _drop, _merge = ns["_migrate_book_ratings"](ratings, df)
    print(f"migration: {raw_n} stored -> {len(ratings)} usable "
          f"({_mig} re-matched by name, {_drop} no longer in the catalogue, "
          f"{_merge} merged)")
    if not ratings:
        sys.exit("nothing survived migration -- the export predates a corpus "
                 "swap and the titles no longer resolve.")

    tags_df = None
    if "TAGS" in art and Path(art["TAGS"]).exists():
        t = pd.read_parquet(art["TAGS"])
        t["book_id"] = t["book_id"].astype(str)
        tags_df = t.pivot_table(index="book_id", columns="tag", values="score",
                                fill_value=0.0)

    print(f"corpus {len(df):,} rows, aligned")
    print(f"profile {len(ratings)} rated books\n")

    # ---- 1. what the PROFILE looks like ------------------------------------
    print("=" * 72)
    print("1. PROFILE COMPOSITION  (is one series dominating the anchors?)")
    print("=" * 72)
    liked = [r for r in ratings
             if r.get("rating", 0) >= 3.5
             and r.get("corpus_index") is not None
             and 0 <= int(r["corpus_index"]) < len(df)]
    print(f"  positively rated (>= 3.5): {len(liked)} of {len(ratings)}")
    if not liked:
        sys.exit("  nothing positively rated -- the ranker has no anchors to work from.")

    prof_series, prof_authors = Counter(), Counter()
    series_titles = defaultdict(list)
    for r in liked:
        row = df.iloc[int(r["corpus_index"])]
        sk = series_key(row["title"], row.get("primary_author", ""))
        if sk:
            prof_series[sk] += 1
            series_titles[sk].append(str(row["title"]))
        prof_authors[_norm(row.get("primary_author", ""))] += 1

    s_share, s_key = share(prof_series, len(liked))
    a_share, a_key = share(prof_authors, len(liked))
    print(f"  distinct series among liked : {len(prof_series)}")
    print(f"  largest series share        : {s_share:.1%}  ({prof_series[s_key]} volumes)")
    print(f"  largest author share        : {a_share:.1%}")
    print()
    print("  series carrying 2+ rated volumes (these are the inflated anchors):")
    multi = [(k, c) for k, c in prof_series.most_common() if c > 1]
    if not multi:
        print("    none -- series inflation is NOT this profile's problem.")
    for k, c in multi[:8]:
        print(f"    {c:>3} volumes  {k.split('||')[0][:44]}")
        for t in series_titles[k][:3]:
            print(f"               {t[:60]}")

    # ---- 2. what comes BACK -------------------------------------------------
    print("\n" + "=" * 72)
    print(f"2. RECOMMENDATIONS  (top {args.n}, grouped by the anchor that pulled them)")
    print("=" * 72)
    recs, _ = ns["recommend_books"](ratings, df, emb, index,
                                    tags_df=tags_df, n=args.n)
    if not len(recs):
        sys.exit("  ranker returned nothing.")

    rec_series, rec_authors, rec_years = Counter(), Counter(), []
    for _, r in recs.iterrows():
        sk = series_key(r.get("title", ""), r.get("author", ""))
        if sk:
            rec_series[sk] += 1
        rec_authors[_norm(r.get("author", ""))] += 1
        y = r.get("year")
        if pd.notna(y):
            rec_years.append(int(y))

    # Group by anchor: the round-robin takes per_anchor_cap items per rated
    # book, so the list is really N mini-lists stacked. Reading it grouped shows
    # which ANCHOR is misfiring, which a flat list hides completely.
    by_anchor = defaultdict(list)
    for _, r in recs.iterrows():
        by_anchor[str(r.get("because_of", "?"))].append(r)

    anchor_nf = {}
    for r in liked:
        row = df.iloc[int(r["corpus_index"])]
        anchor_nf[str(r.get("matched_title", row["title"]))] = is_nonfiction(row.get("genres"))

    for anchor, items in by_anchor.items():
        nf = anchor_nf.get(anchor)
        tag = "NONFIC" if nf else ("fiction" if nf is False else "  ?   ")
        print(f"\n  [{tag}] {anchor[:56]}")
        for r in items:
            yr = int(r["year"]) if pd.notna(r.get("year")) else "?"
            rnf = is_nonfiction(r.get("genres"))
            mark = "!" if (nf is True and rnf is False) else " "
            print(f"    {mark} {str(r.get('title',''))[:40]:42s} "
                  f"{str(r.get('author',''))[:20]:22s} {yr}")
    print("\n  ! = non-fiction anchor returning fiction")

    # ---- 3. concentration ---------------------------------------------------
    print("\n" + "=" * 72)
    print("3. CONCENTRATION  (which cause is it?)")
    print("=" * 72)
    n = len(recs)
    rs_share, rs_key = share(rec_series, n)
    ra_share, ra_key = share(rec_authors, n)
    print(f"  distinct series in output : {len(rec_series)} across {n} recs")
    print(f"  largest series share      : {rs_share:.1%}"
          f"   {rs_key.split('||')[0][:36] if rs_key else ''}")
    print(f"  largest author share      : {ra_share:.1%}")
    if rec_years:
        print(f"  publication year spread   : {min(rec_years)}-{max(rec_years)}, "
              f"median {int(np.median(rec_years))}")

    # Fiction/non-fiction bleed. A memoir anchor returning novels is not a
    # near-miss, it is the wrong shelf -- and plot-summary embeddings have no
    # way to know the difference, because a book ABOUT the Mumbai underworld
    # and a novel SET in one describe the same subject matter.
    bleed = same = 0
    for anchor, items in by_anchor.items():
        if anchor_nf.get(anchor) is not True:
            continue
        for r in items:
            rnf = is_nonfiction(r.get("genres"))
            if rnf is False:
                bleed += 1
            elif rnf is True:
                same += 1
    if bleed or same:
        tot = bleed + same
        print(f"  non-fiction anchors       : {bleed}/{tot} of their recs are FICTION"
              f"  ({bleed/max(tot,1):.0%})")

    print()
    if multi and rs_key and rs_key == s_key:
        print("  >> SERIES INFLATION. The most-rated series in the profile is also")
        print("     the most-returned in the output. The anchor count is doing")
        print("     this, not the scoring. Fix is profile-side: one anchor per")
        print("     series, weighted by volumes rated. No re-embed needed.")
    elif bleed and bleed / max(bleed + same, 1) >= 0.5:
        print("  >> FICTION/NON-FICTION BLEED. Non-fiction anchors are mostly")
        print("     returning novels. Subject matter matches; the kind of book")
        print("     does not. The corpus carries a genre folksonomy that could")
        print("     gate this, and the embedding alone never will.")
    elif rs_share >= 0.30 or ra_share >= 0.30:
        print("  >> NARROW OUTPUT without matching profile inflation. The caps are")
        print("     not binding the way they should. Ranking problem.")
    else:
        print("  >> Output is well spread by series and author. If it still feels")
        print("     wrong, read the grouped list above and ask which ANCHOR is")
        print("     misfiring -- the round-robin gives each rated book an equal")
        print("     share, so one bad anchor costs a fixed slice of every list.")

    print("\n" + "=" * 72)
    print("READ THE TITLES, NOT JUST THE PERCENTAGES")
    print("=" * 72)
    print("  A healthy-looking spread with 20 titles you would never read is a")
    print("  corpus problem wearing a ranking problem's numbers.")


if __name__ == "__main__":
    main()
