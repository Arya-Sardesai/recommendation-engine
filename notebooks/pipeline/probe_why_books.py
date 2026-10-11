"""
probe_why_books.py -- why did THIS anchor return THESE books?

THE COMPLAINT (live Space, 53-book profile)
    The Ax (Westlake: a laid-off executive kills off his job competition)
      -> Things We Left Behind   (Lucy Score, small-town romance)
      -> Secret Service          (Tal Bauer, gay romance, 15 ratings)
      -> Demon Copperhead
    every pick stamped "tag-matched: funny, emotional, adventurous"

Two suspects, and they need different fixes:
  RETRIEVAL  the romance blurbs really are nearest The Ax in embedding space
             ("vengeance-seeking mogul ... building an empire" shares the
             plot nouns of a revenge-on-capitalism thriller). Fix: a genre /
             shelf penalty, like the shelf gate.
  RE-RANK    retrieval returns sensible crime books, and the tag layer lifts
             the romances past them because they carry the broad reader moods
             (funny, emotional) that dominate the taste signature. Fix: tag
             weight, or stop scoring tags this generic.

This prints, for one anchor:
  1. the RAW nearest neighbours, before any scoring -- what the embedding sees
  2. the picks under the live config, with the tag layer off, and with the
     shelf gate on
so it is obvious which stage introduced the bad books.

READ-ONLY.

USAGE
    python notebooks\\pipeline\\probe_why_books.py --ratings <export.json> --anchor "The Ax"
    python notebooks\\pipeline\\probe_why_books.py --ratings <export.json> --anchor "The Ax" --app ..\\hf-space\\app.py

The ratings file: on the Space, "Back up or restore your ratings" -> download.
tests\\movies_ratings.json is the July export and does not contain The Ax.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"
FORMS = PROCESSED / "book_forms_v3.parquet"


def load_book_ratings(path):
    """Accept the Space's backup file, a ?debug=1 dump, or a bare list."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), dict):          # export_payload()
            v = raw["data"].get("ratings")
            if isinstance(v, list):
                return v
        v = raw.get("re_books") or raw.get("ratings")
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return v
    sys.exit(f"no book ratings found in {path}")


def genres_str(g, k=3):
    try:
        return ", ".join(list(g)[:k])
    except TypeError:
        return ""


def explain_score(ns, ratings, df, emb, index, tags_df, a, ai,
                  pick_idx, raw_idx, extra=(), neutral=3.0, cross_weight=0.35,
                  quality_weight=0.12, min_avg_rating=3.6, shrink=50,
                  tag_weight=0.25, mix=0.5, retrieve_k=150, min_ratings=10,
                  sim_scale=1.0):
    """Recompute recommend_books' score terms for chosen candidates under one
    anchor, so it is visible WHICH term carries a bad pick. Mirrors the live
    defaults; mu (the pool mean the tag term is centred on) is approximated
    over every anchor's unfiltered retrieval pool."""
    liked = [r for r in ratings if r["rating"] >= 3.5]
    disliked = [r for r in ratings if r["rating"] < neutral]
    L = np.array([emb[r["corpus_index"]] for r in liked], dtype="float32")
    Lw = np.array([(r["rating"] - neutral) / 2 for r in liked], dtype="float32")
    D = (np.array([emb[r["corpus_index"]] for r in disliked], dtype="float32")
         if disliked else None)
    tau = index.search(D, 200)[0][:, -1] if D is not None else None
    Dw = np.array([(neutral - r["rating"]) / 2 for r in disliked], dtype="float32")
    me = [i for i, r in enumerate(liked) if int(r["corpus_index"]) == ai][0]
    av = L[me]
    damp = np.clip(L @ av, 0, None) ** 2

    # tag profiles
    cols = tags_df.columns
    tv, tw = [], []
    for r in ratings:
        w = r["rating"] - neutral
        if w and str(r["book_id"]) in tags_df.index:
            tv.append(tags_df.loc[str(r["book_id"])].values); tw.append(w)
    tw = np.array(tw, dtype="float32")
    glob = (np.array(tv, dtype="float32") * tw[:, None]).sum(0) / np.abs(tw).sum()
    prof = glob
    abid = str(a["book_id"])
    if mix and abid in tags_df.index:
        at = tags_df.loc[abid].values.astype("float32")
        prof = (1 - mix) * glob / np.linalg.norm(glob) + mix * at / np.linalg.norm(at)
        print(f"\n  anchor's own top tags: "
              f"{', '.join(pd.Series(at, index=cols).nlargest(6).index)}")
    else:
        print("\n  anchor has NO tags -> per-anchor mix falls back to the global profile")

    def tsim(p, bid):
        if bid not in tags_df.index:
            return None
        b = tags_df.loc[bid].values
        nb = np.linalg.norm(b)
        return float(p @ b / (np.linalg.norm(p) * nb)) if nb else None

    psims, pools = index.search(L, retrieve_k)
    pool_ids = {str(df.iloc[j]["book_id"]) for j in pools.ravel() if j >= 0}
    mu_g = np.mean([x for x in (tsim(glob, b) for b in pool_ids) if x is not None])
    mu_a = np.mean([x for x in (tsim(prof, b) for b in pool_ids) if x is not None])
    my_pool = [int(j) for j in pools[me]]
    pmu = float(psims[me][pools[me] >= 0].mean())
    C = float(df.loc[df["ratings_count"] >= min_ratings, "average_rating"].mean())
    rated_auth = {str(r.get("matched_author", "")).lower() for r in ratings}

    cand = list(dict.fromkeys(pick_idx + raw_idx))
    for t in extra:
        m = df.index[df["title"].astype(str).str.lower().str.contains(t.lower(), regex=False)]
        cand += [int(m[0])] if len(m) else []

    print("\n" + "=" * 78)
    print(f"3. SCORE BREAKDOWN under this anchor (new default: floor k=200, "
          f"sim x{sim_scale:g}, mix {mix:g})")
    print("=" * 78)
    print(f"  {'title':<30}{'sim':>6}{'cross':>7}{'neg':>7}{'qual':>6}"
          f"{'tagG':>7}{'tagA':>7}{'total':>7}  notes")
    for j in cand:
        b = df.iloc[j]
        c = emb[j]
        sim = pmu + sim_scale * (float(av @ c) - pmu)
        cr = (L @ c) * Lw * damp
        cr[me] = -9
        k = int(np.argmax(cr)); cross = cross_weight * max(float(cr[k]), 0)
        neg, negby = 0.0, ""
        if D is not None:
            nv = np.clip((D @ c - tau) * Dw, 0, None)
            neg = float(nv.max()); negby = disliked[int(nv.argmax())].get("matched_title", "")
        n_ = float(b["ratings_count"]); ar = b.get("average_rating")
        notes = []
        qual = 0.0
        if ar is not None and not pd.isna(ar):
            adj = (n_ * float(ar) + shrink * C) / (n_ + shrink)
            if adj < min_avg_rating:
                notes.append(f"QUALITY FLOOR ({adj:.2f})")
            qual = quality_weight * max(0.0, (adj - min_avg_rating) / (5 - min_avg_rating))
        if b.get("source") != "targeted_authors" and n_ < min_ratings:
            notes.append("BELOW RATINGS FLOOR")
        if str(b["primary_author"]).lower() in rated_auth:
            notes.append("RATED AUTHOR")
        if j not in my_pool:
            notes.append("not in this anchor's pool")
        bid = str(b["book_id"])
        tg, ta = tsim(glob, bid), tsim(prof, bid)
        tG = tag_weight * ((tg if tg is not None else 0) - mu_g)
        tA = tag_weight * ((ta if ta is not None else 0) - mu_a)
        if tg is None:
            notes.append("untagged")
        total = sim + cross - neg + qual + tA
        tag = "*" if j in pick_idx else " "
        print(f" {tag}{str(b['title'])[:29]:<30}{sim:>6.3f}{cross:>+7.3f}{-neg:>+7.3f}"
              f"{qual:>+6.3f}{tG:>+7.3f}{tA:>+7.3f}{total:>7.3f}  "
              f"cross<-{str(liked[k].get('matched_title'))[:22]}"
              + (f"; neg<-{negby[:18]}" if neg > 0.02 else "")
              + (f"; {', '.join(notes)}" if notes else ""))
    print("  * = picked under the new default. sim is AFTER the stretch. tagG =")
    print("  global tag term, tagA = per-anchor term (the one in total; equal")
    print("  to tagG when mix is 0). Title/author penalties not shown.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--anchor", required=True, help="title (substring) of a rated book")
    ap.add_argument("--app", default=str(SPACE_APP))
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--dismiss", action="append", default=[],
                    help="title (substring) to click 'Not interested' on; "
                         "repeatable. Shows the anchor's picks at several "
                         "skip weights.")
    ap.add_argument("--explain", action="append", default=[],
                    help="extra title (substring) to break down; repeatable")
    args = ap.parse_args()

    app = Path(args.app)
    ns = load_ranker(app, require=("recommend_books", "_migrate_book_ratings"))
    art = resolve_artifacts(app, PROCESSED, kind="book", verbose=False)

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, "ALIGNMENT"
    t = pd.read_parquet(art["TAGS"])
    t["book_id"] = t["book_id"].astype(str)
    tags_df = t.pivot_table(index="book_id", columns="tag", values="score",
                            fill_value=0.0)
    forms = {}
    if FORMS.exists():
        f = pd.read_parquet(FORMS)
        forms = dict(zip(f["book_id"].astype(str), f["form"].astype(str)))

    ratings = load_book_ratings(args.ratings)
    ratings, *_ = ns["_migrate_book_ratings"](ratings, df)
    hits = [r for r in ratings
            if args.anchor.lower() in str(r.get("matched_title", "")).lower()]
    if not hits:
        sys.exit(f"no rated book matching {args.anchor!r} in that file")
    a = hits[0]
    ai = int(a["corpus_index"])
    arow = df.iloc[ai]
    print(f"profile {len(ratings)} books")
    print(f"ANCHOR  {arow['title']}  ({arow['primary_author']})  "
          f"rated {a['rating']}  form={forms.get(str(arow['book_id']), '?')}")
    print(f"        genres: {genres_str(arow.get('genres'), 6)}")
    print(f"        {str(arow.get('description', ''))[:220]}...\n")

    # ---- 1. raw retrieval ---------------------------------------------------
    print("=" * 78)
    print(f"1. RAW NEAREST NEIGHBOURS of the anchor (pure cosine, no scoring)")
    print("=" * 78)
    D, I = index.search(emb[ai:ai + 1], args.k + 1)
    romance_raw = 0
    for s, j in zip(D[0], I[0]):
        if j == ai:
            continue
        r = df.iloc[j]
        g = genres_str(r.get("genres"))
        is_rom = "romance" in g.lower()
        romance_raw += is_rom
        print(f"  {s:.3f}  {str(r['title'])[:38]:40s} {str(r['primary_author'])[:18]:20s}"
              f" rc={int(r['ratings_count']):<5} {g[:34]}")
    print(f"\n  romance-shelved among raw top {args.k}: {romance_raw}")

    # ---- 2. what the ranker keeps, under three configs ----------------------
    def picks(**kw):
        recs, sig = ns["recommend_books"](ratings, df, emb, index,
                                          tags_df=tags_df, n=120, **kw)
        mine = recs[recs["because_of"].astype(str) == str(a.get("matched_title"))]
        return mine, sig

    OLD = {"anchor_tag_mix": 0.0, "neg_floor_k": 0, "sim_scale": 1.0}
    configs = [("OLD (what is live)", OLD),
               ("dislike floor only", {"sim_scale": 1.0}),
               ("floor + sim x3 (new default)", {}),
               ("floor + sim x5", {"sim_scale": 5.0}),
               ("floor + sim x3 + per-anchor tags 0.5", {"anchor_tag_mix": 0.5}),
               ("floor + sim x3, tags OFF", {"tag_weight": 0.0}),
               ("+ tag idf 1.0", {"tag_idf": 1.0}),
               ("+ genre 0.10", {"genre_weight": 0.10}),
               ("+ genre 0.20", {"genre_weight": 0.20}),
               ("+ idf 1.0 + genre 0.10", {"tag_idf": 1.0, "genre_weight": 0.10})]
    if forms and "shelf_penalty" in ns["recommend_books"].__code__.co_varnames:
        configs.append(("new default + shelf gate", {"forms": forms}))

    print("\n" + "=" * 78)
    print("2. WHAT THE RANKER SURFACES FOR THIS ANCHOR")
    print("=" * 78)
    default_pick_idx = []
    for name, kw in configs:
        mine, sig = picks(**kw)
        if "new default" in name:
            default_pick_idx = [int(x) for x in mine["corpus_index"]]
        print(f"\n  [{name}]")
        if (name.startswith("OLD") or "idf" in name) and sig is not None and len(sig):
            print(f"    taste signature: {', '.join(sig.index[:6])}")
        for _, r in mine.iterrows():
            g = genres_str(r.get("genres"))
            mt = r.get("match_tags")
            mt = ", ".join(mt) if isinstance(mt, (list, tuple)) and len(mt) else ""
            print(f"    {str(r['title'])[:40]:42s} {str(r['author'])[:18]:20s} {g[:30]:32s}"
                  + (f" [{mt}]" if mt else ""))
        if not len(mine):
            print("    (anchor got no slots under this config)")

    # ---- 2b. not-interested simulation ---------------------------------------
    if args.dismiss:
        sk = []
        for t in args.dismiss:
            m = df.index[df["title"].astype(str).str.lower()
                         .str.contains(t.lower(), regex=False)]
            if len(m):
                sk.append(str(df.iloc[int(m[0])]["book_id"]))
                print(f"\n  dismissing: {df.iloc[int(m[0])]['title']}")
            else:
                print(f"\n  no corpus title matching {t!r}")
        print("\n" + "=" * 78)
        print("2b. AFTER CLICKING 'NOT INTERESTED'  (0 = hide only, the old behaviour;")
        print("    0.25 = shipped default; 1.0 = as strong as a 1-star rating)")
        print("=" * 78)
        for w in (0.0, 0.25, 0.5, 1.0):
            mine, _ = picks(skip_ids=sk, skip_weight=w)
            print(f"\n  [skip weight {w}]")
            for _, r in mine.iterrows():
                g = genres_str(r.get("genres"))
                print(f"    {str(r['title'])[:40]:42s} {str(r['author'])[:18]:20s} {g[:30]}")

    # ---- 3. score breakdown --------------------------------------------------
    import inspect
    _d = {k: v.default for k, v in
          inspect.signature(ns["recommend_books"]).parameters.items()}
    explain_score(ns, ratings, df, emb, index, tags_df, a, ai,
                  default_pick_idx, [int(j) for j in I[0] if j != ai][:6],
                  extra=args.explain, mix=_d.get("anchor_tag_mix", 0.0),
                  sim_scale=_d.get("sim_scale", 1.0))

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  Romances already in section 1  -> RETRIEVAL: the embedding thinks")
    print("    they are this book's neighbours. A ranking change cannot fix")
    print("    that; it needs a penalty on the shelf, like the shelf gate.")
    print("  Section 1 is crime/satire but LIVE shows romance -> RE-RANK: the")
    print("    tag layer lifted them. If 'tag layer OFF' looks right, the fix")
    print("    is the tags, not the corpus.")


if __name__ == "__main__":
    main()
