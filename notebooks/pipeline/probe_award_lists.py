"""
probe_award_lists.py -- what does the book award bonus do to MY "For you" list?

Runs your real profile through the books ranker exactly as the Space does
(same defaults, shelf gate, your "Not interested" clicks) with the award bonus
off, at 0.06, and at 0.08, then shows:

  1. the top 20 at 0.08: trophy marks, how far each book moved, award text
  2. books the bonus PULLED IN (not in the top 20 without it)
  3. books it PUSHED OUT (in the top 20 without it, gone at 0.08)

What to look for: pushed-out books you'd actually read, replaced by prize
winners you wouldn't. A couple is fine; a list that reads like a prize shelf
instead of your taste means drop to 0.06.

READ-ONLY.

USAGE (repo root):
    python notebooks\\pipeline\\probe_award_lists.py --ratings %USERPROFILE%\\Downloads\\my-recommendation-profile.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402
from probe_why_books import load_book_ratings  # noqa: E402

SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"


def load_skips(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        return list(raw["data"].get("skipped") or [])
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="the Space's backup file")
    ap.add_argument("--app", default=str(SPACE_APP))
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    app = Path(args.app)
    ns = load_ranker(app, require=("recommend_books", "book_award_summaries",
                                   "_migrate_book_ratings"), verbose=False)
    art = resolve_artifacts(app, PROCESSED, kind="book", verbose=False)
    if "AWARDS" not in art or not art["AWARDS"].exists():
        sys.exit("book_awards.parquet missing -- run build_book_awards.py first")

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, "ALIGNMENT"
    t = pd.read_parquet(art["TAGS"])
    t["book_id"] = t["book_id"].astype(str)
    tags_df = t.pivot_table(index="book_id", columns="tag", values="score",
                            fill_value=0.0)
    forms = None
    if "FORMS" in art and art["FORMS"].exists():
        f = pd.read_parquet(art["FORMS"])
        forms = dict(zip(f["book_id"].astype(str), f["form"].astype(str)))
    awards = ns["book_award_summaries"](pd.read_parquet(art["AWARDS"]))

    ratings = load_book_ratings(args.ratings)
    ratings, *_ = ns["_migrate_book_ratings"](ratings, df)
    skips = load_skips(args.ratings)
    print(f"profile: {len(ratings)} books rated, {len(skips)} 'Not interested'\n")

    def top(win, nom):
        recs, _ = ns["recommend_books"](ratings, df, emb, index, tags_df=tags_df,
                                        n=args.n, skip_ids=skips, forms=forms,
                                        awards=awards, award_win=win, award_nom=nom)
        return recs.reset_index(drop=True)

    def aw(bid):
        try:
            return awards.get(int(bid))
        except (TypeError, ValueError):
            return None

    def mark(bid):
        a = aw(bid)
        if not a:
            return "  "
        return "W " if a["major"] == "won" else ("S " if a["major"] == "nominated" else "m ")

    off, mid, on = top(0.0, 0.0), top(0.06, 0.03), top(0.08, 0.04)
    rank_off = {str(b): i for i, b in enumerate(off["book_id"])}
    rank_mid = {str(b): i for i, b in enumerate(mid["book_id"])}

    print("=" * 100)
    print(f"1. YOUR TOP {args.n} AT 0.08 / 0.04    W = won a major prize, S = shortlisted, "
          f"m = minor award only")
    print("   move = vs no bonus;  at 0.06 = its rank at 0.06 / 0.03")
    print("=" * 100)
    for i, r in on.iterrows():
        bid = str(r["book_id"])
        if bid in rank_off:
            d = rank_off[bid] - i
            move = "  =  " if d == 0 else (f" +{d:<3}" if d > 0 else f" {d:<4}")
        else:
            move = " NEW "
        m6 = f"{rank_mid[bid] + 1:>3}" if bid in rank_mid else "  -"
        a = aw(bid)
        tip = f"   [{a['tip'][:40]}]" if a else ""
        because = str(r.get("because_of", ""))[:26]
        print(f"{i + 1:>3}. {mark(bid)}{str(r['title'])[:36]:38s}{str(r['author'])[:18]:20s}"
              f"{move}  at 0.06:{m6}   <- {because:28s}{tip}")

    ids_on, ids_off = set(map(str, on["book_id"])), set(map(str, off["book_id"]))
    pulled = on[~on["book_id"].astype(str).isin(ids_off)]
    pushed = off[~off["book_id"].astype(str).isin(ids_on)]

    print("\n" + "=" * 100)
    print(f"2. PULLED IN by the bonus ({len(pulled)}) -- would you read these?")
    print("=" * 100)
    for _, r in pulled.iterrows():
        a = aw(r["book_id"])
        print(f"   {mark(r['book_id'])}{str(r['title'])[:42]:44s}{str(r['author'])[:22]:24s}"
              f"{(a or {}).get('tip', '')[:40]}")
    print("\n" + "=" * 100)
    print(f"3. PUSHED OUT ({len(pushed)}) -- would you miss these?")
    print("=" * 100)
    for _, r in pushed.iterrows():
        print(f"   {mark(r['book_id'])}{str(r['title'])[:42]:44s}{str(r['author'])[:22]:24s}"
              f"was #{rank_off[str(r['book_id'])] + 1}")

    def share(d):
        return sum(1 for b in d["book_id"] if (aw(b) or {}).get("major")) / max(len(d), 1)
    print(f"\naward books in your top {args.n}: no bonus {share(off):.0%}  ·  "
          f"0.06 {share(mid):.0%}  ·  0.08 {share(on):.0%}")
    print("If section 2 is mostly books you'd skip and section 3 has books you'd read,")
    print("set BOOK_AWARD_WIN = 0.06 / BOOK_AWARD_NOM = 0.03 instead.")


if __name__ == "__main__":
    main()
