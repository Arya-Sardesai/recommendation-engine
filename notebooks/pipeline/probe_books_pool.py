"""
probe_books_pool.py — is the candidate pool romance-heavy because the CORPUS is,
or because RETRIEVAL prefers it?

THE COMPLAINT
Unrelated anchors converge on recent contemporary romance. How to Kidnap the
Rich (Indian comic crime) returns Dating Dr. Dil. Love in the Time of Cholera
returns The Next Best Fling. Scott Pilgrim returns three 2022-2024 romances.
Median publication year of a 25-rec list: 2019.

THE TWO CAUSES NEED OPPOSITE FIXES
  CORPUS   The harvest paginated Hardcover by users_count desc, and Hardcover's
           readership skews contemporary romance. If romance is 30% of every
           servable book, every neighbourhood is romance-adjacent and no
           ranking change fixes it. Fix is a harvest or a corpus-side quota.
  RETRIEVAL Romance blurbs are formulaic and dense in relationship vocabulary,
           so they may sit close to everything regardless of subject. If
           romance is over-represented in the retrieved pool RELATIVE to its
           corpus share, that is an embedding effect and a re-rank can fix it.

The test is the ratio. Corpus share is the baseline; pool share is what
retrieval actually returns. Lift = pool / corpus. Lift near 1.0 means the
ranker is faithfully reflecting a skewed corpus. Lift well above 1.0 means
retrieval is amplifying it.

READ-ONLY. No writes, no API.

USAGE
    python notebooks\\pipeline\\probe_books_pool.py --ratings tests\\movies_ratings.json
    python notebooks\\pipeline\\probe_books_pool.py --ratings tests\\movies_ratings.json --k 100
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from ranker_loader import load_ranker, resolve_artifacts  # noqa: E402

SPACE_APP = REPO_ROOT.parent / "hf-space" / "app.py"
PROCESSED = REPO_ROOT / "data" / "processed"

MIN_RATINGS = 10          # app.py's serving floor for books
WATCH = ["romance", "contemporary romance", "fantasy", "young adult",
         "romantasy", "fiction", "nonfiction", "mystery", "thriller",
         "horror", "science fiction", "historical fiction", "memoir",
         "literary fiction", "graphic novels", "manga", "classics"]


def genres_of(v):
    if v is None:
        return set()
    if isinstance(v, str):
        return {g.strip().lower() for g in v.split(",") if g.strip()}
    try:
        return {str(g).strip().lower() for g in v}
    except TypeError:
        return set()


def load_book_ratings(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        v = raw.get("re_books")
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            return v
    return raw if isinstance(raw, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--k", type=int, default=50,
                    help="neighbours retrieved per anchor (the pool, pre-ranking)")
    args = ap.parse_args()

    ns = load_ranker(SPACE_APP, require=("recommend_books", "_migrate_book_ratings"))
    art = resolve_artifacts(SPACE_APP, PROCESSED, kind="book", verbose=False)

    import faiss
    df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
    emb = np.load(art["EMB"]).astype("float32")
    index = faiss.read_index(str(art["FAISS"]))
    assert len(df) == emb.shape[0] == index.ntotal, "ALIGNMENT"

    ratings = load_book_ratings(Path(args.ratings))
    ratings, _m, _d, _g = ns["_migrate_book_ratings"](ratings, df)
    liked = [r for r in ratings
             if r.get("rating", 0) >= 3.5
             and r.get("corpus_index") is not None
             and 0 <= int(r["corpus_index"]) < len(df)]
    print(f"corpus {len(df):,} rows | {len(liked)} liked anchors\n")

    rc = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    serv_mask = (rc >= MIN_RATINGS).to_numpy()
    serv_idx = np.where(serv_mask)[0]
    gsets = [genres_of(g) for g in df["genres"]]
    years = pd.to_numeric(df["publication_year"], errors="coerce")

    # ---- 1. corpus baseline -------------------------------------------------
    print("=" * 72)
    print(f"1. CORPUS BASELINE  (servable = ratings_count >= {MIN_RATINGS})")
    print("=" * 72)
    n_serv = len(serv_idx)
    print(f"  servable books: {n_serv:,} of {len(df):,}")
    corpus_share = {}
    cnt = Counter()
    for i in serv_idx:
        for g in gsets[i]:
            cnt[g] += 1
    for g in WATCH:
        corpus_share[g] = cnt.get(g, 0) / max(n_serv, 1)
    print("\n  top 15 genres among servable books:")
    for g, c in cnt.most_common(15):
        print(f"    {g:<26} {c:>7,}  {c/n_serv:6.1%}")

    sy = years.to_numpy()[serv_idx]
    sy = sy[~np.isnan(sy)]
    if len(sy):
        print(f"\n  publication year, servable: median {int(np.median(sy))}, "
              f"{(sy >= 2015).mean():.0%} are 2015 or later")

    # ---- 2. the retrieved pool ---------------------------------------------
    print("\n" + "=" * 72)
    print(f"2. RETRIEVED POOL  (top {args.k} neighbours per anchor, before ranking)")
    print("=" * 72)
    pool = Counter()
    pool_years, pool_n = [], 0
    per_anchor = []
    for r in liked:
        ci = int(r["corpus_index"])
        _, I = index.search(emb[ci:ci + 1], args.k + 1)
        hits = [int(j) for j in I[0] if j != ci and serv_mask[j]]
        rom = 0
        for j in hits:
            pool_n += 1
            for g in gsets[j]:
                pool[g] += 1
            if {"romance", "contemporary romance", "romantasy"} & gsets[j]:
                rom += 1
            y = years.iloc[j]
            if pd.notna(y):
                pool_years.append(float(y))
        if hits:
            per_anchor.append((str(r.get("matched_title", "?")), rom / len(hits)))

    if not pool_n:
        sys.exit("empty pool -- nothing retrieved above the serving floor.")

    print(f"  pooled neighbours: {pool_n:,}")
    if pool_years:
        py = np.array(pool_years)
        print(f"  publication year, pool: median {int(np.median(py))}, "
              f"{(py >= 2015).mean():.0%} are 2015 or later")

    # ---- 3. the ratio that decides it --------------------------------------
    print("\n" + "=" * 72)
    print("3. LIFT = pool share / corpus share")
    print("=" * 72)
    print(f"  {'genre':<26}{'corpus':>9}{'pool':>9}{'lift':>8}")
    print("  " + "-" * 50)
    for g in WATCH:
        cs = corpus_share.get(g, 0.0)
        ps = pool.get(g, 0) / pool_n
        if cs == 0 and ps == 0:
            continue
        lift = ps / cs if cs > 0 else float("inf")
        flag = "  <<" if lift >= 1.5 and ps >= 0.05 else ""
        print(f"  {g:<26}{cs:>8.1%}{ps:>9.1%}{lift:>8.2f}{flag}")

    print("\n  anchors by romance share of their own neighbourhood:")
    for t, s in sorted(per_anchor, key=lambda x: -x[1])[:10]:
        print(f"    {s:>5.0%}  {t[:52]}")

    rom_c = sum(corpus_share.get(g, 0) for g in ("romance", "contemporary romance", "romantasy"))
    rom_p = sum(pool.get(g, 0) for g in ("romance", "contemporary romance", "romantasy")) / pool_n
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  romance-family share: corpus {rom_c:.1%}, pool {rom_p:.1%}")
    if rom_c >= 0.20 and rom_p / max(rom_c, 1e-9) < 1.3:
        print("  >> CORPUS. Romance is that big a slice of what is servable, and")
        print("     retrieval is reflecting it rather than amplifying it. A")
        print("     re-rank moves deck chairs; the fix is corpus-side -- harvest")
        print("     differently, or raise the serving floor so the long tail of")
        print("     lightly-read romance stops qualifying.")
    elif rom_p / max(rom_c, 1e-9) >= 1.3:
        print("  >> RETRIEVAL. Romance is over-represented in the pool relative to")
        print("     its corpus share, so the embedding is pulling it in. Formulaic")
        print("     blurbs sitting close to everything is the likely mechanism.")
        print("     A genre-family penalty at re-rank can fix this without a")
        print("     re-embed, because the candidates ARE being retrieved.")
    else:
        print("  >> Romance is not the driver. Read the per-anchor list above and")
        print("     find which anchors are actually misfiring.")


if __name__ == "__main__":
    main()
