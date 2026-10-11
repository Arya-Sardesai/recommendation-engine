"""
probe_shelf_gate.py -- does the shelf gate stop non-fiction anchors returning novels?

BEFORE THE GATE (probe_books_profile, this profile): 60% of the recommendations
pulled by non-fiction anchors were fiction. Dongri to Dubai -> The Empire of
Gold, The Kingdom of Copper.

This runs the same profile with the gate off and at two strengths, and reports
the fiction share under non-fiction anchors using the LLM form labels (the
Hardcover genres are too noisy to judge by -- they file The Great Gatsby as
non-fiction). It prints the gated list grouped by anchor so the titles can be
read, which on this project has caught more than every percentage combined.

goodbooks cannot measure this: nearly every goodbooks anchor is fiction, so
the gate barely fires there. Run eval_goodbooks too, but only as a check that
nothing got WORSE (baseline recall@20 0.017).

USAGE
    python notebooks\\pipeline\\probe_shelf_gate.py --ratings tests\\movies_ratings.json
    python notebooks\\pipeline\\probe_shelf_gate.py --ratings tests\\movies_ratings.json --app app.py
"""
import argparse
import json
import sys
from collections import defaultdict
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
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        v = raw.get("re_books")
        if isinstance(v, str):
            v = json.loads(v)
        return v if isinstance(v, list) else []
    return raw if isinstance(raw, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--app", default=str(SPACE_APP),
                    help="app.py carrying the gate (default: the hf-space copy)")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    app = Path(args.app)
    ns = load_ranker(app, require=("recommend_books", "_migrate_book_ratings"))
    import inspect
    if "shelf_penalty" not in inspect.signature(ns["recommend_books"]).parameters:
        sys.exit(f"{app} has no shelf gate -- copy the new app.py there first, "
                 f"or pass --app pointing at it")
    nonfic = ns["_NONFIC_FORMS"]
    fic = ns["_FICTION_FORMS"]

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

    if not FORMS.exists():
        sys.exit(f"{FORMS} not found -- run build_book_forms.py first")
    fdf = pd.read_parquet(FORMS)
    forms = dict(zip(fdf["book_id"].astype(str), fdf["form"].astype(str)))

    ratings = load_book_ratings(args.ratings)
    ratings, *_ = ns["_migrate_book_ratings"](ratings, df)
    title_form = {str(r.get("matched_title")): forms.get(str(r.get("book_id")))
                  for r in ratings}
    nf_anchors = [k for k, v in title_form.items() if v in nonfic]
    print(f"profile {len(ratings)} books | non-fiction anchors by LLM form: "
          f"{len(nf_anchors)}")
    for a in nf_anchors:
        print(f"    [{title_form[a]}] {a[:60]}")

    def run(label, **kw):
        recs, _ = ns["recommend_books"](ratings, df, emb, index,
                                        tags_df=tags_df, n=args.n, **kw)
        by = defaultdict(list)
        for _, r in recs.iterrows():
            by[str(r.get("because_of"))].append(r)
        tot = bad = 0
        for a, items in by.items():
            if title_form.get(a) not in nonfic:
                continue
            for r in items:
                f = forms.get(str(r["book_id"]))
                tot += 1
                bad += f in fic
        share = bad / tot if tot else 0.0
        print(f"  {label:<22} fiction under non-fiction anchors: "
              f"{bad}/{tot} ({share:.0%})")
        return recs, by

    print("\n" + "=" * 72)
    base, _ = run("gate off", forms=None)
    run("gate 0.10 (shipped)", forms=forms, shelf_penalty=0.10)
    gated, by = run("gate 0.20", forms=forms, shelf_penalty=0.20)
    print("=" * 72)

    base_ids = set(base["book_id"].astype(str))
    print("\nGATED LIST (0.20), grouped by anchor. NEW = not in the ungated list.")
    for a, items in by.items():
        af = title_form.get(a) or "?"
        print(f"\n  [{af}] {a[:58]}")
        for r in items:
            f = forms.get(str(r["book_id"])) or "?"
            mark = "NEW" if str(r["book_id"]) not in base_ids else "   "
            print(f"    {mark} {str(r['title'])[:40]:42s} {str(r['author'])[:18]:20s} {f}")

    print("\nREAD THE NON-FICTION ANCHORS. Fiction share should fall, and what")
    print("replaces it should be real non-fiction on the same subject -- not")
    print("unrelated memoirs pulled in because they are the only non-fiction")
    print("nearby. If the replacements are worse than the novels were, the")
    print("gate is too strong: keep 0.10, or drop it.")


if __name__ == "__main__":
    main()
