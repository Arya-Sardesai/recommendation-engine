import pandas as pd, numpy as np, json, faiss, sys
sys.path.insert(0, "tests")
from ranker_loader import load_ranker, resolve_artifacts
from pathlib import Path

APP = Path("../hf-space/app.py")
ns = load_ranker(APP, require=("recommend_books",))
art = resolve_artifacts(APP, Path("data/processed"), kind="book")
df = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
emb = np.load(art["EMB"]).astype("float32")
ix = faiss.read_index(str(art["FAISS"]))

links = {int(k): int(v) for k, v in json.load(open("tests/goodbooks_links.json")).items()}
rt = pd.read_csv("data/raw/books/goodbooks-10k/ratings.csv")
rt = rt[rt["book_id"].isin(links)]
uid = rt.groupby("user_id").size().sort_values().index[-1]
g = rt[rt["user_id"] == uid].head(30)

ratings = [{"book_id": str(df.iloc[links[int(b)]]["book_id"]),
            "matched_title": df.iloc[links[int(b)]]["title"],
            "matched_author": df.iloc[links[int(b)]].get("primary_author", ""),
            "corpus_index": links[int(b)], "rating": float(r)}
           for b, r in zip(g["book_id"], g["rating"])]

for n in (20, 50, 100):
    recs, _ = ns["recommend_books"](ratings, df, emb, ix, n=n,
                                    exclude_rated_authors=False, per_author_cap=0,
                                    author_adjacent_penalty=0.0,
                                    genre_cap=10**9, per_anchor_cap=10**9)
    print(f"asked {n:>3} -> got {len(recs):>3}")