import pandas as pd, numpy as np, faiss, sys
sys.path.insert(0, "tests")
from ranker_loader import load_ranker, resolve_artifacts
from pathlib import Path

APP = Path("../hf-space/app.py")
ns  = load_ranker(APP, require=("recommend_movies",))
art = resolve_artifacts(APP, Path("data/processed"), kind="movie")
df  = pd.read_parquet(art["CORPUS"]).reset_index(drop=True)
emb = np.load(art["EMB"]).astype("float32")
ix  = faiss.read_index(str(art["FAISS"]))

tags = None
if "TAGS" in art and Path(art["TAGS"]).exists():
    t = pd.read_parquet(art["TAGS"])
    t["movie_id"] = t["movie_id"].astype(str)
    tags = t.pivot_table(index="movie_id", columns="tag", values="score", fill_value=0.0)

rt    = pd.read_csv("data/raw/movies/ml-25m/ratings.csv", nrows=500_000)
links = pd.read_csv("data/raw/movies/ml-25m/links.csv")
tmdb  = dict(zip(links["movieId"], links["tmdbId"]))
idx   = {str(i): p for p, i in enumerate(df["id"].astype(str))}

# find a user with enough mapped ratings to be worth ranking
for uid, g in rt.groupby("userId"):
    ratings = []
    for m, r in zip(g["movieId"], g["rating"]):
        tid = tmdb.get(m)
        if pd.isna(tid):
            continue
        key = str(int(tid))
        if key in idx:
            p = idx[key]
            ratings.append({"movie_id": key, "corpus_index": p, "rating": float(r),
                            "matched_title": df.iloc[p]["title"],
                            "matched_director": df.iloc[p].get("director", "")})
    if len(ratings) >= 25:
        print(f"user {uid}: {len(ratings)} mapped ratings")
        break

kw = dict(n=50, exclude_rated_directors=False, genre_cap=10**9, tags_df=tags)
a, _ = ns["recommend_movies"](ratings, df, emb, ix, **kw)
b, _ = ns["recommend_movies"](ratings, df, emb, ix, min_runtime=0, kids_filter=False, **kw)

print("identical top-50:", list(a["id"]) == list(b["id"]))
extra = [t for t in b["title"] if t not in set(a["title"])]
print("in 'both OFF' but not shipped:", extra[:8] or "none")