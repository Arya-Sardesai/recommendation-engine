"""Smoke-test recommend_movies end-to-end without running the Streamlit app.

Extracts the needed defs from app.py via AST, builds a synthetic corpus with
distinct modes (crime / musical / anime / bollywood / surreal), a rating
profile shaped like Arya's (crime-heavy 5*s, small 4* modes, a 1* surreal),
and checks:
  1. runs without error (would have caught the `kinship` NameError)
  2. small modes appear in the results
  3. crime-thriller family cap holds (<= genre_cap in first pass)
  4. no candidate below the vote_average floor
  5. nothing similar to the 1* dislike surfaces
"""
import ast
import numpy as np
import pandas as pd
import faiss

src = open("app.py").read()
tree = ast.parse(src)
wanted = {"LANG_NAMES", "_TITLE_STOPWORDS"}
wanted_funcs = {"lang_label", "_title_tokens", "_is_title_dup", "recommend_movies"}
ns = {"np": np, "pd": pd, "faiss": faiss}
for node in tree.body:
    take = False
    if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
        take = True
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in wanted:
                take = True
    if take:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), ns)

recommend_movies = ns["recommend_movies"]

# ---- synthetic corpus: 5 modes x 40 films, 8-dim unit vectors ----
rng = np.random.default_rng(7)
modes = {
    "crime":    (np.array([1,0,0,0,0,0,0,0.]), "Crime, Drama"),
    "musical":  (np.array([0,1,0,0,0,0,0,0.]), "Music, Comedy"),
    "anime":    (np.array([0,0,1,0,0,0,0,0.]), "Animation, Adventure"),
    "bolly":    (np.array([0,0,0,1,0,0,0,0.]), "Comedy, Romance"),
    "surreal":  (np.array([0,0,0,0,1,0,0,0.]), "Horror, Fantasy"),
}
rows, vecs = [], []
mid = 0
for mname, (axis, genres) in modes.items():
    for i in range(40):
        v = axis + rng.normal(0, 0.18, 8)
        v = v / np.linalg.norm(v)
        vecs.append(v)
        # a few deliberate duds (low vote_average) in each mode
        va = 4.2 if i % 13 == 0 else float(np.clip(rng.normal(7.0, 0.8), 5.6, 9.0))
        rows.append({
            "id": mid, "title": f"{mname}_{i}", "genres": genres,
            "vote_count": 1000, "vote_average": va, "release_year": 2000 + i % 20,
            "original_language": "en", "director": f"dir_{mname}_{i}",
            "cast": [f"actor{i}"], "studio": "S", "poster_path": "", "overview": "o",
        })
        mid += 1
# cross-pollinated candidates: bollywood films that also carry anime energy
for i in range(6):
    v = modes["bolly"][0] * 0.8 + modes["anime"][0] * 0.55 + rng.normal(0, 0.05, 8)
    v = v / np.linalg.norm(v)
    vecs.append(v)
    rows.append({"id": mid, "title": f"bolly_anime_{i}", "genres": "Comedy, Romance",
                 "vote_count": 1000, "vote_average": 7.5, "release_year": 2015,
                 "original_language": "hi", "director": f"dir_ba_{i}", "cast": ["a"],
                 "studio": "S", "poster_path": "", "overview": "o"})
    mid += 1
# musical films contaminated with surreal (the disliked quality)
for i in range(6):
    v = modes["musical"][0] * 0.75 + modes["surreal"][0] * 0.6 + rng.normal(0, 0.05, 8)
    v = v / np.linalg.norm(v)
    vecs.append(v)
    rows.append({"id": mid, "title": f"musical_surreal_{i}", "genres": "Music, Horror",
                 "vote_count": 1000, "vote_average": 7.5, "release_year": 2015,
                 "original_language": "en", "director": f"dir_ms_{i}", "cast": ["a"],
                 "studio": "S", "poster_path": "", "overview": "o"})
    mid += 1

df = pd.DataFrame(rows)
emb = np.array(vecs, dtype="float32")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# ---- ratings: crime-heavy 5*s, small 4* modes, one 1* surreal ----
def rate(i, r):
    return {"movie_id": int(df.iloc[i]["id"]), "matched_title": df.iloc[i]["title"],
            "matched_director": "", "corpus_index": i, "rating": float(r)}
ratings = []
for i in range(10):            # 10 crime films at 5*
    ratings.append(rate(i, 5))
ratings.append(rate(40, 5))    # musical 5* (Chicago-ish)
ratings.append(rate(41, 4))    # musical 4*
ratings.append(rate(80, 4))    # anime 4* x2
ratings.append(rate(81, 4))
ratings.append(rate(120, 4))   # bolly 4* x2
ratings.append(rate(121, 4))
ratings.append(rate(160, 1))   # surreal 1* (Eraserhead)

recs, tags = recommend_movies(ratings, df, emb, index, tags_df=None, n=20)
assert len(recs) > 0, "no recommendations produced"
print(f"RAN OK — {len(recs)} recs\n")

buckets = {}
for _, r in recs.iterrows():
    mode = r["title"].split("_")[0]
    buckets[mode] = buckets.get(mode, 0) + 1
print("mode distribution:", buckets)

# 2. small modes present
assert any(m in buckets for m in ("musical", "anime", "bolly")), "small modes missing!"
non_crime = sum(v for k, v in buckets.items() if k != "crime")
print(f"non-crime picks: {non_crime}/20")
assert non_crime >= 8, f"crime still flooding: only {non_crime} non-crime"

# 3. crime-family cap held
crime_family = sum(1 for _, r in recs.iterrows()
                   if {"Crime", "Thriller"} & {g.strip() for g in r["genres"]})
print(f"crime/thriller-family picks: {crime_family} (cap 6)")
assert crime_family <= 6, "genre family cap violated"

# 4. quality floor
assert (recs["vote_average"] >= 5.5).all(), "dud below floor surfaced"
print("quality floor held (min va = %.1f)" % recs["vote_average"].min())

# 5. dislike suppression: no surreal or musical_surreal picks
bad = [t for t in recs["title"] if t.startswith("surreal") or t.startswith("musical_surreal")]
print("dislike-adjacent picks:", bad or "none")
assert not bad, "disliked-quality candidates surfaced"

# 6. cross-pollination: bolly_anime hybrids should be able to appear/attribute
xp = recs[recs["title"].str.startswith("bolly_anime")]
if len(xp):
    print("cross-pollinated picks:", list(xp["title"]),
          "also_because:", list(xp["also_because"]))
print("\nALL CHECKS PASSED")