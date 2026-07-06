"""Smoke-test recommend_books and recommend_tv end-to-end (same harness style
as smoke_test.py for movies). Catches runtime errors py_compile misses and
asserts the core behaviors: small modes present, negatives suppressed,
quality floor (TV), genre caps, work-dedup (books)."""
import ast
import numpy as np
import pandas as pd
import faiss

src = open("app.py").read()
tree = ast.parse(src)
wanted = {"LANG_NAMES", "_TITLE_STOPWORDS"}
wanted_funcs = {"lang_label", "_title_tokens", "_is_title_dup", "_creator_names",
                "recommend_books", "recommend_tv"}
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

recommend_books = ns["recommend_books"]
recommend_tv = ns["recommend_tv"]
rng = np.random.default_rng(11)


def make_vec(axis, noise=0.18):
    v = axis + rng.normal(0, noise, 8)
    return v / np.linalg.norm(v)


AX = {k: np.eye(8)[i] for i, k in enumerate(
    ["crimeb", "romb", "sfb", "horrb", "crimet", "comt", "kdramat", "surrt"])}

# ============================ BOOKS ============================
rows, vecs = [], []
bid = 0
book_modes = [("crimeb", ["Fiction", "Mystery", "Thriller"]),
              ("romb",   ["Fiction", "Romance"]),
              ("sfb",    ["Fiction", "Science Fiction"]),
              ("horrb",  ["Fiction", "Horror"])]
for mname, genres in book_modes:
    for i in range(35):
        vecs.append(make_vec(AX[mname]))
        rows.append({"book_id": f"b{bid}", "title": f"{mname}_{i}",
                     "primary_author": f"auth_{mname}_{i}", "genres": genres,
                     "publication_year": 2000 + i % 20, "ratings_count": 50000,
                     "work_id": f"w{bid}", "source": "goodreads", "description": "d"})
        bid += 1
# duplicate-work pair (same work_id) to test dedup
vecs.append(make_vec(AX["crimeb"], 0.02)); rows.append({**rows[0], "book_id": "bdupA", "title": "dupwork_A", "work_id": "SAMEWORK", "primary_author": "dupauthA"})
vecs.append(make_vec(AX["crimeb"], 0.02)); rows.append({**rows[0], "book_id": "bdupB", "title": "dupwork_B", "work_id": "SAMEWORK", "primary_author": "dupauthB"})
bdf = pd.DataFrame(rows)
bemb = np.array(vecs, dtype="float32")
bindex = faiss.IndexFlatIP(8); bindex.add(bemb)

def brate(i, r):
    return {"book_id": bdf.iloc[i]["book_id"], "matched_title": bdf.iloc[i]["title"],
            "matched_author": "", "corpus_index": i, "rating": float(r)}
b_ratings = [brate(i, 5) for i in range(8)]            # crime-heavy
b_ratings += [brate(35, 4), brate(36, 4)]              # romance 4*
b_ratings += [brate(70, 4)]                            # sci-fi 4*
b_ratings += [brate(105, 1)]                           # horror 1* (dislike)

# simple synthetic tags: one tag column per mode, score 1 for its mode
tag_rows = []
for i, row in bdf.iterrows():
    mode = row["title"].split("_")[0].replace("dupwork", "crimeb")
    tag_rows.append({"book_id": str(row["book_id"]), "tag": mode, "score": 1.0})
btags = pd.DataFrame(tag_rows).pivot_table(index="book_id", columns="tag", values="score", fill_value=0.0)

recs, tags = recommend_books(b_ratings, bdf, bemb, bindex, tags_df=btags, n=20)
assert len(recs) > 0
print(f"BOOKS RAN OK — {len(recs)} recs")
bb = {}
for _, r in recs.iterrows():
    bb[r["title"].split("_")[0]] = bb.get(r["title"].split("_")[0], 0) + 1
print("  mode distribution:", bb)
assert any(m in bb for m in ("romb", "sfb")), "books small modes missing"
assert not any(t.startswith("horrb") for t in recs["title"]), "disliked horror surfaced"
# work dedup: at most one of the SAMEWORK pair
dup = [t for t in recs["title"] if t.startswith("dupwork")]
assert len(dup) <= 1, "work dedup broken"
print("  small modes present, dislike suppressed, work-dedup held")

# ============================ TV ============================
rows, vecs = [], []
tid = 0
tv_modes = [("crimet", "Crime, Drama"), ("comt", "Comedy"),
            ("kdramat", "Drama, Romance"), ("surrt", "Sci-Fi & Fantasy, Mystery")]
for mname, genres in tv_modes:
    for i in range(35):
        vecs.append(make_vec(AX[mname]))
        va = 4.5 if i % 11 == 0 else float(np.clip(rng.normal(7.4, 0.7), 6.1, 9.2))
        rows.append({"tmdb_id": tid, "name": f"{mname}_{i}", "genres": genres,
                     "created_by": f"cr_{mname}_{i}", "networks": "N",
                     "vote_count": 500, "vote_average": va, "start_year": 2015,
                     "number_of_seasons": 2, "original_language": "en",
                     "poster_path": "", "overview": "o"})
        tid += 1
tdf = pd.DataFrame(rows)
temb = np.array(vecs, dtype="float32")
tindex = faiss.IndexFlatIP(8); tindex.add(temb)

def trate(i, r):
    return {"tv_id": int(tdf.iloc[i]["tmdb_id"]), "matched_title": tdf.iloc[i]["name"],
            "matched_creator": tdf.iloc[i]["created_by"], "corpus_index": i, "rating": float(r)}
t_ratings = [trate(i, 5) for i in range(8)]            # crime-heavy
t_ratings += [trate(35, 4), trate(36, 4)]              # comedy 4*
t_ratings += [trate(70, 4)]                            # kdrama 4*
t_ratings += [trate(105, 1)]                           # surreal 1*

trecs = recommend_tv(t_ratings, tdf, temb, tindex, n=20)
assert len(trecs) > 0
print(f"TV RAN OK — {len(trecs)} recs")
tb = {}
for _, r in trecs.iterrows():
    tb[r["name"].split("_")[0]] = tb.get(r["name"].split("_")[0], 0) + 1
print("  mode distribution:", tb)
assert any(m in tb for m in ("comt", "kdramat")), "tv small modes missing"
assert not any(t.startswith("surrt") for t in trecs["name"]), "disliked surreal surfaced"
assert (trecs["vote_average"] >= 6.0).all(), "tv quality floor violated"
cm = sum(1 for _, r in trecs.iterrows() if {"Crime", "Mystery"} & set(r["genres"]))
print(f"  crime/mystery-family picks: {cm} (cap 6, water-fill may overflow when alternatives exhaust)")
# cap may only be exceeded if every other bucket is itself capped (water-fill)
assert cm <= 6 or (tb.get("comt", 0) >= 6 and tb.get("kdramat", 0) >= 6), \
    "tv genre family cap violated with uncapped alternatives available"
# creator exclusion: no show by a rated creator
rated_creators = {tdf.iloc[i]["created_by"] for i in [0,1,2,3,4,5,6,7,35,36,70,105]}
assert not any(c in rated_creators for c in trecs["creator"]), "rated creator leaked"
print("  small modes present, dislike suppressed, floor+cap+creator-exclusion held")

print("\nALL BOOKS+TV CHECKS PASSED")