"""
Test the v1 corpus: re-match the sample ratings, generate recommendations,
and report how many recs come from the new Hardcover books.
Run from repo root.
"""
from pathlib import Path
import json
import gzip

import numpy as np
import pandas as pd
import faiss
from rapidfuzz import fuzz, process

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

df = pd.read_parquet(PROCESSED_DIR / "books_v1.parquet").reset_index(drop=True)
embeddings = np.load(PROCESSED_DIR / "embeddings_minilm_v1.npy").astype("float32")
index = faiss.read_index(str(PROCESSED_DIR / "faiss_minilm_v1.index"))
print(f"v1 corpus: {len(df):,} books ({(df['source']=='hardcover').sum():,} Hardcover)")

# ratings
ratings_df = pd.read_csv("data/my_ratings.csv")
for col in ["title", "author", "notes"]:
    if col in ratings_df.columns:
        ratings_df[col] = ratings_df[col].astype(str).str.strip()

def find_best_match(qt, qa, corpus_df, title_threshold=80, author_threshold=70):
    titles = corpus_df["title"].fillna("").tolist()
    tm = process.extract(qt, titles, scorer=fuzz.WRatio, limit=20)
    if not tm or tm[0][1] < title_threshold:
        return None
    best, best_score = None, 0
    for matched_title, ts, idx in tm:
        cand = corpus_df.iloc[idx]
        a_s = fuzz.WRatio(qa, cand["primary_author"]) if qa else 100
        combined = ts * 0.6 + a_s * 0.4
        if combined > best_score and a_s >= author_threshold:
            best_score = combined
            best = {"book_id": cand["book_id"], "matched_title": matched_title,
                    "matched_author": cand["primary_author"], "corpus_index": int(idx),
                    "title_score": ts, "author_score": a_s, "source": cand["source"]}
    return best

print("\n=== Matching ratings against v1 corpus ===")
matched, unmatched = [], []
for _, row in ratings_df.iterrows():
    r = find_best_match(row["title"], row["author"], df)
    if r:
        r["rating"] = row["rating"]
        matched.append(r)
        tag = "[HC]" if r["source"] == "hardcover" else ""
        print(f"  OK  {row['title'][:32]:32s} -> {r['matched_title'][:38]:38s} {tag}")
    else:
        unmatched.append(row["title"])
        print(f"  --  {row['title'][:32]:32s} -> NO MATCH")
print(f"\nMatched: {len(matched)}/{len(ratings_df)}  (was 21/28 on v0)")
if unmatched:
    print("Still unmatched:", unmatched)

# recommend (round-robin)
def recommend(ratings, df, embeddings, index, n=20, min_ratings=2000, like_threshold=3.5, per_book=2):
    liked = sorted([r for r in ratings if r["rating"] >= like_threshold], key=lambda x: -x["rating"])
    if not liked:
        return pd.DataFrame()
    lv = np.array([embeddings[r["corpus_index"]] for r in liked]).astype("float32")
    rated_ids = {r["book_id"] for r in ratings}
    rated_authors = {r["matched_author"].lower() for r in ratings if r.get("matched_author")}
    k = 100
    sims, idxs = index.search(lv, k)
    queues = []
    for i, (ss, ii) in enumerate(zip(sims, idxs)):
        q = []
        for s, ix in zip(ss, ii):
            b = df.iloc[ix]
            if b["book_id"] in rated_ids: continue
            if b["primary_author"].lower() in rated_authors: continue
            if b["ratings_count"] < min_ratings: continue
            q.append((int(ix), float(s)))
        queues.append({"like": liked[i], "queue": q})
    results, seen, seen_w = [], set(), set()
    while len(results) < n:
        added = 0
        for pbq in queues:
            taken = 0
            while pbq["queue"] and taken < per_book:
                ix, s = pbq["queue"].pop(0)
                b = df.iloc[ix]
                w = b["work_id"]
                if ix in seen or (w and w in seen_w): continue
                seen.add(ix)
                if w: seen_w.add(w)
                results.append({"title": b["title"], "author": b["primary_author"],
                                "year": b["publication_year"], "source": b["source"],
                                "ratings_count": int(b["ratings_count"]),
                                "because_of": pbq["like"]["matched_title"]})
                taken += 1; added += 1
                if len(results) >= n: break
            if len(results) >= n: break
        if added == 0: break
    return pd.DataFrame(results)

print("\n=== Top 20 recommendations (v1 corpus) ===")
recs = recommend(matched, df, embeddings, index, n=20)
for _, r in recs.iterrows():
    tag = "[HC NEW]" if r["source"] == "hardcover" else ""
    yr = int(r["year"]) if pd.notna(r["year"]) else "?"
    print(f"  {r['title'][:42]:42s} ({yr}) by {r['author'][:20]:20s} {tag}  <- {r['because_of'][:25]}")

hc_count = (recs["source"] == "hardcover").sum()
print(f"\nRecs from new Hardcover books: {hc_count}/20")