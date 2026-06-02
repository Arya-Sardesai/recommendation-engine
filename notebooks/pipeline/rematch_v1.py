"""
Regenerate my_matched_ratings.json against the v1 corpus (indices changed).
Run from repo root.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

PROCESSED_DIR = Path("data/processed")
df = pd.read_parquet(PROCESSED_DIR / "books_v1.parquet").reset_index(drop=True)

ratings_df = pd.read_csv("data/my_ratings.csv")
for col in ["title", "author", "notes"]:
    if col in ratings_df.columns:
        ratings_df[col] = ratings_df[col].astype(str).str.strip()

def find_best_match(qt, qa, corpus_df, tt=80, at=70):
    titles = corpus_df["title"].fillna("").tolist()
    tm = process.extract(qt, titles, scorer=fuzz.WRatio, limit=20)
    if not tm or tm[0][1] < tt:
        return None
    best, bs = None, 0
    for mt, ts, idx in tm:
        c = corpus_df.iloc[idx]
        a_s = fuzz.WRatio(qa, c["primary_author"]) if qa else 100
        comb = ts*0.6 + a_s*0.4
        if comb > bs and a_s >= at:
            bs = comb
            best = {"book_id": c["book_id"], "matched_title": mt,
                    "matched_author": c["primary_author"], "corpus_index": int(idx)}
    return best

matched = []
for _, row in ratings_df.iterrows():
    r = find_best_match(row["title"], row["author"], df)
    if r:
        r["rating"] = float(row["rating"])
        matched.append(r)

out = PROCESSED_DIR / "my_matched_ratings.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(matched, f, indent=2)
print(f"Saved {len(matched)} matched ratings to {out}")