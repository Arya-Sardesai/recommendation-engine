"""
rescale_v1.py — scale Hardcover users_count to be comparable to Goodreads ratings_count.
Scale factor 25 (settled value: tested 15/25/40/60, robust to choice).
Run from repo root: python notebooks/pipeline/rescale_v1.py
"""
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
SCALE = 25

print("Loading books_v1.parquet...")
df = pd.read_parquet(PROCESSED / "books_v1.parquet")

hc_mask = df["source"] == "hardcover"
print(f"Hardcover rows: {hc_mask.sum():,}")

df.loc[hc_mask, "ratings_count"] = (df.loc[hc_mask, "ratings_count"] * SCALE).astype(int)

print(f"After scaling (sample HC ratings_count range): {df.loc[hc_mask, 'ratings_count'].min():,} - {df.loc[hc_mask, 'ratings_count'].max():,}")

df.to_parquet(PROCESSED / "books_v1.parquet", index=False)
print("Saved books_v1.parquet with scaled ratings_count.")
