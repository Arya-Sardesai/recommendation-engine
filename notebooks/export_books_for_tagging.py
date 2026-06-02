"""
Export a CSV of books to tag via chat (title + description + existing genres).
Pick top-N most popular books that don't already have rich metadata.

Run from repo root.
"""
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
OUTPUT = Path("data/processed/books_for_tagging.csv")

# How many books to export per batch
N = 100

print("Loading corpus...")
df = pd.read_parquet(PROCESSED / "books_v1.parquet")
print(f"  {len(df):,} books total")

# Skip already-tagged books
MASTER_TAGS = PROCESSED / "book_tags.parquet"
if MASTER_TAGS.exists():
    already_tagged = set(pd.read_parquet(MASTER_TAGS)["book_id"].astype(str))
    print(f"  {len(already_tagged):,} books already tagged, will skip")
    df = df[~df["book_id"].astype(str).isin(already_tagged)]

# Filter: must have a description (tags need it), must be recommendable
candidates = df[
    (df["description"].notna()) &
    (df["description"].str.len() >= 100) &  # substantive description
    (df["ratings_count"] >= 2000)  # recommendable threshold
].copy()
print(f"  {len(candidates):,} books with description & recommendable")

# Sort by popularity, take top N
top = candidates.nlargest(N, "ratings_count")

# Clean up genres list -> string
top["genres_str"] = top["genres"].apply(
    lambda g: ", ".join(g) if hasattr(g, "__iter__") and not isinstance(g, str) else ""
)

# Trim description for chat readability (keep first 500 chars)
top["description_short"] = top["description"].str.slice(0, 500)

# Export
export = top[["book_id", "title", "primary_author", "publication_year",
              "genres_str", "description_short"]].rename(
    columns={"primary_author": "author", "description_short": "description"}
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
export.to_csv(OUTPUT, index=False)
print(f"\nExported {len(export)} books to: {OUTPUT}")
print(f"Sample:")
print(export.head(3).to_string())