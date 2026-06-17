import pandas as pd, json
from pathlib import Path
from collections import Counter
# re-collect dropped tags by re-parsing isn't needed — just inspect what we have:
v2 = pd.read_parquet("data/processed/book_tags_v2.parquet")
print("books:", v2.book_id.nunique(), "| rows:", len(v2), "| distinct tags:", v2.tag.nunique())
print("\ntags per book distribution:")
print(v2.groupby("book_id").size().describe())
print("\nany book with very few tags?", (v2.groupby("book_id").size() <= 2).sum(), "books have <=2 tags")