# notebooks/pipeline/merge_tagged_batch.py
"""
Append a new tagged batch to the master book_tags.parquet.
Idempotent — re-running with the same batch won't duplicate.
"""
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
MASTER = PROCESSED / "book_tags.parquet"

# Path to the batch CSV you just got from chat
BATCH = Path("data/processed/book_tagged_batch_014.csv")  # adjust per batch

print(f"Loading batch: {BATCH.name}")
new_batch = pd.read_csv(BATCH)
print(f"  {len(new_batch):,} (book_id, tag, score) rows")
print(f"  {new_batch['book_id'].nunique():,} books tagged in this batch")

# Validate format
required = {"book_id", "tag", "score"}
missing = required - set(new_batch.columns)
if missing:
    raise ValueError(f"Batch missing columns: {missing}")

# Coerce types
new_batch["book_id"] = new_batch["book_id"].astype(str)
new_batch["tag"] = new_batch["tag"].astype(str)
new_batch["score"] = new_batch["score"].astype(float)

if MASTER.exists():
    master = pd.read_parquet(MASTER)
    print(f"\nExisting master: {len(master):,} rows, {master['book_id'].nunique():,} books")
    combined = pd.concat([master, new_batch], ignore_index=True)
    # If same (book_id, tag) appears in old and new batch, keep the NEWEST (last)
    combined = combined.drop_duplicates(subset=["book_id", "tag"], keep="last")
else:
    combined = new_batch

combined.to_parquet(MASTER, index=False)
print(f"\nMaster after merge: {len(combined):,} rows, {combined['book_id'].nunique():,} books tagged")