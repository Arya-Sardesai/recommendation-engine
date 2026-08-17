import pandas as pd
b = pd.read_parquet("data/processed/books_v2_hc.parquet")
print(len(b), "rows")
print(list(b.columns))
print("work_id present:", "work_id" in b.columns)
if "work_id" in b.columns:
    print("work_id == book_id on:", (b["work_id"].astype(str) == b["book_id"].astype(str)).mean())
    print(b[["book_id","work_id","title"]].head())