import pandas as pd
b = pd.read_parquet("data/processed/books_v2_hc.parquet")
bad = b["primary_author"].str.contains("©|Ã|â€|Â", regex=True, na=False)
print(bad.sum(), "authors with mojibake")
print(b.loc[bad, "primary_author"].value_counts().head(15))