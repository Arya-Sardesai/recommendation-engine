import pandas as pd
old = pd.read_parquet(r"data\processed\book_tags.parquet")
new = pd.read_parquet(r"data\processed\book_tags_knn.parquet")
print(old.columns.tolist()); print(old.dtypes)
print(new.columns.tolist()); print(new.dtypes)
print(len(old), len(new), new.iloc[0].to_dict())