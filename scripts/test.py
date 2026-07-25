import pandas as pd

for name, path in [("MOVIES", "data/processed/movies.parquet"),
                   ("TV", "data/processed/tv_corpus.parquet")]:
    df = pd.read_parquet(path)
    print(f"\n=== {name} ({len(df)} rows) ===")
    print("columns:", list(df.columns))
    date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower() or "air" in c.lower()]
    for c in date_cols:
        s = pd.to_datetime(df[c], errors="coerce") if df[c].dtype == object else df[c]
        print(f"\n{c}: max = {s.max()}")
        print(s.sort_values(ascending=False).head(15).to_string())