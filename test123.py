import pandas as pd
df = pd.read_parquet("data/processed/movies_v2_meta.parquet")
for t in ["Blade Runner", "Fight Club", "The Shining", "Dune",
          "Knives Out", "True Romance", "The Grand Budapest Hotel", "Collateral"]:
    r = df[df["title"] == t]
    if len(r):
        kw = str(r.iloc[0]["keywords_tmdb"])
        flag = "based on novel" in kw.lower() or "based on book" in kw.lower()
        print(f"  {t:28s} adaptation={str(flag):5s}  {kw[:70]}")