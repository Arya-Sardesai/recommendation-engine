import pandas as pd, zipfile

zip_path = "data/raw/archive (1).zip"

with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    print("files in zip:", names)
    csv_name = [n for n in names if n.lower().endswith(".csv")][0]
    with z.open(csv_name) as f:
        df = pd.read_csv(f)

print("shape:", df.shape)
print("columns:", df.columns.tolist())

gate = ["id","name","overview","imdb_id","created_by","networks",
        "keywords","genres","first_air_date","original_language",
        "number_of_seasons","number_of_episodes","vote_count","vote_average"]
print("\npresence / null%:")
for c in gate:
    if c in df.columns:
        print(f"  {c:20s} PRESENT   null={df[c].isna().mean()*100:5.1f}%")
    else:
        print(f"  {c:20s} MISSING")

print("\n--- sample values (format check) ---")
for c in ["overview","created_by","networks","genres","keywords"]:
    if c in df.columns:
        v = df[c].dropna().iloc[0] if df[c].notna().any() else "(all null)"
        print(f"{c}: {str(v)[:220]}")