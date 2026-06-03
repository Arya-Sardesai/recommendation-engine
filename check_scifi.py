import pandas as pd
master = pd.read_parquet('data/processed/book_tags.parquet')
df = pd.read_parquet('data/processed/books_v1.parquet')
df['book_id'] = df['book_id'].astype(str)
tagged_ids = set(master['book_id'].astype(str))

keywords = ['neuromancer', 'foundation', 'dune', 'hyperion', 'snow crash', 'three-body', 'children of time', 'leviathan wakes', 'expanse', 'martian', 'kindred', 'left hand of darkness', 'parable of', 'altered carbon']

for kw in keywords:
    hits = df[df['title'].str.contains(kw, case=False, na=False)]
    tagged_hits = hits[hits['book_id'].isin(tagged_ids)]
    if len(tagged_hits) > 0:
        titles = tagged_hits['title'].head(3).tolist()
        print(kw, ':', len(tagged_hits), 'tagged. Titles:', titles)
