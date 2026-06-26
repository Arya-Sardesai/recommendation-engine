import pandas as pd
df = pd.read_parquet('data/processed/books_v1.parquet')
queries = ['convenience store woman', 'yellowface', 'morisaki bookshop', 'days at the morisaki']
for q in queries:
    hits = df[df['title'].str.contains(q, case=False, na=False, regex=False)]
    print(f'\n{q}: {len(hits)} hits')
    if len(hits) > 0:
        print(hits[['book_id', 'title', 'primary_author', 'publication_year', 'ratings_count', 'source']].head(5).to_string(index=False))
