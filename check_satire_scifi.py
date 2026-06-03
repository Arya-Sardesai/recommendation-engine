import pandas as pd
master = pd.read_parquet('data/processed/book_tags.parquet')
df = pd.read_parquet('data/processed/books_v1.parquet')
df['book_id'] = df['book_id'].astype(str)
candidates = pd.read_csv('data/processed/scifi_candidates_for_tagging.csv')
candidate_ids = set(candidates['book_id'].astype(str))
tagged_ids = set(master['book_id'].astype(str))

queries = ['hitchhiker', 'slaughterhouse', 'cat\'s cradle', 'vonnegut', 'sirens of titan']
for q in queries:
    hits = df[df['title'].str.contains(q, case=False, na=False)]
    tagged_hits = hits[hits['book_id'].isin(tagged_ids)]
    for _, row in tagged_hits.iterrows():
        in_cand = row['book_id'] in candidate_ids
        print(row['title'][:60], '| tagged:', True, '| in candidates:', in_cand)
